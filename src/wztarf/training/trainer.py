"""Train WZ-TARF with supervised trajectory, route, topology, and safety losses."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from wztarf.evaluation.metrics_runner import compute_all_metrics
from wztarf.losses.supervised import (
    LossWeights,
    supervised_loss,
)
from wztarf.reporting.run_logger import RunLogger
from wztarf.training.checkpointing import (
    CheckpointState,
    load_checkpoint,
    save_checkpoint,
)


def _move_to_device(
    value: Any,
    device: torch.device,
) -> Any:
    """Recursively move tensors to a device while preserving metadata."""
    if isinstance(
        value,
        torch.Tensor,
    ):
        return value.to(
            device=device,
            non_blocking=True,
        )

    if isinstance(
        value,
        Mapping,
    ):
        return {
            key: _move_to_device(
                item,
                device,
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            _move_to_device(
                item,
                device,
            )
            for item in value
        ]

    if isinstance(
        value,
        tuple,
    ):
        return tuple(
            _move_to_device(
                item,
                device,
            )
            for item in value
        )

    return value


def _current_learning_rate(
    optimizer: torch.optim.Optimizer,
) -> float:
    """Return the learning rate of the first optimizer parameter group."""
    if not optimizer.param_groups:
        return float("nan")

    return float(
        optimizer.param_groups[0]["lr"]
    )


class Trainer:
    """Coordinate supervised WZ-TARF optimization, validation, and checkpoints."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_weights: LossWeights,
        device: str | torch.device,
        checkpoint_dir: str | Path,
        scheduler: Any | None = None,
        logger: RunLogger | None = None,
        config: Mapping[str, Any] | None = None,
        beta_assign: float = 0.25,
        classification_temperature: float = 1.0,
        fps: int = 5,
        dynamics_horizon_steps: int = 10,
        worker_threshold_m: float = 2.0,
        wz_temperature_m: float = 0.25,
        road_tolerance_m: float = 0.25,
        diversity_separation_m: float = 1.0,
        goal_association_tolerance_m: float = 0.25,
        road_gt_tolerance_m: float = 0.25,
        grad_clip_norm: float | None = 5.0,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        scheduler_metric: str | None = None,
    ) -> None:
        """Create a supervised trainer.

        `scheduler_metric` controls scheduler stepping:

        - None: call `scheduler.step()` once per epoch.
        - metric name: call `scheduler.step(validation_metrics[name])`.

        This avoids embedding assumptions about a specific scheduler class.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_weights = loss_weights

        self.device = torch.device(
            device
        )

        self.model.to(
            self.device
        )

        self.checkpoint_dir = Path(
            checkpoint_dir
        ).expanduser()

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.logger = logger
        self.config = (
            dict(config)
            if config is not None
            else {}
        )

        self.beta_assign = beta_assign
        self.classification_temperature = classification_temperature
        self.fps = fps
        self.dynamics_horizon_steps = dynamics_horizon_steps
        self.worker_threshold_m = worker_threshold_m
        self.wz_temperature_m = wz_temperature_m
        self.road_tolerance_m = road_tolerance_m
        self.diversity_separation_m = diversity_separation_m

        self.goal_association_tolerance_m = (
            goal_association_tolerance_m
        )
        self.road_gt_tolerance_m = (
            road_gt_tolerance_m
        )

        self.grad_clip_norm = grad_clip_norm
        self.scheduler_metric = scheduler_metric

        if (
            grad_clip_norm is not None
            and grad_clip_norm <= 0
        ):
            raise ValueError(
                "grad_clip_norm must be positive or None."
            )

        self.use_amp = (
            bool(
                use_amp
            )
            and
            self.device.type == "cuda"
        )

        self.amp_dtype = amp_dtype

        # Gradient scaling is useful for CUDA float16 training. BF16 does not
        # normally require scaling.
        self.scaler = None

        if (
            self.use_amp
            and
            self.device.type == "cuda"
            and
            self.amp_dtype == torch.float16
        ):
            self.scaler = torch.amp.GradScaler(
                "cuda"
            )

        self.global_step = 0

    def _compute_loss(
        self,
        model_output: Mapping[str, Any],
        batch: Mapping[str, Any],
    ):
        """Call the central supervised loss assembly."""
        return supervised_loss(
            model_output,
            batch,
            weights=self.loss_weights,
            beta_assign=self.beta_assign,
            classification_temperature=self.classification_temperature,
            fps=self.fps,
            dynamics_horizon_steps=self.dynamics_horizon_steps,
            worker_threshold_m=self.worker_threshold_m,
            wz_temperature_m=self.wz_temperature_m,
            road_tolerance_m=self.road_tolerance_m,
            diversity_separation_m=self.diversity_separation_m,
            goal_association_tolerance_m=(
                self.goal_association_tolerance_m
            ),
            road_gt_tolerance_m=self.road_gt_tolerance_m,
        )

    def train_epoch(
        self,
        dataloader: DataLoader,
        *,
        epoch: int,
    ) -> dict[str, float]:
        """Train for one complete epoch."""
        self.model.train()

        running: dict[str, float] = defaultdict(
            float
        )

        num_batches = 0

        for batch in dataloader:
            batch = _move_to_device(
                batch,
                self.device,
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                model_output = self.model(
                    batch
                )

                loss_output = self._compute_loss(
                    model_output,
                    batch,
                )

                loss = loss_output.total

            if not bool(
                torch.isfinite(
                    loss
                ).item()
            ):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch}, "
                    f"global step {self.global_step}."
                )

            if self.scaler is not None:
                self.scaler.scale(
                    loss
                ).backward()

                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(
                        self.optimizer
                    )

                    clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.grad_clip_norm,
                        error_if_nonfinite=True,
                    )

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

            else:
                loss.backward()

                if self.grad_clip_norm is not None:
                    clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.grad_clip_norm,
                        error_if_nonfinite=True,
                    )

                self.optimizer.step()

            self.global_step += 1
            num_batches += 1

            running["loss"] += float(
                loss.detach().item()
            )

            for name, value in loss_output.components.items():
                running[
                    f"loss_{name}"
                ] += float(
                    value.detach().item()
                )

        if num_batches == 0:
            raise ValueError(
                "Training DataLoader produced zero batches."
            )

        result = {
            key: value / num_batches
            for key, value in running.items()
        }

        result["learning_rate"] = _current_learning_rate(
            self.optimizer
        )

        return result

    @torch.inference_mode()
    def validate(
        self,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """Run validation loss and the complete forecasting metric suite."""
        self.model.eval()

        running_loss: dict[str, float] = defaultdict(
            float
        )

        pred_batches: list[torch.Tensor] = []
        probability_batches: list[torch.Tensor] = []
        gt_batches: list[torch.Tensor] = []

        wz_batches: list[torch.Tensor] = []
        worker_batches: list[torch.Tensor] = []

        num_batches = 0

        for batch in dataloader:
            batch = _move_to_device(
                batch,
                self.device,
            )

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                output = self.model(
                    batch
                )

                loss_output = self._compute_loss(
                    output,
                    batch,
                )

            running_loss["val_loss"] += float(
                loss_output.total.detach().item()
            )

            for name, value in loss_output.components.items():
                running_loss[
                    f"val_loss_{name}"
                ] += float(
                    value.detach().item()
                )

            pred_batches.append(
                output["pred_xy"]
                .detach()
                .float()
                .cpu()
            )

            probability_batches.append(
                output["mode_prob"]
                .detach()
                .float()
                .cpu()
            )

            gt_batches.append(
                batch["future_xy"]
                .detach()
                .float()
                .cpu()
            )

            if "wz_feat" in batch:
                wz_batches.append(
                    batch["wz_feat"]
                    .detach()
                    .float()
                    .cpu()
                )

            if "wz_worker_feat" in batch:
                worker_batches.append(
                    batch["wz_worker_feat"]
                    .detach()
                    .float()
                    .cpu()
                )

            num_batches += 1

        if num_batches == 0:
            raise ValueError(
                "Validation DataLoader produced zero batches."
            )

        validation = {
            key: value / num_batches
            for key, value in running_loss.items()
        }

        pred_xy = torch.cat(
            pred_batches,
            dim=0,
        )

        mode_prob = torch.cat(
            probability_batches,
            dim=0,
        )

        gt_xy = torch.cat(
            gt_batches,
            dim=0,
        )

        wz_feat = (
            torch.cat(
                wz_batches,
                dim=0,
            )
            if wz_batches
            else None
        )

        worker_feat = (
            torch.cat(
                worker_batches,
                dim=0,
            )
            if worker_batches
            else None
        )

        metrics = compute_all_metrics(
            pred_xy=pred_xy,
            gt_xy=gt_xy,
            mode_prob=mode_prob,
            wz_feat=wz_feat,
            worker_feat=worker_feat,
            fps=self.fps,
            miss_threshold_m=2.0,
            worker_threshold_m=self.worker_threshold_m,
        )

        validation.update(
            metrics
        )

        return validation

    def _step_scheduler(
        self,
        validation_metrics: Mapping[str, float],
    ) -> None:
        """Advance the configured scheduler once per epoch."""
        if self.scheduler is None:
            return

        if self.scheduler_metric is None:
            self.scheduler.step()
            return

        if self.scheduler_metric not in validation_metrics:
            raise KeyError(
                f"Scheduler metric '{self.scheduler_metric}' "
                "was not produced during validation."
            )

        self.scheduler.step(
            validation_metrics[
                self.scheduler_metric
            ]
        )

    def resume(
        self,
        checkpoint_path: str | Path,
        *,
        strict: bool = True,
    ) -> CheckpointState:
        """Restore model, optimizer, scheduler, scaler, and global step."""
        state = load_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location=self.device,
            strict=strict,
        )

        self.global_step = state.global_step

        return state

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        epochs: int,
        start_epoch: int = 1,
        selection_metric: str = "minADE_6",
        selection_mode: str = "min",
        patience: int | None = None,
        resume_from: str | Path | None = None,
    ) -> dict[str, Any]:
        """Train, validate, checkpoint, and early-stop the supervised model.

        Args:
            selection_metric:
                Validation metric used to select `best.pt`.

            selection_mode:
                `"min"` for error/loss metrics and `"max"` for metrics where
                larger is better.

            patience:
                Stop after this many consecutive non-improving epochs.
                `None` disables early stopping.

        Returns:
            Final run summary.
        """
        if epochs <= 0:
            raise ValueError(
                "epochs must be positive."
            )

        if selection_mode not in {
            "min",
            "max",
        }:
            raise ValueError(
                "selection_mode must be 'min' or 'max'."
            )

        if (
            patience is not None
            and patience <= 0
        ):
            raise ValueError(
                "patience must be positive or None."
            )

        best_metric: float | None = None
        best_epoch: int | None = None
        bad_epochs = 0

        if resume_from is not None:
            state = self.resume(
                resume_from
            )

            start_epoch = max(
                start_epoch,
                state.epoch + 1,
            )

            best_metric = state.best_metric

        start_time = time.perf_counter()

        if self.logger is not None:
            self.logger.log(
                f"Starting supervised training at epoch {start_epoch}."
            )

        last_epoch = start_epoch - 1
        final_validation: dict[str, float] = {}

        for epoch in range(
            start_epoch,
            epochs + 1,
        ):
            last_epoch = epoch

            train_metrics = self.train_epoch(
                train_loader,
                epoch=epoch,
            )

            validation_metrics = self.validate(
                val_loader
            )

            final_validation = validation_metrics

            if selection_metric not in validation_metrics:
                raise KeyError(
                    f"Selection metric '{selection_metric}' "
                    "was not produced during validation."
                )

            current_metric = float(
                validation_metrics[
                    selection_metric
                ]
            )

            if not math.isfinite(
                current_metric
            ):
                raise FloatingPointError(
                    f"Selection metric '{selection_metric}' "
                    f"is non-finite at epoch {epoch}."
                )

            if best_metric is None:
                improved = True

            elif selection_mode == "min":
                improved = (
                    current_metric
                    <
                    best_metric
                )

            else:
                improved = (
                    current_metric
                    >
                    best_metric
                )

            if improved:
                best_metric = current_metric
                best_epoch = epoch
                bad_epochs = 0

            else:
                bad_epochs += 1

            # Step before checkpointing so resumed training restores the
            # scheduler state corresponding to the completed epoch.
            self._step_scheduler(
                validation_metrics
            )

            if improved:
                save_checkpoint(
                    self.checkpoint_dir
                    /
                    "best.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch,
                    global_step=self.global_step,
                    best_metric=best_metric,
                    config=self.config,
                    extra={
                        "selection_metric": selection_metric,
                        "validation_metrics": dict(
                            validation_metrics
                        ),
                    },
                )

            # Always maintain a resumable last-state checkpoint.
            save_checkpoint(
                self.checkpoint_dir
                /
                "last.pt",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch,
                global_step=self.global_step,
                best_metric=best_metric,
                config=self.config,
                extra={
                    "selection_metric": selection_metric,
                    "validation_metrics": dict(
                        validation_metrics
                    ),
                },
            )

            if self.logger is not None:
                self.logger.log_metrics(
                    train_metrics,
                    split="train",
                    epoch=epoch,
                    step=self.global_step,
                )

                self.logger.log_metrics(
                    validation_metrics,
                    split="val",
                    epoch=epoch,
                    step=self.global_step,
                )

                self.logger.log(
                    f"Epoch {epoch}: "
                    f"{selection_metric}={current_metric:.6f}, "
                    f"best={best_metric:.6f}, "
                    f"lr={_current_learning_rate(self.optimizer):.8g}"
                )

            if (
                patience is not None
                and bad_epochs >= patience
            ):
                if self.logger is not None:
                    self.logger.log(
                        f"Early stopping after epoch {epoch}."
                    )

                break

        duration_hours = (
            time.perf_counter()
            -
            start_time
        ) / 3600.0

        summary: dict[str, Any] = {
            "last_epoch": last_epoch,
            "best_epoch": best_epoch,
            "best_metric_name": selection_metric,
            "best_metric": best_metric,
            "global_step": self.global_step,
            "training_duration_hours": duration_hours,
            "best_checkpoint": str(
                (
                    self.checkpoint_dir
                    /
                    "best.pt"
                ).resolve()
            ),
            "last_checkpoint": str(
                (
                    self.checkpoint_dir
                    /
                    "last.pt"
                ).resolve()
            ),
            "final_validation": final_validation,
        }

        if self.logger is not None:
            self.logger.save_run_summary(
                summary
            )

            self.logger.log(
                "Supervised training finished."
            )

        return summary
