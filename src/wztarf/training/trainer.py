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

from wztarf.evaluation.metrics_runner import (
    compute_all_metrics,
    compute_grouped_forecasting_metrics,
)
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




def _metadata_from_batch(
    batch: Mapping[str, Any],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Normalize collated metadata and preserve source path for grouping."""
    raw = batch.get(
        "meta"
    )

    if isinstance(raw, list):
        if len(raw) != batch_size:
            raise ValueError(
                "Metadata list length does not match batch size."
            )

        result = [
            dict(item)
            if isinstance(item, Mapping)
            else {"value": item}
            for item in raw
        ]

    elif isinstance(raw, Mapping):
        result = []

        for index in range(batch_size):
            item: dict[str, Any] = {}

            for key, value in raw.items():
                if isinstance(value, torch.Tensor):
                    selected = value[index]

                    item[key] = (
                        selected.detach().cpu().item()
                        if selected.numel() == 1
                        else selected.detach().cpu().tolist()
                    )

                elif isinstance(
                    value,
                    (list, tuple),
                ):
                    item[key] = value[index]

                else:
                    item[key] = value

            result.append(
                item
            )

    else:
        result = [
            {}
            for _ in range(batch_size)
        ]

    source = batch.get(
        "source_path"
    )

    if isinstance(
        source,
        (list, tuple),
    ):
        if len(source) == batch_size:
            for index, value in enumerate(
                source
            ):
                if value is not None:
                    result[index][
                        "source_path"
                    ] = str(value)

    elif source is not None:
        for item in result:
            item[
                "source_path"
            ] = str(source)

    return result

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
        amp_dtype: torch.dtype = torch.bfloat16,
        scheduler_metric: str | None = None,
        composite_fde_weight: float = 0.25,
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

        self.goal_association_tolerance_m = (
            goal_association_tolerance_m
        )
        self.road_gt_tolerance_m = (
            road_gt_tolerance_m
        )

        self.grad_clip_norm = grad_clip_norm
        self.scheduler_metric = scheduler_metric
        self.composite_fde_weight = float(
            composite_fde_weight
        )

        if (
            not math.isfinite(
                self.composite_fde_weight
            )
            or self.composite_fde_weight < 0.0
        ):
            raise ValueError(
                "composite_fde_weight must be finite and non-negative."
            )

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
            self.scaler = torch.cuda.amp.GradScaler()
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


    def _first_nonfinite_gradient(
        self,
    ) -> str:
        """Describe the first named parameter containing NaN/Inf gradient."""
        for name, parameter in self.model.named_parameters():
            gradient = parameter.grad

            if gradient is None:
                continue

            finite = torch.isfinite(
                gradient
            )

            if bool(
                finite.all()
            ):
                continue

            bad_count = int(
                (~finite).sum().detach().cpu()
            )

            module_name = (
                name.rsplit(
                    ".",
                    1,
                )[0]
                if "." in name
                else "<root>"
            )

            return (
                f"parameter={name} | "
                f"module={module_name} | "
                f"shape={tuple(gradient.shape)} | "
                f"nonfinite={bad_count}/{gradient.numel()}"
            )

        return "parameter=<not-found>"


    def calibrate_loss_gradients(
        self,
        dataloader: DataLoader,
        *,
        max_batches: int = 16,
        target_endpoint_gradient_ratio: float = 0.5,
    ) -> dict[str, Any]:
        """Measure trajectory-vs-endpoint gradient magnitudes without updates.

        The target endpoint gradient ratio is 0.5 because the final metric
        thresholds are ADE < 0.8 m and FDE < 1.6 m:

            (FDE coefficient) / (ADE coefficient)
            = (1 / 1.6) / (1 / 0.8)
            = 0.5

        This diagnostic then corrects that coefficient for the native gradient
        magnitudes of the two losses.

        No optimizer step is performed.
        """

        if max_batches <= 0:
            raise ValueError("max_batches must be positive.")

        was_training = self.model.training
        self.model.train()

        named_parameters = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]

        if not named_parameters:
            raise RuntimeError("Model has no trainable parameters.")

        parameter_names = [
            name
            for name, _ in named_parameters
        ]

        parameters = [
            parameter
            for _, parameter in named_parameters
        ]

        decoder_indices = [
            index
            for index, name in enumerate(parameter_names)
            if name.startswith("direct_trajectory_decoder.")
        ]

        shared_indices = [
            index
            for index, name in enumerate(parameter_names)
            if not name.startswith("direct_trajectory_decoder.")
        ]

        if not decoder_indices:
            raise RuntimeError(
                "No parameters with prefix "
                "'direct_trajectory_decoder.' were found."
            )

        def group_stats(
            trajectory_grads,
            endpoint_grads,
            indices,
        ) -> dict[str, float]:
            trajectory_sq = 0.0
            endpoint_sq = 0.0
            dot = 0.0

            trajectory_elements = 0
            endpoint_elements = 0
            overlap_elements = 0

            for index in indices:
                grad_t = trajectory_grads[index]
                grad_e = endpoint_grads[index]

                if grad_t is not None:
                    gt = grad_t.detach().float()
                    trajectory_sq += float(
                        torch.sum(gt * gt).cpu()
                    )
                    trajectory_elements += gt.numel()

                if grad_e is not None:
                    ge = grad_e.detach().float()
                    endpoint_sq += float(
                        torch.sum(ge * ge).cpu()
                    )
                    endpoint_elements += ge.numel()

                if grad_t is not None and grad_e is not None:
                    gt = grad_t.detach().float()
                    ge = grad_e.detach().float()

                    dot += float(
                        torch.sum(gt * ge).cpu()
                    )

                    overlap_elements += min(
                        gt.numel(),
                        ge.numel(),
                    )

            trajectory_norm = math.sqrt(
                max(trajectory_sq, 0.0)
            )

            endpoint_norm = math.sqrt(
                max(endpoint_sq, 0.0)
            )

            raw_ratio = (
                endpoint_norm / trajectory_norm
                if trajectory_norm > 0.0
                else float("nan")
            )

            candidate_endpoint_weight = (
                target_endpoint_gradient_ratio
                * trajectory_norm
                / endpoint_norm
                if endpoint_norm > 0.0
                else float("nan")
            )

            current_weighted_ratio = (
                float(self.loss_weights.endpoint)
                * endpoint_norm
                /
                (
                    float(self.loss_weights.trajectory)
                    * trajectory_norm
                )
                if (
                    trajectory_norm > 0.0
                    and float(self.loss_weights.trajectory) > 0.0
                )
                else float("nan")
            )

            cosine = (
                dot
                / (
                    trajectory_norm
                    * endpoint_norm
                )
                if (
                    trajectory_norm > 0.0
                    and endpoint_norm > 0.0
                )
                else float("nan")
            )

            return {
                "trajectory_grad_norm": trajectory_norm,
                "endpoint_grad_norm": endpoint_norm,
                "endpoint_over_trajectory": raw_ratio,
                "gradient_cosine": cosine,
                "candidate_endpoint_weight": candidate_endpoint_weight,
                "current_weighted_endpoint_over_trajectory": (
                    current_weighted_ratio
                ),
                "trajectory_grad_elements": float(
                    trajectory_elements
                ),
                "endpoint_grad_elements": float(
                    endpoint_elements
                ),
                "overlap_grad_elements": float(
                    overlap_elements
                ),
            }

        records: dict[str, list[dict[str, float]]] = {
            "direct_decoder": [],
            "shared_upstream": [],
        }

        trajectory_losses: list[float] = []
        endpoint_losses: list[float] = []
        winner_counts: dict[int, int] = defaultdict(int)

        batches_used = 0

        for batch_index, batch in enumerate(
            dataloader,
            start=1,
        ):
            if batch_index > max_batches:
                break

            batch = _move_to_device(
                batch,
                self.device,
            )

            self.model.zero_grad(
                set_to_none=True
            )

            # Deliberately disable AMP for measurement precision.
            with torch.autocast(
                device_type=self.device.type,
                enabled=False,
            ):
                model_output = self.model(batch)

                loss_output = self._compute_loss(
                    model_output,
                    batch,
                )

                trajectory_component = (
                    loss_output.components["trajectory"]
                )

                endpoint_component = (
                    loss_output.components["endpoint"]
                )

            if not torch.isfinite(
                trajectory_component
            ):
                raise FloatingPointError(
                    f"Non-finite trajectory loss on batch {batch_index}."
                )

            if not torch.isfinite(
                endpoint_component
            ):
                raise FloatingPointError(
                    f"Non-finite endpoint loss on batch {batch_index}."
                )

            trajectory_grads = torch.autograd.grad(
                trajectory_component,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )

            endpoint_grads = torch.autograd.grad(
                endpoint_component,
                parameters,
                retain_graph=False,
                allow_unused=True,
            )

            records["direct_decoder"].append(
                group_stats(
                    trajectory_grads,
                    endpoint_grads,
                    decoder_indices,
                )
            )

            records["shared_upstream"].append(
                group_stats(
                    trajectory_grads,
                    endpoint_grads,
                    shared_indices,
                )
            )

            trajectory_losses.append(
                float(
                    trajectory_component
                    .detach()
                    .float()
                    .cpu()
                )
            )

            endpoint_losses.append(
                float(
                    endpoint_component
                    .detach()
                    .float()
                    .cpu()
                )
            )

            for winner in (
                loss_output.winner_idx
                .detach()
                .view(-1)
                .cpu()
                .tolist()
            ):
                winner_counts[int(winner)] += 1

            batches_used += 1

            decoder_record = records[
                "direct_decoder"
            ][-1]

            shared_record = records[
                "shared_upstream"
            ][-1]

            print(
                "[GRAD-CAL] "
                f"batch={batch_index}/{max_batches} | "
                f"Ltraj={trajectory_losses[-1]:.6f} | "
                f"Lend={endpoint_losses[-1]:.6f} | "
                f"decoder ratio="
                f"{decoder_record['endpoint_over_trajectory']:.4f} | "
                f"decoder cos="
                f"{decoder_record['gradient_cosine']:.4f} | "
                f"lambda*="
                f"{decoder_record['candidate_endpoint_weight']:.4f} | "
                f"shared ratio="
                f"{shared_record['endpoint_over_trajectory']:.4f} | "
                f"shared cos="
                f"{shared_record['gradient_cosine']:.4f} | "
                f"lambda*="
                f"{shared_record['candidate_endpoint_weight']:.4f}",
                flush=True,
            )

            del trajectory_grads
            del endpoint_grads
            del loss_output
            del model_output

        if batches_used == 0:
            raise RuntimeError(
                "Gradient calibration used zero batches."
            )

        def finite_values(values):
            return [
                float(value)
                for value in values
                if math.isfinite(float(value))
            ]

        def median(values):
            values = sorted(
                finite_values(values)
            )

            if not values:
                return float("nan")

            n = len(values)

            if n % 2 == 1:
                return values[n // 2]

            return 0.5 * (
                values[n // 2 - 1]
                +
                values[n // 2]
            )

        def mean(values):
            values = finite_values(values)

            if not values:
                return float("nan")

            return sum(values) / len(values)

        summary_groups = {}

        for group_name, group_records in records.items():
            summary_groups[group_name] = {}

            for field in (
                "trajectory_grad_norm",
                "endpoint_grad_norm",
                "endpoint_over_trajectory",
                "gradient_cosine",
                "candidate_endpoint_weight",
                "current_weighted_endpoint_over_trajectory",
            ):
                values = [
                    record[field]
                    for record in group_records
                ]

                summary_groups[group_name][
                    f"median_{field}"
                ] = median(values)

                summary_groups[group_name][
                    f"mean_{field}"
                ] = mean(values)

        candidate_values = []

        for group_records in records.values():
            candidate_values.extend(
                record["candidate_endpoint_weight"]
                for record in group_records
            )

        result = {
            "batches_used": batches_used,
            "beta_assign_used": float(
                self.beta_assign
            ),
            "current_trajectory_weight": float(
                self.loss_weights.trajectory
            ),
            "current_endpoint_weight": float(
                self.loss_weights.endpoint
            ),
            "target_endpoint_gradient_ratio": float(
                target_endpoint_gradient_ratio
            ),
            "median_trajectory_loss": median(
                trajectory_losses
            ),
            "median_endpoint_loss": median(
                endpoint_losses
            ),
            "winner_counts": dict(
                sorted(winner_counts.items())
            ),
            "groups": summary_groups,
            "pooled_median_candidate_endpoint_weight": median(
                candidate_values
            ),
            "note": (
                "shared_upstream means trainable parameters outside "
                "direct_trajectory_decoder that actually receive these "
                "trajectory/endpoint gradients."
            ),
        }

        self.model.zero_grad(
            set_to_none=True
        )

        if not was_training:
            self.model.eval()

        return result


    def train_epoch(
        self,
        dataloader: DataLoader,
        *,
        epoch: int,
    ) -> dict[str, float]:
        """Train for one complete epoch."""
        self.model.train()

        # === WZTARF RESET NONFINITE COUNTER V1 ===
        self._nonfinite_grad_skips = 0

        # === WZTARF RESET NONFINITE LOSS COUNTER V1 ===
        self._nonfinite_loss_skips = 0

        running_loss = 0.0
        running_components: dict[str, torch.Tensor] = {}

        num_batches = 0
        epoch_start = time.perf_counter()
        total_batches = len(dataloader)

        for batch_index, batch in enumerate(
            dataloader,
            start=1,
        ):
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

            # One scalar synchronization per training batch.  The previous
            # implementation synchronized once for finiteness, again for the
            # running loss, again for logging, and once for every individual
            # loss component.
            loss_scalar = float(loss.detach())

            # === WZTARF NONFINITE FORWARD LOSS GUARD V1 ===
            if not math.isfinite(loss_scalar):
                self._nonfinite_loss_skips = (
                    getattr(
                        self,
                        "_nonfinite_loss_skips",
                        0,
                    )
                    + 1
                )

                _component_parts = []

                for (
                    _component_name,
                    _component_value,
                ) in loss_output.components.items():
                    try:
                        _component_scalar = float(
                            _component_value.detach()
                        )

                        _component_parts.append(
                            f"{_component_name}="
                            f"{_component_scalar}"
                        )
                    except Exception:
                        _component_parts.append(
                            f"{_component_name}=<?>"
                        )

                print(
                    f"[Phase B][NONFINITE-LOSS] "
                    f"Epoch {epoch} | "
                    f"batch {batch_index}/{total_batches} | "
                    f"global_step={self.global_step} | "
                    f"loss={loss_scalar} | "
                    + ", ".join(_component_parts),
                    flush=True,
                )

                self.optimizer.zero_grad(
                    set_to_none=True
                )

                if self._nonfinite_loss_skips >= 8:
                    raise FloatingPointError(
                        f"Encountered "
                        f"{self._nonfinite_loss_skips} "
                        f"non-finite forward-loss batches "
                        f"in epoch {epoch}."
                    )

                continue

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

                # === WZTARF SURGICAL NONFINITE GUARD V1 ===
                _skip_step = False
                _grad_norm_scalar = None

                if self.grad_clip_norm is not None:
                    _grad_norm = clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.grad_clip_norm,
                        error_if_nonfinite=False,
                    )

                    _grad_norm_scalar = float(
                        _grad_norm.detach()
                    )

                    if not math.isfinite(
                        _grad_norm_scalar
                    ):
                        _skip_step = True

                if _skip_step:
                    _nonfinite_source = (
                        self._first_nonfinite_gradient()
                    )

                    self._nonfinite_grad_skips = (
                        getattr(
                            self,
                            "_nonfinite_grad_skips",
                            0,
                        )
                        + 1
                    )

                    _parts = []

                    for _name, _value in loss_output.components.items():
                        try:
                            _scalar = float(
                                _value.detach()
                            )
                            _parts.append(
                                f"{_name}={_scalar:.6g}"
                            )
                        except Exception:
                            _parts.append(
                                f"{_name}=<?>"
                            )

                    print(
                        f"[Phase B][NONFINITE-GRAD-SOURCE] "
                        f"Epoch {epoch} | "
                        f"batch {batch_index}/{total_batches} | "
                        f"{_nonfinite_source}",
                        flush=True,
                    )

                    print(
                        f"[Phase B][NONFINITE-GRAD] "
                        f"Epoch {epoch} | "
                        f"batch {batch_index}/{total_batches} | "
                        f"loss={loss_scalar:.6g} | "
                        f"grad_norm={_grad_norm_scalar} | "
                        f"skip_count="
                        f"{self._nonfinite_grad_skips} | "
                        f"components="
                        + ", ".join(_parts),
                        flush=True,
                    )

                    self.optimizer.zero_grad(
                        set_to_none=True
                    )

                    if self._nonfinite_grad_skips >= 8:
                        raise FloatingPointError(
                            "Eight non-finite-gradient batches "
                            "were encountered. Aborting rather "
                            "than risking unstable training."
                        )

                    continue

                self.optimizer.step()

            self.global_step += 1
            num_batches += 1

            running_loss += loss_scalar

            if (
                batch_index == 1
                or batch_index % 50 == 0
                or batch_index == total_batches
            ):
                elapsed = time.perf_counter() - epoch_start
                rate = batch_index / max(elapsed, 1e-8)
                remaining = (
                    total_batches - batch_index
                ) / max(rate, 1e-8)

                print(
                    f"[Phase B][TRAIN] "
                    f"Epoch {epoch} | "
                    f"{batch_index}/{total_batches} | "
                    f"loss={loss_scalar:.4f} | "
                    f"{rate:.2f} batch/s | "
                    f"ETA={remaining / 60.0:.1f} min",
                    flush=True,
                )

            # Keep component reductions on-device for the whole epoch and
            # transfer only the final sums.  This removes N loss-component
            # CUDA synchronizations from every optimizer step.
            for name, value in loss_output.components.items():
                key = f"loss_{name}"
                detached = value.detach().float()
                if key in running_components:
                    running_components[key] = running_components[key] + detached
                else:
                    running_components[key] = detached.clone()

        if num_batches == 0:
            raise ValueError(
                "Training DataLoader produced zero batches."
            )

        result = {"loss": running_loss / num_batches}
        result.update(
            {
                key: float(value.cpu()) / num_batches
                for key, value in running_components.items()
            }
        )

        result["learning_rate"] = _current_learning_rate(
            self.optimizer
        )

        result[
            "NONFINITE-GRAD"
        ] = float(
            self._nonfinite_grad_skips
        )

        result[
            "NONFINITE-LOSS"
        ] = float(
            self._nonfinite_loss_skips
        )

        print(
            f"[Phase B][NUMERICS] "
            f"Epoch {epoch} | "
            f"NONFINITE-GRAD={self._nonfinite_grad_skips} | "
            f"NONFINITE-LOSS={self._nonfinite_loss_skips}",
            flush=True,
        )

        return result

    @torch.inference_mode()
    def validate(
        self,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """Run validation loss and the complete forecasting metric suite."""
        self.model.eval()

        running_loss: dict[str, torch.Tensor] = {}

        pred_batches: list[torch.Tensor] = []
        probability_batches: list[torch.Tensor] = []
        gt_batches: list[torch.Tensor] = []

        wz_batches: list[torch.Tensor] = []
        worker_batches: list[torch.Tensor] = []
        metadata: list[dict[str, Any]] = []

        num_batches = 0
        validation_start = time.perf_counter()
        total_batches = len(dataloader)

        for batch_index, batch in enumerate(
            dataloader,
            start=1,
        ):
            batch = _move_to_device(
                batch,
                self.device,
            )

            metadata.extend(
                _metadata_from_batch(
                    batch,
                    int(
                        batch["future_xy"].shape[0]
                    ),
                )
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

            total_detached = loss_output.total.detach().float()
            if "val_loss" in running_loss:
                running_loss["val_loss"] = running_loss["val_loss"] + total_detached
            else:
                running_loss["val_loss"] = total_detached.clone()

            for name, value in loss_output.components.items():
                key = f"val_loss_{name}"
                detached = value.detach().float()
                if key in running_loss:
                    running_loss[key] = running_loss[key] + detached
                else:
                    running_loss[key] = detached.clone()

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

            if (
                batch_index == 1
                or batch_index % 50 == 0
                or batch_index == total_batches
            ):
                elapsed = time.perf_counter() - validation_start
                rate = batch_index / max(elapsed, 1e-8)
                remaining = (
                    total_batches - batch_index
                ) / max(rate, 1e-8)

                print(
                    f"[Phase B][VAL] "
                    f"{batch_index}/{total_batches} | "
                    f"loss={float(loss_output.total.detach()):.4f} | "
                    f"{rate:.2f} batch/s | "
                    f"ETA={remaining / 60.0:.1f} min",
                    flush=True,
                )

        if num_batches == 0:
            raise ValueError(
                "Validation DataLoader produced zero batches."
            )

        validation = {
            key: float(value.cpu()) / num_batches
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

        grouped_metrics = compute_grouped_forecasting_metrics(
            pred_xy=pred_xy,
            gt_xy=gt_xy,
            metadata=metadata,
        )

        validation.update(
            grouped_metrics
        )

        # Frozen before test inspection:
        # J_val = minADE_6 + 0.25 * minFDE_6 by default.
        validation[
            "J_val"
        ] = (
            float(
                validation["minADE_6"]
            )
            +
            self.composite_fde_weight
            *
            float(
                validation["minFDE_6"]
            )
        )

        macro_ade = validation.get(
            "macro_minADE_6_scenario_x_workzone"
        )

        macro_fde = validation.get(
            "macro_minFDE_6_scenario_x_workzone"
        )

        if (
            macro_ade is not None
            and macro_fde is not None
            and math.isfinite(
                float(macro_ade)
            )
            and math.isfinite(
                float(macro_fde)
            )
        ):
            validation[
                "J_val_macro_scenario_x_workzone"
            ] = (
                float(macro_ade)
                +
                self.composite_fde_weight
                *
                float(macro_fde)
            )

        for group_name in (
            "scenario",
            "workzone",
            "scenario_x_workzone",
            "participant",
        ):
            coverage = float(
                validation.get(
                    f"metadata_coverage_{group_name}",
                    0.0,
                )
            )

            if coverage < 0.999:
                print(
                    f"[Phase B][VAL][METADATA-WARNING] "
                    f"{group_name} coverage="
                    f"{coverage:.3f}. "
                    f"Macro metric may not represent every sample.",
                    flush=True,
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
        selection_metric: str = "J_val",
        selection_mode: str = "min",
        patience: int | None = None,
        resume_from: str | Path | None = None,
    ) -> dict[str, Any]:
        """Train with frozen V3 composite primary checkpoint selection.

        Primary checkpoint:
            J_val = minADE_6 + composite_fde_weight * minFDE_6

        Always maintains:
            best_minADE.pt
            best_minFDE.pt
            best_composite.pt
            best.pt              (alias of best_composite)
            last.pt

        Test metrics never participate in checkpoint selection.
        """
        if epochs <= 0:
            raise ValueError(
                "epochs must be positive."
            )

        if selection_mode != "min":
            raise ValueError(
                "V3 checkpoint selection is frozen to minimization."
            )

        if selection_metric not in {
            "J_val",
            "composite",
        }:
            raise ValueError(
                "V3 primary checkpoint selection is frozen to J_val. "
                "Set training.selection_metric: J_val."
            )

        if (
            patience is not None
            and patience <= 0
        ):
            raise ValueError(
                "patience must be positive or None."
            )

        best_ade: float | None = None
        best_fde: float | None = None
        best_composite: float | None = None

        best_ade_epoch: int | None = None
        best_fde_epoch: int | None = None
        best_composite_epoch: int | None = None

        bad_epochs = 0

        if resume_from is not None:
            state = self.resume(
                resume_from
            )

            start_epoch = max(
                start_epoch,
                state.epoch + 1,
            )

            # CheckpointState exposes one historical best value.  For a V3
            # last/best-composite checkpoint this is the primary composite.
            best_composite = state.best_metric

        start_time = time.perf_counter()

        if self.logger is not None:
            self.logger.log(
                f"Starting supervised training at epoch {start_epoch}. "
                f"Frozen J_val = minADE_6 + "
                f"{self.composite_fde_weight:.6g} * minFDE_6."
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

            _val_parts = []

            for _name, _value in sorted(
                validation_metrics.items()
            ):
                try:
                    _scalar = float(
                        _value
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                _val_parts.append(
                    f"{_name}={_scalar:.6f}"
                )

            print(
                f"[Phase B][VAL][SUMMARY] "
                f"Epoch {epoch} | "
                + " | ".join(
                    _val_parts
                ),
                flush=True,
            )

            current_ade = float(
                validation_metrics[
                    "minADE_6"
                ]
            )

            current_fde = float(
                validation_metrics[
                    "minFDE_6"
                ]
            )

            current_composite = float(
                validation_metrics[
                    "J_val"
                ]
            )

            for name, value in (
                (
                    "minADE_6",
                    current_ade,
                ),
                (
                    "minFDE_6",
                    current_fde,
                ),
                (
                    "J_val",
                    current_composite,
                ),
            ):
                if not math.isfinite(
                    value
                ):
                    raise FloatingPointError(
                        f"Validation metric {name} "
                        f"is non-finite at epoch {epoch}."
                    )

            improved_ade = (
                best_ade is None
                or current_ade < best_ade
            )

            improved_fde = (
                best_fde is None
                or current_fde < best_fde
            )

            improved_composite = (
                best_composite is None
                or current_composite
                <
                best_composite
            )

            if improved_ade:
                best_ade = current_ade
                best_ade_epoch = epoch

            if improved_fde:
                best_fde = current_fde
                best_fde_epoch = epoch

            if improved_composite:
                best_composite = current_composite
                best_composite_epoch = epoch
                bad_epochs = 0

            else:
                bad_epochs += 1

            # Scheduler state in every saved checkpoint corresponds to the
            # just-completed epoch.
            self._step_scheduler(
                validation_metrics
            )

            common_extra = {
                "validation_metrics": dict(
                    validation_metrics
                ),
                "composite_formula": (
                    "minADE_6 + "
                    f"{self.composite_fde_weight} * minFDE_6"
                ),
                "composite_fde_weight": (
                    self.composite_fde_weight
                ),
                "best_minADE_6": best_ade,
                "best_minFDE_6": best_fde,
                "best_J_val": best_composite,
                "best_minADE_epoch": best_ade_epoch,
                "best_minFDE_epoch": best_fde_epoch,
                "best_composite_epoch": (
                    best_composite_epoch
                ),
            }

            if improved_ade:
                save_checkpoint(
                    self.checkpoint_dir
                    /
                    "best_minADE.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch,
                    global_step=self.global_step,
                    best_metric=best_ade,
                    config=self.config,
                    extra={
                        **common_extra,
                        "selection_metric": "minADE_6",
                        "checkpoint_role": "best_minADE",
                    },
                )

            if improved_fde:
                save_checkpoint(
                    self.checkpoint_dir
                    /
                    "best_minFDE.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch,
                    global_step=self.global_step,
                    best_metric=best_fde,
                    config=self.config,
                    extra={
                        **common_extra,
                        "selection_metric": "minFDE_6",
                        "checkpoint_role": "best_minFDE",
                    },
                )

            if improved_composite:
                for filename, role in (
                    (
                        "best_composite.pt",
                        "best_composite",
                    ),
                    (
                        "best.pt",
                        "best_composite_alias",
                    ),
                ):
                    save_checkpoint(
                        self.checkpoint_dir
                        /
                        filename,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        epoch=epoch,
                        global_step=self.global_step,
                        best_metric=best_composite,
                        config=self.config,
                        extra={
                            **common_extra,
                            "selection_metric": "J_val",
                            "checkpoint_role": role,
                        },
                    )

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
                best_metric=best_composite,
                config=self.config,
                extra={
                    **common_extra,
                    "selection_metric": "J_val",
                    "checkpoint_role": "last",
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
                    f"minADE_6={current_ade:.6f}, "
                    f"minFDE_6={current_fde:.6f}, "
                    f"J_val={current_composite:.6f}, "
                    f"best_J_val={best_composite:.6f}, "
                    f"lr={_current_learning_rate(self.optimizer):.8g}"
                )

            if (
                patience is not None
                and bad_epochs >= patience
            ):
                if self.logger is not None:
                    self.logger.log(
                        f"Early stopping after epoch {epoch} "
                        f"using frozen J_val selection."
                    )

                break

        duration_hours = (
            time.perf_counter()
            -
            start_time
        ) / 3600.0

        summary: dict[str, Any] = {
            "last_epoch": last_epoch,

            "primary_selection_metric": "J_val",
            "composite_fde_weight": (
                self.composite_fde_weight
            ),

            "best_composite_epoch": (
                best_composite_epoch
            ),
            "best_composite": (
                best_composite
            ),

            "best_minADE_epoch": (
                best_ade_epoch
            ),
            "best_minADE_6": (
                best_ade
            ),

            "best_minFDE_epoch": (
                best_fde_epoch
            ),
            "best_minFDE_6": (
                best_fde
            ),

            "global_step": self.global_step,
            "training_duration_hours": (
                duration_hours
            ),

            "best_checkpoint": str(
                (
                    self.checkpoint_dir
                    /
                    "best_composite.pt"
                ).resolve()
            ),

            "best_alias_checkpoint": str(
                (
                    self.checkpoint_dir
                    /
                    "best.pt"
                ).resolve()
            ),

            "best_minADE_checkpoint": str(
                (
                    self.checkpoint_dir
                    /
                    "best_minADE.pt"
                ).resolve()
            ),

            "best_minFDE_checkpoint": str(
                (
                    self.checkpoint_dir
                    /
                    "best_minFDE.pt"
                ).resolve()
            ),

            "last_checkpoint": str(
                (
                    self.checkpoint_dir
                    /
                    "last.pt"
                ).resolve()
            ),

            "final_validation": (
                final_validation
            ),
        }

        if self.logger is not None:
            self.logger.save_run_summary(
                summary
            )

            self.logger.log(
                "Supervised training finished."
            )

        return summary
