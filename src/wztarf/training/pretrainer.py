"""Train scene representations with the three Phase A objectives."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from wztarf.data.future_topology_targets import (
    build_future_topology_targets,
)
from wztarf.data.map_coverage import (
    build_map_coverage_mask_batched,
)
from wztarf.losses.route_set_objectives import (
    route_set_coverage_loss,
)
from wztarf.pretraining.future_contrastive import (
    build_false_negative_mask,
    future_contrastive_loss,
)
from wztarf.pretraining.masked_reconstruction import (
    masked_reconstruction_loss,
)
from wztarf.pretraining.masking import (
    MaskingConfig,
    build_mask_plan,
)
from wztarf.pretraining.topology_reconstruction import (
    topology_reconstruction_loss,
)
from wztarf.pretraining.topology_targets import build_topology_targets
from wztarf.reporting.run_logger import RunLogger
from wztarf.training.checkpointing import (
    CheckpointState,
    load_checkpoint,
    save_checkpoint,
)


@dataclass(frozen=True)
class PretrainingWeights:
    """Weights for the five V3 Phase-A objectives."""

    masked_reconstruction: float = 1.0
    future_contrastive: float = 1.0

    # WZ-derived objective. Must remain zero for static No-WZ.
    topology: float = 1.0

    # Future + permanent-map objectives. Legal for Full and No-WZ.
    route_coverage: float = 0.5
    route_progress: float = 0.5

    def __post_init__(self) -> None:
        """Reject negative objective weights."""
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(
                    f"Pretraining weight '{name}' cannot be negative."
                )


def _move_to_device(
    value: Any,
    device: torch.device,
) -> Any:
    """Recursively move tensors while preserving Python metadata."""
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


def _require(
    mapping: Mapping[str, Any],
    key: str,
    *,
    owner: str,
) -> Any:
    """Retrieve one required pretraining field."""
    if key not in mapping:
        raise KeyError(
            f"{owner} is missing required field '{key}'."
        )

    return mapping[
        key
    ]


class Pretrainer:
    """Coordinate masked, future-contrastive, and topology pretraining."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str | torch.device,
        checkpoint_dir: str | Path,
        weights: PretrainingWeights | None = None,
        masking_config: MaskingConfig | None = None,
        scheduler: Any | None = None,
        logger: RunLogger | None = None,
        config: Mapping[str, Any] | None = None,
        grad_clip_norm: float | None = 5.0,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
        fac_temperature: float = 0.1,
        fac_exclusion_seconds: float = 5.0,
        fac_symmetric: bool = False,
        fps: int = 5,
        scheduler_metric: str | None = None,
        mask_seed: int = 2023,
    ) -> None:
        """Create the Phase A pretrainer."""
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

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

        self.weights = (
            weights
            if weights is not None
            else PretrainingWeights()
        )

        self.masking_config = (
            masking_config
            if masking_config is not None
            else MaskingConfig()
        )

        self.logger = logger

        self.config = (
            dict(config)
            if config is not None
            else {}
        )

        self.grad_clip_norm = grad_clip_norm
        self.fac_temperature = fac_temperature
        self.fac_exclusion_seconds = fac_exclusion_seconds
        self.fac_symmetric = fac_symmetric
        self.fps = int(
            fps
        )
        self.scheduler_metric = scheduler_metric
        self.mask_seed = int(
            mask_seed
        )

        if (
            grad_clip_norm is not None
            and grad_clip_norm <= 0
        ):
            raise ValueError(
                "grad_clip_norm must be positive or None."
            )

        if fac_temperature <= 0:
            raise ValueError(
                "fac_temperature must be positive."
            )

        if fac_exclusion_seconds < 0:
            raise ValueError(
                "fac_exclusion_seconds cannot be negative."
            )

        if self.fps <= 0:
            raise ValueError(
                "fps must be positive."
            )

        self.use_amp = (
            bool(
                use_amp
            )
            and
            self.device.type == "cuda"
        )

        self.amp_dtype = amp_dtype

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

    def _pretraining_forward(
        self,
        batch: Mapping[str, Any],
        mask_plan: Any,
    ) -> Mapping[str, Any]:
        """Run the model's explicit pretraining forward path.

        The architecture model should expose:

            model.pretraining_forward(batch, mask_plan)

        so Phase A can reuse the same scene encoders while attaching
        training-only reconstruction and future-encoding heads.
        """
        function = getattr(
            self.model,
            "pretraining_forward",
            None,
        )

        if not callable(
            function
        ):
            raise AttributeError(
                "The pretraining model must implement "
                "pretraining_forward(batch, mask_plan)."
            )

        output = function(
            batch,
            mask_plan,
        )

        if not isinstance(
            output,
            Mapping,
        ):
            raise TypeError(
                "pretraining_forward() must return a mapping."
            )

        return output

    def _sequence_information(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[list[Any], torch.Tensor]:
        """Extract drive identity and anchor time for overlap suppression."""

        metadata = batch.get(
            "meta"
        )

        if not isinstance(
            metadata,
            list,
        ):
            raise KeyError(
                "Future-contrastive pretraining requires per-sample meta."
            )

        sequence_keys = (
            "sequence_id",
            "scene_id",
            "scenario_id",
            "drive_id",
            "episode_id",
            "scenario",
        )

        time_keys = (
            "anchor_time_s",
            "time_s",
            "timestamp_s",
            "t_anchor",
        )

        frame_keys = (
            "anchor_frame",
            "frame_idx",
            "frame_index",
        )

        sequence_ids: list[Any] = []
        anchor_times: list[float] = []

        for index, item in enumerate(
            metadata
        ):
            if not isinstance(
                item,
                Mapping,
            ):
                raise TypeError(
                    f"meta[{index}] must be mapping-like."
                )

            sequence = next(
                (
                    item[key]
                    for key in sequence_keys
                    if key in item
                ),
                None,
            )

            if sequence is None:
                raise KeyError(
                    f"meta[{index}] contains no recognized sequence identifier."
                )

            time_s = next(
                (
                    float(
                        item[key]
                    )
                    for key in time_keys
                    if key in item
                ),
                None,
            )

            if time_s is None:
                frame = next(
                    (
                        item[key]
                        for key in frame_keys
                        if key in item
                    ),
                    None,
                )

                if frame is None:
                    raise KeyError(
                        f"meta[{index}] contains no recognized anchor "
                        "time or frame index."
                    )

                time_s = (
                    float(
                        frame
                    )
                    /
                    float(
                        self.fps
                    )
                )

            sequence_ids.append(
                sequence
            )

            anchor_times.append(
                time_s
            )
    
        return (
            sequence_ids,
            torch.tensor(
                anchor_times,
                dtype=torch.float32,
                device=self.device,
            ),
        )

    def _compute_losses(
        self,
        output: Mapping[str, Any],
        batch: Mapping[str, Any],
        mask_plan: Any,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Compute all enabled Phase A objectives."""
        components: dict[str, torch.Tensor] = {}

        reference = next(
            value
            for value in output.values()
            if isinstance(
                value,
                torch.Tensor,
            )
        )

        total = reference.sum() * 0.0

        # --------------------------------------------------------------
        # Masked cross-modal reconstruction
        # --------------------------------------------------------------

        if self.weights.masked_reconstruction > 0:
            predictions = _require(
                output,
                "reconstruction_predictions",
                owner="pretraining output",
            )

            targets = _require(
                output,
                "reconstruction_targets",
                owner="pretraining output",
            )

            # Convert the MaskPlan dataclass into the names expected by the
            # reconstruction heads.
            masks = {
                "motion": mask_plan.motion,
                "controls": mask_plan.controls,
                "gaze": mask_plan.gaze,
                "agents": mask_plan.agents,
                "lanes": mask_plan.lanes,
                "workzone": mask_plan.workzone,
                "workers": mask_plan.workers,
            }

            valid_masks = output.get(
                "reconstruction_valid_masks"
            )

            components[
                "masked_reconstruction"
            ] = masked_reconstruction_loss(
                predictions,
                targets,
                masks,
                modality_weights=output.get(
                    "reconstruction_weights"
                ),
                loss_types=output.get(
                    "reconstruction_loss_types"
                ),
                valid_masks=valid_masks,
            )

            total = (
                total
                +
                self.weights.masked_reconstruction
                *
                components[
                    "masked_reconstruction"
                ]
            )

        # --------------------------------------------------------------
        # Horizon-aware future contrastive alignment
        # --------------------------------------------------------------

        if self.weights.future_contrastive > 0:
            context_embeddings = _require(
                output,
                "context_embeddings",
                owner="pretraining output",
            )

            future_embeddings = _require(
                output,
                "future_embeddings",
                owner="pretraining output",
            )

            sequence_ids, anchor_time_s = self._sequence_information(
                batch
            )

            allowed_mask = build_false_negative_mask(
                sequence_ids,
                anchor_time_s,
                exclusion_seconds=self.fac_exclusion_seconds,
                device=self.device,
            )

            components[
                "future_contrastive"
            ] = future_contrastive_loss(
                context_embeddings,
                future_embeddings,
                allowed_mask=allowed_mask,
                horizon_weights=output.get(
                    "future_horizon_weights"
                ),
                negative_weights=output.get(
                    "future_negative_weights"
                ),
                temperature=self.fac_temperature,
                symmetric=self.fac_symmetric,
            )

            total = (
                total
                +
                self.weights.future_contrastive
                *
                components[
                    "future_contrastive"
                ]
            )

        # --------------------------------------------------------------
        # WorkZone-conditioned topology reconstruction
        # --------------------------------------------------------------

        if self.weights.topology > 0:
            targets = build_topology_targets(
                lane_feat=batch[
                    "lane_feat"
                ],
                lane_point_mask=batch[
                    "lane_point_mask"
                ],
                lane_mask=batch[
                    "lane_mask"
                ],
                lane_edge_index=batch[
                    "lane_edge_index"
                ],
                lane_edge_mask=batch[
                    "lane_edge_mask"
                ],
                wz_feat=batch[
                    "wz_feat"
                ],
            )

            valid_lane = (
                targets.lane_mask
                &
                output[
                    "topology_lane_mask"
                ].bool()
            )

            valid_edge = (
                targets.edge_mask
                &
                output[
                    "topology_edge_mask"
                ].bool()
            )

            components[
                "topology"
            ] = topology_reconstruction_loss(
                lane_overlap_pred=output[
                    "lane_overlap_pred"
                ],
                lane_overlap_target=targets.lane_overlap,
                lane_distance_pred=output[
                    "lane_distance_pred"
                ],
                lane_distance_target=targets.lane_distance,
                lane_mask=valid_lane,
                edge_compat_logits=output[
                    "edge_compat_logits"
                ],
                edge_compat_target=targets.edge_compatibility,
                edge_mask=valid_edge,
            )

            total = (
                total
                +
                self.weights.topology
                *
                components[
                    "topology"
                ]
            )

        # ==============================================================
        # V3 PHASE-A ROUTE SET + PROGRESS SUPERVISION
        #
        # IMPORTANT:
        # Targets below use only future_xy + permanent lane geometry.
        # They never use wz_feat, therefore the same objectives are valid
        # for the Full model and the static No-WZ ablation.
        # ==============================================================

        if (
            self.weights.route_coverage > 0
            or self.weights.route_progress > 0
        ):
            route_node_occupancy = _require(
                output,
                "route_node_occupancy",
                owner="pretraining output",
            )

            route_edge_occupancy = _require(
                output,
                "route_edge_occupancy",
                owner="pretraining output",
            )

            route_goal_prob = _require(
                output,
                "route_goal_prob",
                owner="pretraining output",
            )

            route_anchors = _require(
                output,
                "route_anchors",
                owner="pretraining output",
            )

            # ----------------------------------------------------------
            # Future -> permanent-map route pseudo-target.
            #
            # Use the original unmasked map from the batch for TARGETS.
            # lane_feat[..., :2] is the permanent lane XY geometry.
            # ----------------------------------------------------------

            permanent_lane_xy = batch[
                "lane_feat"
            ][
                ...,
                :2,
            ].float()

            map_coverage = build_map_coverage_mask_batched(
                points=batch["future_xy"],
                lane_feat=batch["lane_feat"],
                lane_point_mask=batch["lane_point_mask"],
                lane_mask=batch["lane_mask"],
                margin_m=2.25,
            )

            route_targets = build_future_topology_targets(
                future_xy=batch["future_xy"],
                lane_centerline=permanent_lane_xy,
                lane_point_mask=batch["lane_point_mask"],
                lane_mask=batch["lane_mask"],
                lane_edge_index=batch["lane_edge_index"],
                lane_edge_mask=batch["lane_edge_mask"],
                map_coverage=map_coverage,
                match_radius_m=2.25,
                transition_penalty=1.0,
                jump_penalty=4.0,
            )

            # ----------------------------------------------------------
            # Smooth best-of-six route-set coverage.
            # ----------------------------------------------------------

            if self.weights.route_coverage > 0:
                coverage = route_set_coverage_loss(
                    route_edge_occupancy=route_edge_occupancy,
                    route_node_occupancy=route_node_occupancy,
                    goal_prob=route_goal_prob,
                    route_anchors=route_anchors,
                    future_xy=batch["future_xy"],
                    targets=route_targets,
                    fps=int(self.model.config.fps),
                    temperature=0.35,
                    edge_weight=1.0,
                    goal_weight=1.0,
                    horizon_weight=1.0,
                    horizon_scale_m=5.0,
                )

                components["route_coverage"] = coverage.total
                components["route_coverage_edge"] = coverage.edge
                components["route_coverage_goal"] = coverage.goal
                components["route_coverage_horizon"] = coverage.horizon

                total = (
                    total
                    +
                    self.weights.route_coverage
                    *
                    components["route_coverage"]
                )

            # ----------------------------------------------------------
            # Explicit monotonic 1/3/5-second progress supervision.
            #
            # route_anchors are generated FROM route_progress, so this
            # directly trains the s1/s3/s5 monotonic progress mechanism.
            # ----------------------------------------------------------

            if self.weights.route_progress > 0:
                # ======================================================
                # TRUE SCALAR ROUTE-PROGRESS SUPERVISION
                #
                # The route-progress head predicts monotonic traveled
                # distances s1, s3, s5 along each route hypothesis.
                #
                # Supervise those scalar distances against cumulative
                # traveled distance of the realized future trajectory.
                #
                # This is intentionally distinct from route coverage's
                # horizon term, which supervises XY route anchors.
                # ======================================================

                predicted_progress = _require(
                    output,
                    "route_progress",
                    owner="pretraining output",
                ).float()

                future_xy = batch[
                    "future_xy"
                ].float()

                # Ego is at (0,0) at prediction time. Include distance
                # from the origin to the first future observation.
                future_delta = torch.zeros_like(
                    future_xy
                )

                future_delta[
                    :,
                    0,
                ] = future_xy[
                    :,
                    0,
                ]

                future_delta[
                    :,
                    1:,
                ] = (
                    future_xy[
                        :,
                        1:,
                    ]
                    -
                    future_xy[
                        :,
                        :-1,
                    ]
                )

                cumulative_progress = torch.cumsum(
                    torch.linalg.vector_norm(
                        future_delta,
                        dim=-1,
                    ),
                    dim=1,
                )

                fps = int(
                    self.model.config.fps
                )

                horizon_indices = torch.tensor(
                    (
                        min(
                            fps * 1 - 1,
                            future_xy.shape[1] - 1,
                        ),
                        min(
                            fps * 3 - 1,
                            future_xy.shape[1] - 1,
                        ),
                        min(
                            fps * 5 - 1,
                            future_xy.shape[1] - 1,
                        ),
                    ),
                    dtype=torch.long,
                    device=future_xy.device,
                )

                gt_progress = cumulative_progress.index_select(
                    1,
                    horizon_indices,
                )

                progress_error = torch.abs(
                    predicted_progress
                    -
                    gt_progress[
                        :,
                        None,
                        :,
                    ]
                )

                horizon_weights = torch.tensor(
                    (
                        0.5,
                        1.0,
                        2.0,
                    ),
                    dtype=progress_error.dtype,
                    device=progress_error.device,
                )

                progress_scale_m = 5.0

                mode_cost = (
                    (
                        progress_error
                        /
                        progress_scale_m
                    )
                    *
                    horizon_weights[
                        None,
                        None,
                        :,
                    ]
                ).sum(
                    dim=-1
                ) / horizon_weights.sum()

                tau = 0.35

                progress_per_sample = (
                    -tau
                    *
                    (
                        torch.logsumexp(
                            -mode_cost / tau,
                            dim=1,
                        )
                        -
                        math.log(
                            mode_cost.shape[1]
                        )
                    )
                )

                components[
                    "route_progress"
                ] = progress_per_sample.mean()

                total = (
                    total
                    +
                    self.weights.route_progress
                    *
                    components[
                        "route_progress"
                    ]
                )

        return (
            total,
            components,
        )

    def run_epoch(
        self,
        dataloader: DataLoader,
        *,
        epoch: int,
        training: bool,
    ) -> dict[str, float]:
        """Run one pretraining or validation epoch."""
        if training:
            self.model.train()
        else:
            self.model.eval()

        mask_generator = None

        if not training:
            mask_generator = torch.Generator(
                device=self.device
            )

            mask_generator.manual_seed(
                self.mask_seed
            )

        running: dict[str, float] = defaultdict(
            float
        )

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

            mask_plan = build_mask_plan(
                batch,
                config=self.masking_config,
                generator=mask_generator,
            )

            if training:
                self.optimizer.zero_grad(
                    set_to_none=True
                )

            with torch.set_grad_enabled(
                training
            ):
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    output = self._pretraining_forward(
                        batch,
                        mask_plan,
                    )

                    total_loss, components = self._compute_losses(
                        output,
                        batch,
                        mask_plan,
                    )

            if not bool(
                torch.isfinite(
                    total_loss
                ).item()
            ):
                mode = (
                    "training"
                    if training
                    else "validation"
                )

                raise FloatingPointError(
                    f"Non-finite pretraining {mode} loss "
                    f"at epoch {epoch}."
                )

            if training:
                if self.scaler is not None:
                    self.scaler.scale(
                        total_loss
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
                    total_loss.backward()

                    if self.grad_clip_norm is not None:
                        clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.grad_clip_norm,
                            error_if_nonfinite=True,
                        )

                    self.optimizer.step()

                self.global_step += 1

            prefix = (
                ""
                if training
                else "val_"
            )

            running[
                f"{prefix}loss"
            ] += float(
                total_loss.detach().item()
            )

            for name, value in components.items():
                running[
                    f"{prefix}loss_{name}"
                ] += float(
                    value.detach().item()
                )

            num_batches += 1
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

                mode = "TRAIN" if training else "VAL"

                print(
                    f"[Phase A][{mode}] "
                    f"Epoch {epoch} | "
                    f"{batch_index}/{total_batches} | "
                    f"loss={total_loss.detach().item():.4f} | "
                    f"{rate:.2f} batch/s | "
                    f"ETA={remaining / 60.0:.1f} min",
                    flush=True,
                )

        if num_batches == 0:
            raise ValueError(
                "Pretraining DataLoader produced zero batches."
            )

        return {
            key: value / num_batches
            for key, value in running.items()
        }

    def _step_scheduler(
        self,
        validation_metrics: Mapping[str, float],
    ) -> None:
        """Advance the learning-rate scheduler once per epoch."""
        if self.scheduler is None:
            return

        if self.scheduler_metric is None:
            self.scheduler.step()
            return

        if self.scheduler_metric not in validation_metrics:
            raise KeyError(
                f"Scheduler metric '{self.scheduler_metric}' "
                "was not produced."
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
        """Resume the complete Phase A optimization state."""
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
        selection_metric: str = "val_loss",
        patience: int | None = None,
        resume_from: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run complete Phase A representation pretraining."""
        if epochs <= 0:
            raise ValueError(
                "epochs must be positive."
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
        last_epoch = start_epoch - 1

        if self.logger is not None:
            self.logger.log(
                f"Starting Phase A pretraining at epoch {start_epoch}."
            )

        final_validation: dict[str, float] = {}

        for epoch in range(
            start_epoch,
            epochs + 1,
        ):
            last_epoch = epoch

            train_metrics = self.run_epoch(
                train_loader,
                epoch=epoch,
                training=True,
            )

            with torch.inference_mode():
                validation_metrics = self.run_epoch(
                    val_loader,
                    epoch=epoch,
                    training=False,
                )

            final_validation = validation_metrics

            if selection_metric not in validation_metrics:
                raise KeyError(
                    f"Selection metric '{selection_metric}' "
                    "was not produced."
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
                    f"Non-finite selection metric at epoch {epoch}."
                )

            improved = (
                best_metric is None
                or current_metric < best_metric
            )

            if improved:
                best_metric = current_metric
                best_epoch = epoch
                bad_epochs = 0

            else:
                bad_epochs += 1
    
            # Advance the scheduler before checkpointing so every checkpoint
            # contains the complete optimizer state after this epoch.
            self._step_scheduler(
                validation_metrics
            )
    
            if improved:
                save_checkpoint(
                    self.checkpoint_dir
                    /
                    "pretrain_best.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch,
                    global_step=self.global_step,
                    best_metric=best_metric,
                    config=self.config,
                    extra={
                        "phase": "pretraining",
                        "selection_metric": selection_metric,
                        "validation_metrics": dict(
                            validation_metrics
                        ),
                    },
                )

            save_checkpoint(
                self.checkpoint_dir
                /
                "pretrain_last.pt",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch,
                global_step=self.global_step,
                best_metric=best_metric,
                config=self.config,
                extra={
                    "phase": "pretraining",
                    "selection_metric": selection_metric,
                    "validation_metrics": dict(
                        validation_metrics
                    ),
                },
            )

            if self.logger is not None:
                self.logger.log_metrics(
                    train_metrics,
                    split="pretrain_train",
                    epoch=epoch,
                    step=self.global_step,
                )

                self.logger.log_metrics(
                    validation_metrics,
                    split="pretrain_val",
                    epoch=epoch,
                    step=self.global_step,
                )

                self.logger.log(
                    f"Pretrain epoch {epoch}: "
                    f"{selection_metric}={current_metric:.6f}, "
                    f"best={best_metric:.6f}"
                )
                
            if (
                patience is not None
                and bad_epochs >= patience
            ):
                if self.logger is not None:
                    self.logger.log(
                        f"Phase A early stopping after epoch {epoch}."
                    )

                break

        duration_hours = (
            time.perf_counter()
            -
            start_time
        ) / 3600.0

        summary = {
            "phase": "pretraining",
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
                    "pretrain_best.pt"
                ).resolve()
            ),
            "last_checkpoint": str(
                (
                    self.checkpoint_dir
                    /
                    "pretrain_last.pt"
                ).resolve()
            ),
            "final_validation": final_validation,
        }

        if self.logger is not None:
            self.logger.save_run_summary(
                summary
            )

            self.logger.log(
                "Phase A pretraining finished."
            )

        return summary