"""Assemble the individual supervised WZ-TARF objectives into one loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .classification import classification_loss
from .directional import directional_loss
from .diversity import diversity_loss
from .dynamics import dynamics_loss
from .endpoint import endpoint_loss
from .lane_goal import lane_goal_loss
from .refinement import refinement_loss
from .road_compliance import road_compliance_loss
from .route import route_loss
from .trajectory import (
    mode_assignment_cost,
    trajectory_loss,
)
from .worker_clearance import worker_clearance_loss
from .workzone_geometry import workzone_geometry_loss
from wztarf.data.targets import build_supervised_targets

@dataclass(frozen=True)
class LossWeights:
    """Weights multiplying each supervised objective.

    Values are intentionally explicit so experiments record every active
    objective rather than relying on hidden defaults.
    """

    trajectory: float
    endpoint: float
    classification: float
    lane: float
    route: float
    angle: float
    dynamics: float
    diversity: float
    road: float
    wz_geometry: float
    worker: float
    refinement: float

    def __post_init__(self) -> None:
        """Reject negative loss weights."""
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(
                    f"Loss weight '{name}' cannot be negative."
                )


@dataclass
class SupervisedLossOutput:
    """Structured output returned by the supervised loss assembly."""

    total: torch.Tensor
    components: dict[str, torch.Tensor]
    winner_idx: torch.Tensor
    assignment_cost: torch.Tensor


def _require(
    mapping: Mapping[str, Any],
    key: str,
    *,
    owner: str,
) -> Any:
    """Retrieve a required field with a useful error message."""
    if key not in mapping:
        raise KeyError(
            f"{owner} is missing required field '{key}'."
        )

    return mapping[key]


def _future_anchor_targets(
    gt_xy: torch.Tensor,
    fps: int,
) -> torch.Tensor:
    """Extract GT route anchors exactly at 1 s, 3 s, and 5 s."""
    indices = [
        fps - 1,
        3 * fps - 1,
        5 * fps - 1,
    ]

    if indices[-1] >= gt_xy.shape[1]:
        raise ValueError(
            "Ground-truth horizon is shorter than 5 seconds."
        )

    return gt_xy[
        :,
        indices,
        :,
    ]


def _extract_wz(
    wz_feat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract four polygon corners and their sample-level validity."""
    if wz_feat.ndim != 3 or wz_feat.shape[1] < 4 or wz_feat.shape[-1] < 3:
        raise ValueError(
            "wz_feat must have shape [B, >=4, >=3]."
        )

    polygon = wz_feat[
        :,
        :4,
        :2,
    ]

    valid = (
        wz_feat[
            :,
            :4,
            2,
        ]
        >
        0
    ).all(dim=1)

    return polygon, valid


def _extract_workers(
    worker_feat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract worker XY coordinates and validity masks."""
    if worker_feat.ndim != 3 or worker_feat.shape[-1] < 3:
        raise ValueError(
            "wz_worker_feat must have shape [B, W, >=3]."
        )

    return (
        worker_feat[..., :2],
        worker_feat[..., 2] > 0,
    )


def supervised_loss(
    model_output: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    weights: LossWeights,
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
) -> SupervisedLossOutput:
    """Compute the complete supervised WZ-TARF objective.

    Minimum required model output:
        pred_xy:
            `[B, K, T, 2]`

    Additional outputs are required only when their corresponding loss weight
    is greater than zero:

        mode_logits
        mode_prob
        goal_logits
        route_anchors
        goal_offset
        dynamics_xy
        coarse_xy
        refinement_delta

    Minimum required batch input:
        future_xy:
            `[B, T, 2]`

    Auxiliary training targets are required only for relevant enabled losses:

        goal_target
        goal_valid
        goal_offset_target
        lane_goal_mask
        road_reliability_mask

    Raw scene tensors used directly by safety/road losses:

        lane_feat
        lane_point_mask
        lane_mask
        wz_feat
        wz_worker_feat
    """
    pred_xy = _require(
        model_output,
        "pred_xy",
        owner="model_output",
    )

    gt_xy = _require(
        batch,
        "future_xy",
        owner="batch",
    )

    if not isinstance(pred_xy, torch.Tensor):
        raise TypeError(
            "model_output['pred_xy'] must be a tensor."
        )

    if not isinstance(gt_xy, torch.Tensor):
        raise TypeError(
            "batch['future_xy'] must be a tensor."
        )

    assignment_cost = mode_assignment_cost(
        pred_xy,
        gt_xy,
        beta_assign=beta_assign,
    )

    winner_idx = (
        assignment_cost
        .detach()
        .argmin(dim=1)
    )

    components: dict[str, torch.Tensor] = {}

    # --------------------------------------------------------------
    # Core trajectory regression
    # --------------------------------------------------------------

    components["trajectory"] = trajectory_loss(
        pred_xy,
        gt_xy,
        winner_idx=winner_idx,
    )

    components["endpoint"] = endpoint_loss(
        pred_xy,
        gt_xy,
        winner_idx,
    )

    # --------------------------------------------------------------
    # Mode classification
    # --------------------------------------------------------------

    if weights.classification > 0:
        mode_logits = _require(
            model_output,
            "mode_logits",
            owner="model_output",
        )

        components["classification"] = classification_loss(
            mode_logits,
            assignment_cost,
            temperature=classification_temperature,
        )

    # --------------------------------------------------------------
    # Terminal lane / MAP_EXIT classification
    # --------------------------------------------------------------

    if weights.lane > 0:
        components["lane"] = lane_goal_loss(
            goal_logits=_require(
                model_output,
                "goal_logits",
                owner="model_output",
            ),
            goal_target=_require(
                batch,
                "goal_target",
                owner="batch",
            ),
            goal_valid=_require(
                batch,
                "goal_valid",
                owner="batch",
            ),
            winner_idx=winner_idx,
        )

    # --------------------------------------------------------------
    # 1 s / 3 s / 5 s route anchors and in-map goal progress
    # --------------------------------------------------------------

    if weights.route > 0:
        route_anchors = _require(
            model_output,
            "route_anchors",
            owner="model_output",
        )

        anchor_target = _future_anchor_targets(
            gt_xy,
            fps=fps,
        )

        has_offset = (
            "goal_offset" in model_output
            and
            "goal_offset_target" in batch
            and
            "lane_goal_mask" in batch
        )

        if has_offset:
            components["route"] = route_loss(
                route_anchors,
                anchor_target,
                winner_idx,
                goal_offset_pred=model_output["goal_offset"],
                goal_offset_target=batch["goal_offset_target"],
                lane_goal_mask=batch["lane_goal_mask"],
            )

        else:
            components["route"] = route_loss(
                route_anchors,
                anchor_target,
                winner_idx,
            )

    # --------------------------------------------------------------
    # Direction
    # --------------------------------------------------------------

    if weights.angle > 0:
        components["angle"] = directional_loss(
            pred_xy,
            gt_xy,
            winner_idx,
        )

    # --------------------------------------------------------------
    # Shared control-dynamics anchor
    # --------------------------------------------------------------

    if weights.dynamics > 0:
        components["dynamics"] = dynamics_loss(
            _require(
                model_output,
                "dynamics_xy",
                owner="model_output",
            ),
            gt_xy,
            horizon_steps=dynamics_horizon_steps,
        )

    # --------------------------------------------------------------
    # Optional route diversity
    # --------------------------------------------------------------

    if weights.diversity > 0:
        components["diversity"] = diversity_loss(
            _require(
                model_output,
                "route_anchors",
                owner="model_output",
            ),
            min_separation_m=diversity_separation_m,
        )

    # --------------------------------------------------------------
    # Coverage-aware ordinary road compliance
    # --------------------------------------------------------------

    if weights.road > 0:
        components["road"] = road_compliance_loss(
            pred_xy=pred_xy,
            road_reliability_mask=_require(
                batch,
                "road_reliability_mask",
                owner="batch",
            ),
            lane_feat=_require(
                batch,
                "lane_feat",
                owner="batch",
            ),
            lane_point_mask=_require(
                batch,
                "lane_point_mask",
                owner="batch",
            ),
            lane_mask=_require(
                batch,
                "lane_mask",
                owner="batch",
            ),
            epsilon_pred_m=road_tolerance_m,
        )

    # --------------------------------------------------------------
    # WorkZone geometry
    # --------------------------------------------------------------

    mode_prob: torch.Tensor | None = None

    if (
        weights.wz_geometry > 0
        or
        weights.worker > 0
    ):
        if "mode_prob" in model_output:
            mode_prob = model_output["mode_prob"]

        elif "mode_logits" in model_output:
            mode_prob = torch.softmax(
                model_output["mode_logits"],
                dim=-1,
            )

        else:
            raise KeyError(
                "Safety losses require either 'mode_prob' "
                "or 'mode_logits' in model_output."
            )

    if weights.wz_geometry > 0:
        wz_polygon, wz_valid = _extract_wz(
            _require(
                batch,
                "wz_feat",
                owner="batch",
            )
        )

        assert mode_prob is not None

        components["wz_geometry"] = workzone_geometry_loss(
            pred_xy,
            mode_prob,
            wz_polygon,
            wz_valid=wz_valid,
            temperature_m=wz_temperature_m,
        )

    # --------------------------------------------------------------
    # Worker clearance
    # --------------------------------------------------------------

    if weights.worker > 0:
        worker_xy, worker_mask = _extract_workers(
            _require(
                batch,
                "wz_worker_feat",
                owner="batch",
            )
        )

        assert mode_prob is not None

        components["worker"] = worker_clearance_loss(
            pred_xy,
            mode_prob,
            worker_xy,
            worker_mask,
            threshold_m=worker_threshold_m,
        )

    # --------------------------------------------------------------
    # Optional staged refinement
    # --------------------------------------------------------------

    if weights.refinement > 0:
        components["refinement"] = refinement_loss(
            coarse_xy=_require(
                model_output,
                "coarse_xy",
                owner="model_output",
            ),
            refinement_delta=_require(
                model_output,
                "refinement_delta",
                owner="model_output",
            ),
            gt_xy=gt_xy,
            winner_idx=winner_idx,
        )

    # --------------------------------------------------------------
    # Weighted total
    # --------------------------------------------------------------

    weighted_terms = {
        "trajectory": weights.trajectory,
        "endpoint": weights.endpoint,
        "classification": weights.classification,
        "lane": weights.lane,
        "route": weights.route,
        "angle": weights.angle,
        "dynamics": weights.dynamics,
        "diversity": weights.diversity,
        "road": weights.road,
        "wz_geometry": weights.wz_geometry,
        "worker": weights.worker,
        "refinement": weights.refinement,
    }

    total = pred_xy.sum() * 0.0

    for name, value in components.items():
        total = (
            total
            +
            weighted_terms[name]
            *
            value
        )

    return SupervisedLossOutput(
        total=total,
        components=components,
        winner_idx=winner_idx,
        assignment_cost=assignment_cost,
    )
