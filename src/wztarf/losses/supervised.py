"""Assemble the individual supervised WZ-TARF objectives into one loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from wztarf.losses.route_set_objectives import (
    route_set_coverage_loss,
    topological_route_diversity_loss,
)

from wztarf.data.future_topology_targets import build_future_topology_targets
from wztarf.losses.future_topology import future_topology_supervision_loss
from wztarf.losses.mode_ranking import mode_ranking_loss

from .classification import classification_loss
from .directional import directional_loss
from .diversity import diversity_loss
from .dynamics import dynamics_loss
from .endpoint import endpoint_loss
from .lane_goal import lane_goal_loss
from .refinement import refinement_loss
from .route_progress_supervision import route_progress_supervision_loss
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

    # V3 mode ranking.
    behavior: float
    ranking_quality: float
    ranking_pairwise: float

    lane: float
    topology: float
    topo_diversity: float
    route_coverage: float
    route: float
    angle: float
    dynamics: float
    diversity: float
    road: float
    wz_geometry: float
    worker: float
    refinement: float

    # V3: bind longitudinal progress to the SAME route mode.
    route_progress_supervision: float = 0.0

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

    Raw scene tensors used to construct map-aware targets and safety losses:

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

    # --------------------------------------------------------------
    # DIRECT-K6 METRIC-ALIGNED REGRESSION WINNERS
    #
    # Evaluation computes minADE_K and minFDE_K independently.
    # A single ADE+beta*FDE winner unnecessarily couples the two
    # regression objectives. Keep the legacy assignment winner for
    # classification / auxiliary compatibility, but route the two
    # core regression losses to their exact metric winners.
    # --------------------------------------------------------------

    metric_displacement = torch.linalg.vector_norm(
        pred_xy
        -
        gt_xy[:, None],
        dim=-1,
    )

    trajectory_winner_idx = (
        metric_displacement
        .mean(dim=-1)
        .detach()
        .argmin(dim=1)
    )

    endpoint_winner_idx = (
        metric_displacement[
            :,
            :,
            -1,
        ]
        .detach()
        .argmin(dim=1)
    )

    generated_targets = None

    need_map_targets = (
        weights.lane > 0
        or weights.road > 0
        or (
            weights.route > 0
            and "goal_offset" in model_output
        )
    )

    if need_map_targets:
        retained_lane_mask = _require(
            model_output,
            "lane_mask",
            owner="model_output",
        )

        generated_targets = build_supervised_targets(
            gt_xy=gt_xy,
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
            retained_lane_mask=retained_lane_mask,
            association_tolerance_m=goal_association_tolerance_m,
            road_gt_tolerance_m=road_gt_tolerance_m,
        )

    components: dict[str, torch.Tensor] = {}

    # ==============================================================
    # V3 FUTURE TOPOLOGY SUPERVISION
    # ==============================================================
    components["topology"] = gt_xy.new_zeros(())

    if weights.topology > 0.0:
        with torch.no_grad():
            future_topology_targets = build_future_topology_targets(
                future_xy=gt_xy,
                lane_centerline=batch["lane_feat"][..., :2],
                lane_point_mask=batch["lane_point_mask"],
                lane_mask=retained_lane_mask.bool(),
                lane_edge_index=batch["lane_edge_index"],
                lane_edge_mask=batch["lane_edge_mask"],
                map_coverage=batch.get(
                    "static_map_coverage"
                ),
                match_radius_m=2.25,
                transition_penalty=0.05,
                jump_penalty=2.0,
            )

        topology_output = future_topology_supervision_loss(
            node_viability=_require(
                model_output,
                "node_viability",
                owner="model_output",
            ),
            edge_viability=_require(
                model_output,
                "edge_viability",
                owner="model_output",
            ),
            route_edge_occupancy=_require(
                model_output,
                "route_edge_occupancy",
                owner="model_output",
            ),
            goal_prob=_require(
                model_output,
                "goal_prob",
                owner="model_output",
            ),
            targets=future_topology_targets,
            blocked_edge_compatibility=batch.get(
                "static_topology_edge_compatibility"
            ),
            blocked_edge_mask=batch.get(
                "static_topology_edge_mask"
            ),
            blocked_negative_weight=0.25,
        )

        components["topology"] = (
            topology_output.total.to(
                gt_xy.dtype
            )
        )

    # --------------------------------------------------------------
    # Core trajectory regression
    # --------------------------------------------------------------

    components["trajectory"] = trajectory_loss(
        pred_xy,
        gt_xy,
        winner_idx=trajectory_winner_idx,
    )

    components["endpoint"] = endpoint_loss(
        pred_xy,
        gt_xy,
        endpoint_winner_idx,
    )

    # --------------------------------------------------------------
    # V3 PER-ROUTE LONGITUDINAL PROGRESS SUPERVISION
    #
    # Bind progress[k] to the GT projection coordinate on route[k].
    # This prevents route/progress mode swapping under set-level losses.
    # --------------------------------------------------------------

    components["route_progress_supervision"] = (
        gt_xy.new_zeros(())
    )

    if weights.route_progress_supervision > 0.0:
        components["route_progress_supervision"] = (
            route_progress_supervision_loss(
                route_progress=_require(
                    model_output,
                    "route_progress",
                    owner="model_output",
                ),
                route_progress_sequence=_require(
                    model_output,
                    "route_progress_sequence",
                    owner="model_output",
                ),
                dense_route_guide=_require(
                    model_output,
                    "dense_route_guide",
                    owner="model_output",
                ),
                route_walk_xy=_require(
                    model_output,
                    "route_walk_xy",
                    owner="model_output",
                ),
                route_walk_s=_require(
                    model_output,
                    "route_walk_s",
                    owner="model_output",
                ),
                future_xy=gt_xy,
                fps=fps,
                scale_m=5.0,
            )
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
        assert generated_targets is not None
        
        components["lane"] = lane_goal_loss(
            goal_logits=_require(
                model_output,
                "goal_logits",
                owner="model_output",
            ),
            goal_target=generated_targets.goal_target,
            goal_valid=generated_targets.goal_valid,
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

        if (
            "goal_offset" in model_output
            and generated_targets is not None
        ):
            # Longitudinal offsets are lane-relative coordinates.
            # Supervise the offset only when the WTA trajectory mode
            # selected the same lane as the GT lane target. Otherwise
            # we would compare distances measured along different lanes.
            goal_logits_for_offset = _require(
                model_output,
                "goal_logits",
                owner="model_output",
            )

            route_batch_idx = torch.arange(
                gt_xy.shape[0],
                device=gt_xy.device,
            )

            winner_goal_logits = goal_logits_for_offset[
                route_batch_idx,
                winner_idx,
            ]

            # Last goal-logit class is MAP_EXIT.
            num_lane_classes = winner_goal_logits.shape[-1] - 1

            winner_goal_class = winner_goal_logits.argmax(
                dim=-1
            )

            winner_goal_lane = winner_goal_logits[
                :,
                :num_lane_classes,
            ].argmax(
                dim=-1
            )

            winner_uses_lane = (
                winner_goal_class
                !=
                num_lane_classes
            )

            # Longitudinal offset is meaningful only when:
            #   1. GT endpoint has an associated retained lane,
            #   2. winner predicts a lane rather than MAP_EXIT,
            #   3. winner's lane equals the GT lane.
            route_offset_mask = (
                generated_targets.lane_goal_mask.bool()
                &
                winner_uses_lane
                &
                (
                    winner_goal_lane
                    ==
                    generated_targets.goal_target
                )
            )

            # === V3 ROUTE-PROGRESS: RETIRE LANE-LOCAL OFFSET LOSS ===
            route_offset_mask = torch.zeros_like(
                route_offset_mask,
                dtype=torch.bool,
            )

            components["route"] = route_loss(
                route_anchors,
                anchor_target,
                winner_idx,
                horizon_weights=(0.5, 1.0, 2.0),
                goal_offset_pred=model_output["goal_offset"],
                goal_offset_target=generated_targets.goal_offset_target,
                lane_goal_mask=route_offset_mask,
            )

        else:
            components["route"] = route_loss(
                route_anchors,
                anchor_target,
                winner_idx,
                horizon_weights=(0.5, 1.0, 2.0),
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
        assert generated_targets is not None
        
        components["road"] = road_compliance_loss(
            pred_xy=pred_xy,
            road_reliability_mask=generated_targets.road_reliability_mask,
            lane_feat=batch["lane_feat"],
            lane_point_mask=batch["lane_point_mask"],
            lane_mask=batch["lane_mask"],
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

    # ==============================================================
    # V3 TOPOLOGICAL DIVERSITY + SET-LEVEL ROUTE COVERAGE
    # ==============================================================
    components["topo_diversity"] = gt_xy.new_zeros(())
    components["route_coverage"] = gt_xy.new_zeros(())

    if weights.topo_diversity > 0.0:
        components["topo_diversity"] = topological_route_diversity_loss(
            route_edge_occupancy=_require(
                model_output,
                "route_edge_occupancy",
                owner="model_output",
            ),
            route_viability=_require(
                model_output,
                "route_viability",
                owner="model_output",
            ),
            edge_viability=_require(
                model_output,
                "edge_viability",
                owner="model_output",
            ),
            lane_edge_index=batch["lane_edge_index"],
            lane_edge_mask=batch["lane_edge_mask"],
            lane_mask=retained_lane_mask.bool(),
        )

    if weights.route_coverage > 0.0:
        if "future_topology_targets" not in locals():
            with torch.no_grad():
                future_topology_targets = build_future_topology_targets(
                    future_xy=gt_xy,
                    lane_centerline=batch["lane_feat"][..., :2],
                    lane_point_mask=batch["lane_point_mask"],
                    lane_mask=retained_lane_mask.bool(),
                    lane_edge_index=batch["lane_edge_index"],
                    lane_edge_mask=batch["lane_edge_mask"],
                    map_coverage=batch.get(
                        "static_map_coverage"
                    ),
                    match_radius_m=2.25,
                    transition_penalty=0.05,
                    jump_penalty=2.0,
                )

        coverage_output = route_set_coverage_loss(
            route_edge_occupancy=_require(
                model_output,
                "route_edge_occupancy",
                owner="model_output",
            ),
            route_node_occupancy=_require(
                model_output,
                "route_node_occupancy",
                owner="model_output",
            ),
            goal_prob=_require(
                model_output,
                "goal_prob",
                owner="model_output",
            ),
            route_anchors=_require(
                model_output,
                "route_anchors",
                owner="model_output",
            ),
            future_xy=gt_xy,
            targets=future_topology_targets,
            fps=5,
            temperature=0.35,
            edge_weight=1.0,
            goal_weight=1.0,
            horizon_weight=1.0,
            horizon_scale_m=5.0,
        )

        components["route_coverage"] = (
            coverage_output.total.to(
                gt_xy.dtype
            )
        )

    # ==============================================================
    # V3 QUALITY-AWARE MODE RANKING
    # ==============================================================

    components["behavior"] = gt_xy.new_zeros(())
    components["ranking_quality"] = gt_xy.new_zeros(())
    components["ranking_pairwise"] = gt_xy.new_zeros(())

    ranking_active = (
        weights.behavior > 0.0
        or
        weights.ranking_quality > 0.0
        or
        weights.ranking_pairwise > 0.0
    )

    if ranking_active:
        route_cost_for_ranking = (
            coverage_output.mode_cost
            if "coverage_output" in locals()
            else None
        )

        ranking_output = mode_ranking_loss(
            behavior_logits=_require(
                model_output,
                "behavior_logits",
                owner="model_output",
            ),
            quality_score=_require(
                model_output,
                "quality_score",
                owner="model_output",
            ),
            ranking_logits=_require(
                model_output,
                "ranking_logits",
                owner="model_output",
            ),
            pred_xy=pred_xy,
            gt_xy=gt_xy,
            route_cost=route_cost_for_ranking,
            fps=fps,
        )

        components["behavior"] = (
            ranking_output.behavior.to(
                gt_xy.dtype
            )
        )

        components["ranking_quality"] = (
            ranking_output.quality.to(
                gt_xy.dtype
            )
        )

        components["ranking_pairwise"] = (
            ranking_output.pairwise.to(
                gt_xy.dtype
            )
        )

    # ==============================================================
    # SINGLE AUTHORITATIVE WEIGHTED TOTAL
    #
    # Do this only after EVERY component has been constructed.
    # ==============================================================

    weighted_terms = {
        "trajectory": weights.trajectory,
        "endpoint": weights.endpoint,
        "classification": weights.classification,
        "behavior": weights.behavior,
        "ranking_quality": weights.ranking_quality,
        "ranking_pairwise": weights.ranking_pairwise,
        "lane": weights.lane,
        "topology": weights.topology,
        "topo_diversity": weights.topo_diversity,
        "route_coverage": weights.route_coverage,
        "route_progress_supervision": weights.route_progress_supervision,
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
        if name not in weighted_terms:
            raise KeyError(
                f"No configured loss weight exists for component '{name}'."
            )

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
