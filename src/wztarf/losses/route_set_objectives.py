"""Set-level objectives for topology-diverse route hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from wztarf.data.future_topology_targets import (
    FutureTopologyTargets,
)


@dataclass(frozen=True)
class RouteCoverageLoss:
    total: torch.Tensor
    edge: torch.Tensor
    goal: torch.Tensor
    horizon: torch.Tensor
    mode_cost: torch.Tensor


def _valid_graph_edges(
    lane_edge_index: torch.Tensor,
    lane_edge_mask: torch.Tensor,
    lane_mask: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    num_lanes = lane_mask.shape[1]

    src = lane_edge_index[
        :,
        0,
    ].long()

    dst = lane_edge_index[
        :,
        1,
    ].long()

    valid = (
        lane_edge_mask.bool()
        &
        (src >= 0)
        &
        (dst >= 0)
        &
        (src < num_lanes)
        &
        (dst < num_lanes)
    )

    safe_src = src.clamp(
        0,
        max(
            num_lanes - 1,
            0,
        ),
    )

    safe_dst = dst.clamp(
        0,
        max(
            num_lanes - 1,
            0,
        ),
    )

    valid &= (
        lane_mask.bool().gather(
            1,
            safe_src,
        )
        &
        lane_mask.bool().gather(
            1,
            safe_dst,
        )
    )

    return (
        safe_src,
        safe_dst,
        valid,
    )


def topological_route_diversity_loss(
    *,
    route_edge_occupancy: torch.Tensor,
    route_viability: torch.Tensor,
    edge_viability: torch.Tensor,
    lane_edge_index: torch.Tensor,
    lane_edge_mask: torch.Tensor,
    lane_mask: torch.Tensor,
    viability_threshold: float = 0.5,
) -> torch.Tensor:
    """Penalize route overlap only up to graph alternative capacity.

    If only three structurally distinct routes appear available, only the
    three most viable route slots are required to be distinct. The remaining
    modes are allowed to reuse topology with different progress/dynamics.
    """

    (
        batch_size,
        num_modes,
        num_edges,
    ) = route_edge_occupancy.shape

    num_lanes = lane_mask.shape[1]

    safe_src, _, valid_edge = (
        _valid_graph_edges(
            lane_edge_index,
            lane_edge_mask,
            lane_mask,
        )
    )

    # ----------------------------------------------------------
    # Estimate number of distinct viable alternatives.
    #
    # capacity =
    # 1 + ? max(out_degree_viable - 1, 0)
    # ----------------------------------------------------------

    viable_for_capacity = (
        valid_edge
        &
        (
            edge_viability.detach()
            >=
            float(
                viability_threshold
            )
        )
    )

    out_degree = torch.zeros(
        batch_size,
        num_lanes,
        dtype=torch.long,
        device=route_edge_occupancy.device,
    )

    out_degree.scatter_add_(
        1,
        safe_src,
        viable_for_capacity.long(),
    )

    extra_choices = torch.relu(
        out_degree - 1
    ).sum(
        dim=-1
    )

    route_capacity = (
        1
        +
        extra_choices
    ).clamp(
        min=1,
        max=num_modes,
    )

    # Select only the required number of primary route slots.
    # Selection is detached; gradients still flow through their occupancies.
    primary_mode_mask = torch.zeros(
        batch_size,
        num_modes,
        dtype=torch.bool,
        device=route_edge_occupancy.device,
    )

    detached_viability = (
        route_viability.detach()
    )

    for b in range(
        batch_size
    ):
        capacity_b = int(
            route_capacity[
                b
            ].item()
        )

        selected = torch.topk(
            detached_viability[
                b
            ],
            k=capacity_b,
            dim=-1,
        ).indices

        primary_mode_mask[
            b,
            selected,
        ] = True

    # ----------------------------------------------------------
    # Emphasize branch edges rather than shared approach/trunk edges.
    # ----------------------------------------------------------

    edge_value = (
        edge_viability.float()
        *
        valid_edge.to(
            edge_viability.dtype
        )
    )

    outgoing_sum = edge_value.new_zeros(
        batch_size,
        num_lanes,
    )

    outgoing_sum.scatter_add_(
        1,
        safe_src,
        edge_value,
    )

    outgoing_max = edge_value.new_zeros(
        batch_size,
        num_lanes,
    )

    outgoing_max.scatter_reduce_(
        1,
        safe_src,
        edge_value,
        reduce="amax",
        include_self=True,
    )

    branch_excess = torch.relu(
        outgoing_sum
        -
        outgoing_max
    )

    branch_edge_weight = (
        branch_excess.gather(
            1,
            safe_src,
        )
        *
        valid_edge.to(
            branch_excess.dtype
        )
    )

    edge_weight = (
        branch_edge_weight
        +
        0.05
        *
        valid_edge.to(
            branch_edge_weight.dtype
        )
    )

    weighted_route = (
        route_edge_occupancy.float()
        *
        torch.sqrt(
            edge_weight[
                :,
                None,
                :,
            ].clamp_min(
                1.0e-8
            )
        )
    )

    norm = torch.linalg.vector_norm(
        weighted_route,
        dim=-1,
    ).clamp_min(
        1.0e-6
    )

    similarity = torch.einsum(
        "bke,bje->bkj",
        weighted_route,
        weighted_route,
    ) / (
        norm[
            :,
            :,
            None,
        ]
        *
        norm[
            :,
            None,
            :,
        ]
    )

    upper = torch.triu(
        torch.ones(
            num_modes,
            num_modes,
            dtype=torch.bool,
            device=route_edge_occupancy.device,
        ),
        diagonal=1,
    )[
        None
    ]

    pair_mask = (
        upper
        &
        primary_mode_mask[
            :,
            :,
            None,
        ]
        &
        primary_mode_mask[
            :,
            None,
            :,
        ]
    )

    pair_viability_sq = (
        (
            route_viability.float()[
                :,
                :,
                None,
            ]
            *
            route_viability.float()[
                :,
                None,
                :,
            ]
        ).clamp_min(
            0.0
        )
    
    ).clamp_min(0.0)

    # V3 SAFE TOPO-DIVERSITY SQRT
    # sqrt(0) is finite forward but has a singular backward derivative.
    # Keep dead route pairs exactly zero while flooring only sqrt's input.
    pair_viability = torch.where(
        pair_viability_sq > 0.0,
        torch.sqrt(
            pair_viability_sq.clamp_min(1.0e-8)
        ),
        torch.zeros_like(pair_viability_sq),
    )

    weight = (
        pair_mask.to(
            similarity.dtype
        )
        *
        pair_viability
    )

    return (
        similarity
        *
        weight
    ).sum() / weight.sum().clamp_min(
        1.0
    )


def route_set_coverage_loss(
    *,
    route_edge_occupancy: torch.Tensor,
    route_node_occupancy: torch.Tensor,
    goal_prob: torch.Tensor,
    route_anchors: torch.Tensor,
    future_xy: torch.Tensor,
    targets: FutureTopologyTargets,
    fps: int = 5,
    temperature: float = 0.35,
    edge_weight: float = 1.0,
    goal_weight: float = 1.0,
    horizon_weight: float = 1.0,
    horizon_scale_m: float = 5.0,
) -> RouteCoverageLoss:
    """Smooth best-of-K route coverage objective."""

    if temperature <= 0.0:
        raise ValueError(
            "temperature must be positive."
        )

    (
        batch_size,
        num_modes,
        _,
    ) = route_edge_occupancy.shape

    num_lanes = (
        route_node_occupancy.shape[-1]
    )

    # ----------------------------------------------------------
    # EDGE COST
    # ----------------------------------------------------------

    gt_edge = targets.edge_positive[
        :,
        None,
        :,
    ].float()

    gt_edge_count = gt_edge.sum(
        dim=-1
    ).clamp_min(
        1.0
    )

    edge_probability = (
        route_edge_occupancy.float()
        .clamp(
            1.0e-6,
            1.0 - 1.0e-6,
        )
    )

    edge_cost = (
        -torch.log(
            edge_probability
        )
        *
        gt_edge
    ).sum(
        dim=-1
    ) / gt_edge_count

    has_positive_edge = (
        targets.edge_positive.any(
            dim=-1
        )[
            :,
            None,
        ]
    )

    edge_cost = torch.where(
        has_positive_edge,
        edge_cost,
        torch.zeros_like(
            edge_cost
        ),
    )

    # ----------------------------------------------------------
    # TERMINAL ROUTE / MAP_EXIT COST
    # ----------------------------------------------------------

    safe_terminal_lane = (
        targets.terminal_lane.clamp(
            0,
            max(
                num_lanes - 1,
                0,
            ),
        )
    )

    gather_lane = safe_terminal_lane[
        :,
        None,
        None,
    ].expand(
        batch_size,
        num_modes,
        1,
    )

    terminal_lane_probability = (
        route_node_occupancy.gather(
            2,
            gather_lane,
        )
        .squeeze(
            -1
        )
        .float()
        .clamp(
            1.0e-6,
            1.0,
        )
    )

    exit_probability = (
        goal_prob[
            ...,
            -1
        ]
        .float()
        .clamp(
            1.0e-6,
            1.0,
        )
    )

    has_terminal_lane = (
        targets.terminal_lane
        >=
        0
    )[
        :,
        None,
    ]

    goal_probability = torch.where(
        targets.map_exit[
            :,
            None,
        ],
        exit_probability,
        torch.where(
            has_terminal_lane,
            terminal_lane_probability,
            torch.ones_like(
                terminal_lane_probability
            ),
        ),
    )

    goal_cost = -torch.log(
        goal_probability.clamp_min(
            1.0e-6
        )
    )

    # ----------------------------------------------------------
    # 1 / 3 / 5 s ROUTE-PROGRESS ANCHOR COST
    # ----------------------------------------------------------

    future_steps = (
        future_xy.shape[1]
    )

    horizon_indices = torch.tensor(
        [
            min(
                fps,
                future_steps,
            ) - 1,
            min(
                3 * fps,
                future_steps,
            ) - 1,
            min(
                5 * fps,
                future_steps,
            ) - 1,
        ],
        dtype=torch.long,
        device=future_xy.device,
    )

    gt_anchor = future_xy.index_select(
        1,
        horizon_indices,
    )

    horizon_error = (
        torch.linalg.vector_norm(
            route_anchors.float()
            -
            gt_anchor[
                :,
                None,
            ].float(),
            dim=-1,
        )
    )

    horizon_weights = torch.tensor(
        [
            0.5,
            1.0,
            2.0,
        ],
        dtype=horizon_error.dtype,
        device=horizon_error.device,
    )

    horizon_cost = (
        horizon_error
        *
        horizon_weights[
            None,
            None,
            :,
        ]
    ).sum(
        dim=-1
    ) / horizon_weights.sum()

    horizon_cost = (
        horizon_cost
        /
        float(
            horizon_scale_m
        )
    )

    # ----------------------------------------------------------
    # SMOOTH BEST-OF-6
    # ----------------------------------------------------------

    mode_cost = (
        float(
            edge_weight
        )
        *
        edge_cost
        +
        float(
            goal_weight
        )
        *
        goal_cost
        +
        float(
            horizon_weight
        )
        *
        horizon_cost
    )

    # -tau log(mean(exp(-C/tau)))
    #
    # Approximates min(C_k) but keeps gradient for near-correct modes.
    coverage = (
        -float(
            temperature
        )
        *
        torch.logsumexp(
            -mode_cost
            /
            float(
                temperature
            ),
            dim=1,
        )
        +
        float(
            temperature
        )
        *
        math.log(
            float(
                num_modes
            )
        )
    )

    return RouteCoverageLoss(
        total=coverage.mean(),
        edge=edge_cost.mean(),
        goal=goal_cost.mean(),
        horizon=horizon_cost.mean(),
        mode_cost=mode_cost,
    )
