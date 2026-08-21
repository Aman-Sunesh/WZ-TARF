"""Pseudo-GT temporary-topology targets from the observed future path."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FutureTopologyTargets:
    """Graph targets inferred from one realized future trajectory."""

    lane_sequence: torch.Tensor
    lane_sequence_valid: torch.Tensor

    node_positive: torch.Tensor
    edge_positive: torch.Tensor

    transition_edge_index: torch.Tensor
    transition_mask: torch.Tensor

    terminal_lane: torch.Tensor
    map_exit: torch.Tensor


def _future_to_lane_distance(
    future_xy: torch.Tensor,
    lane_centerline: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
    *,
    time_chunk: int = 5,
) -> torch.Tensor:
    """Approximate future-to-lane distance with dense centerline points.

    Returns:
        [B,T,L]

    The calculation is target-only, so no gradients are required. Time
    chunking prevents creation of one enormous [B,T,L,P,2] tensor.
    """

    if future_xy.ndim != 3 or future_xy.shape[-1] != 2:
        raise ValueError(
            "future_xy must have shape [B,T,2]."
        )

    if (
        lane_centerline.ndim != 4
        or
        lane_centerline.shape[-1] != 2
    ):
        raise ValueError(
            "lane_centerline must have shape [B,L,P,2]."
        )

    batch_size, future_steps, _ = (
        future_xy.shape
    )

    (
        lane_batch,
        num_lanes,
        num_points,
        _,
    ) = lane_centerline.shape

    if lane_batch != batch_size:
        raise ValueError(
            "future/lane batch sizes differ."
        )

    if lane_point_mask.shape != (
        batch_size,
        num_lanes,
        num_points,
    ):
        raise ValueError(
            "lane_point_mask has wrong shape."
        )

    if lane_mask.shape != (
        batch_size,
        num_lanes,
    ):
        raise ValueError(
            "lane_mask has wrong shape."
        )

    valid_point = (
        lane_point_mask.bool()
        &
        lane_mask[
            :,
            :,
            None,
        ].bool()
    )

    chunks = []

    for start in range(
        0,
        future_steps,
        time_chunk,
    ):
        end = min(
            start + time_chunk,
            future_steps,
        )

        point = future_xy[
            :,
            start:end,
            None,
            None,
            :,
        ]

        center = lane_centerline[
            :,
            None,
            :,
            :,
            :,
        ]

        distance_sq = (
            point
            -
            center
        ).square().sum(
            dim=-1
        )

        distance_sq = (
            distance_sq.masked_fill(
                ~valid_point[
                    :,
                    None,
                    :,
                    :,
                ],
                float("inf"),
            )
        )

        lane_distance_sq = (
            distance_sq.min(
                dim=-1
            ).values
        )

        chunks.append(
            torch.sqrt(
                lane_distance_sq
                .clamp_min(0.0)
            )
        )

    return torch.cat(
        chunks,
        dim=1,
    )


@torch.no_grad()
def build_future_topology_targets(
    *,
    future_xy: torch.Tensor,
    lane_centerline: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
    lane_edge_index: torch.Tensor,
    lane_edge_mask: torch.Tensor,
    map_coverage: torch.Tensor | None = None,
    match_radius_m: float = 2.25,
    transition_penalty: float = 0.05,
    jump_penalty: float = 2.0,
) -> FutureTopologyTargets:
    """Project the realized future onto a graph-compatible lane sequence.

    Viterbi transition policy:

        same lane       -> 0
        represented edge -> small transition penalty
        arbitrary jump   -> larger jump penalty

    The jump fallback makes pseudo-label generation robust to occasional
    incomplete graph connectivity. Jumps do NOT become positive edge labels.
    """

    if match_radius_m <= 0.0:
        raise ValueError(
            "match_radius_m must be positive."
        )

    if transition_penalty < 0.0:
        raise ValueError(
            "transition_penalty cannot be negative."
        )

    if jump_penalty <= transition_penalty:
        raise ValueError(
            "jump_penalty must exceed transition_penalty."
        )

    (
        batch_size,
        future_steps,
        _,
    ) = future_xy.shape

    num_lanes = lane_centerline.shape[1]
    num_edges = lane_edge_index.shape[-1]

    lane_mask = lane_mask.bool()
    lane_edge_mask = lane_edge_mask.bool()

    # ----------------------------------------------------------
    # Emission distance d(y_t, lane_i).
    # ----------------------------------------------------------

    emission = _future_to_lane_distance(
        future_xy.float(),
        lane_centerline.float(),
        lane_point_mask,
        lane_mask,
    )

    emission = emission.masked_fill(
        ~lane_mask[
            :,
            None,
            :,
        ],
        float("inf"),
    )

    # ----------------------------------------------------------
    # Build graph-constrained transition matrix [B,L,L].
    # ----------------------------------------------------------

    transition = torch.full(
        (
            batch_size,
            num_lanes,
            num_lanes,
        ),
        float(jump_penalty),
        dtype=emission.dtype,
        device=emission.device,
    )

    diagonal = torch.arange(
        num_lanes,
        device=emission.device,
    )

    transition[
        :,
        diagonal,
        diagonal,
    ] = 0.0

    src = lane_edge_index[
        :,
        0,
    ].long()

    dst = lane_edge_index[
        :,
        1,
    ].long()

    valid_edge = (
        lane_edge_mask
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

    valid_edge &= (
        lane_mask.gather(
            1,
            safe_src,
        )
        &
        lane_mask.gather(
            1,
            safe_dst,
        )
    )

    batch_grid = torch.arange(
        batch_size,
        device=emission.device,
    )[:, None].expand(
        batch_size,
        num_edges,
    )

    transition[
        batch_grid[valid_edge],
        safe_src[valid_edge],
        safe_dst[valid_edge],
    ] = float(
        transition_penalty
    )

    # ----------------------------------------------------------
    # Batched Viterbi.
    # ----------------------------------------------------------

    dp = emission[
        :,
        0,
    ]

    back_pointer = []

    for t in range(
        1,
        future_steps,
    ):
        candidate = (
            dp[
                :,
                :,
                None,
            ]
            +
            transition
        )

        best_cost, best_prev = (
            candidate.min(
                dim=1
            )
        )

        dp = (
            best_cost
            +
            emission[
                :,
                t,
            ]
        )

        back_pointer.append(
            best_prev
        )

    current = dp.argmin(
        dim=-1
    )

    lane_sequence = torch.empty(
        batch_size,
        future_steps,
        dtype=torch.long,
        device=future_xy.device,
    )

    lane_sequence[
        :,
        -1,
    ] = current

    batch_index = torch.arange(
        batch_size,
        device=future_xy.device,
    )

    for t in range(
        future_steps - 2,
        -1,
        -1,
    ):
        current = back_pointer[
            t
        ][
            batch_index,
            current,
        ]

        lane_sequence[
            :,
            t,
        ] = current

    selected_distance = emission.gather(
        2,
        lane_sequence[
            ...,
            None,
        ],
    ).squeeze(
        -1
    )

    lane_sequence_valid = (
        torch.isfinite(
            selected_distance
        )
        &
        (
            selected_distance
            <=
            float(match_radius_m)
        )
    )

    # ----------------------------------------------------------
    # MAP_EXIT.
    #
    # Prefer the exact cached represented-map coverage whenever
    # available. Otherwise fall back to terminal lane distance.
    # ----------------------------------------------------------

    if map_coverage is not None:
        if map_coverage.shape != (
            batch_size,
            future_steps,
        ):
            raise ValueError(
                "map_coverage must have shape [B,T]."
            )

        map_exit = ~map_coverage[
            :,
            -1,
        ].bool()

    else:
        map_exit = (
            ~lane_sequence_valid[
                :,
                -1,
            ]
        )

    # ----------------------------------------------------------
    # Positive node occupancy.
    #
    # IMPORTANT: unvisited nodes are UNLABELED, not negatives.
    # ----------------------------------------------------------

    node_positive = torch.zeros(
        batch_size,
        num_lanes,
        dtype=torch.bool,
        device=future_xy.device,
    )

    safe_sequence = (
        lane_sequence.clamp(
            0,
            max(
                num_lanes - 1,
                0,
            ),
        )
    )

    node_positive.scatter_(
        1,
        safe_sequence,
        lane_sequence_valid,
    )

    # scatter_ can overwrite repeated destinations. Restore robust
    # "any visited timestep" semantics with scatter_add.
    node_count = torch.zeros(
        batch_size,
        num_lanes,
        dtype=torch.long,
        device=future_xy.device,
    )

    node_count.scatter_add_(
        1,
        safe_sequence,
        lane_sequence_valid.long(),
    )

    node_positive = (
        node_count > 0
    )

    # ----------------------------------------------------------
    # Match realized lane changes to represented graph edges.
    # ----------------------------------------------------------

    if future_steps > 1:
        route_src = lane_sequence[
            :,
            :-1,
        ]

        route_dst = lane_sequence[
            :,
            1:,
        ]

        pair_valid = (
            lane_sequence_valid[
                :,
                :-1,
            ]
            &
            lane_sequence_valid[
                :,
                1:,
            ]
            &
            (
                route_src
                !=
                route_dst
            )
        )

        edge_match = (
            valid_edge[
                :,
                None,
                :,
            ]
            &
            (
                safe_src[
                    :,
                    None,
                    :,
                ]
                ==
                route_src[
                    :,
                    :,
                    None,
                ]
            )
            &
            (
                safe_dst[
                    :,
                    None,
                    :,
                ]
                ==
                route_dst[
                    :,
                    :,
                    None,
                ]
            )
        )

        edge_match &= pair_valid[
            :,
            :,
            None,
        ]

        transition_mask = (
            edge_match.any(
                dim=-1
            )
        )

        transition_edge_index = (
            edge_match.float()
            .argmax(
                dim=-1
            )
            .long()
        )

        transition_edge_index = (
            torch.where(
                transition_mask,
                transition_edge_index,
                torch.full_like(
                    transition_edge_index,
                    -1,
                ),
            )
        )

    else:
        transition_mask = torch.zeros(
            batch_size,
            0,
            dtype=torch.bool,
            device=future_xy.device,
        )

        transition_edge_index = torch.empty(
            batch_size,
            0,
            dtype=torch.long,
            device=future_xy.device,
        )

    edge_positive = torch.zeros(
        batch_size,
        num_edges,
        dtype=torch.bool,
        device=future_xy.device,
    )

    if future_steps > 1:
        safe_transition_edge = (
            transition_edge_index
            .clamp(
                0,
                max(
                    num_edges - 1,
                    0,
                ),
            )
        )

        edge_count = torch.zeros(
            batch_size,
            num_edges,
            dtype=torch.long,
            device=future_xy.device,
        )

        edge_count.scatter_add_(
            1,
            safe_transition_edge,
            transition_mask.long(),
        )

        edge_positive = (
            edge_count > 0
        )

    # ----------------------------------------------------------
    # Last reliable represented lane.
    # ----------------------------------------------------------

    time_index = torch.arange(
        future_steps,
        device=future_xy.device,
    )[
        None,
        :
    ].expand(
        batch_size,
        future_steps,
    )

    valid_time_index = torch.where(
        lane_sequence_valid,
        time_index,
        torch.full_like(
            time_index,
            -1,
        ),
    )

    last_valid_time = (
        valid_time_index.max(
            dim=-1
        ).values
    )

    terminal_lane = torch.full(
        (
            batch_size,
        ),
        -1,
        dtype=torch.long,
        device=future_xy.device,
    )

    has_terminal_lane = (
        last_valid_time >= 0
    )

    if future_steps > 0:
        safe_last_time = (
            last_valid_time.clamp_min(
                0
            )
        )

        gathered_terminal = (
            lane_sequence[
                batch_index,
                safe_last_time,
            ]
        )

        terminal_lane = torch.where(
            has_terminal_lane,
            gathered_terminal,
            terminal_lane,
        )

    return FutureTopologyTargets(
        lane_sequence=lane_sequence,
        lane_sequence_valid=lane_sequence_valid,
        node_positive=node_positive,
        edge_positive=edge_positive,
        transition_edge_index=transition_edge_index,
        transition_mask=transition_mask,
        terminal_lane=terminal_lane,
        map_exit=map_exit,
    )
