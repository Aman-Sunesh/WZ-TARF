"""Construct map-coverage, road-reliability, and MAP_EXIT training targets."""

from __future__ import annotations
from dataclasses import dataclass

import torch

from wztarf.geometry.lanes import (
    lane_bounds,
    reconstruct_lane_polygon,
)
from wztarf.geometry.workzone import (
    distance_to_polygon,
    points_in_polygon,
)

@dataclass(frozen=True)
class GoalTarget:
    """Terminal-goal classification target for one sample.

    Attributes:
        class_index:
            Lane-class index or the MAP_EXIT class index.
            `-1` indicates an ambiguous target that should be masked.

        valid:
            Whether terminal-goal classification should be supervised.

        is_map_exit:
            Whether the target corresponds to continuation beyond the
            available local map.
    """

    class_index: int
    valid: bool
    is_map_exit: bool


def build_map_coverage_mask(
    future_xy: torch.Tensor,
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
    margin_m: float = 0.0,
) -> torch.Tensor:
    """Return a per-timestep coarse local-map coverage mask.

    Args:
        future_xy:
            Ground-truth future `[T, 2]`.

        lane_feat:
            Lane tensor `[L, P, 8]`.

        lane_point_mask:
            Valid lane-point mask `[L, P]`.

        lane_mask:
            Valid lane mask `[L]`.

        margin_m:
            Optional small expansion of the map bounding box.

    Returns:
        Boolean tensor `[T]`.

    The mask only answers whether a future point remains inside the spatial
    extent represented by the supplied local map. It does not say whether
    the point is on-road.
    """
    if future_xy.ndim != 2 or future_xy.shape[-1] != 2:
        raise ValueError(
            "future_xy must have shape [T, 2]."
        )

    bounds = lane_bounds(
        lane_feat,
        lane_point_mask,
        lane_mask,
        margin_m=margin_m,
    )

    xmin, ymin, xmax, ymax = bounds

    x = future_xy[:, 0]
    y = future_xy[:, 1]

    return (
        (x >= xmin)
        &
        (x <= xmax)
        &
        (y >= ymin)
        &
        (y <= ymax)
    )


def distance_to_lane_union(
    points: torch.Tensor,
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
) -> torch.Tensor:
    """Return distance from each point to the represented lane union."""
    distances: list[torch.Tensor] = []

    for lane_index in range(
        lane_feat.shape[0]
    ):
        if not bool(
            lane_mask[
                lane_index
            ]
        ):
            continue

        polygon = reconstruct_lane_polygon(
            lane_feat[
                lane_index
            ],
            lane_point_mask[
                lane_index
            ],
        )

        if polygon is None:
            continue

        boundary_distance = distance_to_polygon(
            points,
            polygon,
        )

        inside = points_in_polygon(
            points,
            polygon,
        )

        lane_distance = torch.where(
            inside,
            torch.zeros_like(
                boundary_distance
            ),
            boundary_distance,
        )

        distances.append(
            lane_distance
        )

    if not distances:
        return torch.full(
            (
                points.shape[0],
            ),
            float("inf"),
            dtype=points.dtype,
            device=points.device,
        )

    return torch.stack(
        distances,
        dim=0,
    ).min(
        dim=0
    ).values


def build_road_reliability_mask(
    future_xy: torch.Tensor,
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
    *,
    epsilon_gt_m: float = 0.25,
    coverage_margin_m: float = 0.0,
) -> torch.Tensor:
    """Return timesteps where lane geometry is reliable enough for supervision.

    A timestep is considered reliable only when:

    1. the ground-truth future point remains inside local-map coverage; and
    2. the GT point is inside or sufficiently close to the reconstructed
       lane union.

    The default 0.25 m tolerance is for training-label reliability only.
    It is not an off-road evaluation threshold.
    """
    if epsilon_gt_m < 0:
        raise ValueError(
            "epsilon_gt_m cannot be negative."
        )

    coverage = build_map_coverage_mask(
        future_xy,
        lane_feat,
        lane_point_mask,
        lane_mask,
        margin_m=coverage_margin_m,
    )

    distance = distance_to_lane_union(
        future_xy,
        lane_feat,
        lane_point_mask,
        lane_mask,
    )

    geometry_reliable = (
        distance
        <=
        epsilon_gt_m
    )

    return (
        coverage
        &
        geometry_reliable
    )


def build_terminal_goal_target(
    *,
    endpoint_is_covered: bool,
    terminal_lane_index: int | None,
    num_retained_lanes: int,
) -> GoalTarget:
    """Construct the deterministic terminal-lane or MAP_EXIT target.

    Target rules:

    1. Endpoint outside reliable map coverage:
           target = MAP_EXIT

    2. Endpoint inside map coverage and reliably associated with a lane:
           target = that lane index

    3. Endpoint inside map coverage but no reliable lane association:
           target is ambiguous and classification should be masked.

    `MAP_EXIT` is assigned class index `num_retained_lanes`, immediately
    after all retained lane classes.
    """
    if num_retained_lanes < 0:
        raise ValueError(
            "num_retained_lanes cannot be negative."
        )

    map_exit_index = num_retained_lanes

    if not endpoint_is_covered:
        return GoalTarget(
            class_index=map_exit_index,
            valid=True,
            is_map_exit=True,
        )

    if terminal_lane_index is None:
        return GoalTarget(
            class_index=-1,
            valid=False,
            is_map_exit=False,
        )

    if not 0 <= terminal_lane_index < num_retained_lanes:
        raise ValueError(
            f"terminal_lane_index={terminal_lane_index} is outside "
            f"[0, {num_retained_lanes})."
        )

    return GoalTarget(
        class_index=terminal_lane_index,
        valid=True,
        is_map_exit=False,
    )

# === WZTARF FAST BATCHED GEOMETRY V1 ===

def _batched_lane_polygon_edges(
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build padded polygon edge tensors for all lanes in a batch.

    Returns:
        edge_start: [B, L, E, 2]
        edge_end:   [B, L, E, 2]
        edge_valid: [B, L, E]

    The edges exactly represent the same lane polygon used by
    reconstruct_lane_polygon(): left boundary forward, end cap,
    right boundary backward, start cap.
    """
    if lane_feat.ndim != 4 or lane_feat.shape[-1] < 8:
        raise ValueError("lane_feat must have shape [B, L, P, F] with F >= 8.")

    if lane_point_mask.shape != lane_feat.shape[:3]:
        raise ValueError("lane_point_mask must have shape [B, L, P].")

    if lane_mask.shape != lane_feat.shape[:2]:
        raise ValueError("lane_mask must have shape [B, L].")

    _, _, num_points, _ = lane_feat.shape

    center = lane_feat[..., 0:2]
    left = center + lane_feat[..., 4:6]
    right = center + lane_feat[..., 6:8]

    valid = (
        lane_point_mask.bool()
        &
        lane_mask.bool().unsqueeze(-1)
    )

    counts = valid.sum(dim=-1)

    # Compact arbitrary valid-point masks while preserving original order.
    rank = valid.long().cumsum(dim=-1) - 1
    scatter_index = (
        rank.clamp_min(0)
        .unsqueeze(-1)
        .expand(-1, -1, -1, 2)
    )

    valid_value = valid.unsqueeze(-1).to(center.dtype)

    def _pack(value: torch.Tensor) -> torch.Tensor:
        packed = torch.zeros_like(value)
        packed.scatter_add_(
            2,
            scatter_index,
            value * valid_value,
        )
        return packed

    left = _pack(left)
    right = _pack(right)

    segment_position = torch.arange(
        max(num_points - 1, 0),
        device=lane_feat.device,
    ).view(1, 1, -1)

    segment_valid = (
        segment_position
        <
        (counts - 1).clamp_min(0).unsqueeze(-1)
    )

    last_index = (counts - 1).clamp_min(0)

    last_gather = (
        last_index
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(-1, -1, 1, 2)
    )

    last_left = left.gather(
        2,
        last_gather,
    ).squeeze(2)

    last_right = right.gather(
        2,
        last_gather,
    ).squeeze(2)

    has_polygon = counts >= 2

    # left boundary forward
    left_start = left[:, :, :-1]
    left_end = left[:, :, 1:]

    # right boundary backward
    right_start = right[:, :, 1:]
    right_end = right[:, :, :-1]

    # Polygon:
    # left forward -> end cap -> right backward -> start cap.
    edge_start = torch.cat(
        (
            left_start,
            last_left.unsqueeze(2),
            right_start,
            right[:, :, 0:1],
        ),
        dim=2,
    )

    edge_end = torch.cat(
        (
            left_end,
            last_right.unsqueeze(2),
            right_end,
            left[:, :, 0:1],
        ),
        dim=2,
    )

    edge_valid = torch.cat(
        (
            segment_valid,
            has_polygon.unsqueeze(-1),
            segment_valid,
            has_polygon.unsqueeze(-1),
        ),
        dim=2,
    )

    return edge_start, edge_end, edge_valid


def _point_to_lane_chunk(
    points: torch.Tensor,
    edge_start: torch.Tensor,
    edge_end: torch.Tensor,
    edge_valid: torch.Tensor,
) -> torch.Tensor:
    """Distance from points to each lane in one lane chunk.

    Args:
        points:      [B, N, 2]
        edge_start:  [B, C, E, 2]
        edge_end:    [B, C, E, 2]
        edge_valid:  [B, C, E]

    Returns:
        Lane distance [B, C, N].
        Points inside a lane polygon receive distance zero.
    """
    query = points[:, None, :, None, :]

    start = edge_start[:, :, None, :, :]
    end = edge_end[:, :, None, :, :]

    segment = end - start
    offset = query - start

    segment_length_sq = (
        segment.square()
        .sum(dim=-1)
        .clamp_min(1e-12)
    )

    alpha = (
        (offset * segment).sum(dim=-1)
        /
        segment_length_sq
    ).clamp(
        0.0,
        1.0,
    )

    closest = (
        start
        +
        alpha.unsqueeze(-1) * segment
    )

    distance_sq = (
        (query - closest)
        .square()
        .sum(dim=-1)
    )

    distance_sq = torch.where(
        edge_valid[:, :, None, :],
        distance_sq,
        torch.full_like(
            distance_sq,
            float("inf"),
        ),
    )

    boundary_distance = (
        distance_sq
        .amin(dim=-1)
        .sqrt()
    )

    # Vectorized ray casting.
    x = points[:, None, :, None, 0]
    y = points[:, None, :, None, 1]

    x1 = edge_start[:, :, None, :, 0]
    y1 = edge_start[:, :, None, :, 1]

    x2 = edge_end[:, :, None, :, 0]
    y2 = edge_end[:, :, None, :, 1]

    crosses_vertical = (
        (y1 > y)
        !=
        (y2 > y)
    )

    intersection_x = (
        (x2 - x1)
        *
        (y - y1)
        /
        (y2 - y1 + 1e-12)
        +
        x1
    )

    crossing = (
        crosses_vertical
        &
        (x < intersection_x)
        &
        edge_valid[:, :, None, :]
    )

    inside = (
        crossing.long().sum(dim=-1)
        %
        2
        ==
        1
    )

    return torch.where(
        inside,
        torch.zeros_like(boundary_distance),
        boundary_distance,
    )


def distance_to_lanes_batched(
    points: torch.Tensor,
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
    *,
    lane_chunk_size: int = 8,
) -> torch.Tensor:
    """Distance from every point to every represented lane.

    Args:
        points: [B, N, 2]

    Returns:
        [B, L, N]
    """
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("points must have shape [B, N, 2].")

    edge_start, edge_end, edge_valid = (
        _batched_lane_polygon_edges(
            lane_feat,
            lane_point_mask,
            lane_mask,
        )
    )

    num_lanes = lane_feat.shape[1]

    chunks: list[torch.Tensor] = []

    for start_index in range(
        0,
        num_lanes,
        lane_chunk_size,
    ):
        end_index = min(
            start_index + lane_chunk_size,
            num_lanes,
        )

        chunks.append(
            _point_to_lane_chunk(
                points,
                edge_start[:, start_index:end_index],
                edge_end[:, start_index:end_index],
                edge_valid[:, start_index:end_index],
            )
        )

    return torch.cat(
        chunks,
        dim=1,
    )


def distance_to_lane_union_batched(
    points: torch.Tensor,
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
    *,
    lane_chunk_size: int = 8,
) -> torch.Tensor:
    """Distance to the union of all represented lanes for batched points.

    Args:
        points: [B, N, 2]

    Returns:
        [B, N]
    """
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("points must have shape [B, N, 2].")

    edge_start, edge_end, edge_valid = (
        _batched_lane_polygon_edges(
            lane_feat,
            lane_point_mask,
            lane_mask,
        )
    )

    batch_size = points.shape[0]
    num_points = points.shape[1]
    num_lanes = lane_feat.shape[1]

    union_distance = torch.full(
        (batch_size, num_points),
        float("inf"),
        dtype=points.dtype,
        device=points.device,
    )

    for start_index in range(
        0,
        num_lanes,
        lane_chunk_size,
    ):
        end_index = min(
            start_index + lane_chunk_size,
            num_lanes,
        )

        lane_distance = _point_to_lane_chunk(
            points,
            edge_start[:, start_index:end_index],
            edge_end[:, start_index:end_index],
            edge_valid[:, start_index:end_index],
        )

        chunk_distance = lane_distance.amin(
            dim=1
        )

        union_distance = torch.minimum(
            union_distance,
            chunk_distance,
        )

    return union_distance


def build_map_coverage_mask_batched(
    points: torch.Tensor,
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
    *,
    margin_m: float = 0.0,
) -> torch.Tensor:
    """Vectorized map bounding-box coverage.

    Args:
        points: [B, N, 2]

    Returns:
        [B, N] boolean coverage mask.
    """
    if margin_m < 0:
        raise ValueError("margin_m cannot be negative.")

    center = lane_feat[..., 0:2]
    left = center + lane_feat[..., 4:6]
    right = center + lane_feat[..., 6:8]

    valid = (
        lane_point_mask.bool()
        &
        lane_mask.bool().unsqueeze(-1)
    )

    geometry = torch.stack(
        (
            center,
            left,
            right,
        ),
        dim=-2,
    )

    valid_geometry = valid.unsqueeze(
        -1
    ).unsqueeze(
        -1
    )

    minimum = torch.where(
        valid_geometry,
        geometry,
        torch.full_like(
            geometry,
            float("inf"),
        ),
    ).amin(
        dim=(1, 2, 3)
    )

    maximum = torch.where(
        valid_geometry,
        geometry,
        torch.full_like(
            geometry,
            float("-inf"),
        ),
    ).amax(
        dim=(1, 2, 3)
    )

    minimum = minimum - margin_m
    maximum = maximum + margin_m

    has_geometry = valid.any(
        dim=(1, 2)
    )

    return (
        has_geometry[:, None]
        &
        (points[..., 0] >= minimum[:, None, 0])
        &
        (points[..., 0] <= maximum[:, None, 0])
        &
        (points[..., 1] >= minimum[:, None, 1])
        &
        (points[..., 1] <= maximum[:, None, 1])
    )


def selected_lane_longitudinal_offset_batched(
    points: torch.Tensor,
    lane_indices: torch.Tensor,
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
) -> torch.Tensor:
    """Project one point per batch item onto a selected raw lane.

    Args:
        points:       [B, 2]
        lane_indices: [B]

    Returns:
        Arc-length offsets [B].
    """
    batch_size, _, num_points, _ = lane_feat.shape

    batch_index = torch.arange(
        batch_size,
        device=lane_feat.device,
    )

    selected_feat = lane_feat[
        batch_index,
        lane_indices.long(),
    ]

    selected_mask = lane_point_mask[
        batch_index,
        lane_indices.long(),
    ].bool()

    center = selected_feat[..., 0:2]

    counts = selected_mask.sum(
        dim=-1
    )

    rank = (
        selected_mask.long()
        .cumsum(dim=-1)
        -
        1
    )

    scatter_index = (
        rank.clamp_min(0)
        .unsqueeze(-1)
        .expand(-1, -1, 2)
    )

    packed = torch.zeros_like(
        center
    )

    packed.scatter_add_(
        1,
        scatter_index,
        center
        *
        selected_mask.unsqueeze(-1).to(center.dtype),
    )

    start = packed[:, :-1]
    end = packed[:, 1:]

    segment = end - start

    position = torch.arange(
        max(num_points - 1, 0),
        device=lane_feat.device,
    ).unsqueeze(0)

    valid_segment = (
        position
        <
        (counts - 1).clamp_min(0).unsqueeze(-1)
    )

    length = torch.linalg.vector_norm(
        segment,
        dim=-1,
    )

    length_sq = (
        segment.square()
        .sum(dim=-1)
        .clamp_min(1e-8)
    )

    alpha = (
        (
            (points[:, None, :] - start)
            *
            segment
        ).sum(dim=-1)
        /
        length_sq
    ).clamp(
        0.0,
        1.0,
    )

    projection = (
        start
        +
        alpha.unsqueeze(-1) * segment
    )

    distance_sq = (
        (projection - points[:, None, :])
        .square()
        .sum(dim=-1)
    )

    distance_sq = torch.where(
        valid_segment,
        distance_sq,
        torch.full_like(
            distance_sq,
            float("inf"),
        ),
    )

    closest = distance_sq.argmin(
        dim=-1
    )

    valid_length = (
        length
        *
        valid_segment.to(length.dtype)
    )

    cumulative_before = (
        torch.cumsum(
            valid_length,
            dim=-1,
        )
        -
        valid_length
    )

    gather_index = closest.unsqueeze(-1)

    offset = (
        cumulative_before.gather(
            1,
            gather_index,
        ).squeeze(1)
        +
        alpha.gather(
            1,
            gather_index,
        ).squeeze(1)
        *
        length.gather(
            1,
            gather_index,
        ).squeeze(1)
    )

    return torch.where(
        counts >= 2,
        offset,
        torch.zeros_like(offset),
    )

# === END WZTARF FAST BATCHED GEOMETRY V1 ===
