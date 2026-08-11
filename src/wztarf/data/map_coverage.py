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
