"""Construct map-coverage, road-reliability, and MAP_EXIT training targets."""

from __future__ import annotations
from dataclasses import dataclass
import torch
from wztarf.geometry.workzone import points_in_polygon


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


def lane_polygons_from_features(
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
) -> list[torch.Tensor]:
    """Reconstruct lane polygons from the final 8-D lane representation.

    Expected lane feature layout:

        lane_feat[L, P, 8]

    Geometry uses:

        center = lane_feat[..., 0:2]
        left   = center + lane_feat[..., 4:6]
        right  = center + lane_feat[..., 6:8]

    Each polygon is constructed as:

        left boundary forward
        +
        right boundary reversed

    Invalid lanes and lanes with fewer than two valid points are skipped.
    """
    if lane_feat.ndim != 3:
        raise ValueError(
            "lane_feat must have shape [L, P, F]."
        )

    if lane_feat.shape[-1] < 8:
        raise ValueError(
            "lane_feat must contain at least 8 features."
        )

    if lane_point_mask.shape != lane_feat.shape[:2]:
        raise ValueError(
            "lane_point_mask must have shape [L, P]."
        )

    if lane_mask.shape != lane_feat.shape[:1]:
        raise ValueError(
            "lane_mask must have shape [L]."
        )

    polygons: list[torch.Tensor] = []

    for lane_index in range(lane_feat.shape[0]):
        if not bool(lane_mask[lane_index]):
            continue

        valid_points = lane_point_mask[lane_index].bool()

        if int(valid_points.sum()) < 2:
            continue

        lane = lane_feat[lane_index, valid_points]

        center = lane[:, 0:2]

        left = (
            center
            +
            lane[:, 4:6]
        )

        right = (
            center
            +
            lane[:, 6:8]
        )

        polygon = torch.cat(
            (
                left,
                torch.flip(right, dims=(0,)),
            ),
            dim=0,
        )

        polygons.append(polygon)

    return polygons


def _lane_support_points(
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
) -> torch.Tensor:
    """Return all valid center, left-boundary, and right-boundary points."""
    points: list[torch.Tensor] = []

    for lane_index in range(lane_feat.shape[0]):
        if not bool(lane_mask[lane_index]):
            continue

        valid_points = lane_point_mask[lane_index].bool()

        if not bool(valid_points.any()):
            continue

        lane = lane_feat[lane_index, valid_points]

        center = lane[:, 0:2]
        left = center + lane[:, 4:6]
        right = center + lane[:, 6:8]

        points.extend(
            (
                center,
                left,
                right,
            )
        )

    if not points:
        raise ValueError(
            "Cannot determine map support because no valid lane points exist."
        )

    return torch.cat(points, dim=0)


def map_bounds_from_lane_features(
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
    margin_m: float = 0.0,
) -> torch.Tensor:
    """Return the local-map bounding box `[xmin, ymin, xmax, ymax]`.

    The bounding box is a coarse coverage indicator. It is not treated as
    drivable-area geometry.
    """
    if margin_m < 0:
        raise ValueError("margin_m cannot be negative.")

    points = _lane_support_points(
        lane_feat,
        lane_point_mask,
        lane_mask,
    )

    minimum = points.amin(dim=0)
    maximum = points.amax(dim=0)

    margin = torch.tensor(
        margin_m,
        dtype=points.dtype,
        device=points.device,
    )

    return torch.stack(
        (
            minimum[0] - margin,
            minimum[1] - margin,
            maximum[0] + margin,
            maximum[1] + margin,
        )
    )


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

    bounds = map_bounds_from_lane_features(
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


def _point_to_polygon_edge_distance(
    points: torch.Tensor,
    polygon: torch.Tensor,
) -> torch.Tensor:
    """Return minimum point-to-polygon-edge distance for each point."""
    start = polygon

    end = torch.roll(
        polygon,
        shifts=-1,
        dims=0,
    )

    segment = (
        end
        -
        start
    )

    # [N, 1, 2] - [1, S, 2] -> [N, S, 2]
    point_offset = (
        points[:, None, :]
        -
        start[None, :, :]
    )

    segment_sq = (
        segment.square().sum(dim=-1)
        .clamp_min(1e-12)
    )

    projection = (
        point_offset
        *
        segment[None, :, :]
    ).sum(dim=-1)

    projection = (
        projection
        /
        segment_sq[None, :]
    ).clamp(0.0, 1.0)

    closest = (
        start[None, :, :]
        +
        projection[..., None]
        *
        segment[None, :, :]
    )

    distance = torch.linalg.vector_norm(
        points[:, None, :]
        -
        closest,
        dim=-1,
    )

    return distance.min(dim=1).values


def distance_to_lane_union(
    points: torch.Tensor,
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
) -> torch.Tensor:
    """Return distance from each point to the represented lane union.

    Distance is exactly zero when a point lies inside at least one
    reconstructed lane polygon.

    For points outside all lane polygons, the result is minimum Euclidean
    distance to any represented lane boundary.

    Args:
        points:
            `[N, 2]`.

    Returns:
        `[N]` distances in meters.
    """
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError(
            "points must have shape [N, 2]."
        )

    polygons = lane_polygons_from_features(
        lane_feat,
        lane_point_mask,
        lane_mask,
    )

    if not polygons:
        raise ValueError(
            "Cannot compute lane-union distance because no valid "
            "lane polygons exist."
        )

    best_distance = torch.full(
        (points.shape[0],),
        float("inf"),
        dtype=points.dtype,
        device=points.device,
    )

    inside_any = torch.zeros(
        points.shape[0],
        dtype=torch.bool,
        device=points.device,
    )

    for polygon in polygons:
        inside = points_in_polygon(
            points,
            polygon,
        )

        inside_any |= inside

        edge_distance = _point_to_polygon_edge_distance(
            points,
            polygon,
        )

        best_distance = torch.minimum(
            best_distance,
            edge_distance,
        )

    return torch.where(
        inside_any,
        torch.zeros_like(best_distance),
        best_distance,
    )


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
