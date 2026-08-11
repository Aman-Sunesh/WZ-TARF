"""Compute polygon membership and distance for restricted WorkZone geometry."""

from __future__ import annotations

import torch


def _validate_points_polygon(
    points: torch.Tensor,
    polygon: torch.Tensor,
) -> None:
    """Validate point and polygon tensor shapes."""
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError(
            "points must have shape [N, 2]."
        )

    if polygon.ndim != 2 or polygon.shape[-1] != 2:
        raise ValueError(
            "polygon must have shape [P, 2]."
        )

    if polygon.shape[0] < 3:
        raise ValueError(
            "A polygon requires at least three vertices."
        )

    if not torch.isfinite(points).all():
        raise ValueError(
            "points contains NaN or infinite values."
        )

    if not torch.isfinite(polygon).all():
        raise ValueError(
            "polygon contains NaN or infinite values."
        )


def points_in_polygon(
    points: torch.Tensor,
    polygon: torch.Tensor,
) -> torch.Tensor:
    """Return whether each point lies inside the polygon.

    Uses the standard ray-casting algorithm.

    Args:
        points:
            Query points `[N, 2]`.

        polygon:
            Ordered polygon vertices `[P, 2]`.

    Returns:
        Boolean tensor `[N]`.

    This function is intended for exact evaluation logic such as WZ-GVR.
    """
    _validate_points_polygon(
        points,
        polygon,
    )

    x = points[:, 0]
    y = points[:, 1]

    px = polygon[:, 0]
    py = polygon[:, 1]

    inside = torch.zeros(
        points.shape[0],
        dtype=torch.bool,
        device=points.device,
    )

    previous = (
        polygon.shape[0]
        -
        1
    )

    for current in range(polygon.shape[0]):
        x_current = px[current]
        y_current = py[current]

        x_previous = px[previous]
        y_previous = py[previous]

        crosses_vertical_range = (
            (y_current > y)
            !=
            (y_previous > y)
        )

        intersection_x = (
            (x_previous - x_current)
            *
            (y - y_current)
            /
            (
                y_previous
                -
                y_current
                +
                1e-12
            )
            +
            x_current
        )

        crosses_ray = (
            crosses_vertical_range
            &
            (x < intersection_x)
        )

        inside ^= crosses_ray

        previous = current

    return inside


def distance_to_polygon(
    points: torch.Tensor,
    polygon: torch.Tensor,
) -> torch.Tensor:
    """Return minimum Euclidean distance to the polygon boundary.

    Args:
        points:
            Query points `[N, 2]`.

        polygon:
            Ordered polygon vertices `[P, 2]`.

    Returns:
        Distance `[N]` in the same spatial units as the inputs.

    Points inside the polygon still receive their distance to the closest
    polygon edge. Use `signed_distance_to_polygon()` when inside/outside
    information is required.
    """
    _validate_points_polygon(
        points,
        polygon,
    )

    segment_start = polygon

    segment_end = torch.roll(
        polygon,
        shifts=-1,
        dims=0,
    )

    segment = (
        segment_end
        -
        segment_start
    )

    # [N, P, 2]
    offset = (
        points[:, None, :]
        -
        segment_start[None, :, :]
    )

    segment_length_sq = (
        segment.square()
        .sum(dim=-1)
        .clamp_min(1e-12)
    )

    projection = (
        offset
        *
        segment[None, :, :]
    ).sum(dim=-1)

    projection = (
        projection
        /
        segment_length_sq[None, :]
    ).clamp(
        0.0,
        1.0,
    )

    closest = (
        segment_start[None, :, :]
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

    return distance.min(
        dim=1
    ).values


def signed_distance_to_polygon(
    points: torch.Tensor,
    polygon: torch.Tensor,
) -> torch.Tensor:
    """Return signed distance to the restricted polygon boundary.

    Sign convention used throughout WZ-TARF:

        positive = outside restricted WorkZone
        negative = inside restricted WorkZone
        zero     = polygon boundary

    Args:
        points:
            Query points `[N, 2]`.

        polygon:
            Ordered polygon vertices `[P, 2]`.

    Returns:
        Signed distance tensor `[N]`.
    """
    distance = distance_to_polygon(
        points,
        polygon,
    )

    inside = points_in_polygon(
        points,
        polygon,
    )

    return torch.where(
        inside,
        -distance,
        distance,
    )
