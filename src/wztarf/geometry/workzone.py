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

def points_on_polygon_boundary(
    points: torch.Tensor,
    polygon: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return whether each point lies on any polygon edge."""
    _validate_points_polygon(
        points,
        polygon,
    )

    edge_start = polygon
    edge_end = torch.roll(
        polygon,
        shifts=-1,
        dims=0,
    )

    edge = (
        edge_end
        -
        edge_start
    )

    rel = (
        points[:, None, :]
        -
        edge_start[None, :, :]
    )

    cross = (
        edge[None, :, 0]
        *
        rel[..., 1]
        -
        edge[None, :, 1]
        *
        rel[..., 0]
    ).abs()

    edge_length = torch.linalg.vector_norm(
        edge,
        dim=-1,
    )

    tolerance = (
        eps
        *
        (
            1.0
            +
            edge_length
        )
    )[None]

    collinear = (
        cross
        <=
        tolerance
    )

    dot = (
        rel
        *
        edge[None]
    ).sum(
        dim=-1
    )

    length_sq = (
        edge
        *
        edge
    ).sum(
        dim=-1
    )[None]

    within = (
        (dot >= -eps)
        &
        (dot <= length_sq + eps)
    )

    return (
        collinear
        &
        within
    ).any(
        dim=1
    )


def segments_intersect_polygon(
    start: torch.Tensor,
    end: torch.Tensor,
    polygon: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> bool:
    """Return whether one line segment touches or crosses a polygon."""
    if start.shape != (2,) or end.shape != (2,):
        raise ValueError(
            "start and end must have shape [2]."
        )

    if bool(
        points_in_polygon(
            torch.stack(
                (
                    start,
                    end,
                )
            ),
            polygon,
        ).any()
    ):
        return True

    def orientation(
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        return (
            (b[0] - a[0])
            *
            (c[1] - a[1])
            -
            (b[1] - a[1])
            *
            (c[0] - a[0])
        )

    def on_segment(
        a: torch.Tensor,
        b: torch.Tensor,
        p: torch.Tensor,
    ) -> bool:
        cross = orientation(
            a,
            b,
            p,
        )

        if abs(
            float(
                cross.item()
            )
        ) > eps:
            return False

        return (
            float(
                torch.minimum(
                    a[0],
                    b[0],
                ).item()
            )
            -
            eps
            <=
            float(
                p[0].item()
            )
            <=
            float(
                torch.maximum(
                    a[0],
                    b[0],
                ).item()
            )
            +
            eps
            and
            float(
                torch.minimum(
                    a[1],
                    b[1],
                ).item()
            )
            -
            eps
            <=
            float(
                p[1].item()
            )
            <=
            float(
                torch.maximum(
                    a[1],
                    b[1],
                ).item()
            )
            +
            eps
        )

    for index in range(
        polygon.shape[0]
    ):
        a = polygon[index]
        b = polygon[
            (index + 1)
            %
            polygon.shape[0]
        ]

        o1 = orientation(
            start,
            end,
            a,
        )

        o2 = orientation(
            start,
            end,
            b,
        )

        o3 = orientation(
            a,
            b,
            start,
        )

        o4 = orientation(
            a,
            b,
            end,
        )

        proper = (
            (
                float(o1.item())
                *
                float(o2.item())
            )
            <
            0
            and
            (
                float(o3.item())
                *
                float(o4.item())
            )
            <
            0
        )

        if proper:
            return True

        if (
            on_segment(
                start,
                end,
                a,
            )
            or
            on_segment(
                start,
                end,
                b,
            )
            or
            on_segment(
                a,
                b,
                start,
            )
            or
            on_segment(
                a,
                b,
                end,
            )
        ):
            return True

    return False


def polygons_intersect(
    polygon_a: torch.Tensor,
    polygon_b: torch.Tensor,
) -> bool:
    """Return whether two polygons touch or overlap."""
    if bool(
        points_in_polygon(
            polygon_a,
            polygon_b,
        ).any()
    ):
        return True

    if bool(
        points_in_polygon(
            polygon_b,
            polygon_a,
        ).any()
    ):
        return True

    for index in range(
        polygon_a.shape[0]
    ):
        start = polygon_a[index]
        end = polygon_a[
            (index + 1)
            %
            polygon_a.shape[0]
        ]

        if segments_intersect_polygon(
            start,
            end,
            polygon_b,
        ):
            return True

    return False
