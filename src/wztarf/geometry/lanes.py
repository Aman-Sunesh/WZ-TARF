"""Reconstruct and inspect lane geometry from the serialized lane tensors."""

from __future__ import annotations

import torch


def _validate_lane_inputs(
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
) -> None:
    """Validate the canonical lane geometry inputs."""
    if lane_feat.ndim != 3:
        raise ValueError(
            "lane_feat must have shape [L, P, F]."
        )

    if lane_feat.shape[-1] < 8:
        raise ValueError(
            "lane_feat must contain at least 8 features per point."
        )

    if lane_point_mask.shape != lane_feat.shape[:2]:
        raise ValueError(
            "lane_point_mask must have shape [L, P]."
        )

    if lane_mask.shape != lane_feat.shape[:1]:
        raise ValueError(
            "lane_mask must have shape [L]."
        )


def reconstruct_lane_polygons(
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
) -> list[torch.Tensor]:
    """Reconstruct one polygon for each valid lane.

    The current final lane representation uses:

        lane_feat[..., 0:2] = centerline XY
        lane_feat[..., 4:6] = left-boundary offset from centerline
        lane_feat[..., 6:8] = right-boundary offset from centerline

    For each lane:

        left_boundary  = centerline + left_offset
        right_boundary = centerline + right_offset

    The polygon is constructed by traversing the left boundary forward and
    the right boundary backward.

    Args:
        lane_feat:
            Lane tensor `[L, P, 8]`.

        lane_point_mask:
            Valid point mask `[L, P]`.

        lane_mask:
            Valid lane mask `[L]`.

    Returns:
        List of polygons. Each polygon has shape `[N, 2]`.

    Invalid lanes and lanes containing fewer than two valid points are skipped.
    """
    _validate_lane_inputs(
        lane_feat,
        lane_point_mask,
        lane_mask,
    )

    polygons: list[torch.Tensor] = []

    for lane_index in range(lane_feat.shape[0]):
        if not bool(lane_mask[lane_index]):
            continue

        polygon = reconstruct_lane_polygon(
            lane_feat[
                lane_index
            ],
            lane_point_mask[
                lane_index
            ],
        )

        if polygon is not None:
            polygons.append(
                polygon
            )

    return polygons


def lane_support_points(
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
    *,
    include_boundaries: bool = True,
) -> torch.Tensor:
    """Collect valid points represented by the local lane map.

    Args:
        include_boundaries:
            When True, return centerline, left-boundary, and right-boundary
            points. Otherwise only centerline points are returned.

    Returns:
        Tensor `[N, 2]` containing all selected geometry points.

    Raises:
        ValueError:
            If no valid lane geometry is available.
    """
    _validate_lane_inputs(
        lane_feat,
        lane_point_mask,
        lane_mask,
    )

    collected: list[torch.Tensor] = []

    for lane_index in range(lane_feat.shape[0]):
        if not bool(lane_mask[lane_index]):
            continue

        valid = lane_point_mask[lane_index].bool()

        if not bool(valid.any()):
            continue

        lane = lane_feat[
            lane_index,
            valid,
        ]

        center = lane[:, 0:2]

        collected.append(center)

        if include_boundaries:
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

            collected.extend(
                (
                    left,
                    right,
                )
            )

    if not collected:
        raise ValueError(
            "No valid lane geometry is available."
        )

    return torch.cat(
        collected,
        dim=0,
    )


def lane_bounds(
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
    *,
    margin_m: float = 0.0,
) -> torch.Tensor:
    """Return the bounding box of represented local lane geometry.

    Returns:

        [xmin, ymin, xmax, ymax]

    The bounding box represents map coverage only. It must not be interpreted
    as the drivable area itself.
    """
    if margin_m < 0:
        raise ValueError(
            "margin_m cannot be negative."
        )

    points = lane_support_points(
        lane_feat,
        lane_point_mask,
        lane_mask,
        include_boundaries=True,
    )

    minimum = points.amin(dim=0)
    maximum = points.amax(dim=0)

    return torch.stack(
        (
            minimum[0] - margin_m,
            minimum[1] - margin_m,
            maximum[0] + margin_m,
            maximum[1] + margin_m,
        )
    )


def reconstruct_lane_polygon(
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
) -> torch.Tensor | None:
    """Reconstruct one lane polygon from center and boundary offsets."""
    if lane_feat.ndim != 2 or lane_feat.shape[-1] < 8:
        raise ValueError(
            "lane_feat must have shape [P, F] with F >= 8."
        )

    if lane_point_mask.shape != lane_feat.shape[:1]:
        raise ValueError(
            "lane_point_mask must have shape [P]."
        )

    valid = lane_point_mask.bool()

    if int(
        valid.sum().item()
    ) < 2:
        return None

    feature = lane_feat[
        valid
    ]

    center = feature[
        :,
        0:2,
    ]

    left = (
        center
        +
        feature[
            :,
            4:6,
        ]
    )

    right = (
        center
        +
        feature[
            :,
            6:8,
        ]
    )

    return torch.cat(
        (
            left,
            torch.flip(
                right,
                dims=(0,),
            ),
        ),
        dim=0,
    )


def polyline_longitudinal_offset(
    point: torch.Tensor,
    polyline: torch.Tensor,
) -> torch.Tensor:
    """Project one point onto a polyline and return arc-length position."""
    if point.shape != (2,):
        raise ValueError(
            "point must have shape [2]."
        )

    if polyline.ndim != 2 or polyline.shape[-1] != 2:
        raise ValueError(
            "polyline must have shape [P, 2]."
        )

    if polyline.shape[0] < 2:
        return point.sum() * 0.0

    start = polyline[:-1]
    end = polyline[1:]

    segment = (
        end
        -
        start
    )

    length = torch.linalg.vector_norm(
        segment,
        dim=-1,
    )

    length_sq = (
        segment
        *
        segment
    ).sum(
        dim=-1
    ).clamp_min(
        1e-8
    )

    alpha = (
        (
            point[None]
            -
            start
        )
        *
        segment
    ).sum(
        dim=-1
    ) / length_sq

    alpha = alpha.clamp(
        0.0,
        1.0,
    )

    projection = (
        start
        +
        alpha[:, None]
        *
        segment
    )

    distance = torch.linalg.vector_norm(
        projection
        -
        point[None],
        dim=-1,
    )

    closest = distance.argmin()

    cumulative = torch.cat(
        (
            torch.zeros(
                1,
                dtype=point.dtype,
                device=point.device,
            ),
            torch.cumsum(
                length,
                dim=0,
            ),
        )
    )

    return (
        cumulative[
            closest
        ]
        +
        alpha[
            closest
        ]
        *
        length[
            closest
        ]
    )
