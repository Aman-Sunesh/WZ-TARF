"""Construct geometry-derived WorkZone topology pseudo-targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from wztarf.data.map_coverage import (
    _batched_lane_polygon_edges,
)


@dataclass
class TopologyTargets:
    """Targets and validity masks for topology reconstruction."""

    lane_overlap: torch.Tensor
    lane_distance: torch.Tensor
    edge_compatibility: torch.Tensor
    lane_mask: torch.Tensor
    edge_mask: torch.Tensor


def _cross(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """2-D cross product."""
    return (
        a[..., 0] * b[..., 1]
        -
        a[..., 1] * b[..., 0]
    )


def _orientation(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """Signed 2-D orientation."""
    return _cross(
        b - a,
        c - a,
    )


def _on_segment(
    a: torch.Tensor,
    b: torch.Tensor,
    p: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Vectorized equivalent of the original workzone.on_segment test."""
    cross = _orientation(
        a,
        b,
        p,
    ).abs()

    collinear = cross <= eps

    within_x = (
        (p[..., 0] >= torch.minimum(a[..., 0], b[..., 0]) - eps)
        &
        (p[..., 0] <= torch.maximum(a[..., 0], b[..., 0]) + eps)
    )

    within_y = (
        (p[..., 1] >= torch.minimum(a[..., 1], b[..., 1]) - eps)
        &
        (p[..., 1] <= torch.maximum(a[..., 1], b[..., 1]) + eps)
    )

    return (
        collinear
        &
        within_x
        &
        within_y
    )


def _points_in_wz_polygon(
    points: torch.Tensor,
    polygon: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Ray-cast arbitrary batched points against the four-corner WZ polygon.

    Args:
        points:
            [B, ..., 2]

        polygon:
            [B, 4, 2]

    Returns:
        [B, ...] boolean tensor.
    """
    batch_size = points.shape[0]

    shape = (
        batch_size,
        *([1] * (points.ndim - 2)),
        4,
        2,
    )

    edge_start = polygon.reshape(shape)

    edge_end = torch.roll(
        polygon,
        shifts=-1,
        dims=1,
    ).reshape(shape)

    query = points.unsqueeze(-2)

    x = query[..., 0]
    y = query[..., 1]

    x1 = edge_start[..., 0]
    y1 = edge_start[..., 1]

    x2 = edge_end[..., 0]
    y2 = edge_end[..., 1]

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
    )

    inside = (
        crossing.long()
        .sum(dim=-1)
        %
        2
        ==
        1
    )

    edge = edge_end - edge_start
    rel = query - edge_start

    cross = _cross(
        edge,
        rel,
    ).abs()

    edge_length = torch.linalg.vector_norm(
        edge,
        dim=-1,
    )

    tolerance = (
        eps
        *
        (1.0 + edge_length)
    )

    collinear = cross <= tolerance

    dot = (
        rel
        *
        edge
    ).sum(dim=-1)

    length_sq = (
        edge
        *
        edge
    ).sum(dim=-1)

    boundary = (
        collinear
        &
        (dot >= -eps)
        &
        (dot <= length_sq + eps)
    ).any(dim=-1)

    return inside | boundary


def _points_in_lane_polygons(
    points: torch.Tensor,
    lane_edge_start: torch.Tensor,
    lane_edge_end: torch.Tensor,
    lane_edge_valid: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Test WZ vertices against every reconstructed lane polygon.

    Args:
        points:
            [B, Q, 2]

    Returns:
        [B, L, Q].
    """
    query = points[:, None, :, None, :]

    start = lane_edge_start[:, :, None, :, :]
    end = lane_edge_end[:, :, None, :, :]

    x = query[..., 0]
    y = query[..., 1]

    x1 = start[..., 0]
    y1 = start[..., 1]

    x2 = end[..., 0]
    y2 = end[..., 1]

    valid = lane_edge_valid[:, :, None, :]

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
        valid
    )

    inside = (
        crossing.long()
        .sum(dim=-1)
        %
        2
        ==
        1
    )

    edge = end - start
    rel = query - start

    cross = _cross(
        edge,
        rel,
    ).abs()

    edge_length = torch.linalg.vector_norm(
        edge,
        dim=-1,
    )

    tolerance = (
        eps
        *
        (1.0 + edge_length)
    )

    dot = (
        rel
        *
        edge
    ).sum(dim=-1)

    length_sq = (
        edge
        *
        edge
    ).sum(dim=-1)

    boundary = (
        valid
        &
        (cross <= tolerance)
        &
        (dot >= -eps)
        &
        (dot <= length_sq + eps)
    ).any(dim=-1)

    return inside | boundary


def _segments_intersect_wz(
    start: torch.Tensor,
    end: torch.Tensor,
    polygon: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Vectorized equivalent of segments_intersect_polygon().

    Args:
        start:
            [B, ..., 2]

        end:
            [B, ..., 2]

        polygon:
            [B, 4, 2]

    Returns:
        [B, ...] boolean tensor.
    """
    endpoint_inside = (
        _points_in_wz_polygon(
            start,
            polygon,
            eps=eps,
        )
        |
        _points_in_wz_polygon(
            end,
            polygon,
            eps=eps,
        )
    )

    batch_size = start.shape[0]

    shape = (
        batch_size,
        *([1] * (start.ndim - 2)),
        4,
        2,
    )

    poly_start = polygon.reshape(shape)

    poly_end = torch.roll(
        polygon,
        shifts=-1,
        dims=1,
    ).reshape(shape)

    seg_start = start.unsqueeze(-2)
    seg_end = end.unsqueeze(-2)

    o1 = _orientation(
        seg_start,
        seg_end,
        poly_start,
    )

    o2 = _orientation(
        seg_start,
        seg_end,
        poly_end,
    )

    o3 = _orientation(
        poly_start,
        poly_end,
        seg_start,
    )

    o4 = _orientation(
        poly_start,
        poly_end,
        seg_end,
    )

    proper = (
        (o1 * o2 < 0)
        &
        (o3 * o4 < 0)
    )

    touching = (
        _on_segment(
            seg_start,
            seg_end,
            poly_start,
            eps=eps,
        )
        |
        _on_segment(
            seg_start,
            seg_end,
            poly_end,
            eps=eps,
        )
        |
        _on_segment(
            poly_start,
            poly_end,
            seg_start,
            eps=eps,
        )
        |
        _on_segment(
            poly_start,
            poly_end,
            seg_end,
            eps=eps,
        )
    )

    edge_hit = (
        proper
        |
        touching
    ).any(dim=-1)

    return endpoint_inside | edge_hit


def _point_to_segment_distance(
    point: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    """Broadcasted Euclidean point-to-segment distance."""
    segment = end - start

    length_sq = (
        segment.square()
        .sum(dim=-1)
        .clamp_min(1e-12)
    )

    alpha = (
        (
            (point - start)
            *
            segment
        ).sum(dim=-1)
        /
        length_sq
    ).clamp(
        0.0,
        1.0,
    )

    closest = (
        start
        +
        alpha.unsqueeze(-1)
        *
        segment
    )

    return torch.linalg.vector_norm(
        point - closest,
        dim=-1,
    )


def _lane_wz_distance(
    lane_edge_start: torch.Tensor,
    lane_edge_end: torch.Tensor,
    lane_edge_valid: torch.Tensor,
    wz_polygon: torch.Tensor,
) -> torch.Tensor:
    """Exact non-overlap polygon distance used by the original targets.

    Computes both:
        lane polygon vertices -> WZ edges
        WZ vertices -> lane polygon edges

    and returns the smaller value.
    """
    batch_size = lane_edge_start.shape[0]

    wz_start = wz_polygon
    wz_end = torch.roll(
        wz_polygon,
        shifts=-1,
        dims=1,
    )

    # --------------------------------------------------------------
    # Original d1:
    # distance_to_polygon(lane_polygon, wz_polygon).min()
    #
    # edge_start contains every reconstructed lane polygon vertex.
    # --------------------------------------------------------------

    lane_vertex = lane_edge_start.unsqueeze(-2)

    wz_shape = (
        batch_size,
        1,
        1,
        4,
        2,
    )

    wz_seg_start = wz_start.reshape(
        wz_shape
    )

    wz_seg_end = wz_end.reshape(
        wz_shape
    )

    lane_to_wz = _point_to_segment_distance(
        lane_vertex,
        wz_seg_start,
        wz_seg_end,
    ).amin(dim=-1)

    lane_to_wz = torch.where(
        lane_edge_valid,
        lane_to_wz,
        torch.full_like(
            lane_to_wz,
            float("inf"),
        ),
    ).amin(dim=-1)

    # --------------------------------------------------------------
    # Original d2:
    # distance_to_polygon(wz_polygon, lane_polygon).min()
    # --------------------------------------------------------------

    wz_vertex = wz_polygon[
        :,
        None,
        :,
        None,
        :,
    ]

    lane_seg_start = lane_edge_start[
        :,
        :,
        None,
        :,
        :,
    ]

    lane_seg_end = lane_edge_end[
        :,
        :,
        None,
        :,
        :,
    ]

    wz_to_lane = _point_to_segment_distance(
        wz_vertex,
        lane_seg_start,
        lane_seg_end,
    )

    wz_to_lane = torch.where(
        lane_edge_valid[
            :,
            :,
            None,
            :,
        ],
        wz_to_lane,
        torch.full_like(
            wz_to_lane,
            float("inf"),
        ),
    )

    wz_to_lane = (
        wz_to_lane
        .amin(dim=-1)
        .amin(dim=-1)
    )

    return torch.minimum(
        lane_to_wz,
        wz_to_lane,
    )


def build_topology_targets(
    *,
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
    lane_edge_index: torch.Tensor,
    lane_edge_mask: torch.Tensor,
    wz_feat: torch.Tensor,
) -> TopologyTargets:
    """Build topology pseudo-targets with batched accelerator geometry."""
    batch_size, num_lanes = lane_feat.shape[:2]
    num_edges = lane_edge_index.shape[-1]

    dtype = lane_feat.dtype
    device = lane_feat.device

    wz_valid = (
        wz_feat[
            :,
            :4,
            2,
        ]
        >
        0
    ).all(dim=1)

    wz_polygon = wz_feat[
        :,
        :4,
        :2,
    ]

    lane_point_valid = lane_point_mask.bool()

    lane_target_mask = (
        lane_mask.bool()
        &
        (
            lane_point_valid.sum(dim=-1)
            >=
            2
        )
        &
        wz_valid[:, None]
    )

    lane_edge_start, lane_edge_end, lane_polygon_edge_valid = (
        _batched_lane_polygon_edges(
            lane_feat,
            lane_point_mask,
            lane_mask,
        )
    )

    lane_polygon_edge_valid = (
        lane_polygon_edge_valid
        &
        lane_target_mask.unsqueeze(-1)
    )

    # --------------------------------------------------------------
    # Lane polygon <-> WZ polygon intersection.
    #
    # Equivalent to polygons_intersect():
    #   1. lane vertex inside WZ
    #   2. WZ vertex inside lane
    #   3. any lane edge intersects WZ
    # --------------------------------------------------------------

    lane_vertex_inside_wz = (
        _points_in_wz_polygon(
            lane_edge_start,
            wz_polygon,
        )
        &
        lane_polygon_edge_valid
    ).any(dim=-1)

    wz_vertex_inside_lane = (
        _points_in_lane_polygons(
            wz_polygon,
            lane_edge_start,
            lane_edge_end,
            lane_polygon_edge_valid,
        )
    ).any(dim=-1)

    lane_edge_hits_wz = (
        _segments_intersect_wz(
            lane_edge_start,
            lane_edge_end,
            wz_polygon,
        )
        &
        lane_polygon_edge_valid
    ).any(dim=-1)

    overlap_bool = (
        lane_target_mask
        &
        (
            lane_vertex_inside_wz
            |
            wz_vertex_inside_lane
            |
            lane_edge_hits_wz
        )
    )

    overlap = overlap_bool.to(dtype)

    polygon_distance = _lane_wz_distance(
        lane_edge_start,
        lane_edge_end,
        lane_polygon_edge_valid,
        wz_polygon,
    )

    distance = torch.where(
        lane_target_mask,
        torch.where(
            overlap_bool,
            torch.zeros_like(
                polygon_distance
            ),
            polygon_distance,
        ),
        torch.zeros_like(
            polygon_distance
        ),
    )

    # --------------------------------------------------------------
    # Edge targets.
    # --------------------------------------------------------------

    src = lane_edge_index[
        :,
        0,
    ].long()

    dst = lane_edge_index[
        :,
        1,
    ].long()

    index_valid = (
        (src >= 0)
        &
        (src < num_lanes)
        &
        (dst >= 0)
        &
        (dst < num_lanes)
    )

    safe_src = src.clamp(
        0,
        max(num_lanes - 1, 0),
    )

    safe_dst = dst.clamp(
        0,
        max(num_lanes - 1, 0),
    )

    src_lane_valid = lane_target_mask.gather(
        1,
        safe_src,
    )

    dst_lane_valid = lane_target_mask.gather(
        1,
        safe_dst,
    )

    edge_target_mask = (
        lane_edge_mask.bool()
        &
        index_valid
        &
        src_lane_valid
        &
        dst_lane_valid
    )

    src_overlap = overlap_bool.gather(
        1,
        safe_src,
    )

    dst_overlap = overlap_bool.gather(
        1,
        safe_dst,
    )

    blocked = (
        src_overlap
        |
        dst_overlap
    )

    # First and last valid centerline points for every lane.
    center = lane_feat[
        ...,
        :2,
    ]

    first_index = (
        lane_point_valid.long()
        .argmax(dim=-1)
    )

    reverse_index = (
        torch.flip(
            lane_point_valid,
            dims=(-1,),
        )
        .long()
        .argmax(dim=-1)
    )

    last_index = (
        lane_point_valid.shape[-1]
        -
        1
        -
        reverse_index
    )

    first_gather = (
        first_index[
            ...,
            None,
            None,
        ]
        .expand(
            -1,
            -1,
            1,
            2,
        )
    )

    last_gather = (
        last_index[
            ...,
            None,
            None,
        ]
        .expand(
            -1,
            -1,
            1,
            2,
        )
    )

    first_center = center.gather(
        2,
        first_gather,
    ).squeeze(2)

    last_center = center.gather(
        2,
        last_gather,
    ).squeeze(2)

    src_connector = last_center.gather(
        1,
        safe_src.unsqueeze(-1).expand(
            -1,
            -1,
            2,
        ),
    )

    dst_connector = first_center.gather(
        1,
        safe_dst.unsqueeze(-1).expand(
            -1,
            -1,
            2,
        ),
    )

    connector_hits_wz = _segments_intersect_wz(
        src_connector,
        dst_connector,
        wz_polygon,
    )

    blocked = (
        blocked
        |
        connector_hits_wz
    )

    edge_compatibility = torch.where(
        edge_target_mask
        &
        ~blocked,
        torch.ones(
            batch_size,
            num_edges,
            dtype=dtype,
            device=device,
        ),
        torch.zeros(
            batch_size,
            num_edges,
            dtype=dtype,
            device=device,
        ),
    )

    return TopologyTargets(
        lane_overlap=overlap,
        lane_distance=distance,
        edge_compatibility=edge_compatibility,
        lane_mask=lane_target_mask,
        edge_mask=edge_target_mask,
    )
