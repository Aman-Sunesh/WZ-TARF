"""Encode WZ boundary, polygon, sign, and worker geometry as typed tokens."""

from __future__ import annotations

import torch
from torch import nn


def _lane_segment_geometry(
    lane_feat: torch.Tensor | None,
    lane_point_mask: torch.Tensor | None,
    lane_mask: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
] | None:
    """Build lane-center segments and represented corridor half-widths."""
    values = (
        lane_feat,
        lane_point_mask,
        lane_mask,
    )

    if all(
        value is None
        for value in values
    ):
        return None

    if any(
        value is None
        for value in values
    ):
        raise ValueError(
            "lane_feat, lane_point_mask, and lane_mask "
            "must be supplied together."
        )

    assert lane_feat is not None
    assert lane_point_mask is not None
    assert lane_mask is not None

    if lane_feat.ndim != 4 or lane_feat.shape[-1] < 8:
        raise ValueError(
            "lane_feat must have shape [B, L, P, F] with F >= 8."
        )

    if lane_point_mask.shape != lane_feat.shape[:3]:
        raise ValueError(
            "lane_point_mask must have shape [B, L, P]."
        )

    if lane_mask.shape != lane_feat.shape[:2]:
        raise ValueError(
            "lane_mask must have shape [B, L]."
        )

    center = lane_feat[
        ...,
        0:2,
    ]

    valid_point = (
        lane_point_mask.bool()
        &
        lane_mask.bool()[
            :,
            :,
            None,
        ]
    )

    segment_start = center[
        :,
        :,
        :-1,
    ]

    segment_end = center[
        :,
        :,
        1:,
    ]

    segment_valid = (
        valid_point[
            :,
            :,
            :-1,
        ]
        &
        valid_point[
            :,
            :,
            1:,
        ]
    )

    left_width = torch.linalg.vector_norm(
        lane_feat[
            ...,
            4:6,
        ],
        dim=-1,
    )

    right_width = torch.linalg.vector_norm(
        lane_feat[
            ...,
            6:8,
        ],
        dim=-1,
    )

    point_half_width = 0.5 * (
        left_width
        +
        right_width
    )

    segment_half_width = 0.5 * (
        point_half_width[
            :,
            :,
            :-1,
        ]
        +
        point_half_width[
            :,
            :,
            1:,
        ]
    )

    return (
        segment_start,
        segment_end,
        segment_half_width,
        segment_valid,
    )


def _point_segment_distance(
    point: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    """Return Euclidean point-to-segment distance with broadcasting."""
    segment = (
        end
        -
        start
    )

    length_sq = (
        segment.square()
        .sum(
            dim=-1
        )
        .clamp_min(
            1e-8
        )
    )

    alpha = (
        (
            point
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

    closest = (
        start
        +
        alpha[
            ...,
            None,
        ]
        *
        segment
    )

    return torch.linalg.vector_norm(
        point
        -
        closest,
        dim=-1,
    )


def _points_to_lane_distance(
    points: torch.Tensor,
    lane_geometry: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ] | None,
) -> torch.Tensor:
    """Return distance from query points to the represented lane corridor."""
    if lane_geometry is None:
        return torch.zeros(
            points.shape[:2],
            dtype=points.dtype,
            device=points.device,
        )

    (
        lane_start,
        lane_end,
        lane_half_width,
        lane_valid,
    ) = lane_geometry

    query = points[
        :,
        :,
        None,
        None,
        :,
    ]

    start = lane_start[
        :,
        None,
    ]

    end = lane_end[
        :,
        None,
    ]

    distance = _point_segment_distance(
        query,
        start,
        end,
    )

    distance = (
        distance
        -
        lane_half_width[
            :,
            None,
        ]
    ).clamp_min(
        0.0
    )

    distance = distance.masked_fill(
        ~lane_valid[
            :,
            None,
        ],
        torch.finfo(
            distance.dtype
        ).max,
    )

    minimum = distance.flatten(
        start_dim=2
    ).min(
        dim=-1
    ).values

    has_lane = lane_valid.flatten(
        start_dim=1
    ).any(
        dim=1
    )

    return torch.where(
        has_lane[
            :,
            None,
        ],
        minimum,
        torch.zeros_like(
            minimum
        ),
    )


def _cross_2d(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """Return the scalar 2-D cross product with broadcasting."""
    return (
        first[..., 0]
        *
        second[..., 1]
        -
        first[..., 1]
        *
        second[..., 0]
    )


def _segments_to_lane_distance(
    query_start: torch.Tensor,
    query_end: torch.Tensor,
    lane_geometry: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ] | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """Return lane-corridor distance and overlap for query segments."""
    if lane_geometry is None:
        shape = query_start.shape[:2]

        return (
            torch.zeros(
                shape,
                dtype=query_start.dtype,
                device=query_start.device,
            ),
            torch.zeros(
                shape,
                dtype=torch.bool,
                device=query_start.device,
            ),
        )

    (
        lane_start,
        lane_end,
        lane_half_width,
        lane_valid,
    ) = lane_geometry

    qa = query_start[
        :,
        :,
        None,
        None,
        :,
    ]

    qb = query_end[
        :,
        :,
        None,
        None,
        :,
    ]

    la = lane_start[
        :,
        None,
    ]

    lb = lane_end[
        :,
        None,
    ]

    endpoint_distance = torch.stack(
        (
            _point_segment_distance(
                qa,
                la,
                lb,
            ),
            _point_segment_distance(
                qb,
                la,
                lb,
            ),
            _point_segment_distance(
                la,
                qa,
                qb,
            ),
            _point_segment_distance(
                lb,
                qa,
                qb,
            ),
        ),
        dim=-1,
    ).min(
        dim=-1
    ).values

    q_vector = (
        qb
        -
        qa
    )

    lane_vector = (
        lb
        -
        la
    )

    o1 = _cross_2d(
        q_vector,
        la - qa,
    )

    o2 = _cross_2d(
        q_vector,
        lb - qa,
    )

    o3 = _cross_2d(
        lane_vector,
        qa - la,
    )

    o4 = _cross_2d(
        lane_vector,
        qb - la,
    )

    eps = 1e-8

    query_straddles = (
        (
            (o1 >= -eps)
            &
            (o2 <= eps)
        )
        |
        (
            (o2 >= -eps)
            &
            (o1 <= eps)
        )
    )

    lane_straddles = (
        (
            (o3 >= -eps)
            &
            (o4 <= eps)
        )
        |
        (
            (o4 >= -eps)
            &
            (o3 <= eps)
        )
    )

    query_min = torch.minimum(
        qa,
        qb,
    )

    query_max = torch.maximum(
        qa,
        qb,
    )

    lane_min = torch.minimum(
        la,
        lb,
    )

    lane_max = torch.maximum(
        la,
        lb,
    )

    bbox_overlap = (
        (
            torch.maximum(
                query_min[..., 0],
                lane_min[..., 0],
            )
            <=
            torch.minimum(
                query_max[..., 0],
                lane_max[..., 0],
            )
            +
            eps
        )
        &
        (
            torch.maximum(
                query_min[..., 1],
                lane_min[..., 1],
            )
            <=
            torch.minimum(
                query_max[..., 1],
                lane_max[..., 1],
            )
            +
            eps
        )
    )

    intersects = (
        query_straddles
        &
        lane_straddles
        &
        bbox_overlap
    )

    centerline_distance = torch.where(
        intersects,
        torch.zeros_like(
            endpoint_distance
        ),
        endpoint_distance,
    )

    corridor_distance = (
        centerline_distance
        -
        lane_half_width[
            :,
            None,
        ]
    ).clamp_min(
        0.0
    )

    valid = lane_valid[
        :,
        None,
    ]

    corridor_distance = corridor_distance.masked_fill(
        ~valid,
        torch.finfo(
            corridor_distance.dtype
        ).max,
    )

    minimum = corridor_distance.flatten(
        start_dim=2
    ).min(
        dim=-1
    ).values

    overlap = (
        (
            corridor_distance
            <=
            1e-4
        )
        &
        valid
    ).flatten(
        start_dim=2
    ).any(
        dim=-1
    )

    has_lane = lane_valid.flatten(
        start_dim=1
    ).any(
        dim=1
    )

    minimum = torch.where(
        has_lane[
            :,
            None,
        ],
        minimum,
        torch.zeros_like(
            minimum
        ),
    )

    overlap = (
        overlap
        &
        has_lane[
            :,
            None,
        ]
    )

    return (
        minimum,
        overlap,
    )
    

def _batched_polygon_distance_and_inside(
    points: torch.Tensor,
    polygon: torch.Tensor,
    polygon_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized point-to-polygon distance and inside test.

    Args:
        points: [B, ..., 2]
        polygon: [B, V, 2]
        polygon_valid: [B]

    Returns:
        distance: [B, ...]
        inside: [B, ...]
    """
    if points.ndim < 3 or points.shape[-1] != 2:
        raise ValueError("points must have shape [B, ..., 2].")
    if polygon.ndim != 3 or polygon.shape[-1] != 2:
        raise ValueError("polygon must have shape [B, V, 2].")

    batch_size = points.shape[0]
    tail_shape = points.shape[1:-1]
    flat = points.reshape(batch_size, -1, 2)

    start = polygon[:, None, :, :]
    end = torch.roll(polygon, shifts=-1, dims=1)[:, None, :, :]
    point = flat[:, :, None, :]

    distance = _point_segment_distance(point, start, end).min(dim=-1).values

    px = point[..., 0]
    py = point[..., 1]
    x1 = start[..., 0]
    y1 = start[..., 1]
    x2 = end[..., 0]
    y2 = end[..., 1]

    crosses = (
        ((y1 > py) != (y2 > py))
        & (
            px
            < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-8) + x1
        )
    )
    inside = (crosses.sum(dim=-1) % 2 == 1)

    # Match the reference convention that points on the polygon boundary are
    # considered inside.  This is also numerically more stable for WZ corners.
    inside = inside | (distance <= 1e-6)
    inside = inside & polygon_valid[:, None]
    distance = torch.where(
        polygon_valid[:, None],
        distance,
        torch.zeros_like(distance),
    )

    return (
        distance.reshape(batch_size, *tail_shape),
        inside.reshape(batch_size, *tail_shape),
    )

def _subsample_packed_lane_geometry(
    lane_feat: torch.Tensor | None,
    lane_point_mask: torch.Tensor | None,
    max_points: int | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Subsample packed lane polylines before WZ/lane geometry comparisons.

    WorkZoneEncoder used to compare every WZ edge/sign/worker against all 234
    lane points during every model forward.  These comparisons are geometric,
    not learned.  Until they are moved to the offline cache, a fixed compact
    representation gives a large hot-path reduction while preserving the
    overall lane shape.
    """
    if (
        lane_feat is None
        or lane_point_mask is None
        or max_points is None
        or lane_feat.shape[2] <= max_points
    ):
        return lane_feat, lane_point_mask

    if max_points < 2:
        raise ValueError("lane_geometry_points must be at least 2 or None.")

    batch_size, num_lanes, num_points, feature_dim = lane_feat.shape
    sample_count = min(int(max_points), num_points)
    mask = lane_point_mask.bool()
    count = mask.sum(dim=-1).long()
    slot = torch.arange(sample_count, device=lane_feat.device).view(1, 1, -1)
    dense = slot.expand(batch_size, num_lanes, -1)
    last = (count - 1).clamp_min(0)[..., None]
    uniform = torch.round(
        dense.to(lane_feat.dtype)
        * last.to(lane_feat.dtype)
        / float(max(1, sample_count - 1))
    ).long()
    index = torch.where(count[..., None] >= sample_count, uniform, dense)
    index = index.clamp(0, num_points - 1)
    sampled_mask = dense < count[..., None].clamp_max(sample_count)
    sampled_feat = lane_feat.gather(
        2,
        index[..., None].expand(-1, -1, -1, feature_dim),
    )
    sampled_feat = sampled_feat * sampled_mask[..., None].to(sampled_feat.dtype)
    return sampled_feat, sampled_mask


class WorkZoneEncoder(nn.Module):
    """Encode structured WorkZone geometry with typed self-attention."""

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        lane_geometry_points: int | None = 64,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.lane_geometry_points = lane_geometry_points
        if lane_geometry_points is not None and lane_geometry_points < 2:
            raise ValueError("lane_geometry_points must be at least 2 or None.")

        self.feature_encoder = nn.Sequential(
            nn.Linear(
                12,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                d_model,
            ),
        )

        # edge, polygon, sign, worker
        self.type_embedding = nn.Embedding(
            4,
            d_model,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )

        self.pool_score = nn.Linear(
            d_model,
            1,
        )

    def forward(
        self,
        wz_feat: torch.Tensor,
        worker_feat: torch.Tensor,
        ego_speed: torch.Tensor | None = None,
        lane_feat: torch.Tensor | None = None,
        lane_point_mask: torch.Tensor | None = None,
        lane_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return typed WZ token states and a pooled WZ context."""
        if wz_feat.ndim != 3 or wz_feat.shape[1:] != (5, 3):
            raise ValueError(
                "wz_feat must have shape [B, 5, 3]."
            )

        if worker_feat.ndim != 3 or worker_feat.shape[-1] != 3:
            raise ValueError(
                "worker_feat must have shape [B, W, 3]."
            )

        batch_size = wz_feat.shape[0]

        # ==============================================================
        # V3 FP32 WORKZONE GEOMETRY
        #
        # Geometry is deliberately evaluated from FP32 tensors even when
        # the learned network is running under BF16 autocast.
        # ==============================================================
        wz_feat = wz_feat.float()
        worker_feat = worker_feat.float()

        if ego_speed is not None:
            ego_speed = ego_speed.float()

        if lane_feat is not None:
            lane_feat = lane_feat.float()

        geometry_lane_feat, geometry_lane_mask = _subsample_packed_lane_geometry(
            lane_feat,
            lane_point_mask,
            self.lane_geometry_points,
        )
        lane_geometry = _lane_segment_geometry(
            geometry_lane_feat,
            geometry_lane_mask,
            lane_mask,
        )

     
        if lane_geometry is None:
            lane_available = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=wz_feat.device,
            )
        else:
            lane_available = lane_geometry[
                3
            ].flatten(
                start_dim=1
            ).any(
                dim=1
            )

        corners = wz_feat[
            :,
            :4,
            :2,
        ]

        corner_valid = wz_feat[
            :,
            :4,
            2,
        ] > 0

        next_corner = torch.roll(
            corners,
            shifts=-1,
            dims=1,
        )

        next_valid = torch.roll(
            corner_valid,
            shifts=-1,
            dims=1,
        )

        edge_valid = (
            corner_valid
            &
            next_valid
        )

        edge_vector = (
            next_corner
            -
            corners
        )

        edge_length = torch.linalg.vector_norm(
            edge_vector,
            dim=-1,
        )

        tangent = edge_vector / (
            edge_length[..., None]
            +
            1e-8
        )

        midpoint = (
            corners
            +
            next_corner
        ) / 2.0

        midpoint_distance = torch.linalg.vector_norm(
            midpoint,
            dim=-1,
        )

        midpoint_sin = midpoint[..., 1] / (
            midpoint_distance
            +
            1e-8
        )

        midpoint_cos = midpoint[..., 0] / (
            midpoint_distance
            +
            1e-8
        )

        signed_area = 0.5 * (
            corners[..., 0]
            *
            next_corner[..., 1]
            -
            corners[..., 1]
            *
            next_corner[..., 0]
        ).sum(
            dim=1
        )

        ccw = (
            signed_area
            >=
            0
        )[:, None]

        outward_ccw = torch.stack(
            (
                tangent[..., 1],
                -tangent[..., 0],
            ),
            dim=-1,
        )

        outward_cw = torch.stack(
            (
                -tangent[..., 1],
                tangent[..., 0],
            ),
            dim=-1,
        )

        outward = torch.where(
            ccw[..., None],
            outward_ccw,
            outward_cw,
        )

        boundary_lane_distance, boundary_lane_overlap = (
            _segments_to_lane_distance(
                corners,
                next_corner,
                lane_geometry,
            )
        )
        
        boundary_lane_distance = torch.where(
            edge_valid,
            boundary_lane_distance,
            torch.zeros_like(
                boundary_lane_distance
            ),
        )
        
        boundary_lane_overlap = (
            boundary_lane_overlap
            &
            edge_valid
        )

        edge_features = torch.cat(
            (
                midpoint,
                midpoint_distance[..., None],
                midpoint_sin[..., None],
                midpoint_cos[..., None],
                tangent,
                outward,
                edge_length[..., None],
                boundary_lane_distance[..., None],
                boundary_lane_overlap[
                    ...,
                    None,
                ].to(
                    midpoint.dtype
                ),
            ),
            dim=-1,
        )
        
        polygon_valid = corner_valid.all(
            dim=1
        )

        polygon_center = corners.mean(
            dim=1
        )

        polygon_distance = torch.linalg.vector_norm(
            polygon_center,
            dim=-1,
        )

        polygon_length = 0.5 * (
            edge_length[:, 0]
            +
            edge_length[:, 2]
        )

        polygon_width = 0.5 * (
            edge_length[:, 1]
            +
            edge_length[:, 3]
        )

        orientation = tangent[
            :,
            0,
        ]
        
        polygon_lane_overlap = boundary_lane_overlap.any(
            dim=1
        )
        
        if (
            geometry_lane_feat is not None
            and
            geometry_lane_mask is not None
            and
            lane_mask is not None
        ):
            valid_lane_point = (
                geometry_lane_mask.bool()
                &
                lane_mask.bool()[
                    :,
                    :,
                    None,
                ]
            )
        
            _, represented_inside = _batched_polygon_distance_and_inside(
                geometry_lane_feat[..., :2],
                corners,
                polygon_valid,
            )
            polygon_lane_overlap = (
                polygon_lane_overlap
                | (represented_inside & valid_lane_point).flatten(1).any(dim=1)
            )
        
        polygon_lane_distance = boundary_lane_distance.min(
            dim=1
        ).values
        
        polygon_lane_distance = torch.where(
            polygon_lane_overlap,
            torch.zeros_like(
                polygon_lane_distance
            ),
            polygon_lane_distance,
        )

        polygon_features = torch.stack(
            (
                polygon_center[:, 0],
                polygon_center[:, 1],
                polygon_distance,
                polygon_center[:, 1] / (polygon_distance + 1e-8),
                polygon_center[:, 0] / (polygon_distance + 1e-8),
                polygon_length,
                polygon_width,
                signed_area.abs(),
                orientation[:, 1],
                orientation[:, 0],
            polygon_lane_distance,
            polygon_lane_overlap.to(
                polygon_center.dtype
            ),
            ),
            dim=-1,
        )[:, None, :]

        sign_xy = wz_feat[
            :,
            4,
            :2,
        ]

        sign_valid = wz_feat[
            :,
            4,
            2,
        ] > 0

        sign_distance = torch.linalg.vector_norm(
            sign_xy,
            dim=-1,
        )

        sign_lane_distance = _points_to_lane_distance(
            sign_xy[
                :,
                None,
            ],
            lane_geometry,
        ).squeeze(
            1
        )
        
        sign_on_lane = (
            (
                sign_lane_distance
                <=
                1e-4
            )
            &
            lane_available
        ).to(
            sign_distance.dtype
        )

        sign_boundary_distance = torch.zeros_like(
            sign_distance
        )
        
        sign_inside = torch.zeros_like(
            sign_distance
        )

        sign_ahead = (
            sign_xy[:, 0]
            >
            0
        ).to(
            sign_xy.dtype
        )

        if ego_speed is None:
            time_to_passage = torch.zeros_like(
                sign_distance
            )
        else:
            safe_speed = ego_speed.clamp_min(
                0.1
            )

            time_to_passage = torch.where(
                sign_ahead.bool(),
                sign_xy[:, 0].clamp_min(0.0)
                /
                safe_speed,
                torch.zeros_like(
                    safe_speed
                ),
            )

        worker_xy = worker_feat[
            ...,
            :2,
        ]

        worker_valid = worker_feat[
            ...,
            2,
        ] > 0

        worker_distance = torch.linalg.vector_norm(
            worker_xy,
            dim=-1,
        )

        worker_ahead = (
            worker_xy[..., 0]
            >
            0
        ).to(
            worker_xy.dtype
        )

        if ego_speed is None:
            worker_ttp = torch.zeros_like(
                worker_distance
            )
        else:
            worker_ttp = torch.where(
                worker_ahead.bool(),
                worker_xy[..., 0].clamp_min(0.0)
                /
                ego_speed[:, None].clamp_min(0.1),
                torch.zeros_like(
                    worker_distance
                ),
            )

        worker_boundary_distance = torch.zeros_like(
            worker_distance
        )

        worker_inside = torch.zeros_like(
            worker_distance
        )

        sign_boundary_distance, sign_inside_bool = (
            _batched_polygon_distance_and_inside(
                sign_xy[:, None, :],
                corners,
                polygon_valid,
            )
        )
        sign_boundary_distance = sign_boundary_distance[:, 0]
        sign_inside = (
            sign_inside_bool[:, 0]
            & sign_valid
        ).to(sign_distance.dtype)
        sign_boundary_distance = torch.where(
            sign_valid,
            sign_boundary_distance,
            torch.zeros_like(sign_boundary_distance),
        )

        worker_boundary_distance, worker_inside_bool = (
            _batched_polygon_distance_and_inside(
                worker_xy,
                corners,
                polygon_valid,
            )
        )
        worker_inside = (
            worker_inside_bool
            & worker_valid
        ).to(worker_distance.dtype)
        worker_boundary_distance = torch.where(
            worker_valid,
            worker_boundary_distance,
            torch.zeros_like(worker_boundary_distance),
        )

        worker_lane_distance = _points_to_lane_distance(
            worker_xy,
            lane_geometry,
        )
    
        worker_on_lane = (
            (
                worker_lane_distance
                <=
                1e-4
            )
            &
            lane_available[
                :,
                None,
            ]
        ).to(
            worker_distance.dtype
        )

        sign_signed_boundary_distance = torch.where(
            sign_inside.bool(),
            -sign_boundary_distance,
            sign_boundary_distance,
        )
    
        worker_signed_boundary_distance = torch.where(
            worker_inside.bool(),
            -worker_boundary_distance,
            worker_boundary_distance,
        )
    
        sign_features = torch.stack(
            (
                sign_xy[:, 0],
                sign_xy[:, 1],
                sign_distance,
                sign_xy[:, 1] / (sign_distance + 1e-8),
                sign_xy[:, 0] / (sign_distance + 1e-8),
                sign_ahead,
                time_to_passage,
                sign_lane_distance,
                sign_on_lane,
                sign_boundary_distance,
                sign_inside,
                sign_signed_boundary_distance,
            ),
            dim=-1,
        )[:, None, :]


        # ==============================================================
        # V3 WZ-INDEPENDENT WORKER REPRESENTATION
        #
        # Worker tokens contain only ego-relative and permanent-map-relative
        # information. No quantity derived from wz_feat is allowed here.
        #
        # Keep 12 channels for feature_encoder compatibility.
        # ==============================================================
        worker_wz_neutral = torch.zeros_like(
            worker_distance
        )

        worker_features = torch.stack(
            (
                worker_xy[..., 0],
                worker_xy[..., 1],
                worker_distance,
                worker_xy[..., 1] / (worker_distance + 1e-8),
                worker_xy[..., 0] / (worker_distance + 1e-8),
                worker_ahead,
                worker_ttp,

                # Former worker_inside WZ-relative channel.
                worker_wz_neutral,

                # Permanent-map-relative quantity.
                worker_lane_distance,

                # Former worker_boundary_distance WZ-relative channel.
                worker_wz_neutral,

                # Permanent-map-relative quantity.
                worker_on_lane,

                # Former signed WZ-boundary-distance channel.
                worker_wz_neutral,
            ),
            dim=-1,
        )

        token_features = torch.cat(
            (
                edge_features,
                polygon_features,
                sign_features,
                worker_features,
            ),
            dim=1,
        )

        token_mask = torch.cat(
            (
                edge_valid,
                polygon_valid[:, None],
                sign_valid[:, None],
                worker_valid,
            ),
            dim=1,
        )

        num_workers = worker_feat.shape[1]

        type_id = torch.tensor(
            [
                0,
                0,
                0,
                0,
                1,
                2,
                *(
                    [3]
                    *
                    num_workers
                ),
            ],
            dtype=torch.long,
            device=wz_feat.device,
        )

        tokens = (
            self.feature_encoder(
                token_features
            )
            +
            self.type_embedding(
                type_id
            )[None]
        )

        # ==============================================================
        # V3 SEPARATE WZ / WORKER ATTENTION STREAMS
        #
        # Keep WZ geometry/sign tokens and worker tokens independent.
        # The dummy visible token is stream-local so an all-masked stream
        # cannot create NaNs inside attention.
        # ==============================================================
        wz_token_count = 6

        original_valid = token_mask.any(
            dim=1
        )

        safe_mask = token_mask.clone()
        tokens = tokens.clone()

        wz_original_valid = token_mask[
            :,
            :wz_token_count,
        ].any(
            dim=1
        )

        safe_mask[:, 0] = (
            safe_mask[:, 0]
            |
            ~wz_original_valid
        )

        tokens[:, 0] = torch.where(
            (~wz_original_valid)[:, None],
            torch.zeros_like(
                tokens[:, 0]
            ),
            tokens[:, 0],
        )

        if tokens.shape[1] > wz_token_count:
            worker_original_valid = token_mask[
                :,
                wz_token_count:,
            ].any(
                dim=1
            )

            safe_mask[
                :,
                wz_token_count,
            ] = (
                safe_mask[
                    :,
                    wz_token_count,
                ]
                |
                ~worker_original_valid
            )

            tokens[
                :,
                wz_token_count,
            ] = torch.where(
                (~worker_original_valid)[:, None],
                torch.zeros_like(
                    tokens[
                        :,
                        wz_token_count,
                    ]
                ),
                tokens[
                    :,
                    wz_token_count,
                ],
            )

        stream_block_mask = torch.zeros(
            (
                tokens.shape[1],
                tokens.shape[1],
            ),
            dtype=torch.bool,
            device=tokens.device,
        )

        if tokens.shape[1] > wz_token_count:
            stream_block_mask[
                :wz_token_count,
                wz_token_count:
            ] = True

            stream_block_mask[
                wz_token_count:,
                :wz_token_count
            ] = True

        tokens = self.transformer(
            tokens,
            mask=stream_block_mask,
            src_key_padding_mask=~safe_mask,
        )

        score = self.pool_score(
            tokens
        ).squeeze(-1)

        score = score.masked_fill(
            ~safe_mask,
            torch.finfo(
                score.dtype
            ).min,
        )

        weight = torch.softmax(
            score,
            dim=1,
        )

        weight = (
            weight
            *
            safe_mask.to(
                weight.dtype
            )
        )

        weight = weight / (
            weight.sum(
                dim=1,
                keepdim=True,
            )
            +
            1e-8
        )

        context = (
            tokens
            *
            weight[..., None]
        ).sum(
            dim=1
        )

        context = (
            context
            *
            original_valid[:, None].to(
                context.dtype
            )
        )

        # ==============================================================
        # V3 EXPLICIT WZ / WORKER REPRESENTATIONS
        #
        # Pool each stream independently.  Workers cannot alter wz_context,
        # and WZ geometry cannot alter worker_context through self-attention.
        # ==============================================================
        def _pool_stream(
            stream_tokens: torch.Tensor,
            stream_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if stream_tokens.shape[1] == 0:
                return (
                    torch.zeros(
                        stream_tokens.shape[0],
                        self.d_model,
                        dtype=stream_tokens.dtype,
                        device=stream_tokens.device,
                    ),
                    torch.zeros(
                        stream_tokens.shape[0],
                        dtype=torch.bool,
                        device=stream_tokens.device,
                    ),
                )

            stream_valid = stream_mask.any(
                dim=1
            )

            stream_score = self.pool_score(
                stream_tokens
            ).squeeze(
                -1
            ).float()

            stream_score = stream_score.masked_fill(
                ~stream_mask,
                torch.finfo(
                    stream_score.dtype
                ).min,
            )

            stream_weight = torch.softmax(
                stream_score,
                dim=1,
            )

            stream_weight = (
                stream_weight
                *
                stream_mask.to(
                    stream_weight.dtype
                )
            )

            stream_weight = stream_weight / (
                stream_weight.sum(
                    dim=1,
                    keepdim=True,
                ).clamp_min(
                    1e-8
                )
            )

            stream_context = (
                stream_tokens.float()
                *
                stream_weight[..., None]
            ).sum(
                dim=1
            )

            stream_context = (
                stream_context
                *
                stream_valid[:, None].to(
                    stream_context.dtype
                )
            )

            return (
                stream_context,
                stream_valid,
            )

        wz_context, wz_valid = _pool_stream(
            tokens[
                :,
                :wz_token_count,
            ],
            token_mask[
                :,
                :wz_token_count,
            ],
        )

        worker_context, worker_valid = _pool_stream(
            tokens[
                :,
                wz_token_count:,
            ],
            token_mask[
                :,
                wz_token_count:,
            ],
        )

        token_xy = torch.cat(
            (
                midpoint,
                polygon_center[:, None],
                sign_xy[:, None],
                worker_xy,
            ),
            dim=1,
        )

        return {
            "wz_tokens": tokens,
            "wz_token_mask": token_mask,
            "wz_token_xy": token_xy,
            "wz_context": wz_context,
            "wz_valid": wz_valid,

            # V3 explicitly separated worker representation.
            "worker_tokens": tokens[:, wz_token_count:],
            "worker_token_mask": token_mask[:, wz_token_count:],
            "worker_token_xy": token_xy[:, wz_token_count:],
            "worker_context": worker_context,
            "worker_valid": worker_valid,

            # Explicit geometry-only WZ token views.
            "wz_geometry_tokens": tokens[:, :wz_token_count],
            "wz_geometry_token_mask": token_mask[:, :wz_token_count],
            "wz_geometry_token_xy": token_xy[:, :wz_token_count],
            "boundary_lane_distance": boundary_lane_distance,
            "boundary_lane_overlap": boundary_lane_overlap,
            "polygon_lane_distance": polygon_lane_distance,
            "polygon_lane_overlap": polygon_lane_overlap,
            "sign_lane_distance": sign_lane_distance,
            "worker_lane_distance": worker_lane_distance,
            "worker_boundary_distance": worker_boundary_distance,
            "worker_inside": worker_inside,
        }
