"""Encode WZ boundary, polygon, sign, and worker geometry as typed tokens."""

from __future__ import annotations

import torch
from torch import nn

from wztarf.geometry.workzone import (
    distance_to_polygon,
    points_in_polygon,
)

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
    
class WorkZoneEncoder(nn.Module):
    """Encode structured WorkZone geometry with typed self-attention."""

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.d_model = d_model

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

        lane_geometry = _lane_segment_geometry(
            lane_feat,
            lane_point_mask,
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
            lane_feat is not None
            and
            lane_point_mask is not None
            and
            lane_mask is not None
        ):
            valid_lane_point = (
                lane_point_mask.bool()
                &
                lane_mask.bool()[
                    :,
                    :,
                    None,
                ]
            )
        
            for b in range(batch_size):
                if (
                    bool(
                        polygon_valid[b]
                    )
                    and
                    bool(
                        valid_lane_point[b].any()
                    )
                ):
                    represented_points = lane_feat[
                        b,
                        ...,
                        :2,
                    ][
                        valid_lane_point[b]
                    ]
        
                    if bool(
                        points_in_polygon(
                            represented_points,
                            corners[b],
                        ).any()
                    ):
                        polygon_lane_overlap[
                            b
                        ] = True
        
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

        for b in range(batch_size):
            if not bool(
                polygon_valid[b]
            ):
                continue
    
            if bool(
                sign_valid[b]
            ):
                sign_boundary_distance[
                    b
                ] = distance_to_polygon(
                    sign_xy[
                        b:
                        b + 1
                    ],
                    corners[b],
                )[0]
    
                sign_inside[
                    b
                ] = points_in_polygon(
                    sign_xy[
                        b:
                        b + 1
                    ],
                    corners[b],
                )[0].to(
                    sign_distance.dtype
                )
    
            if worker_xy.shape[1] > 0:
                worker_boundary_distance[
                    b
                ] = distance_to_polygon(
                    worker_xy[b],
                    corners[b],
                )
                worker_inside[
                    b
                ] = points_in_polygon(
                    worker_xy[b],
                    corners[b],
                ).to(
                    worker_distance.dtype
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


        worker_features = torch.stack(
            (
                worker_xy[..., 0],
                worker_xy[..., 1],
                worker_distance,
                worker_xy[..., 1] / (worker_distance + 1e-8),
                worker_xy[..., 0] / (worker_distance + 1e-8),
                worker_ahead,
                worker_ttp,
                worker_inside,
            worker_lane_distance,
            worker_boundary_distance,
            worker_on_lane,
            worker_signed_boundary_distance,
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

        original_valid = token_mask.any(
            dim=1
        )

        safe_mask = token_mask.clone()

        no_valid = ~original_valid

        if bool(
            no_valid.any()
        ):
            safe_mask[
                no_valid,
                0,
            ] = True

            tokens = tokens.clone()
            tokens[
                no_valid,
                0,
            ] = 0.0

        tokens = self.transformer(
            tokens,
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
            "wz_context": context,
            "wz_valid": original_valid,
            "boundary_lane_distance": boundary_lane_distance,
            "boundary_lane_overlap": boundary_lane_overlap,
            "polygon_lane_distance": polygon_lane_distance,
            "polygon_lane_overlap": polygon_lane_overlap,
            "sign_lane_distance": sign_lane_distance,
            "worker_lane_distance": worker_lane_distance,
            "worker_boundary_distance": worker_boundary_distance,
            "worker_inside": worker_inside,
        }
