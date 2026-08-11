"""Construct geometry-derived WorkZone topology pseudo-targets."""

from __future__ import annotations
from dataclasses import dataclass

import torch

from wztarf.geometry.lanes import reconstruct_lane_polygon
from wztarf.geometry.workzone import (
    distance_to_polygon,
    polygons_intersect,
    segments_intersect_polygon,
)


@dataclass
class TopologyTargets:
    """Targets and validity masks for topology reconstruction."""

    lane_overlap: torch.Tensor
    lane_distance: torch.Tensor
    edge_compatibility: torch.Tensor
    lane_mask: torch.Tensor
    edge_mask: torch.Tensor


def build_topology_targets(
    *,
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
    lane_edge_index: torch.Tensor,
    lane_edge_mask: torch.Tensor,
    wz_feat: torch.Tensor,
) -> TopologyTargets:
    """Build topology pseudo-targets directly from represented geometry."""
    batch_size, num_lanes = lane_feat.shape[:2]
    num_edges = lane_edge_index.shape[-1]

    overlap = torch.zeros(
        batch_size,
        num_lanes,
        dtype=lane_feat.dtype,
        device=lane_feat.device,
    )

    distance = torch.zeros_like(
        overlap
    )

    edge_compatibility = torch.zeros(
        batch_size,
        num_edges,
        dtype=lane_feat.dtype,
        device=lane_feat.device,
    )

    lane_target_mask = torch.zeros(
        batch_size,
        num_lanes,
        dtype=torch.bool,
        device=lane_feat.device,
    )

    edge_target_mask = torch.zeros(
        batch_size,
        num_edges,
        dtype=torch.bool,
        device=lane_feat.device,
    )

    for b in range(
        batch_size
    ):
        wz_valid = (
            wz_feat[
                b,
                :4,
                2,
            ]
            >
            0
        ).all()

        if not bool(
            wz_valid
        ):
            continue

        wz_polygon = wz_feat[
            b,
            :4,
            :2,
        ]

        polygons: list[torch.Tensor | None] = [
            None
        ] * num_lanes

        for lane_index in range(
            num_lanes
        ):
            if not bool(
                lane_mask[
                    b,
                    lane_index,
                ]
            ):
                continue

            polygon = reconstruct_lane_polygon(
                lane_feat[
                    b,
                    lane_index,
                ],
                lane_point_mask[
                    b,
                    lane_index,
                ],
            )

            if polygon is None:
                continue

            polygons[
                lane_index
            ] = polygon

            lane_target_mask[
                b,
                lane_index,
            ] = True

            intersects = polygons_intersect(
                polygon,
                wz_polygon,
            )

            overlap[
                b,
                lane_index,
            ] = float(
                intersects
            )

            if intersects:
                distance[
                    b,
                    lane_index,
                ] = 0.0
            else:
                d1 = distance_to_polygon(
                    polygon,
                    wz_polygon,
                ).min()

                d2 = distance_to_polygon(
                    wz_polygon,
                    polygon,
                ).min()

                distance[
                    b,
                    lane_index,
                ] = torch.minimum(
                    d1,
                    d2,
                )

        valid_edges = torch.nonzero(
            lane_edge_mask[
                b
            ].bool(),
            as_tuple=False,
        ).flatten()

        for edge_index in valid_edges.tolist():
            src = int(
                lane_edge_index[
                    b,
                    0,
                    edge_index,
                ].item()
            )

            dst = int(
                lane_edge_index[
                    b,
                    1,
                    edge_index,
                ].item()
            )

            if not (
                0 <= src < num_lanes
                and
                0 <= dst < num_lanes
            ):
                continue

            if not (
                bool(
                    lane_target_mask[
                        b,
                        src,
                    ]
                )
                and
                bool(
                    lane_target_mask[
                        b,
                        dst,
                    ]
                )
            ):
                continue

            edge_target_mask[
                b,
                edge_index,
            ] = True

            blocked = (
                bool(
                    overlap[
                        b,
                        src,
                    ]
                    >
                    0
                )
                or
                bool(
                    overlap[
                        b,
                        dst,
                    ]
                    >
                    0
                )
            )

            if not blocked:
                src_valid = lane_point_mask[
                    b,
                    src,
                ].bool()

                dst_valid = lane_point_mask[
                    b,
                    dst,
                ].bool()

                src_center = lane_feat[
                    b,
                    src,
                    src_valid,
                    :2,
                ]

                dst_center = lane_feat[
                    b,
                    dst,
                    dst_valid,
                    :2,
                ]

                if (
                    src_center.shape[0] > 0
                    and
                    dst_center.shape[0] > 0
                ):
                    blocked = segments_intersect_polygon(
                        src_center[
                            -1
                        ],
                        dst_center[
                            0
                        ],
                        wz_polygon,
                    )

            edge_compatibility[
                b,
                edge_index,
            ] = (
                0.0
                if blocked
                else 1.0
            )

    return TopologyTargets(
        lane_overlap=overlap,
        lane_distance=distance,
        edge_compatibility=edge_compatibility,
        lane_mask=lane_target_mask,
        edge_mask=edge_target_mask,
    )
