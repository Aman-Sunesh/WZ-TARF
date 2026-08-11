"""Construct lane, MAP_EXIT, longitudinal-goal, and road-reliability targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from wztarf.data.map_coverage import distance_to_lane_union
from wztarf.geometry.lanes import (
    lane_bounds,
    polyline_longitudinal_offset,
    reconstruct_lane_polygon,
)
from wztarf.geometry.workzone import (
    distance_to_polygon,
    points_in_polygon,
)


@dataclass
class SupervisedTargets:
    """Training targets derived from GT future and represented map geometry."""

    goal_target: torch.Tensor
    goal_valid: torch.Tensor
    goal_offset_target: torch.Tensor
    lane_goal_mask: torch.Tensor
    road_reliability_mask: torch.Tensor


def _coverage_mask(
    points: torch.Tensor,
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
) -> torch.Tensor:
    """Return conservative map-crop coverage based on represented lane bounds."""
    try:
        bounds = lane_bounds(
            lane_feat,
            lane_point_mask,
            lane_mask,
        )
    except ValueError:
        return torch.zeros(
            points.shape[0],
            dtype=torch.bool,
            device=points.device,
        )

    xmin, ymin, xmax, ymax = bounds.unbind()

    return (
        (points[:, 0] >= xmin)
        &
        (points[:, 0] <= xmax)
        &
        (points[:, 1] >= ymin)
        &
        (points[:, 1] <= ymax)
    )


def build_supervised_targets(
    *,
    gt_xy: torch.Tensor,
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
    retained_lane_mask: torch.Tensor,
    association_tolerance_m: float = 0.25,
    road_gt_tolerance_m: float = 0.25,
) -> SupervisedTargets:
    """Build supervised map targets without treating ambiguity as MAP_EXIT.

    Target policy:

        outside represented map coverage
            -> MAP_EXIT

        inside coverage and reliably associated with retained lane
            -> that lane

        inside coverage but no reliable retained-lane association
            -> terminal lane classification masked
    """
    if gt_xy.ndim != 3 or gt_xy.shape[-1] != 2:
        raise ValueError(
            "gt_xy must have shape [B, T, 2]."
        )

    batch_size, future_steps, _ = gt_xy.shape
    num_lanes = lane_feat.shape[1]

    if retained_lane_mask.shape != (
        batch_size,
        num_lanes,
    ):
        raise ValueError(
            "retained_lane_mask must have shape [B, L]."
        )

    map_exit_class = num_lanes

    goal_target = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=gt_xy.device,
    )

    goal_valid = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=gt_xy.device,
    )

    goal_offset_target = torch.zeros(
        batch_size,
        dtype=gt_xy.dtype,
        device=gt_xy.device,
    )

    lane_goal_mask = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=gt_xy.device,
    )

    road_reliability = torch.zeros(
        batch_size,
        future_steps,
        dtype=torch.bool,
        device=gt_xy.device,
    )

    for b in range(
        batch_size
    ):
        raw_lane_mask = lane_mask[
            b
        ].bool()

        coverage = _coverage_mask(
            gt_xy[
                b
            ],
            lane_feat[
                b
            ],
            lane_point_mask[
                b
            ],
            raw_lane_mask,
        )

        distance = distance_to_lane_union(
            gt_xy[
                b
            ],
            lane_feat[
                b
            ],
            lane_point_mask[
                b
            ],
            raw_lane_mask,
        )

        road_reliability[
            b
        ] = (
            coverage
            &
            (
                distance
                <=
                road_gt_tolerance_m
            )
        )

        endpoint = gt_xy[
            b,
            -1,
        ]

        # Explicit beyond-map case.
        if not bool(
            coverage[
                -1
            ]
        ):
            goal_target[
                b
            ] = map_exit_class

            goal_valid[
                b
            ] = True

            continue

        retained = (
            retained_lane_mask[
                b
            ].bool()
            &
            raw_lane_mask
        )

        best_lane: int | None = None
        best_distance = float("inf")

        for lane_index in torch.nonzero(
            retained,
            as_tuple=False,
        ).flatten().tolist():
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

            inside = bool(
                points_in_polygon(
                    endpoint[
                        None
                    ],
                    polygon,
                )[0]
            )

            if inside:
                lane_distance = 0.0
            else:
                lane_distance = float(
                    distance_to_polygon(
                        endpoint[
                            None
                        ],
                        polygon,
                    )[0].item()
                )

            if lane_distance < best_distance:
                best_distance = lane_distance
                best_lane = lane_index

        # In-map but no reliable retained association remains intentionally
        # masked instead of being mislabeled MAP_EXIT.
        if (
            best_lane is None
            or
            best_distance
            >
            association_tolerance_m
        ):
            continue

        goal_target[
            b
        ] = best_lane

        goal_valid[
            b
        ] = True

        lane_goal_mask[
            b
        ] = True

        valid_points = lane_point_mask[
            b,
            best_lane,
        ].bool()

        centerline = lane_feat[
            b,
            best_lane,
            valid_points,
            :2,
        ]

        goal_offset_target[
            b
        ] = polyline_longitudinal_offset(
            endpoint,
            centerline,
        )

    return SupervisedTargets(
        goal_target=goal_target,
        goal_valid=goal_valid,
        goal_offset_target=goal_offset_target,
        lane_goal_mask=lane_goal_mask,
        road_reliability_mask=road_reliability,
    )
