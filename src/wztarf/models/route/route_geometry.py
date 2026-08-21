"""Differentiable route-walk geometry and monotonic route-progress anchors."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _lane_geometry(
    lane_centerline: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
    *,
    samples_per_lane: int = 4,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Return ego-forward sampled lane geometry, downstream tangent,
    and distance from each lane to the ego origin.

    IMPORTANT
    ---------
    Serialized lane point order is not consistently aligned with
    the ego's forward driving direction.

    The dataset is ego-centered and yaw-aligned, so +X is forward.

    For each lane we therefore:

      1. identify both physical lane endpoints;
      2. compute the OUTWARD tangent from each endpoint;
      3. choose as downstream the endpoint whose outward tangent
         has the larger +X component;
      4. sample from the opposite endpoint toward that downstream
         endpoint;
      5. return a terminal tangent pointing downstream.

    This rule matched the endpoint oracle exactly on the 128-scene
    validation diagnostic.
    """

    if samples_per_lane < 2:
        raise ValueError(
            "samples_per_lane must be >= 2."
        )

    (
        batch_size,
        num_lanes,
        num_points,
        _,
    ) = lane_centerline.shape

    if num_points < 1:
        raise ValueError(
            "lane_centerline must contain at least one point."
        )

    lane32 = lane_centerline.float()

    valid = (
        lane_point_mask.bool()
        &
        lane_mask[
            :,
            :,
            None,
        ].bool()
    )

    point_index = torch.arange(
        num_points,
        device=lane32.device,
    ).view(
        1,
        1,
        num_points,
    )

    first_index = torch.where(
        valid,
        point_index,
        torch.full_like(
            point_index,
            num_points,
        ),
    ).amin(
        dim=-1
    )

    last_index = torch.where(
        valid,
        point_index,
        torch.full_like(
            point_index,
            -1,
        ),
    ).amax(
        dim=-1
    )

    has_point = (
        (first_index < num_points)
        &
        (last_index >= 0)
    )

    safe_first = first_index.clamp(
        min=0,
        max=num_points - 1,
    )

    safe_last = last_index.clamp(
        min=0,
        max=num_points - 1,
    )

    first_next_index = torch.minimum(
        safe_first + 1,
        safe_last,
    )

    last_previous_index = torch.maximum(
        safe_last - 1,
        safe_first,
    )

    def gather_point(
        index: torch.Tensor,
    ) -> torch.Tensor:

        return lane32.gather(
            2,
            index[
                ...,
                None,
                None,
            ].expand(
                batch_size,
                num_lanes,
                1,
                2,
            ),
        ).squeeze(
            2
        )

    first_point = gather_point(
        safe_first
    )

    first_next = gather_point(
        first_next_index
    )

    last_point = gather_point(
        safe_last
    )

    last_previous = gather_point(
        last_previous_index
    )

    # ----------------------------------------------------------
    # OUTWARD tangent from the first endpoint:
    #
    #     interior ----> first
    #
    # therefore first - next.
    # ----------------------------------------------------------

    outward_first = (
        first_point
        -
        first_next
    )

    # ----------------------------------------------------------
    # OUTWARD tangent from the last endpoint:
    #
    #     previous ----> last
    #
    # therefore last - previous.
    # ----------------------------------------------------------

    outward_last = (
        last_point
        -
        last_previous
    )

    first_norm = torch.linalg.vector_norm(
        outward_first,
        dim=-1,
        keepdim=True,
    )

    last_norm = torch.linalg.vector_norm(
        outward_last,
        dim=-1,
        keepdim=True,
    )

    default_tangent = torch.zeros_like(
        outward_first
    )

    default_tangent[
        ...,
        0,
    ] = 1.0

    outward_first_unit = torch.where(
        first_norm > 1.0e-6,
        outward_first
        /
        first_norm.clamp_min(
            1.0e-6
        ),
        default_tangent,
    )

    outward_last_unit = torch.where(
        last_norm > 1.0e-6,
        outward_last
        /
        last_norm.clamp_min(
            1.0e-6
        ),
        default_tangent,
    )

    # ==========================================================
    # PROVEN ORIENTATION RULE
    #
    # Ego frame: +X is forward.
    #
    # Choose whichever endpoint's OUTWARD tangent has the larger
    # +X component.
    # ==========================================================

    downstream_is_last = (
        outward_last_unit[
            ...,
            0,
        ]
        >=
        outward_first_unit[
            ...,
            0,
        ]
    )

    upstream_index = torch.where(
        downstream_is_last,
        safe_first,
        safe_last,
    )

    downstream_index = torch.where(
        downstream_is_last,
        safe_last,
        safe_first,
    )

    downstream_outward = torch.where(
        downstream_is_last[
            ...,
            None,
        ],
        outward_last_unit,
        outward_first_unit,
    )

    # ----------------------------------------------------------
    # Canonical SUCCESSOR geometry:
    # full +X upstream endpoint -> +X downstream endpoint.
    #
    # Root-only closest-point truncation is constructed separately
    # inside forward().
    # ----------------------------------------------------------

    fractions = torch.linspace(
        0.0,
        1.0,
        samples_per_lane,
        dtype=torch.float32,
        device=lane32.device,
    ).view(
        1,
        1,
        samples_per_lane,
    )

    sample_position = (
        upstream_index[
            :,
            :,
            None,
        ].float()
        +
        fractions
        *
        (
            downstream_index
            -
            upstream_index
        )[
            :,
            :,
            None,
        ].float()
    )
    sample_index = torch.round(
        sample_position
    ).long().clamp(
        min=0,
        max=num_points - 1,
    )

    lane_samples = lane32.gather(
        2,
        sample_index[
            ...,
            None,
        ].expand(
            batch_size,
            num_lanes,
            samples_per_lane,
            2,
        ),
    )

    lane_samples = (
        lane_samples
        *
        has_point[
            :,
            :,
            None,
            None,
        ].to(
            lane_samples.dtype
        )
    )

    # ----------------------------------------------------------
    # Terminal tangent must point toward the selected downstream
    # endpoint.
    # ----------------------------------------------------------

    tangent = (
        lane_samples[
            :,
            :,
            -1,
        ]
        -
        lane_samples[
            :,
            :,
            -2,
        ]
    )

    tangent_norm = torch.linalg.vector_norm(
        tangent,
        dim=-1,
        keepdim=True,
    )

    tangent = torch.where(
        tangent_norm > 1.0e-6,
        tangent
        /
        tangent_norm.clamp_min(
            1.0e-6
        ),
        downstream_outward,
    )

    tangent = (
        tangent
        *
        has_point[
            ...,
            None,
        ].to(
            tangent.dtype
        )
    )

    # ----------------------------------------------------------
    # Physical distance from lane to ego origin.
    # ----------------------------------------------------------

    point_distance = (
        torch.linalg.vector_norm(
            lane32,
            dim=-1,
        )
    ).masked_fill(
        ~valid,
        float("inf"),
    )

    origin_distance = point_distance.amin(
        dim=-1
    )

    origin_distance = torch.where(
        torch.isfinite(
            origin_distance
        ),
        origin_distance,
        torch.full_like(
            origin_distance,
            1.0e6,
        ),
    )

    return (
        lane_samples,
        tangent,
        origin_distance,
    )


def _interpolate_polyline(
    points: torch.Tensor,
    cumulative_s: torch.Tensor,
    progress: torch.Tensor,
) -> torch.Tensor:
    """Interpolate 1/3/5 s progress along a differentiable route polyline.

    Segment selection is detached. The selected coordinates, segment lengths,
    and interpolation fraction retain gradients.
    """

    (
        batch_size,
        num_modes,
        num_points,
        _,
    ) = points.shape

    num_horizons = (
        progress.shape[-1]
    )

    flat_points = points.reshape(
        batch_size * num_modes,
        num_points,
        2,
    )

    flat_s = cumulative_s.reshape(
        batch_size * num_modes,
        num_points,
    ).contiguous()

    flat_progress = progress.reshape(
        batch_size * num_modes,
        num_horizons,
    )

    max_s = flat_s[
        :,
        -1:
    ].clamp_min(
        1.0e-4
    )

    eval_progress = torch.minimum(
        flat_progress,
        max_s - 1.0e-5,
    ).clamp_min(
        0.0
    )

    right = torch.searchsorted(
        flat_s.detach(),
        eval_progress.detach().contiguous(),
        right=False,
    ).clamp(
        min=1,
        max=num_points - 1,
    )

    left = right - 1

    left_s = flat_s.gather(
        1,
        left,
    )

    right_s = flat_s.gather(
        1,
        right,
    )

    left_xy = flat_points.gather(
        1,
        left[
            ...,
            None,
        ].expand(
            -1,
            -1,
            2,
        ),
    )

    right_xy = flat_points.gather(
        1,
        right[
            ...,
            None,
        ].expand(
            -1,
            -1,
            2,
        ),
    )

    alpha = (
        (
            eval_progress
            -
            left_s
        )
        /
        (
            right_s
            -
            left_s
        ).clamp_min(
            1.0e-6
        )
    ).clamp(
        0.0,
        1.0,
    )

    anchor = (
        left_xy
        +
        alpha[
            ...,
            None,
        ]
        *
        (
            right_xy
            -
            left_xy
        )
    )

    return anchor.reshape(
        batch_size,
        num_modes,
        num_horizons,
        2,
    )


