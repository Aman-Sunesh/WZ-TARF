"""Penalize predicted road violations only where local map geometry is reliable."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from wztarf.data.map_coverage import (
    distance_to_lane_union_batched,
)


def road_compliance_loss(
    pred_xy: torch.Tensor,
    road_reliability_mask: torch.Tensor,
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
    *,
    epsilon_pred_m: float = 0.25,
) -> torch.Tensor:
    """Compute batched coverage-aware road-compliance loss.

    Geometry semantics are unchanged, but predicted trajectories are processed
    together on the accelerator rather than one batch item / lane at a time.
    """
    if pred_xy.ndim != 4 or pred_xy.shape[-1] != 2:
        raise ValueError(
            "pred_xy must have shape [B, K, T, 2]."
        )

    batch_size, num_modes, future_steps, _ = pred_xy.shape

    if road_reliability_mask.shape != (
        batch_size,
        future_steps,
    ):
        raise ValueError(
            "road_reliability_mask must have shape [B, T]."
        )

    if epsilon_pred_m < 0:
        raise ValueError(
            "epsilon_pred_m cannot be negative."
        )

    points = pred_xy.reshape(
        batch_size,
        num_modes * future_steps,
        2,
    )

    distance = distance_to_lane_union_batched(
        points,
        lane_feat,
        lane_point_mask,
        lane_mask,
    ).reshape(
        batch_size,
        num_modes,
        future_steps,
    )

    penalty = F.relu(
        distance
        -
        epsilon_pred_m
    ).square()

    mask = (
        road_reliability_mask.bool()
        .unsqueeze(1)
        .expand(
            -1,
            num_modes,
            -1,
        )
    )

    numerator = (
        penalty
        *
        mask.to(penalty.dtype)
    ).sum()

    denominator = (
        mask.sum()
        .clamp_min(1)
        .to(penalty.dtype)
    )

    return numerator / denominator
