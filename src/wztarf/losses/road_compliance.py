"""Penalize predicted road violations only where local map geometry is reliable."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from wztarf.data.map_coverage import distance_to_lane_union


def road_compliance_loss(
    pred_xy: torch.Tensor,
    road_reliability_mask: torch.Tensor,
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
    *,
    epsilon_pred_m: float = 0.25,
) -> torch.Tensor:
    """Compute coverage- and reliability-aware road-compliance loss.

    Args:
        pred_xy:
            Candidate trajectories `[B, K, T, 2]`.

        road_reliability_mask:
            GT-derived reliability mask `[B, T]`.

        lane_feat:
            Batched lane geometry `[B, L, P, 8]`.

        lane_point_mask:
            Batched valid-point mask `[B, L, P]`.

        lane_mask:
            Batched valid-lane mask `[B, L]`.

        epsilon_pred_m:
            Small geometric tolerance before a road penalty begins.

    Returns:
        Scalar road-compliance loss.
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

    total_loss = pred_xy.sum() * 0.0
    denominator = 0

    for batch_index in range(batch_size):
        reliable = road_reliability_mask[
            batch_index
        ].bool()

        valid_steps = int(
            reliable.sum().item()
        )

        if valid_steps == 0:
            continue

        points = pred_xy[
            batch_index
        ].reshape(
            num_modes * future_steps,
            2,
        )

        distance = distance_to_lane_union(
            points,
            lane_feat[batch_index],
            lane_point_mask[batch_index],
            lane_mask[batch_index],
        ).reshape(
            num_modes,
            future_steps,
        )

        penalty = F.relu(
            distance
            -
            epsilon_pred_m
        ).square()

        mask = reliable.unsqueeze(0).expand(
            num_modes,
            -1,
        )

        total_loss = (
            total_loss
            +
            penalty[mask].sum()
        )

        denominator += (
            num_modes
            *
            valid_steps
        )

    if denominator == 0:
        return pred_xy.sum() * 0.0

    return total_loss / float(denominator)
