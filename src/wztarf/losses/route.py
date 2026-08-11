"""Supervise 1 s, 3 s, 5 s route anchors and in-map goal progress."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def route_loss(
    route_anchors: torch.Tensor,
    anchor_target: torch.Tensor,
    winner_idx: torch.Tensor,
    *,
    horizon_weights: Sequence[float] | torch.Tensor | None = None,
    goal_offset_pred: torch.Tensor | None = None,
    goal_offset_target: torch.Tensor | None = None,
    lane_goal_mask: torch.Tensor | None = None,
    goal_offset_weight: float = 1.0,
    huber_beta: float = 1.0,
) -> torch.Tensor:
    """Supervise coarse route anchors and optional longitudinal lane goal.

    Args:
        route_anchors:
            Predicted anchors `[B, K, H, 2]`.

        anchor_target:
            Ground-truth anchors `[B, H, 2]`.

        winner_idx:
            Selected WTA route mode `[B]`.

        horizon_weights:
            Optional weight for each semantic horizon.

        goal_offset_pred:
            Optional in-lane goal offsets `[B, K]`.

        goal_offset_target:
            Optional ground-truth offsets `[B]`.

        lane_goal_mask:
            `[B]` mask indicating that an in-map longitudinal goal exists.
            MAP_EXIT and ambiguous targets should be false.

    Returns:
        Scalar route loss.
    """
    if route_anchors.ndim != 4 or route_anchors.shape[-1] != 2:
        raise ValueError(
            "route_anchors must have shape [B, K, H, 2]."
        )

    batch_size, _, num_horizons, _ = route_anchors.shape

    if anchor_target.shape != (
        batch_size,
        num_horizons,
        2,
    ):
        raise ValueError(
            "anchor_target must have shape [B, H, 2]."
        )

    batch_idx = torch.arange(
        batch_size,
        device=route_anchors.device,
    )

    selected = route_anchors[
        batch_idx,
        winner_idx,
    ]

    # [B, H, 2] -> [B, H]
    anchor_error = F.smooth_l1_loss(
        selected,
        anchor_target,
        beta=huber_beta,
        reduction="none",
    ).mean(dim=-1)

    if horizon_weights is None:
        weights = torch.ones(
            num_horizons,
            dtype=route_anchors.dtype,
            device=route_anchors.device,
        )

    else:
        weights = torch.as_tensor(
            horizon_weights,
            dtype=route_anchors.dtype,
            device=route_anchors.device,
        )

        if weights.shape != (num_horizons,):
            raise ValueError(
                "horizon_weights must have shape [H]."
            )

        if (weights < 0).any():
            raise ValueError(
                "horizon_weights cannot be negative."
            )

    if float(weights.sum()) <= 0:
        raise ValueError(
            "At least one horizon weight must be positive."
        )

    anchor_loss = (
        anchor_error
        *
        weights.unsqueeze(0)
    ).sum(dim=-1) / weights.sum()

    total = anchor_loss.mean()

    has_offset_inputs = (
        goal_offset_pred is not None
        or goal_offset_target is not None
        or lane_goal_mask is not None
    )

    if has_offset_inputs:
        if (
            goal_offset_pred is None
            or goal_offset_target is None
            or lane_goal_mask is None
        ):
            raise ValueError(
                "goal_offset_pred, goal_offset_target, and "
                "lane_goal_mask must be provided together."
            )

        if goal_offset_pred.shape != (
            batch_size,
            route_anchors.shape[1],
        ):
            raise ValueError(
                "goal_offset_pred must have shape [B, K]."
            )

        if goal_offset_target.shape != (batch_size,):
            raise ValueError(
                "goal_offset_target must have shape [B]."
            )

        lane_goal_mask = lane_goal_mask.bool()

        if bool(lane_goal_mask.any()):
            selected_offset = goal_offset_pred[
                batch_idx,
                winner_idx,
            ]

            offset_loss = F.smooth_l1_loss(
                selected_offset[lane_goal_mask],
                goal_offset_target[lane_goal_mask],
                beta=huber_beta,
                reduction="mean",
            )

            total = (
                total
                +
                goal_offset_weight
                *
                offset_loss
            )

    return total
