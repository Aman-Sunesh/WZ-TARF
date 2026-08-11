"""Supervise terminal retained-lane or MAP_EXIT goal classification."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def lane_goal_loss(
    goal_logits: torch.Tensor,
    goal_target: torch.Tensor,
    goal_valid: torch.Tensor,
    winner_idx: torch.Tensor,
) -> torch.Tensor:
    """Compute terminal-goal classification for the selected route mode.

    Args:
        goal_logits:
            Goal-class logits `[B, K, C]`.

        goal_target:
            Integer target `[B]`. The MAP_EXIT class is simply one of the
            valid class indices.

        goal_valid:
            Boolean mask `[B]`. False indicates ambiguous in-map supervision
            and removes that sample from this loss.

        winner_idx:
            WTA mode index `[B]`.

    Returns:
        Scalar cross-entropy loss.
    """
    if goal_logits.ndim != 3:
        raise ValueError(
            "goal_logits must have shape [B, K, C]."
        )

    batch_size = goal_logits.shape[0]

    if goal_target.shape != (batch_size,):
        raise ValueError(
            "goal_target must have shape [B]."
        )

    if goal_valid.shape != (batch_size,):
        raise ValueError(
            "goal_valid must have shape [B]."
        )

    if winner_idx.shape != (batch_size,):
        raise ValueError(
            "winner_idx must have shape [B]."
        )

    goal_valid = goal_valid.bool()

    if not bool(goal_valid.any()):
        return goal_logits.sum() * 0.0

    batch_idx = torch.arange(
        batch_size,
        device=goal_logits.device,
    )

    selected_logits = goal_logits[
        batch_idx,
        winner_idx,
    ]

    return F.cross_entropy(
        selected_logits[goal_valid],
        goal_target[goal_valid].long(),
    )
