"""Supervise the final 5 s endpoint of the WTA-selected trajectory."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def endpoint_loss(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    winner_idx: torch.Tensor,
    *,
    huber_beta: float = 1.0,
) -> torch.Tensor:
    """Compute explicit terminal-point Huber loss.

    Args:
        pred_xy:
            Predicted trajectories `[B, K, T, 2]`.

        gt_xy:
            Ground-truth future `[B, T, 2]`.

        winner_idx:
            Selected WTA mode `[B]`.

    Returns:
        Scalar endpoint loss.
    """
    batch_size = pred_xy.shape[0]

    if winner_idx.shape != (batch_size,):
        raise ValueError(
            "winner_idx must have shape [B]."
        )

    batch_idx = torch.arange(
        batch_size,
        device=pred_xy.device,
    )

    pred_endpoint = pred_xy[
        batch_idx,
        winner_idx,
        -1,
    ]

    gt_endpoint = gt_xy[:, -1]

    return F.smooth_l1_loss(
        pred_endpoint,
        gt_endpoint,
        beta=huber_beta,
        reduction="mean",
    )
