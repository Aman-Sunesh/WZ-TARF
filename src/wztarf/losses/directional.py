"""Penalize local trajectory segments that point in the wrong direction."""

from __future__ import annotations

import torch


def directional_loss(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    winner_idx: torch.Tensor,
    *,
    min_movement_m: float = 1e-3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compare local predicted and ground-truth motion directions.

    Only the WTA-selected trajectory is supervised.

    Timesteps whose ground-truth displacement is smaller than
    `min_movement_m` are ignored because direction is unstable when the
    vehicle is effectively stationary.

    Args:
        pred_xy:
            Predicted trajectories `[B, K, T, 2]`.

        gt_xy:
            Ground-truth future `[B, T, 2]`.

        winner_idx:
            WTA-selected mode index `[B]`.

    Returns:
        Scalar cosine-direction loss.
    """
    if pred_xy.ndim != 4 or pred_xy.shape[-1] != 2:
        raise ValueError(
            "pred_xy must have shape [B, K, T, 2]."
        )

    if gt_xy.ndim != 3 or gt_xy.shape[-1] != 2:
        raise ValueError(
            "gt_xy must have shape [B, T, 2]."
        )

    if winner_idx.shape != (pred_xy.shape[0],):
        raise ValueError(
            "winner_idx must have shape [B]."
        )

    batch_idx = torch.arange(
        pred_xy.shape[0],
        device=pred_xy.device,
    )

    selected = pred_xy[
        batch_idx,
        winner_idx,
    ]

    pred_delta = (
        selected[:, 1:]
        -
        selected[:, :-1]
    )

    gt_delta = (
        gt_xy[:, 1:]
        -
        gt_xy[:, :-1]
    )

    pred_norm = torch.linalg.vector_norm(
        pred_delta,
        dim=-1,
    )

    gt_norm = torch.linalg.vector_norm(
        gt_delta,
        dim=-1,
    )

    movement_mask = (
        gt_norm
        >
        min_movement_m
    )

    dot = (
        pred_delta
        *
        gt_delta
    ).sum(dim=-1)

    cosine = (
        dot
        /
        (
            (pred_norm + eps)
            *
            (gt_norm + eps)
        )
    )

    cosine = cosine.clamp(
        min=-1.0,
        max=1.0,
    )

    loss = (
        1.0
        -
        cosine
    )

    if not bool(movement_mask.any()):
        return pred_xy.sum() * 0.0

    return loss[movement_mask].mean()
