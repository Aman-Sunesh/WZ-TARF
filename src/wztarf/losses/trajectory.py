"""Select the best trajectory mode and apply WTA Huber regression."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
) -> None:
    """Validate canonical trajectory tensor shapes."""
    if pred_xy.ndim != 4 or pred_xy.shape[-1] != 2:
        raise ValueError(
            "pred_xy must have shape [B, K, T, 2]."
        )

    if gt_xy.ndim != 3 or gt_xy.shape[-1] != 2:
        raise ValueError(
            "gt_xy must have shape [B, T, 2]."
        )

    if pred_xy.shape[0] != gt_xy.shape[0]:
        raise ValueError(
            "Prediction and GT batch sizes do not match."
        )

    if pred_xy.shape[2] != gt_xy.shape[1]:
        raise ValueError(
            "Prediction and GT horizons do not match."
        )


def mode_assignment_cost(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    *,
    beta_assign: float = 0.25,
) -> torch.Tensor:
    """Compute ADE + beta * FDE for every trajectory mode.

    Returns:
        Assignment cost `[B, K]`.
    """
    _validate(
        pred_xy,
        gt_xy,
    )

    if beta_assign < 0:
        raise ValueError(
            "beta_assign cannot be negative."
        )

    displacement = torch.linalg.vector_norm(
        pred_xy
        -
        gt_xy[:, None],
        dim=-1,
    )

    ade = displacement.mean(
        dim=-1
    )

    fde = displacement[:, :, -1]

    return (
        ade
        +
        beta_assign
        *
        fde
    )


def winner_indices(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    *,
    beta_assign: float = 0.25,
) -> torch.Tensor:
    """Return the WTA mode selected by ADE/FDE assignment cost."""
    cost = mode_assignment_cost(
        pred_xy,
        gt_xy,
        beta_assign=beta_assign,
    )

    # Winner selection itself is discrete and should not carry gradient.
    return cost.detach().argmin(
        dim=1
    )


def trajectory_loss(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    *,
    winner_idx: torch.Tensor | None = None,
    beta_assign: float = 0.25,
    huber_beta: float = 1.0,
) -> torch.Tensor:
    """Apply Huber regression to the WTA-selected trajectory.

    Args:
        pred_xy:
            Predictions `[B, K, T, 2]`.

        gt_xy:
            Ground truth `[B, T, 2]`.

        winner_idx:
            Optional precomputed WTA mode `[B]`.

        beta_assign:
            FDE contribution used only if winner selection is needed.

    Returns:
        Scalar trajectory regression loss.
    """
    _validate(
        pred_xy,
        gt_xy,
    )

    if winner_idx is None:
        winner_idx = winner_indices(
            pred_xy,
            gt_xy,
            beta_assign=beta_assign,
        )

    batch_idx = torch.arange(
        pred_xy.shape[0],
        device=pred_xy.device,
    )

    selected = pred_xy[
        batch_idx,
        winner_idx,
    ]

    return F.smooth_l1_loss(
        selected,
        gt_xy,
        beta=huber_beta,
        reduction="mean",
    )
