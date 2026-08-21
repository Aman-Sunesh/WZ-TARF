"""Supervise the trajectory-local residual refinement module."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def refinement_loss(
    coarse_xy: torch.Tensor,
    refinement_delta: torch.Tensor,
    gt_xy: torch.Tensor,
    winner_idx: torch.Tensor,
    *,
    huber_beta: float = 1.0,
) -> torch.Tensor:
    """Train the refiner to correct the coarse selected trajectory.

    The target residual is:

        ground_truth - coarse_prediction

    Args:
        coarse_xy:
            Coarse trajectories `[B, K, T, 2]`.

        refinement_delta:
            Predicted residual corrections `[B, K, T, 2]`.

        gt_xy:
            Ground-truth future `[B, T, 2]`.

        winner_idx:
            Selected WTA mode `[B]`.

    Returns:
        Scalar residual-refinement loss.
    """
    if coarse_xy.shape != refinement_delta.shape:
        raise ValueError(
            "coarse_xy and refinement_delta must have identical shapes."
        )

    batch_size = coarse_xy.shape[0]

    batch_idx = torch.arange(
        batch_size,
        device=coarse_xy.device,
    )

    selected_coarse = coarse_xy[
        batch_idx,
        winner_idx,
    ]

    selected_delta = refinement_delta[
        batch_idx,
        winner_idx,
    ]

    # The final trajectory loss already supervises coarse + refinement.
    # Detach the coarse prediction here so this auxiliary objective
    # specifically trains the local refiner to correct the current
    # coarse trajectory instead of duplicating the full regression
    # gradient through the coarse decoder.
    target_delta = (
        gt_xy
        -
        selected_coarse.detach()
    )

    return F.smooth_l1_loss(
        selected_delta,
        target_delta,
        beta=huber_beta,
        reduction="mean",
    )
