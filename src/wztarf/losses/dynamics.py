"""Supervise the control-conditioned dynamics anchor at short horizons."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dynamics_loss(
    dynamics_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    *,
    horizon_steps: int = 10,
    huber_beta: float = 1.0,
) -> torch.Tensor:
    """Train the shared dynamics anchor against the near future.

    Args:
        dynamics_xy:
            Predicted dynamics anchor `[B, T, 2]`.

        gt_xy:
            Ground-truth future `[B, T, 2]`.

        horizon_steps:
            Number of future steps used for supervision.
            At 5 Hz, 10 steps corresponds to 2 seconds.

        huber_beta:
            Smooth-L1 transition point.

    Returns:
        Scalar short-horizon dynamics loss.
    """
    if dynamics_xy.shape != gt_xy.shape:
        raise ValueError(
            "dynamics_xy and gt_xy must have identical shapes."
        )

    if dynamics_xy.ndim != 3 or dynamics_xy.shape[-1] != 2:
        raise ValueError(
            "dynamics_xy must have shape [B, T, 2]."
        )

    if not 1 <= horizon_steps <= dynamics_xy.shape[1]:
        raise ValueError(
            "horizon_steps lies outside the prediction horizon."
        )

    return F.smooth_l1_loss(
        dynamics_xy[:, :horizon_steps],
        gt_xy[:, :horizon_steps],
        beta=huber_beta,
        reduction="mean",
    )
