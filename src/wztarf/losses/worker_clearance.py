"""Penalize high-probability trajectories that pass too close to workers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def worker_clearance_loss(
    pred_xy: torch.Tensor,
    mode_prob: torch.Tensor,
    worker_xy: torch.Tensor,
    worker_mask: torch.Tensor,
    *,
    threshold_m: float = 2.0,
) -> torch.Tensor:
    """Compute probability-weighted worker-clearance penalty.

    Args:
        pred_xy:
            Candidate trajectories `[B, K, T, 2]`.

        mode_prob:
            Mode probabilities `[B, K]`.

        worker_xy:
            Worker positions `[B, W, 2]`.

        worker_mask:
            Valid-worker mask `[B, W]`.

        threshold_m:
            Clearance threshold. The project currently uses 2.0 m.

    Returns:
        Scalar worker-safety loss averaged over samples containing workers.
    """
    if threshold_m <= 0:
        raise ValueError(
            "threshold_m must be positive."
        )

    batch_size, num_modes, future_steps, _ = pred_xy.shape

    if mode_prob.shape != (
        batch_size,
        num_modes,
    ):
        raise ValueError(
            "mode_prob must have shape [B, K]."
        )

    if worker_xy.ndim != 3 or worker_xy.shape[-1] != 2:
        raise ValueError(
            "worker_xy must have shape [B, W, 2]."
        )

    if worker_mask.shape != worker_xy.shape[:2]:
        raise ValueError(
            "worker_mask must have shape [B, W]."
        )

    sample_losses: list[torch.Tensor] = []

    for batch_index in range(batch_size):
        valid_workers = worker_xy[
            batch_index,
            worker_mask[batch_index].bool(),
        ]

        if valid_workers.numel() == 0:
            continue

        # [K, T, W]
        distance = torch.linalg.vector_norm(
            pred_xy[batch_index, :, :, None, :]
            -
            valid_workers[None, None, :, :],
            dim=-1,
        )

        # Architecture definition:
        # (1 / T) * sum_t sum_worker max(0, 2 - d)^2
        risk_per_mode = F.relu(
            threshold_m
            -
            distance
        ).square().sum(
            dim=-1
        ).mean(
            dim=-1
        )

        sample_loss = (
            mode_prob[batch_index]
            *
            risk_per_mode
        ).sum()

        sample_losses.append(
            sample_loss
        )

    if not sample_losses:
        return pred_xy.sum() * 0.0

    return torch.stack(
        sample_losses
    ).mean()
