"""Compute Worker Safety Violation Rate for Top-1 predictions."""

from __future__ import annotations

import torch


def per_sample_worker_violation(
    pred_xy: torch.Tensor,
    mode_prob: torch.Tensor,
    worker_xy: torch.Tensor,
    worker_mask: torch.Tensor,
    threshold_m: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return worker-violation flags and worker-presence flags per sample.

    A sample violates when the highest-probability trajectory contains at
    least one ego reference point whose distance to any valid represented
    worker is less than or equal to `threshold_m`.

    Args:
        pred_xy:
            Predicted trajectories `[B, K, T, 2]`.

        mode_prob:
            Mode probabilities `[B, K]`.

        worker_xy:
            Worker locations `[B, W, 2]`.

        worker_mask:
            Valid-worker mask `[B, W]`.

        threshold_m:
            Worker-clearance threshold in meters.

    Returns:
        violation:
            Boolean tensor `[B]`.

        has_worker:
            Boolean tensor `[B]`.
    """
    if pred_xy.ndim != 4 or pred_xy.shape[-1] != 2:
        raise ValueError(
            "pred_xy must have shape [B, K, T, 2]."
        )

    batch_size, num_modes = pred_xy.shape[:2]

    if mode_prob.shape != (batch_size, num_modes):
        raise ValueError(
            "mode_prob must have shape [B, K]."
        )

    if worker_xy.ndim != 3 or worker_xy.shape[-1] != 2:
        raise ValueError(
            "worker_xy must have shape [B, W, 2]."
        )

    if worker_xy.shape[0] != batch_size:
        raise ValueError(
            "worker_xy batch size must match pred_xy."
        )

    if worker_mask.shape != worker_xy.shape[:2]:
        raise ValueError(
            "worker_mask must have shape [B, W]."
        )

    if threshold_m <= 0:
        raise ValueError(
            "threshold_m must be positive."
        )

    worker_mask = worker_mask.bool()

    top_idx = mode_prob.argmax(
        dim=1
    )

    batch_idx = torch.arange(
        batch_size,
        device=pred_xy.device,
    )

    selected = pred_xy[
        batch_idx,
        top_idx,
    ]

    has_worker = worker_mask.any(
        dim=1
    )

    flags = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=pred_xy.device,
    )

    for b in range(batch_size):
        valid_workers = worker_xy[
            b,
            worker_mask[b],
        ]

        if valid_workers.numel() == 0:
            continue

        distance = torch.cdist(
            selected[b],
            valid_workers,
        )

        flags[b] = (
            distance.min()
            <=
            threshold_m
        )

    return flags, has_worker


def wsvr(
    pred_xy: torch.Tensor,
    mode_prob: torch.Tensor,
    worker_xy: torch.Tensor,
    worker_mask: torch.Tensor,
    threshold_m: float = 2.0,
) -> torch.Tensor:
    """Return worker violation rate among samples containing valid workers."""
    flags, has_worker = per_sample_worker_violation(
        pred_xy,
        mode_prob,
        worker_xy,
        worker_mask,
        threshold_m,
    )

    if not bool(has_worker.any()):
        return torch.tensor(
            float("nan"),
            dtype=pred_xy.dtype,
            device=pred_xy.device,
        )

    return flags[
        has_worker
    ].float().mean()
