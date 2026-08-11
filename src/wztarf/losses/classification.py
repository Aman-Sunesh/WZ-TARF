"""Train mode probabilities using soft trajectory-quality targets."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_quality_targets(
    assignment_cost: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Convert per-mode trajectory costs into detached soft targets.

    Args:
        assignment_cost:
            Per-mode quality cost `[B, K]`. Lower is better.

        temperature:
            Softmax temperature. Smaller values produce a sharper target
            distribution.

    Returns:
        Detached probability targets `[B, K]`.
    """
    if assignment_cost.ndim != 2:
        raise ValueError(
            "assignment_cost must have shape [B, K]."
        )

    if temperature <= 0:
        raise ValueError(
            "temperature must be positive."
        )

    targets = torch.softmax(
        -assignment_cost / temperature,
        dim=-1,
    )

    return targets.detach()


def classification_loss(
    mode_logits: torch.Tensor,
    assignment_cost: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute soft quality-aware trajectory mode classification loss.

    Args:
        mode_logits:
            Raw mode scores `[B, K]`.

        assignment_cost:
            ADE/FDE-based mode costs `[B, K]`.

        temperature:
            Temperature used to construct soft target probabilities.

    Returns:
        Scalar cross-entropy loss.
    """
    if mode_logits.ndim != 2:
        raise ValueError(
            "mode_logits must have shape [B, K]."
        )

    if assignment_cost.shape != mode_logits.shape:
        raise ValueError(
            "assignment_cost and mode_logits must have identical shapes."
        )

    targets = soft_quality_targets(
        assignment_cost,
        temperature=temperature,
    )

    log_prob = F.log_softmax(
        mode_logits,
        dim=-1,
    )

    per_sample = -(
        targets
        *
        log_prob
    ).sum(dim=-1)

    return per_sample.mean()
