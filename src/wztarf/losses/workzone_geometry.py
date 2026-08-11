"""Penalize high-probability trajectories that enter restricted WZ geometry."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from wztarf.geometry.workzone import signed_distance_to_polygon


def workzone_geometry_loss(
    pred_xy: torch.Tensor,
    mode_prob: torch.Tensor,
    wz_polygon: torch.Tensor,
    *,
    wz_valid: torch.Tensor | None = None,
    temperature_m: float = 0.25,
) -> torch.Tensor:
    """Compute probability-weighted smooth WorkZone polygon risk.

    Signed-distance convention:

        positive = outside restricted polygon
        negative = inside restricted polygon

    Args:
        pred_xy:
            Candidate trajectories `[B, K, T, 2]`.

        mode_prob:
            Mode probabilities `[B, K]`.

        wz_polygon:
            WorkZone polygon vertices `[B, P, 2]`.

        wz_valid:
            Optional sample-validity mask `[B]`.

        temperature_m:
            Controls smoothness of the signed-distance penalty.

    Returns:
        Scalar WorkZone geometry loss.
    """
    if temperature_m <= 0:
        raise ValueError(
            "temperature_m must be positive."
        )

    batch_size, num_modes, future_steps, _ = pred_xy.shape

    if mode_prob.shape != (
        batch_size,
        num_modes,
    ):
        raise ValueError(
            "mode_prob must have shape [B, K]."
        )

    if wz_polygon.ndim != 3 or wz_polygon.shape[-1] != 2:
        raise ValueError(
            "wz_polygon must have shape [B, P, 2]."
        )

    if wz_polygon.shape[0] != batch_size:
        raise ValueError(
            "wz_polygon batch size does not match predictions."
        )

    if wz_valid is None:
        wz_valid = torch.ones(
            batch_size,
            dtype=torch.bool,
            device=pred_xy.device,
        )

    else:
        if wz_valid.shape != (batch_size,):
            raise ValueError(
                "wz_valid must have shape [B]."
            )

        wz_valid = wz_valid.bool()

    sample_losses: list[torch.Tensor] = []

    for batch_index in range(batch_size):
        if not bool(wz_valid[batch_index]):
            continue

        points = pred_xy[
            batch_index
        ].reshape(
            num_modes * future_steps,
            2,
        )

        sdf = signed_distance_to_polygon(
            points,
            wz_polygon[batch_index],
        ).reshape(
            num_modes,
            future_steps,
        )

        risk_per_mode = F.softplus(
            -sdf / temperature_m
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
