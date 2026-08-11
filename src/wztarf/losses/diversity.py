"""Discourage collapse of the six feasible route hypotheses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def diversity_loss(
    route_anchors: torch.Tensor,
    *,
    min_separation_m: float = 1.0,
    valid_mode_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize route hypotheses that are too similar.

    Diversity is measured using average distance across the predicted
    1 s, 3 s, and 5 s route anchors.

    Args:
        route_anchors:
            Route anchors `[B, K, H, 2]`.

        min_separation_m:
            Minimum desired average route-space separation.

        valid_mode_mask:
            Optional valid-mode mask `[B, K]`.

    Returns:
        Scalar diversity penalty.

    This loss should normally remain disabled until actual mode collapse
    is observed.
    """
    if route_anchors.ndim != 4 or route_anchors.shape[-1] != 2:
        raise ValueError(
            "route_anchors must have shape [B, K, H, 2]."
        )

    if min_separation_m <= 0:
        raise ValueError(
            "min_separation_m must be positive."
        )

    batch_size, num_modes, _, _ = route_anchors.shape

    # [B, K, K, H, 2]
    delta = (
        route_anchors[:, :, None]
        -
        route_anchors[:, None, :]
    )

    # [B, K, K, H]
    distance = torch.linalg.vector_norm(
        delta,
        dim=-1,
    )

    # Average geometric disagreement over the route anchors.
    pair_distance = distance.mean(
        dim=-1
    )

    pair_mask = torch.triu(
        torch.ones(
            num_modes,
            num_modes,
            dtype=torch.bool,
            device=route_anchors.device,
        ),
        diagonal=1,
    )

    pair_mask = pair_mask.unsqueeze(0).expand(
        batch_size,
        -1,
        -1,
    )

    if valid_mode_mask is not None:
        if valid_mode_mask.shape != (
            batch_size,
            num_modes,
        ):
            raise ValueError(
                "valid_mode_mask must have shape [B, K]."
            )

        valid_pair = (
            valid_mode_mask[:, :, None]
            &
            valid_mode_mask[:, None, :]
        )

        pair_mask &= valid_pair

    if not bool(pair_mask.any()):
        return route_anchors.sum() * 0.0

    penalty = F.relu(
        min_separation_m
        -
        pair_distance
    ).square()

    return penalty[pair_mask].mean()
