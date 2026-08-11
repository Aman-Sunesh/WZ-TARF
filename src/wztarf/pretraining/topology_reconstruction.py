"""Train lane representations to reconstruct WorkZone-conditioned topology."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return the mean over valid entries or differentiable zero if none exist."""
    mask = mask.bool()

    if values.shape != mask.shape:
        raise ValueError(
            "values and mask must have identical shapes."
        )

    if not bool(
        mask.any()
    ):
        return values.sum() * 0.0

    return values[
        mask
    ].mean()


def topology_reconstruction_loss(
    *,
    lane_overlap_pred: torch.Tensor,
    lane_overlap_target: torch.Tensor,
    lane_distance_pred: torch.Tensor,
    lane_distance_target: torch.Tensor,
    lane_mask: torch.Tensor,
    edge_compat_logits: torch.Tensor,
    edge_compat_target: torch.Tensor,
    edge_mask: torch.Tensor,
    overlap_weight: float = 1.0,
    distance_weight: float = 1.0,
    edge_weight: float = 1.0,
    huber_beta: float = 1.0,
) -> torch.Tensor:
    """Supervise WorkZone-aware lane and edge geometry reconstruction.

    Args:
        lane_overlap_pred:
            Predicted lane/WZ overlap values `[B, L]`.

        lane_overlap_target:
            Geometry-derived overlap pseudo-targets `[B, L]`.

        lane_distance_pred:
            Predicted lane-to-WZ distances `[B, L]`.

        lane_distance_target:
            Geometry-derived distance pseudo-targets `[B, L]`.

        lane_mask:
            Valid-lane mask `[B, L]`.

        edge_compat_logits:
            Predicted temporary edge compatibility logits `[B, E]`.

        edge_compat_target:
            Geometry-derived soft edge compatibility target `[B, E]`,
            expected in `[0, 1]`.

        edge_mask:
            Valid permanent-edge mask `[B, E]`.

        overlap_weight:
            Weight of lane/WZ overlap reconstruction.

        distance_weight:
            Weight of lane/WZ distance reconstruction.

        edge_weight:
            Weight of temporary edge compatibility reconstruction.

        huber_beta:
            Smooth-L1 transition point for continuous geometry targets.

    Returns:
        Scalar topology reconstruction loss.

    `edge_compat_target` is intentionally a geometry-derived compatibility
    pseudo-target. It must not be interpreted as ground-truth lane-closure
    annotation.
    """
    if lane_overlap_pred.shape != lane_overlap_target.shape:
        raise ValueError(
            "lane_overlap_pred and lane_overlap_target must match."
        )

    if lane_distance_pred.shape != lane_distance_target.shape:
        raise ValueError(
            "lane_distance_pred and lane_distance_target must match."
        )

    if lane_overlap_pred.shape != lane_distance_pred.shape:
        raise ValueError(
            "Lane overlap and distance tensors must use the same [B, L] shape."
        )

    if lane_mask.shape != lane_overlap_pred.shape:
        raise ValueError(
            "lane_mask must have shape [B, L]."
        )

    if edge_compat_logits.shape != edge_compat_target.shape:
        raise ValueError(
            "edge_compat_logits and edge_compat_target must match."
        )

    if edge_mask.shape != edge_compat_logits.shape:
        raise ValueError(
            "edge_mask must have shape [B, E]."
        )

    for name, value in {
        "overlap_weight": overlap_weight,
        "distance_weight": distance_weight,
        "edge_weight": edge_weight,
    }.items():
        if value < 0:
            raise ValueError(
                f"{name} cannot be negative."
            )

    if not torch.isfinite(
        lane_overlap_target
    ).all():
        raise ValueError(
            "lane_overlap_target contains non-finite values."
        )

    if not torch.isfinite(
        lane_distance_target
    ).all():
        raise ValueError(
            "lane_distance_target contains non-finite values."
        )

    valid_edge_targets = edge_compat_target[
        edge_mask.bool()
    ]

    if valid_edge_targets.numel() > 0:
        if (
            (valid_edge_targets < 0).any()
            or
            (valid_edge_targets > 1).any()
        ):
            raise ValueError(
                "Valid edge compatibility targets must lie in [0, 1]."
            )

    lane_mask = lane_mask.bool()
    edge_mask = edge_mask.bool()

    overlap_element = F.smooth_l1_loss(
        lane_overlap_pred,
        lane_overlap_target,
        beta=huber_beta,
        reduction="none",
    )

    overlap_loss = _masked_mean(
        overlap_element,
        lane_mask,
    )

    distance_element = F.smooth_l1_loss(
        lane_distance_pred,
        lane_distance_target,
        beta=huber_beta,
        reduction="none",
    )

    distance_loss = _masked_mean(
        distance_element,
        lane_mask,
    )

    edge_element = F.binary_cross_entropy_with_logits(
        edge_compat_logits,
        edge_compat_target.to(
            dtype=edge_compat_logits.dtype
        ),
        reduction="none",
    )

    edge_loss = _masked_mean(
        edge_element,
        edge_mask,
    )

    total_weight = (
        overlap_weight
        +
        distance_weight
        +
        edge_weight
    )

    if total_weight <= 0:
        return (
            lane_overlap_pred.sum()
            +
            lane_distance_pred.sum()
            +
            edge_compat_logits.sum()
        ) * 0.0

    return (
        overlap_weight
        *
        overlap_loss
        +
        distance_weight
        *
        distance_loss
        +
        edge_weight
        *
        edge_loss
    ) / total_weight
