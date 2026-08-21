"""Per-scene normalized quality and pairwise mode-ranking objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ModeRankingLoss:
    behavior: torch.Tensor
    quality: torch.Tensor
    pairwise: torch.Tensor

    oracle_cost: torch.Tensor
    normalized_oracle_cost: torch.Tensor
    normalized_behavior_cost: torch.Tensor


def robust_normalize_mode_cost(
    cost: torch.Tensor,
    *,
    eps: float = 1.0e-4,
) -> torch.Tensor:
    """Normalize mode cost within each scene using median / IQR.

    If IQR collapses, standard deviation is used as a fallback.
    """

    if cost.ndim != 2:
        raise ValueError(
            "cost must have shape [B,K]."
        )

    value = cost.detach().float()

    median = value.median(
        dim=1,
        keepdim=True,
    ).values

    q25 = torch.quantile(
        value,
        0.25,
        dim=1,
        keepdim=True,
    )

    q75 = torch.quantile(
        value,
        0.75,
        dim=1,
        keepdim=True,
    )

    iqr = (
        q75 - q25
    )

    std = value.std(
        dim=1,
        keepdim=True,
        unbiased=False,
    )

    scale = torch.where(
        iqr > eps,
        iqr,
        torch.where(
            std > eps,
            std,
            torch.ones_like(std),
        ),
    )

    normalized = (
        value
        -
        median
    ) / scale.clamp_min(
        eps
    )

    return normalized.clamp(
        -6.0,
        6.0,
    )


def detached_oracle_mode_cost(
    *,
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    route_cost: torch.Tensor | None = None,
    fps: int = 5,
    ade1_weight: float = 0.10,
    ade3_weight: float = 0.20,
    ade5_weight: float = 0.35,
    fde5_weight: float = 0.35,
    route_weight: float = 0.25,
) -> torch.Tensor:
    """Construct the actual per-mode training oracle.

    Emphasizes late-horizon accuracy without discarding short-horizon
    dynamics.
    """

    if pred_xy.ndim != 4 or pred_xy.shape[-1] != 2:
        raise ValueError(
            "pred_xy must have shape [B,K,T,2]."
        )

    if gt_xy.shape != (
        pred_xy.shape[0],
        pred_xy.shape[2],
        2,
    ):
        raise ValueError(
            "gt_xy has incompatible shape."
        )

    with torch.no_grad():
        error = torch.linalg.vector_norm(
            pred_xy.detach().float()
            -
            gt_xy[
                :,
                None,
            ].detach().float(),
            dim=-1,
        )

        future_steps = error.shape[-1]

        t1 = min(
            fps,
            future_steps,
        )

        t3 = min(
            3 * fps,
            future_steps,
        )

        ade1 = error[
            :,
            :,
            :t1,
        ].mean(
            dim=-1
        )

        ade3 = error[
            :,
            :,
            :t3,
        ].mean(
            dim=-1
        )

        ade5 = error.mean(
            dim=-1
        )

        fde5 = error[
            :,
            :,
            -1,
        ]

        cost = (
            float(ade1_weight)
            *
            ade1
            +
            float(ade3_weight)
            *
            ade3
            +
            float(ade5_weight)
            *
            ade5
            +
            float(fde5_weight)
            *
            fde5
        )

        if route_cost is not None:
            if route_cost.shape != cost.shape:
                raise ValueError(
                    "route_cost must have shape [B,K]."
                )

            cost = (
                cost
                +
                float(route_weight)
                *
                route_cost.detach().float()
            )

    return cost


def mode_ranking_loss(
    *,
    behavior_logits: torch.Tensor,
    quality_score: torch.Tensor,
    ranking_logits: torch.Tensor,
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    route_cost: torch.Tensor | None = None,
    fps: int = 5,
) -> ModeRankingLoss:
    """Train probability and quality as related but separate concepts."""

    expected = pred_xy.shape[:2]

    for name, tensor in (
        (
            "behavior_logits",
            behavior_logits,
        ),
        (
            "quality_score",
            quality_score,
        ),
        (
            "ranking_logits",
            ranking_logits,
        ),
    ):
        if tensor.shape != expected:
            raise ValueError(
                f"{name} must have shape [B,K]."
            )

    oracle_cost = detached_oracle_mode_cost(
        pred_xy=pred_xy,
        gt_xy=gt_xy,
        route_cost=route_cost,
        fps=fps,
    )

    normalized_oracle = (
        robust_normalize_mode_cost(
            oracle_cost
        )
    )

    # ----------------------------------------------------------
    # BEHAVIORAL ROUTE PROBABILITY
    #
    # Prefer topology/route coverage cost when available.
    # This keeps p_k about which route matches observed behavior,
    # rather than only about final decoder geometry.
    # ----------------------------------------------------------

    behavior_cost = (
        route_cost.detach()
        if route_cost is not None
        else oracle_cost
    )

    normalized_behavior = (
        robust_normalize_mode_cost(
            behavior_cost
        )
    )

    behavior_target = torch.softmax(
        -normalized_behavior,
        dim=-1,
    ).detach()

    behavior_loss = -(
        behavior_target
        *
        F.log_softmax(
            behavior_logits.float(),
            dim=-1,
        )
    ).sum(
        dim=-1
    ).mean()

    # ----------------------------------------------------------
    # QUALITY REGRESSION
    #
    # Higher q should mean better forecast.
    # ----------------------------------------------------------

    quality_target = (
        -normalized_oracle
    ).detach()

    quality_loss = F.smooth_l1_loss(
        quality_score.float(),
        quality_target,
        beta=1.0,
        reduction="mean",
    )

    # ----------------------------------------------------------
    # PAIRWISE RANKING
    #
    # If oracle says i is better than j:
    #     S_i > S_j
    # ----------------------------------------------------------

    num_modes = expected[1]

    i, j = torch.triu_indices(
        num_modes,
        num_modes,
        offset=1,
        device=ranking_logits.device,
    )

    cost_i = normalized_oracle[
        :,
        i,
    ]

    cost_j = normalized_oracle[
        :,
        j,
    ]

    score_i = ranking_logits.float()[
        :,
        i,
    ]

    score_j = ranking_logits.float()[
        :,
        j,
    ]

    cost_delta = (
        cost_j
        -
        cost_i
    )

    valid_pair = (
        cost_delta.abs()
        >
        1.0e-3
    )

    direction = torch.sign(
        cost_delta
    )

    score_delta = (
        score_i
        -
        score_j
    )

    pair_loss = F.softplus(
        -direction
        *
        score_delta
    )

    # Give more weight to pairs whose oracle quality differs clearly.
    pair_weight = cost_delta.abs().clamp(
        0.25,
        4.0,
    )

    pair_weight = (
        pair_weight
        *
        valid_pair.to(
            pair_weight.dtype
        )
    )

    pairwise_loss = (
        pair_loss
        *
        pair_weight
    ).sum() / pair_weight.sum().clamp_min(
        1.0
    )

    return ModeRankingLoss(
        behavior=behavior_loss,
        quality=quality_loss,
        pairwise=pairwise_loss,
        oracle_cost=oracle_cost,
        normalized_oracle_cost=normalized_oracle,
        normalized_behavior_cost=normalized_behavior,
    )
