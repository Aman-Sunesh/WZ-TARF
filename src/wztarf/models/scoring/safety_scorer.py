"""Behavior/quality-separated route-mode ranking for WZ-TARF V3."""

from __future__ import annotations

import torch
from torch import nn


class RouteQualityScorer(nn.Module):
    """Rank K route-conditioned forecasts with separate semantics.

    behavior head:
        Estimates relative behavioral plausibility of the route/mode.

    quality head:
        Estimates how well the decoded forecast is expected to perform.

    Final inference score:

        S_k = behavior_logit_k + alpha * quality_score_k

    where alpha is a bounded learned scalar.
    """

    def __init__(
        self,
        d_model: int = 128,
        behavior_feature_dim: int = 8,
        quality_feature_dim: int = 11,
    ) -> None:
        super().__init__()

        self.behavior_feature_dim = behavior_feature_dim
        self.quality_feature_dim = quality_feature_dim

        self.behavior_net = nn.Sequential(
            nn.Linear(
                d_model + behavior_feature_dim,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                d_model // 2,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model // 2,
                1,
            ),
        )

        self.quality_net = nn.Sequential(
            nn.Linear(
                d_model + quality_feature_dim,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                d_model // 2,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model // 2,
                1,
            ),
        )

        # alpha = 0.5 + sigmoid(raw) -> [0.5, 1.5]
        # Starts exactly at 1.0.
        self.quality_alpha_raw = nn.Parameter(
            torch.tensor(0.0)
        )

    def forward(
        self,
        *,
        mode_context: torch.Tensor,
        behavior_features: torch.Tensor,
        quality_features: torch.Tensor,
        route_prior: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return behavioral, quality, and combined ranking outputs."""

        batch_size, num_modes, _ = mode_context.shape

        if behavior_features.shape != (
            batch_size,
            num_modes,
            self.behavior_feature_dim,
        ):
            raise ValueError(
                "behavior_features has wrong shape."
            )

        if quality_features.shape != (
            batch_size,
            num_modes,
            self.quality_feature_dim,
        ):
            raise ValueError(
                "quality_features has wrong shape."
            )

        if route_prior.shape != (
            batch_size,
            num_modes,
        ):
            raise ValueError(
                "route_prior must have shape [B,K]."
            )

        behavior_learned = self.behavior_net(
            torch.cat(
                (
                    mode_context,
                    behavior_features,
                ),
                dim=-1,
            )
        ).squeeze(-1)

        # Explicit route prior acts as an interpretable baseline.
        behavior_logits = (
            behavior_learned
            +
            torch.log(
                route_prior.float().clamp_min(
                    1.0e-6
                )
            ).to(
                behavior_learned.dtype
            )
        )

        behavior_prob = torch.softmax(
            behavior_logits.float(),
            dim=-1,
        ).to(
            behavior_logits.dtype
        )

        quality_score = self.quality_net(
            torch.cat(
                (
                    mode_context,
                    quality_features,
                ),
                dim=-1,
            )
        ).squeeze(-1)

        quality_alpha = (
            0.5
            +
            torch.sigmoid(
                self.quality_alpha_raw
            )
        )

        ranking_logits = (
            behavior_logits
            +
            quality_alpha
            *
            quality_score
        )

        mode_prob = torch.softmax(
            ranking_logits.float(),
            dim=-1,
        ).to(
            ranking_logits.dtype
        )

        return {
            "behavior_logits": behavior_logits,
            "behavior_prob": behavior_prob,
            "quality_score": quality_score,
            "ranking_logits": ranking_logits,
            "mode_prob": mode_prob,
            "quality_alpha": quality_alpha,
        }


# Backward-compatible import name for any external code that still imports it.
SafetyAwareScorer = RouteQualityScorer
