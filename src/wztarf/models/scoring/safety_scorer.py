"""Rank candidate trajectories using route context and explicit safety features."""

from __future__ import annotations

import torch
from torch import nn


class SafetyAwareScorer(nn.Module):
    """Produce normalized K-mode probabilities after trajectory generation."""

    def __init__(
        self,
        d_model: int = 128,
        safety_feature_dim: int = 3,
    ) -> None:
        super().__init__()

        self.safety_feature_dim = safety_feature_dim

        self.net = nn.Sequential(
            nn.Linear(
                d_model
                +
                safety_feature_dim,
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

    def forward(
        self,
        mode_context: torch.Tensor,
        safety_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mode logits and normalized probabilities."""
        if safety_features.shape != (
            mode_context.shape[0],
            mode_context.shape[1],
            self.safety_feature_dim,
        ):
            raise ValueError(
                "safety_features has the wrong shape."
            )

        logits = self.net(
            torch.cat(
                (
                    mode_context,
                    safety_features,
                ),
                dim=-1,
            )
        ).squeeze(-1)

        probability = torch.softmax(
            logits,
            dim=-1,
        )

        return (
            logits,
            probability,
        )
