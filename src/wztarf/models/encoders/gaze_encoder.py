"""Encode gaze history as a confidence-gated route-intent signal."""

from __future__ import annotations

import torch
from torch import nn


class GazeEncoder(nn.Module):
    """Encode gaze without directly concatenating raw gaze into XY decoding."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        confidence_index: int = 2,
    ) -> None:
        super().__init__()

        self.output_dim = output_dim
        self.confidence_index = confidence_index

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            batch_first=True,
        )

        self.projection = nn.Linear(
            hidden_dim,
            output_dim,
        )

        self.reliability_head = nn.Sequential(
            nn.Linear(
                hidden_dim + 1,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return temporal states, pooled gaze context, and reliability."""
        if x.ndim != 3:
            raise ValueError(
                "Gaze input must have shape [B, T, D]."
            )

        if mask.shape != x.shape[:2]:
            raise ValueError(
                "Gaze mask must have shape [B, T]."
            )

        if not 0 <= self.confidence_index < x.shape[-1]:
            raise ValueError(
                "confidence_index is outside the gaze feature dimension."
            )

        raw_states, _ = self.gru(
            x
        )

        confidence = x[
            ...,
            self.confidence_index:
            self.confidence_index + 1,
        ]

        reliability = torch.sigmoid(
            self.reliability_head(
                torch.cat(
                    (
                        raw_states,
                        confidence,
                    ),
                    dim=-1,
                )
            )
        ).squeeze(-1)

        reliability = (
            reliability
            *
            mask.to(
                reliability.dtype
            )
        )

        states = self.projection(
            raw_states
        )

        denominator = (
            reliability.sum(
                dim=1,
                keepdim=True,
            )
            +
            1e-8
        )

        pooled = (
            states
            *
            reliability[..., None]
        ).sum(
            dim=1
        ) / denominator

        has_gaze = mask.bool().any(
            dim=1
        )

        pooled = (
            pooled
            *
            has_gaze[:, None].to(
                pooled.dtype
            )
        )

        return {
            "gaze_states": states,
            "gaze_context": pooled,
            "gaze_reliability": reliability,
            "gaze_valid": has_gaze,
        }
