"""Encode the ego-motion history into temporal and pooled motion states."""

from __future__ import annotations

import torch
from torch import nn


class MotionEncoder(nn.Module):
    """Encode ego motion using a two-layer GRU."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return temporal states `[B,T,H]` and final state `[B,H]`."""
        if x.ndim != 3:
            raise ValueError(
                "Motion input must have shape [B, T, D]."
            )

        states, hidden = self.gru(
            x
        )

        return (
            states,
            hidden[-1],
        )
