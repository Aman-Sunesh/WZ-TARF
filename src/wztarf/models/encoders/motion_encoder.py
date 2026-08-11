"""Temporal encoder for ego motion history."""

import torch
from torch import nn


class MotionEncoder(nn.Module):
    """Two-layer GRU motion encoder with explicit physical-time input."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode `[B,T,D]` motion features into `[B,T,H]` states."""
        out, _ = self.gru(x)
        return out
