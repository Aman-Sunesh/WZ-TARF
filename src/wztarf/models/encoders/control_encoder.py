"""Temporal encoder for observed driver controls."""

import torch
from torch import nn


class ControlEncoder(nn.Module):
    """Compact GRU for steering, throttle, brake, derivatives, validity, and time."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode `[B,T,D]` control features into temporal states."""
        out, _ = self.gru(x)
        return out
