"""Confidence-gated temporal gaze-intent encoder."""

from torch import nn


class GazeEncoder(nn.Module):
    """Encode gaze for route-intent reasoning rather than direct XY decoding."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out
