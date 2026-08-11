"""Shared temporal encoder for sparse surrounding-agent histories."""

from torch import nn


class AgentEncoder(nn.Module):
    """Shared GRU applied only to valid surrounding agents."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out
