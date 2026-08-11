"""Safety-conditioned K-mode scoring head."""

from torch import nn


class SafetyAwareScorer(nn.Module):
    """Rank candidate modes using route intent and differentiable safety risk."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
