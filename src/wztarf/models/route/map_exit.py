"""Beyond-map continuous continuation goal head."""

from torch import nn


class MapExitGoalHead(nn.Module):
    """Predict a continuous continuation goal for MAP_EXIT hypotheses."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, x):
        return self.head(x)
