"""K=6 route queries and terminal-goal prediction."""

from torch import nn


class RouteGoalQueries(nn.Module):
    """Predict retained-lane goals or the explicit MAP_EXIT class."""

    def __init__(self, d_model: int = 128, num_modes: int = 6) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_modes = num_modes

    def forward(self, *args, **kwargs):
        raise NotImplementedError
