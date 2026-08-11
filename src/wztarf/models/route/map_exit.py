"""Predict a continuous terminal continuation goal for MAP_EXIT modes."""

from __future__ import annotations

import torch
from torch import nn


class MapExitGoalHead(nn.Module):
    """Predict beyond-map continuation from route, lane, and ego context."""

    def __init__(
        self,
        d_model: int = 128,
    ) -> None:
        super().__init__()

        self.goal_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.ego_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.head = nn.Sequential(
            nn.Linear(
                d_model,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                2,
            ),
        )

    def forward(
        self,
        mode_context: torch.Tensor,
        goal_context: torch.Tensor,
        ego_context: torch.Tensor,
    ) -> torch.Tensor:
        """Return continuous continuation goals `[B, K, 2]`."""
        if mode_context.shape != goal_context.shape:
            raise ValueError(
                "mode_context and goal_context must match."
            )

        ego = self.ego_projection(
            ego_context
        )[:, None]

        hidden = (
            mode_context
            +
            self.goal_projection(
                goal_context
            )
            +
            ego
        )

        return self.head(
            hidden
        )
