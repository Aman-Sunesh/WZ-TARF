"""Validity-aware temporal fusion of motion and controls."""

from torch import nn


class ControlFusion(nn.Module):
    """Gate projected control states into the ego-motion representation."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, *args, **kwargs):
        raise NotImplementedError
