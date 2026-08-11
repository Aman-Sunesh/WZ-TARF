"""Learned control-conditioned short-horizon dynamics anchor."""

from torch import nn


class DynamicsAnchor(nn.Module):
    """Predict the motion prior shared across route hypotheses."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, *args, **kwargs):
        raise NotImplementedError
