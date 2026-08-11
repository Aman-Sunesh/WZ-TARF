"""Polyline encoding, WZ-aware lane selection, and relation-aware graph propagation."""

from torch import nn


class LaneEncoder(nn.Module):
    """Own lane point encoding, pooling, pruning, and graph propagation."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, *args, **kwargs):
        raise NotImplementedError
