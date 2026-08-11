"""Soft WorkZone-conditioned lane-node and lane-edge viability."""

from torch import nn


class TemporaryTopologyAdapter(nn.Module):
    """Transform the retained permanent graph into a soft temporary graph."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, *args, **kwargs):
        raise NotImplementedError
