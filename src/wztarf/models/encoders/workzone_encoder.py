"""Typed encoder for WZ boundary edges, polygon, warning sign, and workers."""

from torch import nn


class WorkZoneEncoder(nn.Module):
    """Encode structured WorkZone geometric tokens into 128-D representations."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, *args, **kwargs):
        raise NotImplementedError
