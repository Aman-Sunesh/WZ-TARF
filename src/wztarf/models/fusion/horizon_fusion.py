"""Horizon-adaptive role-aware multimodal fusion."""

from torch import nn


class HorizonFusion(nn.Module):
    """Learn separate modality weights for 1 s, 3 s, and 5 s reasoning."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, *args, **kwargs):
        raise NotImplementedError
