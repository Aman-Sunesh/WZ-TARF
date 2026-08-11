"""Optional trajectory-local WZ/lane/worker refinement."""

from torch import nn


class LocalRefiner(nn.Module):
    """Refine each coarse mode using only nearby scene context."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, *args, **kwargs):
        raise NotImplementedError
