"""Top-level WZ-TARF model assembly."""

from torch import nn


class WZTARF(nn.Module):
    """Compose encoders, temporary topology, route queries, and decoders."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch):
        """Return `pred_xy [B,K,T,2]`, `mode_prob [B,K]`, and auxiliaries."""
        raise NotImplementedError
