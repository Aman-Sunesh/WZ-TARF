"""Route-conditioned residual trajectory decoder."""

from torch import nn


class TrajectoryDecoder(nn.Module):
    """Decode 25-step mode-specific residuals around the dynamics anchor."""

    def __init__(self, d_model: int = 128, future_steps: int = 25) -> None:
        super().__init__()
        self.d_model = d_model
        self.future_steps = future_steps

    def forward(self, *args, **kwargs):
        raise NotImplementedError
