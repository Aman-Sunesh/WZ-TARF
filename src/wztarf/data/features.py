"""Derived physical-time features used by temporal encoders."""

import torch


def relative_history_time(history_steps: int = 10, fps: int = 5) -> torch.Tensor:
    """Return `[-1.8, ..., -0.2, 0.0]` for the default 10-step history."""
    return torch.arange(-(history_steps - 1), 1, dtype=torch.float32) / fps


def future_time(future_steps: int = 25, fps: int = 5) -> torch.Tensor:
    """Return future times `[0.2, ..., 5.0]` for the default horizon."""
    return torch.arange(1, future_steps + 1, dtype=torch.float32) / fps
