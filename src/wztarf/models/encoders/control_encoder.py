"""Encode steering, throttle, brake, derivatives, validity, and time."""

from __future__ import annotations

import torch
from torch import nn


class ControlEncoder(nn.Module):
    """Encode the observed control stream with a compact GRU."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return temporal states and the last valid control state."""
        if x.ndim != 3:
            raise ValueError(
                "Control input must have shape [B, T, D]."
            )

        states, _ = self.gru(
            x
        )

        if mask is None:
            return (
                states,
                states[:, -1],
            )

        if mask.shape != x.shape[:2]:
            raise ValueError(
                "Control mask must have shape [B, T]."
            )

        mask = mask.bool()

        indices = torch.arange(
            x.shape[1],
            device=x.device,
        )[None]

        last_index = (
            mask.long()
            *
            indices
        ).max(dim=1).values

        batch_index = torch.arange(
            x.shape[0],
            device=x.device,
        )

        pooled = states[
            batch_index,
            last_index,
        ]

        has_control = mask.any(
            dim=1
        )

        pooled = (
            pooled
            *
            has_control[:, None].to(
                pooled.dtype
            )
        )

        return (
            states,
            pooled,
        )
