"""Fuse motion and controls through a validity-aware temporal gate."""

from __future__ import annotations

import torch
from torch import nn


class ControlFusion(nn.Module):
    """Inject observed controls into motion only when they are informative."""

    def __init__(
        self,
        motion_dim: int = 128,
        control_dim: int = 64,
        gate_feature_dim: int = 6,
        d_model: int = 128,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.gate_feature_dim = gate_feature_dim

        self.motion_projection = nn.Linear(
            motion_dim,
            d_model,
        )

        self.control_projection = nn.Linear(
            control_dim,
            d_model,
        )

        self.gate = nn.Sequential(
            nn.Linear(
                2 * d_model
                +
                gate_feature_dim,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                d_model,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        motion_states: torch.Tensor,
        control_states: torch.Tensor,
        control_mask: torch.Tensor,
        gate_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fused temporal states, pooled ego state, and gate values."""
        if motion_states.shape[:2] != control_states.shape[:2]:
            raise ValueError(
                "Motion and control temporal dimensions must match."
            )

        if control_mask.shape != motion_states.shape[:2]:
            raise ValueError(
                "control_mask must have shape [B, T]."
            )

        if gate_features.shape != (
            motion_states.shape[0],
            motion_states.shape[1],
            self.gate_feature_dim,
        ):
            raise ValueError(
                "gate_features has the wrong shape."
            )

        motion = self.motion_projection(
            motion_states
        )

        control = self.control_projection(
            control_states
        )

        gate = self.gate(
            torch.cat(
                (
                    motion,
                    control,
                    gate_features,
                ),
                dim=-1,
            )
        )

        gate = (
            gate
            *
            control_mask[..., None].to(
                gate.dtype
            )
        )

        fused = (
            motion
            +
            gate
            *
            control
        )

        pooled = fused[
            :,
            -1,
        ]

        return (
            fused,
            pooled,
            gate,
        )
