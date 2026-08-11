"""Predict the short-horizon control-conditioned dynamics trajectory."""

from __future__ import annotations

import torch
from torch import nn


class DynamicsAnchor(nn.Module):
    """Decode a shared motion prior from ego and control context."""

    def __init__(
        self,
        d_model: int = 128,
        control_dim: int = 64,
        future_steps: int = 25,
        fps: int = 5,
    ) -> None:
        super().__init__()

        if future_steps <= 0:
            raise ValueError("future_steps must be positive.")

        if fps <= 0:
            raise ValueError("fps must be positive.")

        self.d_model = d_model
        self.future_steps = future_steps
        self.dt = 1.0 / float(fps)

        self.control_projection = nn.Linear(
            control_dim,
            d_model,
        )

        self.context_fusion = nn.Sequential(
            nn.Linear(
                2 * d_model,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                d_model,
            ),
        )

        self.time_encoder = nn.Sequential(
            nn.Linear(
                1,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                d_model,
            ),
        )

        self.velocity_head = nn.Sequential(
            nn.Linear(
                2 * d_model,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                2,
            ),
        )

        future_time = (
            torch.arange(
                1,
                future_steps + 1,
                dtype=torch.float32,
            )
            /
            float(fps)
        )

        self.register_buffer(
            "future_time_s",
            future_time,
            persistent=False,
        )

    def forward(
        self,
        ego_context: torch.Tensor,
        control_context: torch.Tensor,
    ) -> torch.Tensor:
        """Return integrated future positions `[B, T, 2]`."""
        if ego_context.ndim != 2:
            raise ValueError(
                "ego_context must have shape [B, D]."
            )

        if control_context.ndim != 2:
            raise ValueError(
                "control_context must have shape [B, Dc]."
            )

        if ego_context.shape[0] != control_context.shape[0]:
            raise ValueError(
                "Ego and control batch sizes must match."
            )

        control = self.control_projection(
            control_context
        )

        context = self.context_fusion(
            torch.cat(
                (
                    ego_context,
                    control,
                ),
                dim=-1,
            )
        )

        batch_size = context.shape[0]

        time = self.future_time_s.to(
            dtype=context.dtype,
            device=context.device,
        )

        time_embedding = self.time_encoder(
            time[:, None]
        )

        context = context[:, None, :].expand(
            batch_size,
            self.future_steps,
            self.d_model,
        )

        time_embedding = time_embedding[None].expand(
            batch_size,
            -1,
            -1,
        )

        velocity = self.velocity_head(
            torch.cat(
                (
                    context,
                    time_embedding,
                ),
                dim=-1,
            )
        )

        displacement = (
            velocity
            *
            self.dt
        )

        return torch.cumsum(
            displacement,
            dim=1,
        )
