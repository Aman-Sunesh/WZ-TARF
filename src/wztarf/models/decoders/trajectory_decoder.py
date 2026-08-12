"""Decode route-conditioned trajectories around the shared dynamics anchor."""

from __future__ import annotations

import torch
from torch import nn


def _interpolate_route_anchors(
    route_anchors: torch.Tensor,
    *,
    future_steps: int,
    fps: int,
) -> torch.Tensor:
    """Interpolate 1 s, 3 s, and 5 s anchors to all future timesteps."""
    if route_anchors.shape[-2:] != (3, 2):
        raise ValueError(
            "route_anchors must end with shape [3, 2]."
        )

    batch_size, num_modes = route_anchors.shape[:2]

    times = (
        torch.arange(
            1,
            future_steps + 1,
            dtype=route_anchors.dtype,
            device=route_anchors.device,
        )
        /
        float(fps)
    )

    origin = torch.zeros(
        batch_size,
        num_modes,
        1,
        2,
        dtype=route_anchors.dtype,
        device=route_anchors.device,
    )

    anchors = torch.cat(
        (
            origin,
            route_anchors,
        ),
        dim=2,
    )

    anchor_times = torch.tensor(
        [
            0.0,
            1.0,
            3.0,
            5.0,
        ],
        dtype=route_anchors.dtype,
        device=route_anchors.device,
    )

    # For t in (0, 1] -> anchor interval [0, 1]
    # For t in (1, 3] -> anchor interval [1, 3]
    # For t in (3, 5] -> anchor interval [3, 5]
    right = (
        torch.bucketize(
            times,
            anchor_times[1:],
            right=False,
        )
        +
        1
    )

    left = right - 1

    left_time = anchor_times[left]
    right_time = anchor_times[right]

    alpha = (
        (times - left_time)
        /
        (right_time - left_time)
    ).view(
        1,
        1,
        future_steps,
        1,
    )

    left_point = anchors.index_select(
        2,
        left,
    )
    right_point = anchors.index_select(
        2,
        right,
    )

    return (
        left_point
        +
        alpha
        *
        (
            right_point
            -
            left_point
        )
    )
    
class TrajectoryDecoder(nn.Module):
    """Decode 25-step route-conditioned residual trajectories."""

    def __init__(
        self,
        d_model: int = 128,
        future_steps: int = 25,
        fps: int = 5,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.future_steps = future_steps
        self.fps = fps

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

        self.decoder = nn.Sequential(
            nn.Linear(
                3 * d_model + 4,
                2 * d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                2 * d_model,
                d_model,
            ),
            nn.ReLU(),
        )

        self.residual_head = nn.Linear(
            d_model,
            2,
        )

        self.route_gate = nn.Linear(
            d_model,
            1,
        )

    def forward(
        self,
        mode_context: torch.Tensor,
        horizon_context: torch.Tensor,
        route_anchors: torch.Tensor,
        dynamics_xy: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decode coarse multimodal trajectories.

        Returns:
            `coarse_xy`
            `trajectory_residual`
            `route_guide`
        """
        batch_size, num_modes, d_model = mode_context.shape

        if d_model != self.d_model:
            raise ValueError(
                "mode_context feature dimension does not match d_model."
            )

        if horizon_context.shape != (
            batch_size,
            3,
            self.d_model,
        ):
            raise ValueError(
                "horizon_context must have shape [B, 3, D]."
            )

        if route_anchors.shape != (
            batch_size,
            num_modes,
            3,
            2,
        ):
            raise ValueError(
                "route_anchors must have shape [B, K, 3, 2]."
            )

        if dynamics_xy.shape != (
            batch_size,
            self.future_steps,
            2,
        ):
            raise ValueError(
                "dynamics_xy must have shape [B, T, 2]."
            )

        route_guide = _interpolate_route_anchors(
            route_anchors,
            future_steps=self.future_steps,
            fps=self.fps,
        )

        future_time = (
            torch.arange(
                1,
                self.future_steps + 1,
                dtype=mode_context.dtype,
                device=mode_context.device,
            )
            /
            float(self.fps)
        )

        time_embedding = self.time_encoder(
            future_time[:, None]
        )

        time_embedding = time_embedding[
            None,
            None,
        ].expand(
            batch_size,
            num_modes,
            -1,
            -1,
        )

        # Use the nearest semantic horizon context for each future interval.
        horizon_sequence = []

        for step in range(
            self.future_steps
        ):
            time_s = (
                step + 1
            ) / float(
                self.fps
            )

            if time_s <= 1.0:
                index = 0
            elif time_s <= 3.0:
                index = 1
            else:
                index = 2

            horizon_sequence.append(
                horizon_context[
                    :,
                    index,
                ]
            )

        horizon_sequence = torch.stack(
            horizon_sequence,
            dim=1,
        )

        horizon_sequence = horizon_sequence[
            :,
            None,
        ].expand(
            batch_size,
            num_modes,
            self.future_steps,
            self.d_model,
        )

        mode_sequence = mode_context[
            :,
            :,
            None,
            :,
        ].expand(
            batch_size,
            num_modes,
            self.future_steps,
            self.d_model,
        )

        dynamics = dynamics_xy[
            :,
            None,
        ].expand(
            batch_size,
            num_modes,
            self.future_steps,
            2,
        )

        hidden = self.decoder(
            torch.cat(
                (
                    mode_sequence,
                    horizon_sequence,
                    time_embedding,
                    route_guide,
                    dynamics,
                ),
                dim=-1,
            )
        )

        residual = self.residual_head(
            hidden
        )

        route_gate = torch.sigmoid(
            self.route_gate(
                hidden
            )
        )

        coarse_xy = (
            dynamics
            +
            route_gate
            *
            (
                route_guide
                -
                dynamics
            )
            +
            residual
        )

        return {
            "coarse_xy": coarse_xy,
            "trajectory_residual": residual,
            "route_guide": route_guide,
        }
