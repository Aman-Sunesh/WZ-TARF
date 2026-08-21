"""Route-relative trajectory decoder for WZ-TARF V3."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _interpolate_route_anchors(
    route_anchors: torch.Tensor,
    *,
    future_steps: int,
    fps: int,
) -> torch.Tensor:
    """Backward-compatible fallback from 1/3/5 s anchors."""

    if route_anchors.shape[-2:] != (
        3,
        2,
    ):
        raise ValueError(
            "route_anchors must end with [3,2]."
        )

    batch_size, num_modes = (
        route_anchors.shape[:2]
    )

    times = (
        torch.arange(
            1,
            future_steps + 1,
            dtype=route_anchors.dtype,
            device=route_anchors.device,
        )
        /
        float(
            fps
        )
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

    origin = route_anchors.new_zeros(
        batch_size,
        num_modes,
        1,
        2,
    )

    control = torch.cat(
        (
            origin,
            route_anchors,
        ),
        dim=2,
    )

    right = (
        torch.bucketize(
            times,
            anchor_times[1:],
            right=False,
        )
        +
        1
    ).clamp(
        max=3
    )

    left = right - 1

    alpha = (
        (
            times
            -
            anchor_times[left]
        )
        /
        (
            anchor_times[right]
            -
            anchor_times[left]
        ).clamp_min(
            1.0e-6
        )
    ).view(
        1,
        1,
        future_steps,
        1,
    )

    left_xy = control.index_select(
        2,
        left,
    )

    right_xy = control.index_select(
        2,
        right,
    )

    return (
        left_xy
        +
        alpha
        *
        (
            right_xy
            -
            left_xy
        )
    )


def _route_frame(
    route_guide: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """Construct unit tangent/normal vectors along a dense route."""

    if (
        route_guide.ndim != 4
        or
        route_guide.shape[-1] != 2
    ):
        raise ValueError(
            "route_guide must have shape [B,K,T,2]."
        )

    future_steps = (
        route_guide.shape[2]
    )

    if future_steps < 2:
        raise ValueError(
            "route guide must contain >= 2 timesteps."
        )

    origin = route_guide.new_zeros(
        route_guide.shape[0],
        route_guide.shape[1],
        1,
        2,
    )

    previous = torch.cat(
        (
            origin,
            route_guide[
                :,
                :,
                :-1,
            ],
        ),
        dim=2,
    )

    final_delta = (
        route_guide[
            :,
            :,
            -1:,
        ]
        -
        route_guide[
            :,
            :,
            -2:-1,
        ]
    )

    following = torch.cat(
        (
            route_guide[
                :,
                :,
                1:,
            ],
            route_guide[
                :,
                :,
                -1:,
            ]
            +
            final_delta,
        ),
        dim=2,
    )

    tangent_raw = (
        following
        -
        previous
    )

    tangent_norm = (
        torch.linalg.vector_norm(
            tangent_raw.float(),
            dim=-1,
            keepdim=True,
        ).to(
            tangent_raw.dtype
        )
    )

    fallback = torch.zeros_like(
        tangent_raw
    )

    fallback[
        ...,
        0
    ] = 1.0

    tangent = torch.where(
        tangent_norm > 1.0e-5,
        tangent_raw
        /
        tangent_norm.clamp_min(
            1.0e-6
        ),
        fallback,
    )

    normal = torch.stack(
        (
            -tangent[
                ...,
                1
            ],
            tangent[
                ...,
                0
            ],
        ),
        dim=-1,
    )

    return (
        tangent,
        normal,
    )


class TrajectoryDecoder(nn.Module):
    """Decode trajectories in candidate-route coordinates.

    The model no longer predicts arbitrary XY residuals.

    Instead it predicts local route-relative corrections:

        longitudinal correction: ?s_t
        lateral offset:          d_t

    which are mapped back to XY using the candidate route frame.
    """

    def __init__(
        self,
        d_model: int = 128,
        future_steps: int = 25,
        fps: int = 5,
        longitudinal_correction_m: float = 2.0,
        lateral_correction_m: float = 1.25,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.future_steps = future_steps
        self.fps = fps

        self.longitudinal_correction_m = float(
            longitudinal_correction_m
        )

        self.lateral_correction_m = float(
            lateral_correction_m
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

        # mode + horizon + time + route_xy + dynamics_xy
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

        # ==========================================================
        # V3 ROUTE-RELATIVE DECODER
        #
        # output[0] -> longitudinal correction
        # output[1] -> lateral correction
        # ==========================================================

        self.local_residual_head = nn.Linear(
            d_model,
            2,
        )

        # Keep original learned route preference, but structural route
        # dominance remains a floor.
        self.route_gate = nn.Linear(
            d_model,
            1,
        )

        self.route_gate_slope_raw = nn.Parameter(
            torch.tensor(
                -0.43275213
            )
        )

        self.route_gate_midpoint_raw = nn.Parameter(
            torch.tensor(
                0.0
            )
        )

        self.route_gate_residual_fraction = 0.25

        future_time = (
            torch.arange(
                1,
                future_steps + 1,
                dtype=torch.float32,
            )
            /
            float(
                fps
            )
        )

        horizon_index = torch.where(
            future_time <= 1.0,
            torch.zeros_like(
                future_time,
                dtype=torch.long,
            ),
            torch.where(
                future_time <= 3.0,
                torch.ones_like(
                    future_time,
                    dtype=torch.long,
                ),
                torch.full_like(
                    future_time,
                    2,
                    dtype=torch.long,
                ),
            ),
        )

        self.register_buffer(
            "future_time",
            future_time,
            persistent=False,
        )

        self.register_buffer(
            "horizon_index",
            horizon_index,
            persistent=False,
        )

    def forward(
        self,
        mode_context: torch.Tensor,
        horizon_context: torch.Tensor,
        route_anchors: torch.Tensor,
        dynamics_xy: torch.Tensor,
        *,
        route_guide: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode route-relative coarse trajectories."""

        (
            batch_size,
            num_modes,
            d_model,
        ) = mode_context.shape

        if d_model != self.d_model:
            raise ValueError(
                "mode_context feature dimension mismatch."
            )

        if horizon_context.shape != (
            batch_size,
            3,
            self.d_model,
        ):
            raise ValueError(
                "horizon_context must have shape [B,3,D]."
            )

        if dynamics_xy.shape != (
            batch_size,
            self.future_steps,
            2,
        ):
            raise ValueError(
                "dynamics_xy must have shape [B,T,2]."
            )

        if route_guide is None:
            route_guide = _interpolate_route_anchors(
                route_anchors,
                future_steps=self.future_steps,
                fps=self.fps,
            )

        if route_guide.shape != (
            batch_size,
            num_modes,
            self.future_steps,
            2,
        ):
            raise ValueError(
                "route_guide must have shape [B,K,T,2]."
            )

        (
            route_tangent,
            route_normal,
        ) = _route_frame(
            route_guide
        )

        future_time = self.future_time.to(
            dtype=mode_context.dtype
        )

        time_embedding = self.time_encoder(
            future_time[
                :,
                None,
            ]
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

        horizon_sequence = (
            horizon_context.index_select(
                1,
                self.horizon_index,
            )
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

        mode_sequence = (
            mode_context[
                :,
                :,
                None,
                :
            ].expand(
                batch_size,
                num_modes,
                self.future_steps,
                self.d_model,
            )
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

        learned_route_gate = torch.sigmoid(
            self.route_gate(
                hidden
            )
        )

        route_gate_slope = (
            1.0
            +
            F.softplus(
                self.route_gate_slope_raw
            )
        )

        route_gate_midpoint = (
            2.0
            +
            torch.sigmoid(
                self.route_gate_midpoint_raw
            )
        )

        structural_route_gate = torch.sigmoid(
            route_gate_slope
            *
            (
                future_time.view(
                    1,
                    1,
                    self.future_steps,
                    1,
                )
                -
                route_gate_midpoint
            )
        )

        route_gate = (
            structural_route_gate
            +
            (
                1.0
                -
                structural_route_gate
            )
            *
            self.route_gate_residual_fraction
            *
            learned_route_gate
        )

        # ----------------------------------------------------------
        # Express dynamics prediction in candidate-route coordinates.
        # ----------------------------------------------------------

        dynamics_relative = (
            dynamics
            -
            route_guide
        )

        dynamics_longitudinal = (
            dynamics_relative
            *
            route_tangent
        ).sum(
            dim=-1
        )

        dynamics_lateral = (
            dynamics_relative
            *
            route_normal
        ).sum(
            dim=-1
        )

        # ----------------------------------------------------------
        # Learned residual is ALSO route-relative and bounded.
        # ----------------------------------------------------------

        local_raw = torch.tanh(
            self.local_residual_head(
                hidden
            )
        )

        # Late route dominance also reduces how much a learned residual
        # may distort the route.
        late_residual_scale = (
            1.0
            -
            0.50
            *
            structural_route_gate.squeeze(
                -1
            )
        )

        learned_longitudinal = (
            local_raw[
                ...,
                0
            ]
            *
            self.longitudinal_correction_m
            *
            late_residual_scale
        )

        learned_lateral = (
            local_raw[
                ...,
                1
            ]
            *
            self.lateral_correction_m
            *
            late_residual_scale
        )

        route_weight = route_gate.squeeze(
            -1
        )

        # At early horizons preserve physics/dynamics.
        # At late horizons collapse toward the selected route.
        route_longitudinal_offset = (
            (
                1.0
                -
                route_weight
            )
            *
            dynamics_longitudinal
            +
            learned_longitudinal
        )

        route_lateral_offset = (
            (
                1.0
                -
                route_weight
            )
            *
            dynamics_lateral
            +
            learned_lateral
        )

        coarse_xy = (
            route_guide
            +
            route_longitudinal_offset[
                ...,
                None
            ]
            *
            route_tangent
            +
            route_lateral_offset[
                ...,
                None
            ]
            *
            route_normal
        )

        learned_residual_xy = (
            learned_longitudinal[
                ...,
                None
            ]
            *
            route_tangent
            +
            learned_lateral[
                ...,
                None
            ]
            *
            route_normal
        )

        return {
            "coarse_xy": coarse_xy,

            # Existing compatibility key.
            "trajectory_residual": learned_residual_xy,

            # Explicit route-relative quantities.
            "trajectory_residual_sd": torch.stack(
                (
                    learned_longitudinal,
                    learned_lateral,
                ),
                dim=-1,
            ),
            "route_longitudinal_offset": route_longitudinal_offset,
            "route_lateral_offset": route_lateral_offset,

            "route_guide": route_guide,
            "route_tangent": route_tangent,
            "route_normal": route_normal,

            "route_gate": route_gate,
            "structural_route_gate": structural_route_gate,
            "learned_route_gate": learned_route_gate,
        }
