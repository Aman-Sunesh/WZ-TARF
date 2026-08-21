"""Route-aware local trajectory refiner."""

from __future__ import annotations

import torch
from torch import nn


class LocalRefiner(nn.Module):
    """Refine local geometry without changing route intent.

    The route module owns the route decision.

    The refiner may only make bounded:
        longitudinal correction ?s
        lateral correction ?d

    in the already-selected route frame.
    """

    def __init__(
        self,
        d_model: int = 128,
        local_radius_m: float = 8.0,
        max_longitudinal_correction_m: float = 0.75,
        max_lateral_correction_m: float = 0.45,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.local_radius_m = float(
            local_radius_m
        )

        self.max_longitudinal_correction_m = float(
            max_longitudinal_correction_m
        )

        self.max_lateral_correction_m = float(
            max_lateral_correction_m
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

        # mode context
        # + local scene context
        # + temporal encoding
        # + current local (s,d)
        self.net = nn.Sequential(
            nn.Linear(
                3 * d_model + 2,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                d_model // 2,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model // 2,
                2,
            ),
        )

    def forward(
        self,
        *,
        coarse_xy: torch.Tensor,
        mode_context: torch.Tensor,
        route_guide: torch.Tensor,
        route_tangent: torch.Tensor,
        route_normal: torch.Tensor,
        scene_tokens: torch.Tensor,
        scene_xy: torch.Tensor,
        scene_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return bounded route-relative local corrections."""

        (
            batch_size,
            num_modes,
            future_steps,
            _,
        ) = coarse_xy.shape

        if mode_context.shape != (
            batch_size,
            num_modes,
            self.d_model,
        ):
            raise ValueError(
                "mode_context must have shape [B,K,D]."
            )

        expected_route_shape = (
            batch_size,
            num_modes,
            future_steps,
            2,
        )

        for name, value in (
            (
                "route_guide",
                route_guide,
            ),
            (
                "route_tangent",
                route_tangent,
            ),
            (
                "route_normal",
                route_normal,
            ),
        ):
            if value.shape != expected_route_shape:
                raise ValueError(
                    f"{name} has wrong shape."
                )

        if (
            scene_tokens.ndim != 3
            or
            scene_tokens.shape[
                0
            ]
            !=
            batch_size
            or
            scene_tokens.shape[
                -1
            ]
            !=
            self.d_model
        ):
            raise ValueError(
                "scene_tokens must have shape [B,N,D]."
            )

        num_scene_tokens = (
            scene_tokens.shape[1]
        )

        if scene_xy.shape != (
            batch_size,
            num_scene_tokens,
            2,
        ):
            raise ValueError(
                "scene_xy must have shape [B,N,2]."
            )

        if scene_mask.shape != (
            batch_size,
            num_scene_tokens,
        ):
            raise ValueError(
                "scene_mask must have shape [B,N]."
            )

        # ----------------------------------------------------------
        # Local scene context around EACH coarse trajectory point.
        #
        # Includes:
        # lanes
        # WZ tokens
        # agents
        # and therefore workers where encoded by WZ tokens.
        # ----------------------------------------------------------

        distance = torch.linalg.vector_norm(
            coarse_xy[
                :,
                :,
                :,
                None,
                :
            ]
            -
            scene_xy[
                :,
                None,
                None,
                :,
                :
            ],
            dim=-1,
        )

        valid = (
            scene_mask[
                :,
                None,
                None,
                :
            ].bool()
            &
            (
                distance
                <=
                self.local_radius_m
            )
        )

        local_weight = torch.exp(
            -distance.float()
            /
            max(
                self.local_radius_m,
                1.0e-4,
            )
        )

        local_weight = (
            local_weight
            *
            valid.to(
                local_weight.dtype
            )
        )

        local_weight = (
            local_weight
            /
            local_weight.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(
                1.0e-8
            )
        )

        local_context = torch.einsum(
            "bktn,bnd->bktd",
            local_weight.to(
                scene_tokens.dtype
            ),
            scene_tokens,
        )

        # ----------------------------------------------------------
        # Current coarse route-relative coordinates.
        # ----------------------------------------------------------

        coarse_relative = (
            coarse_xy
            -
            route_guide
        )

        coarse_longitudinal = (
            coarse_relative
            *
            route_tangent
        ).sum(
            dim=-1
        )

        coarse_lateral = (
            coarse_relative
            *
            route_normal
        ).sum(
            dim=-1
        )

        coarse_sd = torch.stack(
            (
                coarse_longitudinal,
                coarse_lateral,
            ),
            dim=-1,
        )

        # ----------------------------------------------------------
        # Temporal encoding.
        # ----------------------------------------------------------

        normalized_time = torch.linspace(
            0.0,
            1.0,
            future_steps,
            dtype=coarse_xy.dtype,
            device=coarse_xy.device,
        )

        time_context = self.time_encoder(
            normalized_time[
                :,
                None,
            ]
        )

        time_context = time_context[
            None,
            None,
        ].expand(
            batch_size,
            num_modes,
            future_steps,
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
                future_steps,
                self.d_model,
            )
        )

        correction_raw = torch.tanh(
            self.net(
                torch.cat(
                    (
                        mode_sequence,
                        local_context,
                        time_context,
                        coarse_sd,
                    ),
                    dim=-1,
                )
            )
        )

        refinement_s = (
            correction_raw[
                ...,
                0
            ]
            *
            self.max_longitudinal_correction_m
        )

        refinement_d = (
            correction_raw[
                ...,
                1
            ]
            *
            self.max_lateral_correction_m
        )

        refinement_sd = torch.stack(
            (
                refinement_s,
                refinement_d,
            ),
            dim=-1,
        )

        refinement_delta = (
            refinement_s[
                ...,
                None
            ]
            *
            route_tangent
            +
            refinement_d[
                ...,
                None
            ]
            *
            route_normal
        )

        return {
            "refinement_delta": refinement_delta,
            "refinement_sd": refinement_sd,
            "local_scene_context": local_context,
            "local_scene_weight": local_weight,
        }
