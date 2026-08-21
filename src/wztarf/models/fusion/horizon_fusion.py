"""Fuse scene roles differently for 1 s, 3 s, and 5 s forecasting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn


class HorizonFusion(nn.Module):
    """Learn horizon-dependent importance for each scene modality."""

    def __init__(
        self,
        d_model: int = 128,
        roles: Sequence[str] = (
            "ego",
            "workzone",
            "lane",
            "gaze",
            "agents",
        ),
        num_horizons: int = 3,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.roles = tuple(
            roles
        )
        self.num_horizons = num_horizons

        self.role_embedding = nn.ParameterDict(
            {
                role: nn.Parameter(
                    torch.zeros(
                        d_model
                    )
                )
                for role in self.roles
            }
        )

        self.scorer = nn.ModuleDict(
            {
                role: nn.Sequential(
                    nn.Linear(
                        2 * d_model,
                        d_model,
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        d_model,
                        1,
                    ),
                )
                for role in self.roles
            }
        )

        self.horizon_embedding = nn.Parameter(
            torch.randn(
                num_horizons,
                d_model,
            )
            *
            0.02
        )

        self.output_norm = nn.LayerNorm(
            d_model
        )

    def forward(
        self,
        role_context: Mapping[str, torch.Tensor],
        role_valid: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `[B,H,D]` fused context and `[B,H,R]` role weights."""
        missing = [
            role
            for role in self.roles
            if role not in role_context
        ]

        if missing:
            raise KeyError(
                f"Missing role contexts: {missing}"
            )

        batch_size = role_context[
            self.roles[0]
        ].shape[0]

        fused_horizons = []
        all_weights = []

        for horizon_index in range(
            self.num_horizons
        ):
            horizon = self.horizon_embedding[
                horizon_index
            ][None].expand(
                batch_size,
                -1,
            )

            representations = []
            scores = []
            validities = []

            for role in self.roles:
                value = role_context[
                    role
                ]

                if value.shape != (
                    batch_size,
                    self.d_model,
                ):
                    raise ValueError(
                        f"Role '{role}' must have shape [B, D]."
                    )

                value = (
                    value
                    +
                    self.role_embedding[
                        role
                    ][None]
                )

                score = self.scorer[
                    role
                ](
                    torch.cat(
                        (
                            value,
                            horizon,
                        ),
                        dim=-1,
                    )
                ).squeeze(-1)

                if (
                    role_valid is not None
                    and
                    role in role_valid
                ):
                    valid = role_valid[
                        role
                    ].bool()

                    if valid.shape != (
                        batch_size,
                    ):
                        raise ValueError(
                            f"Role mask '{role}' must have shape [B]."
                        )
                else:
                    valid = torch.ones(
                        batch_size,
                        dtype=torch.bool,
                        device=value.device,
                    )

                representations.append(
                    value
                )

                scores.append(
                    score
                )

                validities.append(
                    valid
                )

            representations = torch.stack(
                representations,
                dim=1,
            )

            scores = torch.stack(
                scores,
                dim=1,
            )

            validity = torch.stack(
                validities,
                dim=1,
            )

            scores = scores.masked_fill(
                ~validity,
                torch.finfo(
                    scores.dtype
                ).min,
            )

            weights = torch.softmax(
                scores,
                dim=1,
            )

            weights = (
                weights
                *
                validity.to(
                    weights.dtype
                )
            )

            weights = weights / (
                weights.sum(
                    dim=1,
                    keepdim=True,
                )
                +
                1e-8
            )

            fused = (
                representations
                *
                weights[..., None]
            ).sum(
                dim=1
            )

            fused = self.output_norm(
                fused
                +
                horizon
            )

            fused_horizons.append(
                fused
            )

            all_weights.append(
                weights
            )

        return (
            torch.stack(
                fused_horizons,
                dim=1,
            ),
            torch.stack(
                all_weights,
                dim=1,
            ),
        )