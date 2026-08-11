"""Encode the training-only ground-truth future at 1 s, 3 s, and 5 s."""

from __future__ import annotations

import torch
from torch import nn


class FutureEncoder(nn.Module):
    """Encode future XY only during self-supervised pretraining."""

    def __init__(
        self,
        d_model: int = 128,
        fps: int = 5,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.fps = fps

        self.gru = nn.GRU(
            input_size=5,
            hidden_size=d_model,
            batch_first=True,
        )

        self.projections = nn.ModuleDict(
            {
                str(horizon): nn.Sequential(
                    nn.Linear(
                        d_model,
                        d_model,
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        d_model,
                        d_model,
                    ),
                )
                for horizon in (
                    1,
                    3,
                    5,
                )
            }
        )

    def forward(
        self,
        future_xy: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """Return future representations at 1 s, 3 s, and 5 s."""
        if future_xy.ndim != 3 or future_xy.shape[-1] != 2:
            raise ValueError(
                "future_xy must have shape [B, T, 2]."
            )

        delta = torch.zeros_like(
            future_xy
        )

        delta[
            :,
            0,
        ] = future_xy[
            :,
            0,
        ]

        delta[
            :,
            1:,
        ] = (
            future_xy[
                :,
                1:,
            ]
            -
            future_xy[
                :,
                :-1,
            ]
        )

        time = (
            torch.arange(
                1,
                future_xy.shape[1] + 1,
                dtype=future_xy.dtype,
                device=future_xy.device,
            )
            /
            float(
                self.fps
            )
        )

        time = time[
            None,
            :,
            None,
        ].expand(
            future_xy.shape[0],
            -1,
            -1,
        )

        feature = torch.cat(
            (
                future_xy,
                delta,
                time,
            ),
            dim=-1,
        )

        states, _ = self.gru(
            feature
        )

        result: dict[int, torch.Tensor] = {}

        for horizon in (
            1,
            3,
            5,
        ):
            index = (
                horizon
                *
                self.fps
                -
                1
            )

            if index >= states.shape[1]:
                raise ValueError(
                    f"Future horizon is shorter than {horizon} seconds."
                )

            result[
                horizon
            ] = self.projections[
                str(
                    horizon
                )
            ](
                states[
                    :,
                    index,
                ]
            )

        return result
