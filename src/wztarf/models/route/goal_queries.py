"""Generate six route hypotheses over retained lanes plus MAP_EXIT."""

from __future__ import annotations

import math

import torch
from torch import nn


class RouteGoalQueries(nn.Module):
    """Predict lane/MAP_EXIT goals and 1 s, 3 s, 5 s route anchors."""

    def __init__(
        self,
        d_model: int = 128,
        num_modes: int = 6,
        num_horizons: int = 3,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_modes = num_modes
        self.num_horizons = num_horizons

        self.mode_queries = nn.Parameter(
            torch.randn(
                num_modes,
                d_model,
            )
            *
            0.02
        )

        self.global_fusion = nn.Sequential(
            nn.Linear(
                3 * d_model,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                d_model,
            ),
        )

        self.query_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.lane_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.map_exit_logit = nn.Linear(
            d_model,
            1,
        )

        self.map_exit_embedding = nn.Parameter(
            torch.randn(
                d_model
            )
            *
            0.02
        )

        self.mode_norm = nn.LayerNorm(
            d_model
        )

        self.anchor_heads = nn.ModuleList(
            [
                nn.Sequential(
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
                for _ in range(
                    num_horizons
                )
            ]
        )

        self.goal_offset_head = nn.Sequential(
            nn.Linear(
                d_model,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                1,
            ),
        )

    def forward(
        self,
        ego_context: torch.Tensor,
        lane_context: torch.Tensor,
        horizon_context: torch.Tensor,
        lane_states: torch.Tensor,
        lane_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return K route modes and goal distributions."""
        batch_size, num_lanes, _ = lane_states.shape

        global_context = self.global_fusion(
            torch.cat(
                (
                    ego_context,
                    lane_context,
                    horizon_context.mean(
                        dim=1
                    ),
                ),
                dim=-1,
            )
        )

        mode_context = (
            global_context[:, None]
            +
            self.mode_queries[None]
        )

        query = self.query_projection(
            mode_context
        )

        lane_key = self.lane_projection(
            lane_states
        )

        lane_logits = torch.einsum(
            "bkd,bld->bkl",
            query,
            lane_key,
        ) / math.sqrt(
            self.d_model
        )

        lane_logits = lane_logits.masked_fill(
            ~lane_mask[:, None, :].bool(),
            -1e9,
        )

        exit_logit = self.map_exit_logit(
            mode_context
        )

        goal_logits = torch.cat(
            (
                lane_logits,
                exit_logit,
            ),
            dim=-1,
        )

        goal_prob = torch.softmax(
            goal_logits,
            dim=-1,
        )

        lane_prob = goal_prob[
            ...,
            :num_lanes,
        ]

        goal_context = torch.einsum(
            "bkl,bld->bkd",
            lane_prob,
            lane_states,
        )

        exit_prob = goal_prob[
            ...,
            -1:
        ]

        goal_context = (
            goal_context
            +
            exit_prob
            *
            self.map_exit_embedding[
                None,
                None,
                :,
            ]
        )

        mode_context = self.mode_norm(
            mode_context
            +
            goal_context
        )

        anchors = []

        for horizon_index in range(
            self.num_horizons
        ):
            horizon = horizon_context[
                :,
                horizon_index,
            ][:, None].expand(
                -1,
                self.num_modes,
                -1,
            )

            anchors.append(
                self.anchor_heads[
                    horizon_index
                ](
                    torch.cat(
                        (
                            mode_context,
                            horizon,
                        ),
                        dim=-1,
                    )
                )
            )

        route_anchors = torch.stack(
            anchors,
            dim=2,
        )

        goal_offset = self.goal_offset_head(
            mode_context
        ).squeeze(-1)

        return {
            "mode_context": mode_context,
            "goal_context": goal_context,
            "goal_logits": goal_logits,
            "goal_prob": goal_prob,
            "goal_offset": goal_offset,
            "route_anchors": route_anchors,
        }
