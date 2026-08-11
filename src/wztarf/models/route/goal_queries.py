"""Generate six route hypotheses over retained lanes plus MAP_EXIT."""

from __future__ import annotations

import math

import torch
from torch import nn

def _lane_points_from_fraction(
    lane_centerline: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_index: torch.Tensor,
    fraction: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Convert selected lane plus longitudinal fraction into XY goals.

    The fraction is converted to a metric longitudinal offset using the
    selected lane's represented arc length.

    Returns:
        goal_xy:
            [B, K, 2]

        goal_offset:
            Metric arc-length offset [B, K].

        goal_valid:
            Whether the selected lane contains represented geometry [B, K].
    """
    if lane_centerline.ndim != 4 or lane_centerline.shape[-1] != 2:
        raise ValueError(
            "lane_centerline must have shape [B, L, P, 2]."
        )

    if lane_point_mask.shape != lane_centerline.shape[:3]:
        raise ValueError(
            "lane_point_mask must have shape [B, L, P]."
        )

    if lane_index.shape != fraction.shape:
        raise ValueError(
            "lane_index and fraction must have shape [B, K]."
        )

    batch_size, num_lanes, _, _ = lane_centerline.shape

    batch_goal = []
    batch_offset = []
    batch_valid = []

    for b in range(batch_size):
        mode_goal = []
        mode_offset = []
        mode_valid = []

        for k in range(lane_index.shape[1]):
            index = int(
                lane_index[
                    b,
                    k,
                ].item()
            )

            if index < 0 or index >= num_lanes:
                mode_goal.append(
                    fraction[
                        b,
                        k,
                    ].new_zeros(
                        2
                    )
                )

                mode_offset.append(
                    fraction[
                        b,
                        k,
                    ] * 0.0
                )

                mode_valid.append(
                    False
                )

                continue

            valid = lane_point_mask[
                b,
                index,
            ].bool()

            points = lane_centerline[
                b,
                index,
                valid,
            ]

            if points.shape[0] == 0:
                mode_goal.append(
                    fraction[
                        b,
                        k,
                    ].new_zeros(
                        2
                    )
                )

                mode_offset.append(
                    fraction[
                        b,
                        k,
                    ] * 0.0
                )

                mode_valid.append(
                    False
                )

                continue

            if points.shape[0] == 1:
                mode_goal.append(
                    points[0]
                )

                mode_offset.append(
                    fraction[
                        b,
                        k,
                    ] * 0.0
                )

                mode_valid.append(
                    True
                )

                continue

            segment = (
                points[1:]
                -
                points[:-1]
            )

            length = torch.linalg.vector_norm(
                segment,
                dim=-1,
            )

            cumulative = torch.cumsum(
                length,
                dim=0,
            )

            total_length = cumulative[-1]

            offset = (
                fraction[
                    b,
                    k,
                ]
                *
                total_length
            )

            if float(
                total_length.detach().item()
            ) <= 1e-8:
                goal = points[-1]
            else:
                segment_index = torch.searchsorted(
                    cumulative,
                    offset.detach(),
                    right=False,
                )

                segment_index = segment_index.clamp(
                    max=length.shape[0] - 1
                )

                start_offset = torch.where(
                    segment_index > 0,
                    cumulative[
                        (
                            segment_index
                            -
                            1
                        ).clamp_min(
                            0
                        )
                    ],
                    torch.zeros_like(
                        offset
                    ),
                )

                alpha = (
                    offset
                    -
                    start_offset
                ) / length[
                    segment_index
                ].clamp_min(
                    1e-8
                )

                goal = (
                    points[
                        segment_index
                    ]
                    +
                    alpha
                    *
                    segment[
                        segment_index
                    ]
                )

            mode_goal.append(
                goal
            )

            mode_offset.append(
                offset
            )

            mode_valid.append(
                True
            )

        batch_goal.append(
            torch.stack(
                mode_goal,
                dim=0,
            )
        )

        batch_offset.append(
            torch.stack(
                mode_offset,
                dim=0,
            )
        )

        batch_valid.append(
            torch.tensor(
                mode_valid,
                dtype=torch.bool,
                device=lane_centerline.device,
            )
        )

    return (
        torch.stack(
            batch_goal,
            dim=0,
        ),
        torch.stack(
            batch_offset,
            dim=0,
        ),
        torch.stack(
            batch_valid,
            dim=0,
        ),
    )
    
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
        lane_centerline: torch.Tensor,
        lane_point_mask: torch.Tensor,
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

        learned_route_anchors = torch.stack(
            anchors,
            dim=2,
        )

        # --------------------------------------------------------------
        # Geometric terminal graph goal.
        #
        # In-map:
        #     (lane, longitudinal offset)
        #         -> LanePoint(lane, offset)
        #         -> 5 s anchor
        #
        # MAP_EXIT keeps the learned continuous terminal anchor.
        # --------------------------------------------------------------

        goal_class = goal_logits.argmax(
            dim=-1
        )

        goal_is_map_exit = (
            goal_class
            ==
            num_lanes
        )

        selected_lane = lane_logits.argmax(
            dim=-1
        )

        goal_fraction = torch.sigmoid(
            self.goal_offset_head(
                mode_context
            ).squeeze(-1)
        )

        geometric_goal_xy, goal_offset, geometric_goal_valid = (
            _lane_points_from_fraction(
                lane_centerline,
                (
                    lane_point_mask.bool()
                    &
                    lane_mask[
                        :,
                        :,
                        None,
                    ].bool()
                ),
                selected_lane,
                goal_fraction,
            )
        )

        use_geometric_goal = (
            ~goal_is_map_exit
            &
            geometric_goal_valid
        )

        terminal_goal = torch.where(
            use_geometric_goal[
                ...,
                None,
            ],
            geometric_goal_xy,
            learned_route_anchors[
                :,
                :,
                2,
            ],
        )

        route_anchors = torch.cat(
            (
                learned_route_anchors[
                    :,
                    :,
                    :2,
                ],
                terminal_goal[
                    :,
                    :,
                    None,
                ],
            ),
            dim=2,
        )

        return {
            "mode_context": mode_context,
            "goal_context": goal_context,
            "goal_logits": goal_logits,
            "goal_prob": goal_prob,
            "goal_offset": goal_offset,
            "route_anchors": route_anchors,
            "goal_lane_index": selected_lane,
            "goal_is_map_exit": goal_is_map_exit,
            "geometric_goal_xy": geometric_goal_xy,
            "geometric_goal_valid": geometric_goal_valid,
        }
