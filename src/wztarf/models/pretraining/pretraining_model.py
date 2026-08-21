"""Attach training-only Phase A heads around the standard WZ-TARF backbone."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from wztarf.data.features import (
    build_control_features,
    build_gaze_features,
    build_motion_features,
)
from wztarf.models.pretraining.future_encoder import FutureEncoder
from wztarf.models.pretraining.reconstruction_heads import ReconstructionHeads
from wztarf.models.pretraining.topology_heads import TopologyHeads
from wztarf.models.wztarf import (
    WZTARF,
    WZTARFConfig,
)


class WZTARFPretrainingModel(nn.Module):
    """Use the forecasting backbone with heads that exist only in Phase A."""

    def __init__(
        self,
        config: WZTARFConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.backbone = WZTARF(
            config
        )

        self.future_encoder = FutureEncoder(
            d_model=config.d_model,
            fps=config.fps,
        )

        self.context_projection = nn.ModuleDict(
            {
                str(horizon): nn.Sequential(
                    nn.Linear(
                        config.d_model,
                        config.d_model,
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        config.d_model,
                        config.d_model,
                    ),
                )
                for horizon in (
                    1,
                    3,
                    5,
                )
            }
        )

        self.reconstruction = ReconstructionHeads(
            d_model=config.d_model,
            control_dim=config.control_hidden,
        )

        self.topology = TopologyHeads(
            d_model=config.d_model,
            num_edge_types=config.num_edge_types,
        )

    def forward(
        self,
        batch: Mapping[str, Any],
    ):
        """Delegate ordinary forecasting to the backbone."""
        return self.backbone(
            batch
        )

    def pretraining_forward(
        self,
        batch: Mapping[str, Any],
        mask_plan: Any,
    ) -> Mapping[str, Any]:
        """Run masked scene encoding and all Phase A prediction heads."""
        motion_target = build_motion_features(
            batch[
                "ego_hist"
            ],
            fps=self.config.fps,
        )

        control_target = build_control_features(
            batch[
                "control_hist"
            ],
            batch[
                "control_mask"
            ],
            fps=self.config.fps,
        )

        gaze_target = build_gaze_features(
            batch[
                "gaze_feat"
            ],
            batch[
                "gaze_mask"
            ],
            fps=self.config.fps,
        )

        scene = self.backbone.encode_scene(
            batch,
            mask_plan=mask_plan,
            compact_lanes=False,
        )

        reconstruction = self.reconstruction(
            scene=scene,
            batch=batch,
            motion_target=motion_target,
            control_target=control_target,
            gaze_target=gaze_target,
        )

        future_embeddings = self.future_encoder(
            batch[
                "future_xy"
            ]
        )

        context_embeddings = {
            horizon: self.context_projection[
                str(
                    horizon
                )
            ](
                scene[
                    "horizon_context"
                ][
                    :,
                    horizon_index,
                ]
            )
            for horizon_index, horizon in enumerate(
                (
                    1,
                    3,
                    5,
                )
            )
        }

        topology = self.topology(
            lane_states=scene[
                "lane_states"
            ],
            lane_mask=scene[
                "lane_mask"
            ],
            # Full V3 topology pretraining uses the same explicitly
            # separated WZ+worker topology context as Phase B.
            wz_context=scene[
                "topology_context"
            ],
            edge_index=batch[
                "lane_edge_index"
            ],
            edge_type=batch[
                "lane_edge_type"
            ],
            edge_mask=batch[
                "lane_edge_mask"
            ],
        )

        # ==============================================================
        # V3 PHASE-A SHARED ROUTE MACHINERY
        #
        # Use the exact route-query and monotonic route-progress modules
        # that are used during Phase-B forecasting.
        # ==============================================================

        route = self.backbone.route_queries(
            ego_context=scene["ego_context"],
            lane_context=scene["lane_context"],
            horizon_context=scene["horizon_context"],
            lane_states=scene["lane_states"],
            lane_mask=scene["lane_mask"],
            lane_centerline=scene["lane_centerline"],
            lane_point_mask=scene["lane_point_mask"],
            node_viability=scene["node_viability"],
            edge_viability=scene["edge_viability"],
            lane_edge_index=batch["lane_edge_index"],
            lane_edge_mask=batch["lane_edge_mask"],
        )

        progress = self.backbone.route_progress(
            mode_context=route["mode_context"],
            route_node_occupancy=route["route_node_occupancy"],
            route_edge_occupancy=route["route_edge_occupancy"],
            edge_viability=scene["edge_viability"],
            lane_edge_index=batch["lane_edge_index"],
            lane_edge_type=batch["lane_edge_type"],
            lane_edge_mask=batch["lane_edge_mask"],
            lane_centerline=scene["lane_centerline"],
            lane_point_mask=scene["lane_point_mask"],
            lane_mask=scene["lane_mask"],
        )

        return {
            **reconstruction,
            **topology,
            "context_embeddings": context_embeddings,
            "future_embeddings": future_embeddings,

            # Shared Phase-B V3 route machinery.
            "route_node_occupancy": route["route_node_occupancy"],
            "route_edge_occupancy": route["route_edge_occupancy"],
            "route_viability": route["route_viability"],
            "route_goal_prob": route["goal_prob"],
            "route_progress": progress["route_progress"],
            "route_anchors": progress["route_progress_anchors"],
        }
