"""Assemble the complete WZ-TARF trajectory forecasting architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from wztarf.data.features import (
    build_control_features,
    build_gaze_features,
    build_motion_features,
    relative_history_time,
)
from wztarf.geometry.workzone import signed_distance_to_polygon
from wztarf.models.decoders import (
    DynamicsAnchor,
    LocalRefiner,
    TrajectoryDecoder,
)
from wztarf.models.encoders import (
    AgentEncoder,
    ControlEncoder,
    GazeEncoder,
    LaneEncoder,
    MotionEncoder,
    WorkZoneEncoder,
)
from wztarf.models.fusion import (
    ControlFusion,
    HorizonFusion,
)
from wztarf.models.route import (
    MapExitGoalHead,
    RouteGoalQueries,
)
from wztarf.models.scoring import SafetyAwareScorer
from wztarf.models.topology import TemporaryTopologyAdapter


@dataclass(frozen=True)
class WZTARFConfig:
    """Configuration controlling the WZ-TARF model structure."""

    d_model: int = 128

    motion_hidden: int = 128
    control_hidden: int = 64
    gaze_hidden: int = 64
    agent_hidden: int = 64

    num_modes: int = 6

    fps: int = 5
    history_steps: int = 10
    future_steps: int = 25

    top_seed_lanes: int = 4

    lane_graph_layers: int = 2
    topology_layers: int = 2

    num_edge_types: int = 16

    use_refiner: bool = False
    refiner_radius_m: float = 8.0


class WZTARF(nn.Module):
    """WorkZone-Conditioned Topology-Adaptive Route Forecaster."""

    def __init__(
        self,
        config: WZTARFConfig | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = WZTARFConfig()

        self.config = config

        # --------------------------------------------------------------
        # Role-specific temporal and spatial encoders
        # --------------------------------------------------------------

        self.motion_encoder = MotionEncoder(
            input_dim=11,
            hidden_dim=config.motion_hidden,
        )

        self.control_encoder = ControlEncoder(
            input_dim=8,
            hidden_dim=config.control_hidden,
        )

        self.control_fusion = ControlFusion(
            motion_dim=config.motion_hidden,
            control_dim=config.control_hidden,
            gate_feature_dim=6,
            d_model=config.d_model,
        )

        self.gaze_encoder = GazeEncoder(
            input_dim=5,
            hidden_dim=config.gaze_hidden,
            output_dim=config.d_model,
        )

        # 11 raw agent dimensions plus relative physical time.
        self.agent_encoder = AgentEncoder(
            input_dim=12,
            hidden_dim=config.agent_hidden,
            output_dim=config.d_model,
        )

        self.workzone_encoder = WorkZoneEncoder(
            d_model=config.d_model,
        )

        self.lane_encoder = LaneEncoder(
            input_dim=8,
            d_model=config.d_model,
            top_seed_lanes=config.top_seed_lanes,
            graph_layers=config.lane_graph_layers,
            num_edge_types=config.num_edge_types,
        )

        # --------------------------------------------------------------
        # Temporary topology and horizon-aware fusion
        # --------------------------------------------------------------

        self.topology_adapter = TemporaryTopologyAdapter(
            d_model=config.d_model,
            num_edge_types=config.num_edge_types,
            num_layers=config.topology_layers,
        )

        self.horizon_fusion = HorizonFusion(
            d_model=config.d_model,
        )

        # --------------------------------------------------------------
        # Route hypothesis generation
        # --------------------------------------------------------------

        self.route_queries = RouteGoalQueries(
            d_model=config.d_model,
            num_modes=config.num_modes,
        )

        self.map_exit_head = MapExitGoalHead(
            d_model=config.d_model,
        )

        # --------------------------------------------------------------
        # Dynamics and continuous decoding
        # --------------------------------------------------------------

        self.dynamics_anchor = DynamicsAnchor(
            d_model=config.d_model,
            control_dim=config.control_hidden,
            future_steps=config.future_steps,
            fps=config.fps,
        )

        self.trajectory_decoder = TrajectoryDecoder(
            d_model=config.d_model,
            future_steps=config.future_steps,
            fps=config.fps,
        )

        self.local_refiner = (
            LocalRefiner(
                d_model=config.d_model,
                local_radius_m=config.refiner_radius_m,
            )
            if config.use_refiner
            else None
        )

        # --------------------------------------------------------------
        # Final safety-aware mode ranking
        # --------------------------------------------------------------

        self.safety_scorer = SafetyAwareScorer(
            d_model=config.d_model,
            safety_feature_dim=3,
        )

    def _agent_features(
        self,
        agent_hist: torch.Tensor,
    ) -> torch.Tensor:
        """Append relative physical time to each agent history."""
        batch_size, history_steps, num_agents, _ = agent_hist.shape

        time = relative_history_time(
            history_steps=history_steps,
            fps=self.config.fps,
            dtype=agent_hist.dtype,
            device=agent_hist.device,
        )

        time = time[
            None,
            :,
            None,
            None,
        ].expand(
            batch_size,
            history_steps,
            num_agents,
            1,
        )

        return torch.cat(
            (
                agent_hist,
                time,
            ),
            dim=-1,
        )

    def _control_gate_features(
        self,
        motion_features: torch.Tensor,
        control_features: torch.Tensor,
    ) -> torch.Tensor:
        """Build `[Δcontrols, acceleration, yaw-rate]` gate inputs."""
        delta_control = control_features[
            ...,
            3:6,
        ]

        acceleration = motion_features[
            ...,
            4:6,
        ]

        yaw_rate = motion_features[
            ...,
            8:9,
        ]

        return torch.cat(
            (
                delta_control,
                acceleration,
                yaw_rate,
            ),
            dim=-1,
        )

    def _safety_features(
        self,
        pred_xy: torch.Tensor,
        goal_prob: torch.Tensor,
        wz_feat: torch.Tensor,
        worker_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Construct per-mode WZ risk, worker risk, and goal confidence."""
        batch_size, num_modes, future_steps, _ = pred_xy.shape

        wz_risk = torch.zeros(
            batch_size,
            num_modes,
            dtype=pred_xy.dtype,
            device=pred_xy.device,
        )

        worker_risk = torch.zeros_like(
            wz_risk
        )

        for b in range(
            batch_size
        ):
            polygon_valid = bool(
                (
                    wz_feat[
                        b,
                        :4,
                        2,
                    ]
                    >
                    0
                ).all()
            )

            if polygon_valid:
                polygon = wz_feat[
                    b,
                    :4,
                    :2,
                ]

                points = pred_xy[
                    b
                ].reshape(
                    num_modes * future_steps,
                    2,
                )

                signed_distance = signed_distance_to_polygon(
                    points,
                    polygon,
                ).reshape(
                    num_modes,
                    future_steps,
                )

                wz_risk[
                    b
                ] = F.softplus(
                    -signed_distance
                    /
                    0.25
                ).mean(
                    dim=1
                )

            worker_mask = (
                worker_feat[
                    b,
                    :,
                    2,
                ]
                >
                0
            )

            valid_workers = worker_feat[
                b,
                worker_mask,
                :2,
            ]

            if valid_workers.numel() > 0:
                distance = torch.linalg.vector_norm(
                    pred_xy[
                        b,
                        :,
                        :,
                        None,
                        :,
                    ]
                    -
                    valid_workers[
                        None,
                        None,
                        :,
                        :,
                    ],
                    dim=-1,
                )

                worker_risk[
                    b
                ] = F.relu(
                    2.0
                    -
                    distance
                ).square().sum(
                    dim=-1
                ).mean(
                    dim=-1
                )

        goal_confidence = goal_prob.max(
            dim=-1
        ).values

        return torch.stack(
            (
                wz_risk,
                worker_risk,
                goal_confidence,
            ),
            dim=-1,
        )

    def encode_scene(
        self,
        batch: Mapping[str, Any],
        *,
        mask_plan: Any | None = None,
        compact_lanes: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Encode all observed scene modalities before route decoding."""
        motion_features = build_motion_features(
            batch["ego_hist"],
            fps=self.config.fps,
        )

        control_features = build_control_features(
            batch["control_hist"],
            batch["control_mask"],
            fps=self.config.fps,
        )

        gaze_features = build_gaze_features(
            batch["gaze_feat"],
            batch["gaze_mask"],
            fps=self.config.fps,
        )

        control_mask = batch[
            "control_mask"
        ].bool()

        gaze_mask = batch[
            "gaze_mask"
        ].bool()

        agent_mask = batch[
            "agent_mask"
        ].bool()

        lane_point_mask = batch[
            "lane_point_mask"
        ].bool()

        lane_feat = batch[
            "lane_feat"
        ]

        wz_feat = batch[
            "wz_feat"
        ]

        worker_feat = batch[
            "wz_worker_feat"
        ]

        agent_features = self._agent_features(
            batch[
                "agent_hist"
            ]
        )

        if mask_plan is not None:
            motion_features = motion_features.masked_fill(
                mask_plan.motion[
                    ...,
                    None,
                ],
                0.0,
            )

            control_features = control_features.masked_fill(
                mask_plan.controls[
                    ...,
                    None,
                ],
                0.0,
            )

            control_mask = (
                control_mask
                &
                ~mask_plan.controls
            )

            gaze_features = gaze_features.masked_fill(
                mask_plan.gaze[
                    ...,
                    None,
                ],
                0.0,
            )

            gaze_mask = (
                gaze_mask
                &
                ~mask_plan.gaze
            )

            agent_mask = (
                agent_mask
                &
                ~mask_plan.agents
            )

            lane_point_mask = (
                lane_point_mask
                &
                ~mask_plan.lanes
            )

            wz_feat = wz_feat.clone()

            wz_feat[
                ...,
                2,
            ] = torch.where(
                mask_plan.workzone,
                torch.zeros_like(
                    wz_feat[
                        ...,
                        2,
                    ]
                ),
                wz_feat[
                    ...,
                    2,
                ],
            )

            worker_feat = worker_feat.clone()

            worker_feat[
                ...,
                2,
            ] = torch.where(
                mask_plan.workers,
                torch.zeros_like(
                    worker_feat[
                        ...,
                        2,
                    ]
                ),
                worker_feat[
                    ...,
                    2,
                ],
            )

        motion_states, _ = self.motion_encoder(
            motion_features
        )

        control_states, control_context = self.control_encoder(
            control_features,
            control_mask,
        )

        gate_features = self._control_gate_features(
            motion_features,
            control_features,
        )

        ego_states, ego_context, control_gate = self.control_fusion(
            motion_states,
            control_states,
            control_mask,
            gate_features,
        )

        gaze = self.gaze_encoder(
            gaze_features,
            gaze_mask,
        )

        agent = self.agent_encoder(
            agent_features,
            agent_mask,
            ego_context,
        )

        workzone = self.workzone_encoder(
            wz_feat,
            worker_feat,
            ego_speed=motion_features[
                :,
                -1,
                9,
            ],
            lane_feat=lane_feat,
            lane_point_mask=lane_point_mask,
            lane_mask=batch["lane_mask"],
        )

        lane = self.lane_encoder(
        lane_feat,
        lane_point_mask,
            batch["lane_mask"],
            batch["lane_edge_index"],
            batch["lane_edge_type"],
            batch["lane_edge_mask"],
            ego_context,
            workzone["wz_context"],
            lane_attr=batch.get(
                "lane_attr"
            ),
            compact=compact_lanes,
        )

        topology = self.topology_adapter(
            lane["lane_states"],
            lane["lane_mask"],
            batch["lane_edge_index"],
            batch["lane_edge_type"],
            batch["lane_edge_mask"],
            workzone["wz_context"],
            lane_xy=lane["lane_xy"],
            lane_heading=lane["lane_heading"],
        )

        agent_valid = agent[
            "agent_mask"
        ].any(
            dim=1
        )

        role_context = {
            "ego": ego_context,
            "workzone": workzone["wz_context"],
            "lane": topology["lane_context"],
            "gaze": gaze["gaze_context"],
            "agents": agent["agent_context"],
        }

        role_valid = {
            "ego": torch.ones(
                ego_context.shape[0],
                dtype=torch.bool,
                device=ego_context.device,
            ),
            "workzone": workzone["wz_valid"],
            "lane": lane["lane_mask"].any(
                dim=1
            ),
            "gaze": gaze["gaze_valid"],
            "agents": agent_valid,
        }

        horizon_context, horizon_role_weights = self.horizon_fusion(
            role_context,
            role_valid,
        )

        return {
            "motion_features": motion_features,
            "control_features": control_features,
            "ego_states": ego_states,
            "ego_context": ego_context,
            "control_states": control_states,
            "control_context": control_context,
            "control_gate": control_gate,
            "gaze_states": gaze["gaze_states"],
            "gaze_context": gaze["gaze_context"],
            "gaze_reliability": gaze["gaze_reliability"],
            "agent_temporal_states": agent["agent_temporal_states"],
            "agent_states": agent["agent_states"],
            "agent_context": agent["agent_context"],
            "agent_mask": agent["agent_mask"],
            "agent_xy": agent["agent_xy"],
            "wz_tokens": workzone["wz_tokens"],
            "wz_token_mask": workzone["wz_token_mask"],
            "wz_token_xy": workzone["wz_token_xy"],
            "wz_context": workzone["wz_context"],
            "lane_point_states": lane["lane_point_states"],
            "lane_states": topology["lane_states"],
            "lane_context": topology["lane_context"],
            "lane_mask": lane["lane_mask"],
            "lane_xy": lane["lane_xy"],
            "node_viability": topology["node_viability"],
            "edge_viability": topology["edge_viability"],
            "horizon_context": horizon_context,
            "horizon_role_weights": horizon_role_weights,
        }

    def forward(
        self,
        batch: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Run complete K=6 route-to-trajectory forecasting."""
        scene = self.encode_scene(
            batch
        )

        route = self.route_queries(
            ego_context=scene["ego_context"],
            lane_context=scene["lane_context"],
            horizon_context=scene["horizon_context"],
            lane_states=scene["lane_states"],
            lane_mask=scene["lane_mask"],
        )

        map_exit_goal = self.map_exit_head(
            route["mode_context"],
            route["goal_context"],
            scene["ego_context"],
        )

        # Blend the predicted terminal anchor toward the learned continuous
        # continuation goal when the route query assigns MAP_EXIT probability.
        exit_probability = route[
            "goal_prob"
        ][
            ...,
            -1:
        ]

        route_anchors = route[
            "route_anchors"
        ].clone()

        route_anchors[
            :,
            :,
            -1,
        ] = (
            (
                1.0
                -
                exit_probability
            )
            *
            route_anchors[
                :,
                :,
                -1,
            ]
            +
            exit_probability
            *
            map_exit_goal
        )

        dynamics_xy = self.dynamics_anchor(
            scene["ego_context"],
            scene["control_context"],
        )

        decoded = self.trajectory_decoder(
            route["mode_context"],
            scene["horizon_context"],
            route_anchors,
            dynamics_xy,
        )

        coarse_xy = decoded[
            "coarse_xy"
        ]

        refinement_delta: torch.Tensor | None = None

        if self.local_refiner is not None:
            scene_tokens = torch.cat(
                (
                    scene["lane_states"],
                    scene["wz_tokens"],
                    scene["agent_states"],
                ),
                dim=1,
            )

            scene_xy = torch.cat(
                (
                    scene["lane_xy"],
                    scene["wz_token_xy"],
                    scene["agent_xy"],
                ),
                dim=1,
            )

            scene_mask = torch.cat(
                (
                    scene["lane_mask"],
                    scene["wz_token_mask"],
                    scene["agent_mask"],
                ),
                dim=1,
            )

            refinement_delta = self.local_refiner(
                coarse_xy,
                route["mode_context"],
                scene_tokens,
                scene_xy,
                scene_mask,
            )

            pred_xy = (
                coarse_xy
                +
                refinement_delta
            )

        else:
            pred_xy = coarse_xy

        safety_features = self._safety_features(
            pred_xy,
            route["goal_prob"],
            batch["wz_feat"],
            batch["wz_worker_feat"],
        )

        mode_logits, mode_prob = self.safety_scorer(
            route["mode_context"],
            safety_features,
        )

        output = {
            "pred_xy": pred_xy,
            "mode_logits": mode_logits,
            "mode_prob": mode_prob,
            "lane_mask": scene["lane_mask"],

            "coarse_xy": coarse_xy,
            "dynamics_xy": dynamics_xy,

            "route_anchors": route_anchors,
            "goal_logits": route["goal_logits"],
            "goal_prob": route["goal_prob"],
            "goal_offset": route["goal_offset"],
            "map_exit_goal": map_exit_goal,

            "mode_context": route["mode_context"],

            "node_viability": scene["node_viability"],
            "edge_viability": scene["edge_viability"],

            "control_gate": scene["control_gate"],
            "gaze_reliability": scene["gaze_reliability"],
            "horizon_role_weights": scene["horizon_role_weights"],

            "safety_features": safety_features,
        }

        if refinement_delta is not None:
            output[
                "refinement_delta"
            ] = refinement_delta

        return output
