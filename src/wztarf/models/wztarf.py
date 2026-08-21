"""Assemble the complete WZ-TARF trajectory forecasting architecture."""

from __future__ import annotations

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
from wztarf.models.decoders.direct_trajectory import DirectTrajectoryDecoder
from wztarf.models.config import WZTARFConfig
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
from wztarf.models.route import RouteGoalQueries
from wztarf.models.route.route_progress import DifferentiableRouteProgress
from wztarf.models.scoring import RouteQualityScorer
from wztarf.models.topology import TemporaryTopologyAdapter


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

        # Validate structural ablation settings.
        topology_mode = str(
            config.topology_mode
        ).lower()

        if topology_mode not in {
            "workzone",
            "static",
        }:
            raise ValueError(
                "config.topology_mode must be 'workzone' or 'static'."
            )

        for name, probability in (
            (
                "aux_dropout_controls",
                config.aux_dropout_controls,
            ),
            (
                "aux_dropout_gaze",
                config.aux_dropout_gaze,
            ),
            (
                "aux_dropout_workers",
                config.aux_dropout_workers,
            ),
        ):
            if not 0.0 <= float(probability) < 1.0:
                raise ValueError(
                    f"{name} must lie in [0, 1)."
                )

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
            lane_geometry_points=config.wz_lane_geometry_points,
        )

        self.lane_encoder = LaneEncoder(
            input_dim=8,
            d_model=config.d_model,
            top_seed_lanes=config.top_seed_lanes,
            graph_layers=config.lane_graph_layers,
            num_edge_types=config.num_edge_types,
            encode_points=config.lane_encode_points,
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
            roles=(
                "ego",
                "lane",
                "gaze",
                "agents",
            ),
        )

        # --------------------------------------------------------------
        # Route hypothesis generation
        # --------------------------------------------------------------

        self.route_queries = RouteGoalQueries(
            d_model=config.d_model,
            num_modes=config.num_modes,
            topology_bias=config.route_topology_bias,
            competition_strength=config.route_competition_strength,
        )

        self.route_progress = DifferentiableRouteProgress(
            d_model=config.d_model,
            future_steps=config.future_steps,
            fps=config.fps,
            walk_steps=config.route_walk_steps,
            progress_scale_m=config.route_progress_scale_m,
            exit_extension_m=config.route_exit_extension_m,
            use_dense_progress_repair=config.use_dense_progress_repair,
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
            longitudinal_correction_m=(
                config.decoder_longitudinal_correction_m
            ),
            lateral_correction_m=(
                config.decoder_lateral_correction_m
            ),
        )
        self.direct_trajectory_decoder = (
            DirectTrajectoryDecoder(
                d_model=config.d_model,
                num_modes=config.num_modes,
                future_steps=config.future_steps,
                fps=config.fps,
                use_anchor_calibration=config.use_direct_anchor_calibration,
                use_longitudinal_repair=config.use_direct_longitudinal_repair,
            )
            if config.use_direct_decoder
            else None
        )

        self.local_refiner = (
            LocalRefiner(
                d_model=config.d_model,
                local_radius_m=config.refiner_radius_m,
                max_longitudinal_correction_m=(
                    config.refiner_longitudinal_correction_m
                ),
                max_lateral_correction_m=(
                    config.refiner_lateral_correction_m
                ),
            )
            if config.use_refiner
            else None
        )

        # --------------------------------------------------------------
        # Final safety-aware mode ranking
        # --------------------------------------------------------------

        # ==========================================================
        # V3 BEHAVIOR / QUALITY SEPARATED MODE RANKER
        # ==========================================================
        self.mode_ranker = RouteQualityScorer(
            d_model=config.d_model,
            behavior_feature_dim=8,
            quality_feature_dim=11,
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
        """Construct per-mode WZ risk, worker risk, and goal confidence.

        The WorkZone polygon always has four represented corners, and workers
        are padded to a tiny fixed set.  Compute both risks in batched tensor
        form instead of synchronizing Python once per sample.
        """
        batch_size, num_modes, future_steps, _ = pred_xy.shape

        # V3: safety geometry remains FP32 under BF16 training.
        pred_xy = pred_xy.float()
        goal_prob = goal_prob.float()
        wz_feat = wz_feat.float()
        worker_feat = worker_feat.float()

        polygon = wz_feat[:, :4, :2]
        polygon_valid = (wz_feat[:, :4, 2] > 0).all(dim=1)

        points = pred_xy.reshape(batch_size, num_modes * future_steps, 2)
        start = polygon
        end = torch.roll(polygon, shifts=-1, dims=1)
        edge = end - start

        # Point-to-segment distance: [B,N,4].
        point_delta = points[:, :, None, :] - start[:, None, :, :]
        edge_sq = (edge * edge).sum(dim=-1).clamp_min(1e-8)
        alpha = (
            (point_delta * edge[:, None, :, :]).sum(dim=-1)
            / edge_sq[:, None, :]
        ).clamp(0.0, 1.0)
        projection = start[:, None, :, :] + alpha[..., None] * edge[:, None, :, :]
        boundary_distance = torch.linalg.vector_norm(
            points[:, :, None, :] - projection,
            dim=-1,
        ).amin(dim=-1)

        # Batched ray casting for inside/outside classification.
        px = points[..., 0, None]
        py = points[..., 1, None]
        x0 = start[:, None, :, 0]
        y0 = start[:, None, :, 1]
        x1 = end[:, None, :, 0]
        y1 = end[:, None, :, 1]
        crosses_y = (y0 > py) != (y1 > py)
        x_intersection = (x1 - x0) * (py - y0) / (y1 - y0 + 1e-8) + x0
        crosses = crosses_y & (px < x_intersection)
        inside = (crosses.long().sum(dim=-1) % 2) == 1

        signed_distance = torch.where(
            inside,
            -boundary_distance,
            boundary_distance,
        ).reshape(batch_size, num_modes, future_steps)
        wz_risk = F.softplus(-signed_distance / 0.25).mean(dim=-1)
        wz_risk = wz_risk * polygon_valid[:, None].to(wz_risk.dtype)

        worker_xy = worker_feat[..., :2].to(pred_xy.dtype)
        worker_mask = worker_feat[..., 2] > 0
        worker_distance = torch.linalg.vector_norm(
            pred_xy[:, :, :, None, :] - worker_xy[:, None, None, :, :],
            dim=-1,
        )
        worker_penalty = F.relu(2.0 - worker_distance).square()
        worker_penalty = worker_penalty * worker_mask[:, None, None, :].to(
            worker_penalty.dtype
        )
        worker_risk = worker_penalty.sum(dim=-1).mean(dim=-1)

        goal_confidence = goal_prob.max(dim=-1).values
        return torch.stack((wz_risk, worker_risk, goal_confidence), dim=-1)

    def _ranking_feature_sets(
        self,
        *,
        pred_xy: torch.Tensor,
        dynamics_xy: torch.Tensor,
        route_guide: torch.Tensor,
        route_tangent: torch.Tensor,
        route_normal: torch.Tensor,
        route_viability: torch.Tensor,
        goal_prob: torch.Tensor,
        progress_increment: torch.Tensor,
        route_safety_features: torch.Tensor,
        trajectory_safety_features: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Build interpretable behavioral and forecast-quality features."""

        dtype = pred_xy.dtype
        eps = 1.0e-6

        # ----------------------------------------------------------
        # Goal / progress confidence.
        # ----------------------------------------------------------

        goal_probability = (
            goal_prob.float().clamp_min(
                eps
            )
        )

        goal_confidence = (
            goal_probability.max(
                dim=-1
            ).values
        )

        goal_entropy = -(
            goal_probability
            *
            torch.log(
                goal_probability
            )
        ).sum(
            dim=-1
        )

        if goal_probability.shape[-1] > 1:
            goal_entropy = (
                goal_entropy
                /
                torch.log(
                    goal_entropy.new_tensor(
                        float(
                            goal_probability.shape[-1]
                        )
                    )
                )
            )

        endpoint_uncertainty = goal_entropy.clamp(
            0.0,
            1.0,
        )

        progress_value = (
            progress_increment.float()
        )

        progress_mean = progress_value.mean(
            dim=-1
        ).clamp_min(
            eps
        )

        progress_cv = (
            progress_value.std(
                dim=-1,
                unbiased=False,
            )
            /
            progress_mean
        )

        progress_confidence = torch.exp(
            -progress_cv
        ).clamp(
            0.0,
            1.0,
        )

        # ----------------------------------------------------------
        # Route safety PRIOR is based on the candidate route guide,
        # not the final decoded trajectory.
        # ----------------------------------------------------------

        route_wz_risk = torch.log1p(
            route_safety_features[
                ...,
                0,
            ].float().clamp_min(
                0.0
            )
        )

        route_worker_risk = torch.log1p(
            route_safety_features[
                ...,
                1,
            ].float().clamp_min(
                0.0
            )
        )

        route_prior_logits = (
            torch.log(
                route_viability.float().clamp_min(
                    eps
                )
            )
            +
            0.50
            *
            torch.log(
                goal_confidence.clamp_min(
                    eps
                )
            )
            +
            0.25
            *
            torch.log(
                progress_confidence.clamp_min(
                    eps
                )
            )
            -
            0.35
            *
            route_wz_risk
            -
            0.35
            *
            route_worker_risk
        )

        route_prior = torch.softmax(
            route_prior_logits,
            dim=-1,
        )

        # ----------------------------------------------------------
        # Trajectory <-> route consistency.
        # ----------------------------------------------------------

        relative = (
            pred_xy.float()
            -
            route_guide.float()
        )

        lateral = (
            relative
            *
            route_normal.float()
        ).sum(
            dim=-1
        )

        route_consistency_cost = (
            lateral.abs().mean(
                dim=-1
            )
            /
            2.0
        )

        # ----------------------------------------------------------
        # Early dynamics agreement.
        # ----------------------------------------------------------

        dynamics = dynamics_xy[
            :,
            None,
        ].float().expand_as(
            pred_xy.float()
        )

        early_steps = min(
            self.config.fps,
            pred_xy.shape[2],
        )

        dynamics_agreement_cost = (
            torch.linalg.vector_norm(
                pred_xy[
                    :,
                    :,
                    :early_steps,
                ].float()
                -
                dynamics[
                    :,
                    :,
                    :early_steps,
                ],
                dim=-1,
            ).mean(
                dim=-1
            )
            /
            5.0
        )

        # ----------------------------------------------------------
        # Endpoint deviation from route guide.
        # ----------------------------------------------------------

        endpoint_deviation = (
            torch.linalg.vector_norm(
                pred_xy[
                    :,
                    :,
                    -1,
                ].float()
                -
                route_guide[
                    :,
                    :,
                    -1,
                ].float(),
                dim=-1,
            )
            /
            5.0
        )

        # ----------------------------------------------------------
        # Route curvature / complexity.
        # ----------------------------------------------------------

        tangent_change = (
            route_tangent[
                :,
                :,
                1:,
            ].float()
            -
            route_tangent[
                :,
                :,
                :-1,
            ].float()
        )

        route_curvature = torch.linalg.vector_norm(
            tangent_change,
            dim=-1,
        ).mean(
            dim=-1
        )

        # Final decoded safety is a QUALITY feature.
        trajectory_wz_risk = torch.log1p(
            trajectory_safety_features[
                ...,
                0,
            ].float().clamp_min(
                0.0
            )
        )

        trajectory_worker_risk = torch.log1p(
            trajectory_safety_features[
                ...,
                1,
            ].float().clamp_min(
                0.0
            )
        )

        # ----------------------------------------------------------
        # Behavioral probability features.
        #
        # Deliberately excludes post-decoder geometry error signals.
        # ----------------------------------------------------------

        behavior_features = torch.stack(
            (
                route_viability.float(),
                route_prior,
                goal_confidence,
                progress_confidence,
                endpoint_uncertainty,
                route_wz_risk,
                route_worker_risk,
                route_curvature,
            ),
            dim=-1,
        ).to(
            dtype
        )

        # ----------------------------------------------------------
        # Forecast-quality features.
        # ----------------------------------------------------------

        quality_features = torch.stack(
            (
                route_viability.float(),
                route_prior,
                goal_confidence,
                progress_confidence,
                endpoint_uncertainty,
                route_consistency_cost,
                dynamics_agreement_cost,
                endpoint_deviation,
                trajectory_wz_risk,
                trajectory_worker_risk,
                route_curvature,
            ),
            dim=-1,
        ).to(
            dtype
        )

        return (
            behavior_features,
            quality_features,
            route_prior.to(
                dtype
            ),
        )


    def encode_scene(
        self,
        batch: Mapping[str, Any],
        *,
        mask_plan: Any | None = None,
        compact_lanes: bool = True,
        return_aux: bool = True,
    ) -> dict[str, Any]:
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

        # Apply structural modality ablations before encoding.
        if not self.config.use_controls:
            control_features = torch.zeros_like(
                control_features
            )
            control_mask = torch.zeros_like(
                control_mask,
                dtype=torch.bool,
            )

        if not self.config.use_gaze:
            gaze_features = torch.zeros_like(
                gaze_features
            )
            gaze_mask = torch.zeros_like(
                gaze_mask,
                dtype=torch.bool,
            )

        if not self.config.use_workers:
            worker_feat = torch.zeros_like(
                worker_feat
            )

        # No-WZ is a genuinely static-map architecture.  WorkZone geometry
        # is absent, while the worker stream can remain available as its own
        # independent auxiliary modality.
        if self.config.topology_mode == "static":
            wz_feat = torch.zeros_like(
                wz_feat
            )

        # --------------------------------------------------------------
        # Training-only whole-stream modality dropout.
        #
        # Do not apply this on Phase A calls that already have mask_plan.
        # --------------------------------------------------------------
        if (
            self.training
            and mask_plan is None
        ):
            batch_size = ego_context_batch = motion_features.shape[0]
            del ego_context_batch

            control_drop = (
                torch.rand(
                    batch_size,
                    device=motion_features.device,
                )
                <
                float(
                    self.config.aux_dropout_controls
                )
            )

            gaze_drop = (
                torch.rand(
                    batch_size,
                    device=motion_features.device,
                )
                <
                float(
                    self.config.aux_dropout_gaze
                )
            )

            worker_drop = (
                torch.rand(
                    batch_size,
                    device=motion_features.device,
                )
                <
                float(
                    self.config.aux_dropout_workers
                )
            )

            if self.config.use_controls:
                control_features = control_features.masked_fill(
                    control_drop[
                        :,
                        None,
                        None,
                    ],
                    0.0,
                )

                control_mask = (
                    control_mask
                    &
                    ~control_drop[
                        :,
                        None,
                    ]
                )

            if self.config.use_gaze:
                gaze_features = gaze_features.masked_fill(
                    gaze_drop[
                        :,
                        None,
                        None,
                    ],
                    0.0,
                )

                gaze_mask = (
                    gaze_mask
                    &
                    ~gaze_drop[
                        :,
                        None,
                    ]
                )

            if self.config.use_workers:
                worker_feat = torch.where(
                    worker_drop[
                        :,
                        None,
                        None,
                    ],
                    torch.zeros_like(
                        worker_feat
                    ),
                    worker_feat,
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
            return_temporal=return_aux,
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

        # WZ and worker context affect the lane graph only through topology.
        wz_topology_valid = workzone[
            "wz_valid"
        ].to(
            workzone[
                "wz_context"
            ].dtype
        )[:, None]

        worker_topology_valid = workzone[
            "worker_valid"
        ].to(
            workzone[
                "worker_context"
            ].dtype
        )[:, None]

        topology_context_numerator = (
            workzone[
                "wz_context"
            ]
            *
            wz_topology_valid
            +
            workzone[
                "worker_context"
            ]
            *
            worker_topology_valid
        )

        topology_context_denominator = (
            wz_topology_valid
            +
            worker_topology_valid
        ).clamp_min(
            1.0
        )

        topology_context = (
            topology_context_numerator
            /
            topology_context_denominator
        )

        lane = self.lane_encoder(
        lane_feat,
        lane_point_mask,
            batch["lane_mask"],
            batch["lane_edge_index"],
            batch["lane_edge_type"],
            batch["lane_edge_mask"],
            ego_context,

            # V3 causal path:
            # permanent lane encoding is not directly WZ-conditioned.
            torch.zeros_like(
                workzone["wz_context"]
            ),

            lane_attr=batch.get(
                "lane_attr"
            ),
            compact=compact_lanes,
            return_point_states=return_aux,
        )

        topology = self.topology_adapter(
            lane["lane_states"],
            lane["lane_mask"],
            batch["lane_edge_index"],
            batch["lane_edge_type"],
            batch["lane_edge_mask"],
            topology_context,
            lane_xy=lane["lane_xy"],
            lane_heading=lane["lane_heading"],
            topology_mode=self.config.topology_mode,
        )

        agent_valid = agent[
            "agent_mask"
        ].any(
            dim=1
        )

        role_context = {
            "ego": ego_context,
            # WZ affects forecasting through temporary topology,
            # not through an inseparable generic context shortcut.
            "workzone": torch.zeros_like(
                workzone["wz_context"]
            ),
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
            "workzone": torch.zeros_like(
                workzone["wz_valid"],
                dtype=torch.bool,
            ),
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
            "topology_context": topology_context,

            # Explicit independent worker representation.
            "worker_tokens": workzone["worker_tokens"],
            "worker_token_mask": workzone["worker_token_mask"],
            "worker_token_xy": workzone["worker_token_xy"],
            "worker_context": workzone["worker_context"],
            "worker_valid": workzone["worker_valid"],

            # Effective observed modalities after structural ablation /
            # training-only modality dropout.
            "effective_wz_feat": wz_feat,
            "effective_worker_feat": worker_feat,

            "lane_point_states": lane["lane_point_states"],
            "lane_states": topology["lane_states"],
            "lane_context": topology["lane_context"],
            "lane_mask": lane["lane_mask"],
            "lane_xy": lane["lane_xy"],
            "lane_centerline": lane["lane_centerline"],
            "lane_point_mask": lane["lane_point_mask"],
            "node_viability": topology["node_viability"],
            "edge_viability": topology["edge_viability"],
            "edge_viability_raw": topology["edge_viability_raw"],
            "horizon_context": horizon_context,
            "horizon_role_weights": horizon_role_weights,
        }

    def forward(
        self,
        batch: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Run complete K=6 route-to-trajectory forecasting."""
        scene = self.encode_scene(
            batch,
            return_aux=False,
        )

        route = self.route_queries(
            ego_context=scene["ego_context"],
            lane_context=scene["lane_context"],
            horizon_context=scene["horizon_context"],
            lane_states=scene["lane_states"],
            lane_mask=scene["lane_mask"],
            lane_centerline=scene["lane_centerline"],
            lane_point_mask=scene["lane_point_mask"],

            # V3: route generation operates directly on the learned
            # WorkZone-conditioned temporary graph.
            node_viability=scene["node_viability"],
            edge_viability=scene["edge_viability"],
            lane_edge_index=batch["lane_edge_index"],
            lane_edge_mask=batch["lane_edge_mask"],
        )

        # --------------------------------------------------------------
        # V3 ROUTE PROGRESS
        # --------------------------------------------------------------

        progress = self.route_progress(
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

        route_anchors = progress[
            "route_progress_anchors"
        ]

        dynamics_xy = self.dynamics_anchor(
            scene["ego_context"],
            scene["control_context"],
        )

        direct_decoded: dict[str, torch.Tensor] | None = None

        if self.direct_trajectory_decoder is not None:

            direct_decoded = self.direct_trajectory_decoder(
                ego_context=scene["ego_context"],
                horizon_context=scene["horizon_context"],
                lane_states=scene["lane_states"],
                lane_mask=scene["lane_mask"],
                agent_states=scene["agent_states"],
                agent_mask=scene["agent_mask"],
                worker_tokens=scene["worker_tokens"],
                worker_mask=scene["worker_token_mask"],
                wz_feat=scene["effective_wz_feat"],
                dynamics_xy=dynamics_xy,
            )

        decoded = self.trajectory_decoder(
            route["mode_context"],
            scene["horizon_context"],
            route_anchors,
            dynamics_xy,
            route_guide=progress["dense_route_guide"],
        )
        coarse_xy = decoded[
            "coarse_xy"
        ]

        refinement_delta: torch.Tensor | None = None


        refinement_sd: torch.Tensor | None = None
        if self.local_refiner is not None:
            scene_tokens = torch.cat(
                (
                    scene["lane_states"],
                    scene["worker_tokens"],
                    scene["agent_states"],
                ),
                dim=1,
            )

            scene_xy = torch.cat(
                (
                    scene["lane_xy"],
                    scene["worker_token_xy"],
                    scene["agent_xy"],
                ),
                dim=1,
            )

            scene_mask = torch.cat(
                (
                    scene["lane_mask"],
                    scene["worker_token_mask"],
                    scene["agent_mask"],
                ),
                dim=1,
            )

            refined = self.local_refiner(
                coarse_xy=coarse_xy,
                mode_context=route["mode_context"],
                route_guide=decoded["route_guide"],
                route_tangent=decoded["route_tangent"],
                route_normal=decoded["route_normal"],
                scene_tokens=scene_tokens,
                scene_xy=scene_xy,
                scene_mask=scene_mask,
            )

            refinement_delta = refined[
                "refinement_delta"
            ]

            refinement_sd = refined[
                "refinement_sd"
            ]

            pred_xy = (
                coarse_xy
                +
                refinement_delta
            )

        else:
            pred_xy = coarse_xy

        # ==========================================================
        # DIRECT K=6 CARTESIAN BYPASS
        #
        # Route/topology remains available below for diagnostics and optional
        # auxiliary objectives. It no longer determines the forecast.
        # ==========================================================

        if direct_decoded is not None:
            pred_xy = direct_decoded["pred_xy"]

        # ==========================================================
        # V3 QUALITY-AWARE MODE RANKING
        #
        # route_safety_features:
        #     safety of the route proposal itself
        #
        # trajectory_safety_features:
        #     safety of the final decoded trajectory
        # ==========================================================

        route_safety_features = self._safety_features(
            progress["dense_route_guide"],
            route["goal_prob"],
            scene["effective_wz_feat"],
            scene["effective_worker_feat"],
        )

        trajectory_safety_features = self._safety_features(
            pred_xy,
            route["goal_prob"],
            scene["effective_wz_feat"],
            scene["effective_worker_feat"],
        )

        (
            behavior_features,
            quality_features,
            route_prior,
        ) = self._ranking_feature_sets(
            pred_xy=pred_xy,
            dynamics_xy=dynamics_xy,
            route_guide=decoded["route_guide"],
            route_tangent=decoded["route_tangent"],
            route_normal=decoded["route_normal"],
            route_viability=route["route_viability"],
            goal_prob=route["goal_prob"],
            progress_increment=progress["progress_increment"],
            route_safety_features=route_safety_features,
            trajectory_safety_features=trajectory_safety_features,
        )

        ranking = self.mode_ranker(
            mode_context=route["mode_context"],
            behavior_features=behavior_features,
            quality_features=quality_features,
            route_prior=route_prior,
        )

        # Evaluator compatibility:
        # mode_logits / mode_prob now mean FINAL ranking score/probability.
        mode_logits = ranking[
            "ranking_logits"
        ]

        mode_prob = ranking[
            "mode_prob"
        ]

        if direct_decoded is not None:
            mode_logits = direct_decoded["mode_logits"]
            mode_prob = direct_decoded["mode_prob"]

        safety_features = (
            trajectory_safety_features
        )

        output = {
            "pred_xy": pred_xy,
            "mode_logits": mode_logits,
            "mode_prob": mode_prob,

            # V3 separated ranking concepts.
            "behavior_logits": ranking["behavior_logits"],
            "behavior_prob": ranking["behavior_prob"],
            "quality_score": ranking["quality_score"],
            "ranking_logits": ranking["ranking_logits"],
            "ranking_quality_alpha": ranking["quality_alpha"],
            "route_prior": route_prior,
            "behavior_ranking_features": behavior_features,
            "quality_ranking_features": quality_features,
            "route_safety_features": route_safety_features,

            "lane_mask": scene["lane_mask"],

            "coarse_xy": coarse_xy,
            "dynamics_xy": dynamics_xy,

            "direct_delta": (
                direct_decoded["direct_delta"]
                if direct_decoded is not None
                else torch.zeros_like(pred_xy)
            ),

            "direct_residual_delta": (
                direct_decoded["direct_residual_delta"]
                if direct_decoded is not None
                else torch.zeros_like(pred_xy)
            ),

            "direct_pre_repair_delta": (
                direct_decoded["direct_pre_repair_delta"]
                if direct_decoded is not None
                else torch.zeros_like(pred_xy)
            ),

            "direct_pre_repair_pred_xy": (
                direct_decoded["direct_pre_repair_pred_xy"]
                if direct_decoded is not None
                else torch.zeros_like(pred_xy)
            ),

            "direct_longitudinal_repair_delta": (
                direct_decoded["direct_longitudinal_repair_delta"]
                if direct_decoded is not None
                else torch.zeros_like(pred_xy)
            ),

            # Dense hard-route progress diagnostics.
            "route_progress_sequence": progress[
                "route_progress_sequence"
            ],
            "dense_progress_increment": progress[
                "dense_progress_increment"
            ],
            "legacy_route_progress_sequence": progress[
                "legacy_route_progress_sequence"
            ],
            "hard_route_geometry_context": progress[
                "hard_route_geometry_context"
            ],

            "route_gate": decoded["route_gate"],
            "structural_route_gate": decoded["structural_route_gate"],
            "learned_route_gate": decoded["learned_route_gate"],

            # V3 route-relative trajectory diagnostics.
            "dense_route_guide": decoded["route_guide"],
            "route_tangent": decoded["route_tangent"],
            "route_normal": decoded["route_normal"],

            "trajectory_residual_sd": decoded[
                "trajectory_residual_sd"
            ],

            "route_longitudinal_offset": decoded[
                "route_longitudinal_offset"
            ],

            "route_lateral_offset": decoded[
                "route_lateral_offset"
            ],

            "route_anchors": route_anchors,

            # V3 progress diagnostics.
            "route_progress": progress["route_progress"],
            "progress_increment": progress["progress_increment"],
            "route_walk_xy": progress["route_walk_xy"],
            "route_walk_s": progress["route_walk_s"],
            "route_terminal_tangent": progress["route_terminal_tangent"],
            "route_total_length": progress["route_total_length"],

            "goal_logits": route["goal_logits"],
            "goal_prob": route["goal_prob"],

            "mode_context": route["mode_context"],

            "node_viability": scene["node_viability"],
            "edge_viability": scene["edge_viability"],
            "edge_viability_raw": scene["edge_viability_raw"],

            # V3 explicit route hypotheses.
            "route_node_logits": route["route_node_logits"],
            "route_node_occupancy": route["route_node_occupancy"],
            "route_edge_logits": route["route_edge_logits"],
            "route_edge_occupancy": route["route_edge_occupancy"],
            "route_embedding": route["route_embedding"],
            "route_viability": route["route_viability"],

            "control_gate": scene["control_gate"],
            "gaze_reliability": scene["gaze_reliability"],
            "horizon_role_weights": scene["horizon_role_weights"],

            "safety_features": safety_features,
        }

        if refinement_delta is not None:
            output[
                "refinement_delta"
            ] = refinement_delta

        if refinement_sd is not None:
            output[
                "refinement_sd"
            ] = refinement_sd

        return output
