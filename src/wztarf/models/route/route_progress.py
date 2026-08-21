"""Monotonic route-progress prediction over lane geometry."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .route_geometry import _interpolate_polyline, _lane_geometry

class DifferentiableRouteProgress(nn.Module):
    """Convert soft graph routes into monotonic 1/3/5 s progress anchors."""

    def __init__(
        self,
        d_model: int = 128,
        num_horizons: int = 3,
        future_steps: int = 25,
        fps: int = 5,
        walk_steps: int = 10,
        samples_per_lane: int = 4,
        progress_scale_m: float = 8.0,
        exit_extension_m: float = 60.0,
        start_temperature_m: float = 4.0,
        use_dense_progress_repair: bool = True,
    ) -> None:
        super().__init__()

        if walk_steps < 2:
            raise ValueError(
                "walk_steps must be at least 2."
            )

        if samples_per_lane < 2:
            raise ValueError(
                "samples_per_lane must be at least 2."
            )

        if progress_scale_m <= 0.0:
            raise ValueError(
                "progress_scale_m must be positive."
            )

        if exit_extension_m <= 0.0:
            raise ValueError(
                "exit_extension_m must be positive."
            )

        self.d_model = d_model
        self.num_horizons = num_horizons
        self.future_steps = future_steps
        self.fps = fps
        self.use_dense_progress_repair = bool(use_dense_progress_repair)
        self.walk_steps = walk_steps
        self.samples_per_lane = samples_per_lane

        self.progress_scale_m = float(
            progress_scale_m
        )

        self.exit_extension_m = float(
            exit_extension_m
        )

        self.start_temperature_m = float(
            start_temperature_m
        )

        self.progress_head = nn.Sequential(
            nn.Linear(
                d_model,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                num_horizons,
            ),
        )

        # Start neutral:
        # softplus(0) * 8 ~= 5.55 m per positive increment.
        # Tiny nonzero initialization preserves the neutral
        # softplus(0) progress regime while allowing gradients to reach
        # the first MLP layer on the first optimizer step.
        nn.init.normal_(
            self.progress_head[
                -1
            ].weight,
            mean=0.0,
            std=1.0e-3,
        )

        nn.init.zeros_(
            self.progress_head[
                -1
            ].bias
        )

        # ==========================================================
        # V3 DENSE HARD-ROUTE PROGRESS REPAIR
        #
        # The legacy 1/3/5-second progress head is retained as a
        # checkpoint-compatible coarse prior.
        #
        # A new 25-step residual progress model is conditioned on the
        # ACTUAL generated hard route geometry rather than relying only
        # on the nearly-collapsed soft route embedding.
        #
        # geometry feature:
        #   5 sampled hard-route XY points = 10
        #   physical route length          =  1
        #   terminal tangent               =  2
        #                                  ----
        #                                    13
        # ==========================================================

        if self.use_dense_progress_repair:
            self.hard_route_geometry_encoder = nn.Sequential(
                nn.Linear(13, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            self.dense_progress_fusion = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.ReLU(),
                nn.LayerNorm(d_model),
            )
            self.dense_progress_residual_head = nn.Linear(
                d_model,
                future_steps,
            )

            # Historical HEADONLY construction added these 10 tensors fresh.
            # Zero initialization preserves the exact legacy progress sequence
            # at the 247 -> 257 architecture boundary.
            nn.init.zeros_(self.dense_progress_residual_head.weight)
            nn.init.zeros_(self.dense_progress_residual_head.bias)
        else:
            # Historical Phase A/B + ProgressFix architecture.  Register no
            # dense-progress modules at all: 247 tensors / 2,243,801 params.
            self.hard_route_geometry_encoder = None
            self.dense_progress_fusion = None
            self.dense_progress_residual_head = None

        # residual log-scale is bounded:
        #
        #   exp(-3) ~= 0.050
        #   exp(+3) ~= 20.086
        #
        # This is wide enough to represent near-stops as well as rapid
        # travel without allowing unconstrained exponential growth.
        self.dense_progress_log_scale_limit = 3.0

    def forward(
        self,
        *,
        mode_context: torch.Tensor,
        route_node_occupancy: torch.Tensor,
        route_edge_occupancy: torch.Tensor,
        edge_viability: torch.Tensor,
        lane_edge_index: torch.Tensor,
        lane_edge_type: torch.Tensor,
        lane_edge_mask: torch.Tensor,
        lane_centerline: torch.Tensor,
        lane_point_mask: torch.Tensor,
        lane_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # ==========================================================
        # V3 FP32 ROUTE GEOMETRY
        #
        # Occupancy probabilities and polyline geometry stay FP32 even
        # when the surrounding model is under BF16 autocast.
        # ==========================================================
        route_node_occupancy = route_node_occupancy.float()
        route_edge_occupancy = route_edge_occupancy.float()
        edge_viability = edge_viability.float()
        lane_centerline = lane_centerline.float()


        (
            batch_size,
            num_modes,
            d_model,
        ) = mode_context.shape

        num_lanes = (
            lane_centerline.shape[1]
        )

        num_edges = (
            lane_edge_index.shape[-1]
        )

        if d_model != self.d_model:
            raise ValueError(
                "mode_context dimension mismatch."
            )

        if route_node_occupancy.shape != (
            batch_size,
            num_modes,
            num_lanes,
        ):
            raise ValueError(
                "route_node_occupancy must have shape [B,K,L]."
            )

        if route_edge_occupancy.shape != (
            batch_size,
            num_modes,
            num_edges,
        ):
            raise ValueError(
                "route_edge_occupancy must have shape [B,K,E]."
            )

        lane_mask_bool = (
            lane_mask.bool()
        )

        edge_mask_bool = (
            lane_edge_mask.bool()
        )

        edge_type = lane_edge_type.long()

        if edge_type.shape != (
            batch_size,
            num_edges,
        ):
            raise ValueError(
                "lane_edge_type must have shape [B,E]."
            )

        (
            lane_samples,
            lane_tangent,
            origin_distance,
        ) = _lane_geometry(
            lane_centerline.float(),
            lane_point_mask,
            lane_mask_bool,
            samples_per_lane=self.samples_per_lane,
        )

        lane_samples = lane_samples.to(
            mode_context.dtype
        )

        lane_tangent = lane_tangent.to(
            mode_context.dtype
        )

        origin_distance = origin_distance.to(
            mode_context.dtype
        )

        # ======================================================
        # ROOT-ONLY GEOMETRY
        #
        # Successor lanes use full canonical upstream->downstream
        # geometry from _lane_geometry().
        #
        # Only the first/root lane begins at its represented point
        # nearest the ego.
        # ======================================================

        lane32_root = lane_centerline.float()

        root_valid_point = (
            lane_point_mask.bool()
            &
            lane_mask_bool[
                :,
                :,
                None,
            ]
        )

        root_num_points = lane32_root.shape[2]

        root_point_index = torch.arange(
            root_num_points,
            device=lane32_root.device,
        ).view(
            1,
            1,
            root_num_points,
        )

        root_first_index = torch.where(
            root_valid_point,
            root_point_index,
            torch.full_like(
                root_point_index,
                root_num_points,
            ),
        ).amin(
            dim=-1
        )

        root_last_index = torch.where(
            root_valid_point,
            root_point_index,
            torch.full_like(
                root_point_index,
                -1,
            ),
        ).amax(
            dim=-1
        )

        root_safe_first = root_first_index.clamp(
            min=0,
            max=root_num_points - 1,
        )

        root_safe_last = root_last_index.clamp(
            min=0,
            max=root_num_points - 1,
        )

        root_point_distance = (
            torch.linalg.vector_norm(
                lane32_root,
                dim=-1,
            )
        ).masked_fill(
            ~root_valid_point,
            float("inf"),
        )

        root_closest_index = (
            root_point_distance.argmin(
                dim=-1
            )
        )

        root_closest_index = torch.maximum(
            root_closest_index,
            root_safe_first,
        )

        root_closest_index = torch.minimum(
            root_closest_index,
            root_safe_last,
        )

        def _gather_root_point(
            index: torch.Tensor,
        ) -> torch.Tensor:

            return lane32_root.gather(
                2,
                index[
                    ...,
                    None,
                    None,
                ].expand(
                    batch_size,
                    num_lanes,
                    1,
                    2,
                ),
            ).squeeze(
                2
            )

        root_first_xy = _gather_root_point(
            root_safe_first
        )

        root_last_xy = _gather_root_point(
            root_safe_last
        )

        # lane_samples is already oriented upstream -> downstream.
        root_downstream_xy = (
            lane_samples[
                :,
                :,
                -1,
                :
            ].float()
        )

        root_downstream_is_last = (
            torch.linalg.vector_norm(
                root_downstream_xy
                -
                root_last_xy,
                dim=-1,
            )
            <=
            torch.linalg.vector_norm(
                root_downstream_xy
                -
                root_first_xy,
                dim=-1,
            )
        )

        root_downstream_index = torch.where(
            root_downstream_is_last,
            root_safe_last,
            root_safe_first,
        )

        root_fraction = torch.linspace(
            0.0,
            1.0,
            self.samples_per_lane,
            dtype=torch.float32,
            device=lane32_root.device,
        ).view(
            1,
            1,
            self.samples_per_lane,
        )

        root_sample_position = (
            root_closest_index[
                :,
                :,
                None,
            ].float()
            +
            root_fraction
            *
            (
                root_downstream_index
                -
                root_closest_index
            )[
                :,
                :,
                None,
            ].float()
        )

        root_sample_index = torch.round(
            root_sample_position
        ).long().clamp(
            min=0,
            max=root_num_points - 1,
        )

        root_lane_samples = lane32_root.gather(
            2,
            root_sample_index[
                ...,
                None,
            ].expand(
                batch_size,
                num_lanes,
                self.samples_per_lane,
                2,
            ),
        ).to(
            mode_context.dtype
        )

        src = lane_edge_index[
            :,
            0,
        ].long()

        dst = lane_edge_index[
            :,
            1,
        ].long()

        # Only true forward-successor edges belong in a
        # forward route walk.
        #
        # Geometry audit of this dataset:
        #   type 0 -> forward successor
        #   type 1 -> reverse/predecessor
        valid_edge = (
            edge_mask_bool
            &
            (edge_type == 0)
            &
            (src >= 0)
            &
            (dst >= 0)
            &
            (src < num_lanes)
            &
            (dst < num_lanes)
        )

        safe_src = src.clamp(
            0,
            max(
                num_lanes - 1,
                0,
            ),
        )

        safe_dst = dst.clamp(
            0,
            max(
                num_lanes - 1,
                0,
            ),
        )

        valid_edge &= (
            lane_mask_bool.gather(
                1,
                safe_src,
            )
            &
            lane_mask_bool.gather(
                1,
                safe_dst,
            )
        )

        # ======================================================
        # TYPE01 AUTO-ORIENTED PHYSICAL GRAPH
        #
        # Edge type 0 = successor.
        # Edge type 1 = reverse/predecessor representation.
        #
        # They describe the same physical lane-pair relationship.
        #
        # Tournament winner:
        #   type01_auto_oriented
        #   edge P50 = 0.23m
        #   greedy 5s = 0.71m
        #   greedy within5 = 100%
        # ======================================================

        raw_src = lane_edge_index[
            :,
            0,
            :,
        ].long()

        raw_dst = lane_edge_index[
            :,
            1,
            :,
        ].long()

        raw_in_bounds = (
            (raw_src >= 0)
            &
            (raw_src < num_lanes)
            &
            (raw_dst >= 0)
            &
            (raw_dst < num_lanes)
        )

        raw_safe_src = raw_src.clamp(
            min=0,
            max=num_lanes - 1,
        )

        raw_safe_dst = raw_dst.clamp(
            min=0,
            max=num_lanes - 1,
        )

        relation_valid = (
            edge_mask_bool
            &
            raw_in_bounds
            &
            (
                (edge_type == 0)
                |
                (edge_type == 1)
            )
            &
            lane_mask_bool.gather(
                1,
                raw_safe_src,
            )
            &
            lane_mask_bool.gather(
                1,
                raw_safe_dst,
            )
        )

        # ------------------------------------------------------
        # Deduplicate each unordered physical lane pair.
        #
        # If both type0/type1 records exist, prefer type0 so the
        # learned raw edge feature corresponds to the canonical
        # successor record whenever possible.
        # ------------------------------------------------------

        pair_lo = torch.minimum(
            raw_safe_src,
            raw_safe_dst,
        )

        pair_hi = torch.maximum(
            raw_safe_src,
            raw_safe_dst,
        )

        pair_key = (
            pair_lo
            *
            num_lanes
            +
            pair_hi
        )

        preference = (
            edge_type != 0
        ).long()

        max_pair_key = (
            num_lanes
            *
            num_lanes
        )

        sort_key = (
            pair_key
            *
            2
            +
            preference
            +
            (
                ~relation_valid
            ).long()
            *
            (
                2
                *
                max_pair_key
                +
                4
            )
        )

        pair_order = torch.argsort(
            sort_key,
            dim=1,
        )

        sorted_pair_key = pair_key.gather(
            1,
            pair_order,
        )

        sorted_relation_valid = relation_valid.gather(
            1,
            pair_order,
        )

        first_for_pair = (
            sorted_relation_valid.clone()
        )

        if num_edges > 1:

            first_for_pair[
                :,
                1:
            ] = (
                sorted_relation_valid[
                    :,
                    1:
                ]
                &
                (
                    (
                        sorted_pair_key[
                            :,
                            1:
                        ]
                        !=
                        sorted_pair_key[
                            :,
                            :-1
                        ]
                    )
                    |
                    (
                        ~sorted_relation_valid[
                            :,
                            :-1
                        ]
                    )
                )
            )

        keep_physical_edge = torch.zeros_like(
            relation_valid
        )

        keep_physical_edge.scatter_(
            1,
            pair_order,
            first_for_pair,
        )

        # ------------------------------------------------------
        # +X canonical orientation.
        #
        # lane_samples:
        #   [:,:,0]  = +X upstream
        #   [:,:,-1] = +X downstream
        # ------------------------------------------------------

        lane_upstream_xy = (
            lane_samples[
                :,
                :,
                0,
                :
            ].float()
        )

        lane_downstream_xy = (
            lane_samples[
                :,
                :,
                -1,
                :
            ].float()
        )

        def _gather_canonical_lane(
            point_tensor: torch.Tensor,
            index: torch.Tensor,
        ) -> torch.Tensor:

            return point_tensor.gather(
                1,
                index[
                    ...,
                    None,
                ].expand(
                    batch_size,
                    num_edges,
                    2,
                ),
            )

        raw_src_downstream = _gather_canonical_lane(
            lane_downstream_xy,
            raw_safe_src,
        )

        raw_dst_upstream = _gather_canonical_lane(
            lane_upstream_xy,
            raw_safe_dst,
        )

        raw_dst_downstream = _gather_canonical_lane(
            lane_downstream_xy,
            raw_safe_dst,
        )

        raw_src_upstream = _gather_canonical_lane(
            lane_upstream_xy,
            raw_safe_src,
        )

        gap_src_to_dst = torch.linalg.vector_norm(
            raw_src_downstream
            -
            raw_dst_upstream,
            dim=-1,
        )

        gap_dst_to_src = torch.linalg.vector_norm(
            raw_dst_downstream
            -
            raw_src_upstream,
            dim=-1,
        )

        canonical_src_to_dst = (
            gap_src_to_dst
            <=
            gap_dst_to_src
        )

        safe_src = torch.where(
            canonical_src_to_dst,
            raw_safe_src,
            raw_safe_dst,
        )

        safe_dst = torch.where(
            canonical_src_to_dst,
            raw_safe_dst,
            raw_safe_src,
        )

        canonical_gap = torch.where(
            canonical_src_to_dst,
            gap_src_to_dst,
            gap_dst_to_src,
        )

        valid_edge = keep_physical_edge

        # Continuity prior.
        #
        # 0.2-0.5m physical joins remain ~1.
        # large 20-50m false joins are strongly suppressed.
        edge_geometry_prior = (
            torch.exp(
                -
                canonical_gap
                /
                2.0
            )
            *
            valid_edge.to(
                torch.float32
            )
        )
        src_index = safe_src[
            :,
            None,
            :,
        ].expand(
            batch_size,
            num_modes,
            num_edges,
        )

        dst_index = safe_dst[
            :,
            None,
            :,
        ].expand(
            batch_size,
            num_modes,
            num_edges,
        )

        src_support = (
            route_node_occupancy.gather(
                2,
                src_index,
            )
        )

        dst_support = (
            route_node_occupancy.gather(
                2,
                dst_index,
            )
        )

        # ----------------------------------------------------------
        # Differentiable route transition probability.
        #
        # Route membership ? temporary topology ? incident-node support.
        # ----------------------------------------------------------

        edge_weight = (
            route_edge_occupancy
            *
            edge_viability[
                :,
                None,
                :,
            ].to(
                route_edge_occupancy.dtype
            )
            *
            torch.sqrt(
                (
                    src_support
                    *
                    dst_support
                ).clamp_min(
                    1.0e-8
                )
            )
            *
            valid_edge[
                :,
                None,
                :,
            ].to(
                route_edge_occupancy.dtype
            )
        )

        edge_weight = (
            edge_weight
            *
            edge_geometry_prior[
                :,
                None,
                :
            ].to(
                edge_weight.dtype
            )
        )

        flat_transition = (
            mode_context.new_zeros(
                batch_size,
                num_modes,
                num_lanes
                *
                num_lanes,
            )
        )

        flat_edge_index = (
            safe_src
            *
            num_lanes
            +
            safe_dst
        )[
            :,
            None,
            :,
        ].expand(
            batch_size,
            num_modes,
            num_edges,
        )

        flat_transition.scatter_add_(
            2,
            flat_edge_index,
            edge_weight.to(
                flat_transition.dtype
            ),
        )

        transition = (
            flat_transition.reshape(
                batch_size,
                num_modes,
                num_lanes,
                num_lanes,
            )
        )

        row_sum = transition.sum(
            dim=-1,
            keepdim=True,
        )

        identity = torch.eye(
            num_lanes,
            dtype=transition.dtype,
            device=transition.device,
        ).view(
            1,
            1,
            num_lanes,
            num_lanes,
        )

        # ==========================================================
        # PERMANENT-GRAPH FALLBACK
        #
        # Learned WZ topology reweights/suppresses represented
        # forward-successor edges. But if the learned gate collapses
        # numerically to zero, that must NOT turn every lane into a
        # fake dead end.
        #
        # Build a static transition using the valid type-0 permanent
        # graph. True graph dead ends alone fall back to identity.
        # ==========================================================

        # ======================================================
        # ROUTE-SCORE-PRESERVING PERMANENT GRAPH FALLBACK
        #
        # If WZ topology viability collapses, permanent connectivity
        # remains available and each route mode still gets to rank its
        # own outgoing type-0 successor edges.
        # ======================================================

        static_edge_weight = (
            route_edge_occupancy.to(
                flat_transition.dtype
            )
            *
            valid_edge[
                :,
                None,
                :,
            ].to(
                flat_transition.dtype
            )
        )

        static_edge_weight = (
            static_edge_weight
            *
            edge_geometry_prior[
                :,
                None,
                :
            ].to(
                static_edge_weight.dtype
            )
        )

        static_flat = torch.zeros_like(
            flat_transition
        )

        static_flat.scatter_add_(
            2,
            flat_edge_index,
            static_edge_weight,
        )

        static_transition = static_flat.reshape(
            batch_size,
            num_modes,
            num_lanes,
            num_lanes,
        )

        static_row_sum = static_transition.sum(
            dim=-1,
            keepdim=True,
        )

        # Last-resort uniform permanent transition if even the
        # route-edge scores provide zero mass.
        uniform_flat = torch.zeros_like(
            flat_transition
        )

        uniform_weight = (
            valid_edge[
                :,
                None,
                :,
            ]
            .expand(
                batch_size,
                num_modes,
                num_edges,
            )
            .to(
                uniform_flat.dtype
            )
        )

        uniform_flat.scatter_add_(
            2,
            flat_edge_index,
            uniform_weight,
        )

        uniform_transition = uniform_flat.reshape(
            batch_size,
            num_modes,
            num_lanes,
            num_lanes,
        )

        uniform_row_sum = uniform_transition.sum(
            dim=-1,
            keepdim=True,
        )

        uniform_transition = torch.where(
            uniform_row_sum > 1.0e-8,
            uniform_transition
            /
            uniform_row_sum.clamp_min(
                1.0e-8
            ),
            identity,
        )

        static_transition = torch.where(
            static_row_sum > 1.0e-8,
            static_transition
            /
            static_row_sum.clamp_min(
                1.0e-8
            ),
            uniform_transition,
        )
        # Use learned temporary topology whenever it supplies a
        # usable transition. Otherwise retain permanent-map
        # connectivity rather than repeatedly traversing one lane.
        transition = torch.where(
            row_sum > 1.0e-8,
            transition
            /
            row_sum.clamp_min(
                1.0e-8
            ),
            static_transition,
        )

        # ----------------------------------------------------------
        # Start distribution:
        #
        # FINAL K=6 ROOT POLICY
        # ---------------------
        # Each mode receives one distinct physically plausible
        # lane root. Lanes within 10 m of the ego are prioritized,
        # ordered by actual minimum centerline distance.
        #
        # 128-scene validation capacity:
        #   K=1 nearest : 3.76 m @5s
        #   K=6 <=10 m : 1.02 m @5s, 98.4% within 5 m
        #   all <=10 m  : 0.80 m @5s
        #
        # Therefore multimodality begins at root selection instead
        # of making all six hypotheses compete for one soft root.
        # ----------------------------------------------------------

        if num_modes > num_lanes:
            raise ValueError(
                "num_modes cannot exceed num_lanes."
            )

        valid_root = (
            lane_mask_bool
            &
            torch.isfinite(
                origin_distance
            )
            &
            (
                origin_distance
                <
                1.0e5
            )
        )

        near_root = (
            valid_root
            &
            (
                origin_distance
                <=
                10.0
            )
        )

        # <=10m valid lanes always rank before farther valid lanes.
        # Invalid lanes rank last.
        root_priority = (
            origin_distance.float()
            +
            (
                ~near_root
            ).to(
                torch.float32
            )
            *
            1.0e4
            +
            (
                ~valid_root
            ).to(
                torch.float32
            )
            *
            1.0e8
        )

        root_index = torch.argsort(
            root_priority,
            dim=-1,
        )[
            :,
            :num_modes,
        ]

        # Extremely defensive fallback for malformed scenes.
        nearest_valid_index = (
            origin_distance
            .masked_fill(
                ~valid_root,
                float("inf"),
            )
            .argmin(
                dim=-1
            )
        )

        selected_valid = valid_root.gather(
            1,
            root_index,
        )

        root_index = torch.where(
            selected_valid,
            root_index,
            nearest_valid_index[
                :,
                None,
            ],
        )

        start = torch.zeros(
            batch_size,
            num_modes,
            num_lanes,
            dtype=mode_context.dtype,
            device=mode_context.device,
        )

        start.scatter_(
            2,
            root_index[
                ...,
                None,
            ],
            1.0,
        )
        # ======================================================
        # CONNECTED STRAIGHT-THROUGH GRAPH WALK
        #
        # Forward geometry:
        #   one actual lane per route mode at every graph step.
        #
        # Backward:
        #   retain gradients through the soft distribution.
        #
        # This prevents spatial averaging between unrelated branch
        # lanes while preserving trainability.
        # ======================================================

        route_point_groups = [
            mode_context.new_zeros(
                batch_size,
                num_modes,
                1,
                2,
            )
        ]

        node_distribution = start

        previous_lane_index = None

        for step in range(
            self.walk_steps
        ):

            lane_index = (
                node_distribution.argmax(
                    dim=-1,
                    keepdim=True,
                )
            )

            hard_node = torch.zeros_like(
                node_distribution
            ).scatter_(
                -1,
                lane_index,
                1.0,
            )

            # Straight-through estimator.
            walk_node = (
                hard_node
                +
                node_distribution
                -
                node_distribution.detach()
            )

            geometry_samples = (
                root_lane_samples
                if step == 0
                else lane_samples
            )

            expected_samples = torch.einsum(
                "bkl,blpd->bkpd",
                walk_node,
                geometry_samples,
            )

            current_lane_index = (
                lane_index.squeeze(
                    -1
                )
            )

            # A graph dead end becomes an identity transition.
            # Do not draw the same entire lane again at every walk
            # step; stay at its terminal point instead.
            if previous_lane_index is not None:

                repeated_lane = (
                    current_lane_index
                    ==
                    previous_lane_index
                )

                repeated_terminal = (
                    expected_samples[
                        :,
                        :,
                        -1:,
                        :
                    ].expand_as(
                        expected_samples
                    )
                )

                expected_samples = torch.where(
                    repeated_lane[
                        :,
                        :,
                        None,
                        None,
                    ],
                    repeated_terminal,
                    expected_samples,
                )

            route_point_groups.append(
                expected_samples
            )

            previous_lane_index = (
                current_lane_index
            )

            # Keep final node state hard in forward pass so later
            # terminal tangent/extension belongs to a real lane.
            node_distribution = walk_node

            if (
                step + 1
                <
                self.walk_steps
            ):

                next_distribution = torch.einsum(
                    "bki,bkij->bkj",
                    walk_node,
                    transition,
                )

                next_mass = next_distribution.sum(
                    dim=-1,
                    keepdim=True,
                )

                node_distribution = torch.where(
                    next_mass > 1.0e-8,
                    next_distribution
                    /
                    next_mass.clamp_min(
                        1.0e-8
                    ),
                    walk_node,
                )
        route_walk_xy = torch.cat(
            route_point_groups,
            dim=2,
        )

        expected_terminal_tangent = (
            torch.einsum(
                "bkl,bld->bkd",
                node_distribution,
                lane_tangent,
            )
        )

        geometric_tangent = (
            route_walk_xy[
                :,
                :,
                -1,
            ]
            -
            route_walk_xy[
                :,
                :,
                -2,
            ]
        )

        geometric_norm = (
            torch.linalg.vector_norm(
                geometric_tangent,
                dim=-1,
                keepdim=True,
            )
        )

        tangent = torch.where(
            geometric_norm > 1.0e-5,
            geometric_tangent
            /
            geometric_norm.clamp_min(
                1.0e-6
            ),
            expected_terminal_tangent,
        )

        tangent_norm = (
            torch.linalg.vector_norm(
                tangent,
                dim=-1,
                keepdim=True,
            )
        )

        fallback = torch.zeros_like(
            tangent
        )

        fallback[
            ...,
            0
        ] = 1.0

        tangent = torch.where(
            tangent_norm > 1.0e-5,
            tangent
            /
            tangent_norm.clamp_min(
                1.0e-6
            ),
            fallback,
        )

        # ----------------------------------------------------------
        # MAP_EXIT continuation.
        #
        # The route polyline always extends beyond the retained graph
        # using the route's terminal tangent.
        # ----------------------------------------------------------

        extension = (
            route_walk_xy[
                :,
                :,
                -1,
            ]
            +
            self.exit_extension_m
            *
            tangent
        )

        route_walk_xy = torch.cat(
            (
                route_walk_xy,
                extension[
                    :,
                    :,
                    None,
                    :,
                ],
            ),
            dim=2,
        )

        segment = (
            route_walk_xy[
                :,
                :,
                1:,
            ]
            -
            route_walk_xy[
                :,
                :,
                :-1,
            ]
        )

        segment_length = (
            torch.linalg.vector_norm(
                segment.float(),
                dim=-1,
            ).to(
                route_walk_xy.dtype
            )
        )

        route_walk_s = torch.cat(
            (
                segment_length.new_zeros(
                    batch_size,
                    num_modes,
                    1,
                ),
                torch.cumsum(
                    segment_length,
                    dim=-1,
                ),
            ),
            dim=-1,
        )

        # ==========================================================
        # V3 DENSE HARD-ROUTE PROGRESS
        # ==========================================================
        #
        # Stage 1:
        #   retain the legacy positive 1/3/5-second progress predictor
        #   as a coarse prior so pretrained checkpoints remain useful.
        #
        # Stage 2:
        #   reconstruct the exact legacy 25-step piecewise-linear
        #   progress sequence.
        #
        # Stage 3:
        #   encode the actual generated hard route geometry.
        #
        # Stage 4:
        #   predict a bounded per-timestep multiplicative correction to
        #   the legacy positive increments.
        #
        # Result:
        #   25 independent positive increments => monotonic dense s(t).
        # ==========================================================

        legacy_progress_increment = (
            F.softplus(
                self.progress_head(
                    mode_context
                ).float()
            )
            *
            self.progress_scale_m
        ).to(
            mode_context.dtype
        )

        legacy_route_progress = torch.cumsum(
            legacy_progress_increment,
            dim=-1,
        )

        # ----------------------------------------------------------
        # Reconstruct the exact old 0/1/3/5-second interpolation.
        # ----------------------------------------------------------

        future_time = (
            torch.arange(
                1,
                self.future_steps + 1,
                dtype=torch.float32,
                device=mode_context.device,
            )
            /
            float(
                self.fps
            )
        )

        control_time = torch.tensor(
            [
                0.0,
                1.0,
                3.0,
                5.0,
            ],
            dtype=torch.float32,
            device=mode_context.device,
        )

        progress_control = torch.cat(
            (
                legacy_route_progress.new_zeros(
                    batch_size,
                    num_modes,
                    1,
                ),
                legacy_route_progress,
            ),
            dim=-1,
        )

        right = torch.bucketize(
            future_time,
            control_time,
            right=False,
        ).clamp(
            1,
            control_time.numel() - 1,
        )

        left = (
            right
            -
            1
        )

        left_time = control_time.index_select(
            0,
            left,
        )

        right_time = control_time.index_select(
            0,
            right,
        )

        alpha = (
            (
                future_time
                -
                left_time
            )
            /
            (
                right_time
                -
                left_time
            ).clamp_min(
                1.0e-6
            )
        ).to(
            legacy_route_progress.dtype
        )

        left_progress = (
            progress_control.index_select(
                2,
                left,
            )
        )

        right_progress = (
            progress_control.index_select(
                2,
                right,
            )
        )

        legacy_route_progress_sequence = (
            left_progress
            +
            alpha[
                None,
                None,
                :,
            ]
            *
            (
                right_progress
                -
                left_progress
            )
        )

        # ----------------------------------------------------------
        # Encode ACTUAL hard-route geometry.
        #
        # Do not backpropagate progress supervision into hard route
        # geometry merely to make longitudinal targets easier.
        #
        # The physical route ends at -2 because -1 is the explicit
        # 100m map-exit continuation point.
        # ----------------------------------------------------------

        physical_route_length = (
            route_walk_s[
                :,
                :,
                -2
            ]
            .float()
            .clamp_min(
                1.0e-3
            )
        )

        route_fraction = torch.tensor(
            [
                0.0,
                0.25,
                0.50,
                0.75,
                1.0,
            ],
            dtype=torch.float32,
            device=mode_context.device,
        )

        if self.use_dense_progress_repair:
            geometry_query_s = (
                physical_route_length[:, :, None]
                * route_fraction[None, None, :]
            ).to(route_walk_s.dtype)

            hard_route_samples = _interpolate_polyline(
                route_walk_xy,
                route_walk_s,
                geometry_query_s,
            )

            hard_route_geometry_feature = torch.cat(
                (
                    hard_route_samples.float().reshape(
                        batch_size, num_modes, 10
                    ) / 50.0,
                    physical_route_length[:, :, None] / 100.0,
                    tangent.float(),
                ),
                dim=-1,
            ).detach()

            hard_route_geometry_context = self.hard_route_geometry_encoder(
                hard_route_geometry_feature
            )
            dense_progress_context = self.dense_progress_fusion(
                torch.cat(
                    (
                        mode_context,
                        hard_route_geometry_context.to(mode_context.dtype),
                    ),
                    dim=-1,
                )
            )
            dense_log_scale = (
                self.dense_progress_log_scale_limit
                * torch.tanh(
                    self.dense_progress_residual_head(
                        dense_progress_context
                    ).float()
                )
            )
        else:
            # Exact legacy forward path used by the 247-tensor Phase-B and
            # ProgressFix checkpoints.  A zero log-scale makes the public
            # route-progress sequence exactly the legacy interpolation.
            hard_route_geometry_context = torch.zeros_like(mode_context)
            dense_log_scale = legacy_route_progress_sequence.new_zeros(
                batch_size, num_modes, self.future_steps
            ).float()

        # ----------------------------------------------------------
        # Convert legacy cumulative s(t) into positive per-frame
        # increments, reshape them with the dense route-conditioned
        # residual, then integrate again.
        # ----------------------------------------------------------

        legacy_progress_with_origin = torch.cat(
            (
                legacy_route_progress_sequence.new_zeros(
                    batch_size,
                    num_modes,
                    1,
                ),
                legacy_route_progress_sequence,
            ),
            dim=-1,
        )

        legacy_dense_increment = (
            legacy_progress_with_origin[
                :,
                :,
                1:
            ]
            -
            legacy_progress_with_origin[
                :,
                :,
                :-1
            ]
        ).float().clamp_min(
            1.0e-6
        )

        dense_progress_increment = (
            legacy_dense_increment
            *
            torch.exp(
                dense_log_scale
            )
        ).to(
            mode_context.dtype
        )

        route_progress_sequence = torch.cumsum(
            dense_progress_increment,
            dim=-1,
        )

        # ----------------------------------------------------------
        # Preserve the public 1/3/5-second API for ranking and legacy
        # losses while the actual route guide uses all 25 steps.
        # ----------------------------------------------------------

        horizon_index = torch.tensor(
            [
                min(
                    self.fps - 1,
                    self.future_steps - 1,
                ),
                min(
                    3 * self.fps - 1,
                    self.future_steps - 1,
                ),
                min(
                    5 * self.fps - 1,
                    self.future_steps - 1,
                ),
            ],
            dtype=torch.long,
            device=mode_context.device,
        )

        route_progress = (
            route_progress_sequence.index_select(
                2,
                horizon_index,
            )
        )

        progress_increment = torch.cat(
            (
                route_progress[
                    :,
                    :,
                    0:1,
                ],
                route_progress[
                    :,
                    :,
                    1:2,
                ]
                -
                route_progress[
                    :,
                    :,
                    0:1,
                ],
                route_progress[
                    :,
                    :,
                    2:3,
                ]
                -
                route_progress[
                    :,
                    :,
                    1:2,
                ],
            ),
            dim=-1,
        )

        route_progress_anchors = (
            _interpolate_polyline(
                route_walk_xy,
                route_walk_s,
                route_progress,
            )
        )

        dense_route_guide = (
            _interpolate_polyline(
                route_walk_xy,
                route_walk_s,
                route_progress_sequence,
            )
        )

        return {
            "progress_increment": progress_increment,
            "route_progress": route_progress,
            "route_progress_anchors": route_progress_anchors,

            # Dense route-coordinate representation.
            "route_progress_sequence": route_progress_sequence,
            "dense_progress_increment": dense_progress_increment,
            "dense_route_guide": dense_route_guide,

            # Diagnostics / migration checks.
            "legacy_route_progress": legacy_route_progress,
            "legacy_route_progress_sequence": (
                legacy_route_progress_sequence
            ),
            "hard_route_geometry_context": (
                hard_route_geometry_context
            ),

            "route_walk_xy": route_walk_xy,
            "route_walk_s": route_walk_s,
            "route_terminal_tangent": tangent,
            "route_total_length": route_walk_s[
                :,
                :,
                -1,
            ],
        }
