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

    This is the vectorized equivalent of the original B x K Python loop.  It
    keeps the same semantics but removes ``Tensor.item()`` synchronizations
    from the GPU hot path.
    """
    if lane_centerline.ndim != 4 or lane_centerline.shape[-1] != 2:
        raise ValueError("lane_centerline must have shape [B, L, P, 2].")
    if lane_point_mask.shape != lane_centerline.shape[:3]:
        raise ValueError("lane_point_mask must have shape [B, L, P].")
    if lane_index.shape != fraction.shape:
        raise ValueError("lane_index and fraction must have shape [B, K].")

    batch_size, num_lanes, num_points, _ = lane_centerline.shape
    if num_points == 0:
        raise ValueError("lane_centerline must contain at least one point slot.")

    num_modes = lane_index.shape[1]
    lane_in_bounds = (lane_index >= 0) & (lane_index < num_lanes)
    safe_lane_index = lane_index.clamp(0, max(num_lanes - 1, 0)).long()

    batch_index = torch.arange(
        batch_size,
        device=lane_centerline.device,
    )[:, None].expand(batch_size, num_modes)

    points = lane_centerline[batch_index, safe_lane_index]  # [B,K,P,2]
    mask = lane_point_mask[batch_index, safe_lane_index].bool()
    mask = mask & lane_in_bounds[..., None]

    # Compact valid points into the front of each selected polyline.  The
    # canonical dataset is already packed, but scatter_add keeps this robust to
    # occasional holes without introducing Python loops.
    rank = (mask.long().cumsum(dim=-1) - 1).clamp_min(0)
    compact = torch.zeros_like(points)
    compact.scatter_add_(
        2,
        rank[..., None].expand(-1, -1, -1, 2),
        points * mask[..., None].to(points.dtype),
    )

    count = mask.sum(dim=-1)  # [B,K]
    valid_goal = lane_in_bounds & (count > 0)

    if num_points == 1:
        goal_xy = compact[..., 0, :]
        goal_xy = goal_xy * valid_goal[..., None].to(goal_xy.dtype)
        goal_offset = torch.zeros_like(fraction)
        return goal_xy, goal_offset, valid_goal

    segment = compact[..., 1:, :] - compact[..., :-1, :]
    segment_length = torch.linalg.vector_norm(segment, dim=-1)

    segment_slot = torch.arange(
        num_points - 1,
        device=lane_centerline.device,
    ).view(1, 1, -1)
    segment_valid = segment_slot < (count - 1).clamp_min(0)[..., None]
    segment_length = segment_length * segment_valid.to(segment_length.dtype)

    cumulative_end = torch.cumsum(segment_length, dim=-1)
    total_length = cumulative_end[..., -1]
    goal_offset = fraction.clamp(0.0, 1.0) * total_length

    # First segment whose cumulative arc length reaches the requested offset.
    segment_index = (
        cumulative_end < goal_offset[..., None]
    ).sum(dim=-1).clamp(max=num_points - 2)

    gather_seg = segment_index[..., None, None].expand(-1, -1, 1, 2)
    start_point = compact[..., :-1, :].gather(2, gather_seg).squeeze(2)
    selected_segment = segment.gather(2, gather_seg).squeeze(2)

    cumulative_start = torch.cat(
        (torch.zeros_like(cumulative_end[..., :1]), cumulative_end[..., :-1]),
        dim=-1,
    )
    start_offset = cumulative_start.gather(
        -1,
        segment_index[..., None],
    ).squeeze(-1)
    selected_length = segment_length.gather(
        -1,
        segment_index[..., None],
    ).squeeze(-1)

    alpha = (goal_offset - start_offset) / selected_length.clamp_min(1e-8)
    interpolated = start_point + alpha[..., None] * selected_segment

    # Degenerate one-point/zero-length polylines use their last represented
    # point exactly like the original implementation.
    last_index = (count - 1).clamp_min(0).clamp_max(num_points - 1)
    last_point = compact.gather(
        2,
        last_index[..., None, None].expand(-1, -1, 1, 2),
    ).squeeze(2)
    use_interpolation = (count > 1) & (total_length > 1e-8)
    goal_xy = torch.where(
        use_interpolation[..., None],
        interpolated,
        last_point,
    )
    goal_xy = goal_xy * valid_goal[..., None].to(goal_xy.dtype)
    goal_offset = torch.where(
        use_interpolation,
        goal_offset,
        torch.zeros_like(goal_offset),
    )

    return goal_xy, goal_offset, valid_goal


class RouteGoalQueries(nn.Module):
    """Generate K explicit, topology-conditioned route hypotheses.

    Each mode owns a soft lane-node occupancy and a soft edge occupancy over
    the temporary WorkZone graph. These occupancies are differentiable and
    are pooled into the route embedding that conditions the downstream goal
    and trajectory decoder.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_modes: int = 6,
        num_horizons: int = 3,
        topology_bias: float = 1.0,
        competition_strength: float = 0.5,
    ) -> None:
        super().__init__()

        if topology_bias < 0.0:
            raise ValueError(
                "topology_bias must be non-negative."
            )

        if competition_strength < 0.0:
            raise ValueError(
                "competition_strength must be non-negative."
            )

        self.d_model = d_model
        self.num_modes = num_modes
        self.num_horizons = num_horizons
        self.topology_bias = float(topology_bias)
        self.competition_strength = float(
            competition_strength
        )

        # --------------------------------------------------------------
        # V3 route-slot identities.
        #
        # These no longer directly represent arbitrary trajectory modes.
        # Every slot must first construct a topology-conditioned route
        # occupancy, and only that route-conditioned representation is
        # delivered to the trajectory decoder.
        # --------------------------------------------------------------
        self.route_slot_queries = nn.Parameter(
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

        self.edge_projection = nn.Sequential(
            nn.Linear(
                2 * d_model,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                d_model,
            ),
        )

        self.route_embedding_fusion = nn.Sequential(
            nn.Linear(
                2 * d_model,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                d_model,
            ),
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

    @staticmethod
    def _masked_softmax(
        logits: torch.Tensor,
        mask: torch.Tensor,
        *,
        dim: int,
    ) -> torch.Tensor:
        """Numerically safe masked softmax, including all-masked rows."""

        mask = mask.bool()

        # Route/topology probability normalization is deliberately FP32.
        # This also avoids introducing another BF16 nonfinite source.
        logits_fp32 = logits.float().masked_fill(
            ~mask,
            -1.0e9,
        )

        probability = torch.softmax(
            logits_fp32,
            dim=dim,
        )

        probability = (
            probability
            *
            mask.to(
                probability.dtype
            )
        )

        denominator = probability.sum(
            dim=dim,
            keepdim=True,
        )

        probability = torch.where(
            denominator > 0.0,
            probability
            /
            denominator.clamp_min(
                1.0e-8
            ),
            torch.zeros_like(
                probability
            ),
        )

        return probability.to(
            logits.dtype
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
        *,
        node_viability: torch.Tensor | None = None,
        edge_viability: torch.Tensor | None = None,
        lane_edge_index: torch.Tensor | None = None,
        lane_edge_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return K explicit topology-conditioned route hypotheses."""

        batch_size, num_lanes, _ = lane_states.shape

        lane_mask_bool = lane_mask.bool()
        eps = 1.0e-6

        if node_viability is None:
            node_viability = lane_mask_bool.to(
                lane_states.dtype
            )

        if node_viability.shape != (
            batch_size,
            num_lanes,
        ):
            raise ValueError(
                "node_viability must have shape [B, L]."
            )

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

        # --------------------------------------------------------------
        # Six route slots.
        # --------------------------------------------------------------
        slot_context = (
            global_context[:, None]
            +
            self.route_slot_queries[
                None
            ]
        )

        slot_query = self.query_projection(
            slot_context
        )

        lane_key = self.lane_projection(
            lane_states
        )

        # --------------------------------------------------------------
        # Route-node occupancy.
        #
        # Topology viability is a PRIOR in the route allocation itself,
        # rather than merely a feature used somewhere upstream.
        # --------------------------------------------------------------
        base_node_logits = torch.einsum(
            "bkd,bld->bkl",
            slot_query,
            lane_key,
        ) / math.sqrt(
            self.d_model
        )

        topology_node_prior = torch.log(
            node_viability.clamp_min(
                eps
            )
        ).to(
            base_node_logits.dtype
        )

        route_node_logits = (
            base_node_logits
            +
            self.topology_bias
            *
            topology_node_prior[
                :,
                None,
                :,
            ]
        )

        node_mask = lane_mask_bool[
            :,
            None,
            :,
        ].expand(
            batch_size,
            self.num_modes,
            num_lanes,
        )

        base_route_node_occupancy = (
            self._masked_softmax(
                route_node_logits,
                node_mask,
                dim=-1,
            )
        )

        # --------------------------------------------------------------
        # Explicit route-edge occupancy over the temporary graph.
        #
        # Each edge receives:
        #
        #   route-slot compatibility
        #   + WZ temporary-edge prior
        #   + support from incident route nodes
        #
        # Then route slots softly compete for graph edges.
        #
        # This is intentionally NOT a hard edge partition:
        # routes may share a common trunk and specialize at branches.
        # --------------------------------------------------------------
        if (
            lane_edge_index is not None
            and
            lane_edge_mask is not None
        ):
            if (
                lane_edge_index.ndim != 3
                or
                lane_edge_index.shape[1] != 2
            ):
                raise ValueError(
                    "lane_edge_index must have shape [B, 2, E]."
                )

            num_edges = lane_edge_index.shape[
                -1
            ]

            if lane_edge_mask.shape != (
                batch_size,
                num_edges,
            ):
                raise ValueError(
                    "lane_edge_mask must have shape [B, E]."
                )

            if edge_viability is None:
                edge_viability = (
                    lane_edge_mask.to(
                        lane_states.dtype
                    )
                )

            if edge_viability.shape != (
                batch_size,
                num_edges,
            ):
                raise ValueError(
                    "edge_viability must have shape [B, E]."
                )

            src = lane_edge_index[
                :,
                0,
            ].long()

            dst = lane_edge_index[
                :,
                1,
            ].long()

            valid_edge = (
                lane_edge_mask.bool().clone()
            )

            valid_edge &= (
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

            gather_src = safe_src[
                ...,
                None,
            ].expand(
                batch_size,
                num_edges,
                self.d_model,
            )

            gather_dst = safe_dst[
                ...,
                None,
            ].expand(
                batch_size,
                num_edges,
                self.d_model,
            )

            src_state = lane_states.gather(
                1,
                gather_src,
            )

            dst_state = lane_states.gather(
                1,
                gather_dst,
            )

            edge_state = self.edge_projection(
                torch.cat(
                    (
                        src_state,
                        dst_state,
                    ),
                    dim=-1,
                )
            )

            edge_compatibility = torch.einsum(
                "bkd,bed->bke",
                slot_query,
                edge_state,
            ) / math.sqrt(
                self.d_model
            )

            route_src_index = safe_src[
                :,
                None,
                :,
            ].expand(
                batch_size,
                self.num_modes,
                num_edges,
            )

            route_dst_index = safe_dst[
                :,
                None,
                :,
            ].expand(
                batch_size,
                self.num_modes,
                num_edges,
            )

            src_support = (
                base_route_node_occupancy.gather(
                    2,
                    route_src_index,
                )
            )

            dst_support = (
                base_route_node_occupancy.gather(
                    2,
                    route_dst_index,
                )
            )

            node_support = 0.5 * (
                src_support
                +
                dst_support
            )

            edge_prior = torch.log(
                edge_viability.clamp_min(
                    eps
                )
            ).to(
                edge_compatibility.dtype
            )

            route_edge_logits = (
                edge_compatibility
                +
                self.topology_bias
                *
                edge_prior[
                    :,
                    None,
                    :,
                ]
                +
                torch.log(
                    node_support.clamp_min(
                        eps
                    )
                )
            )

            edge_mask = valid_edge[
                :,
                None,
                :,
            ].expand(
                batch_size,
                self.num_modes,
                num_edges,
            )

            # ----------------------------------------------------------
            # Soft route ownership.
            #
            # Each edge asks: "which route slot owns me?"
            #
            # competition_strength=0 would recover independent route
            # attentions. Positive values force specialization.
            # ----------------------------------------------------------
            owner_logits = (
                route_edge_logits
                .float()
                .masked_fill(
                    ~edge_mask,
                    -1.0e9,
                )
            )

            edge_owner = torch.softmax(
                owner_logits,
                dim=1,
            ).to(
                route_edge_logits.dtype
            )

            route_edge_logits = (
                route_edge_logits
                +
                self.competition_strength
                *
                torch.log(
                    edge_owner.clamp_min(
                        eps
                    )
                )
            )

            # === V3 DIFFERENTIABLE MULTI-EDGE ROUTE MEMBERSHIP ===
            #
            # A route contains MULTIPLE edges, so edge membership must not
            # softmax across E. Softmax would force every route to distribute
            # a total probability mass of exactly 1 across its whole path.
            #
            # Sigmoid gives every graph edge an independent differentiable
            # membership probability.
            route_edge_occupancy = torch.sigmoid(
                route_edge_logits.float()
            ).to(
                route_edge_logits.dtype
            )

            route_edge_occupancy = (
                route_edge_occupancy
                *
                edge_mask.to(
                    route_edge_occupancy.dtype
                )
            )

            # Separate normalized attention is used only when pooling edge
            # features into a fixed-dimensional route embedding.
            route_edge_attention = (
                route_edge_occupancy
                /
                route_edge_occupancy.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(
                    eps
                )
            )

            # ----------------------------------------------------------
            # Turn edge ownership into a route-node occupancy.
            #
            # Destination mass is emphasized because it represents the
            # downstream route continuation. Source mass preserves shared
            # trunk structure.
            # ----------------------------------------------------------
            route_node_mass = (
                base_route_node_occupancy.clone()
            )

            route_node_mass.scatter_add_(
                2,
                route_src_index,
                (0.5
                *
                route_edge_attention).to(dtype=route_node_mass.dtype),
            )

            route_node_mass.scatter_add_(
                2,
                route_dst_index,
                (route_edge_attention).to(dtype=route_node_mass.dtype),
            )

            route_node_mass = (
                route_node_mass
                *
                node_viability[
                    :,
                    None,
                    :,
                ].to(
                    route_node_mass.dtype
                )
                *
                lane_mask_bool[
                    :,
                    None,
                    :,
                ].to(
                    route_node_mass.dtype
                )
            )

            node_mass_sum = (
                route_node_mass.sum(
                    dim=-1,
                    keepdim=True,
                )
            )

            route_node_occupancy = torch.where(
                node_mass_sum > eps,
                route_node_mass
                /
                node_mass_sum.clamp_min(
                    eps
                ),
                base_route_node_occupancy,
            )

            edge_embedding = torch.einsum(
                "bke,bed->bkd",
                route_edge_attention,
                edge_state,
            )

            node_expected_viability = (
                route_node_occupancy
                *
                node_viability[
                    :,
                    None,
                    :,
                ].to(
                    route_node_occupancy.dtype
                )
            ).sum(
                dim=-1
            )

            edge_expected_viability = (
                route_edge_attention
                *
                edge_viability[
                    :,
                    None,
                    :,
                ].to(
                    route_edge_occupancy.dtype
                )
            ).sum(
                dim=-1
            )

            has_valid_edge = (
                valid_edge.any(
                    dim=-1
                )[
                    :,
                    None,
                ]
            )

            route_viability = torch.where(
                has_valid_edge,
                0.5
                *
                (
                    node_expected_viability
                    +
                    edge_expected_viability
                ),
                node_expected_viability,
            )

        else:
            # Compatibility fallback for isolated module tests.
            route_node_occupancy = (
                base_route_node_occupancy
            )

            route_edge_logits = (
                lane_states.new_zeros(
                    batch_size,
                    self.num_modes,
                    0,
                )
            )

            route_edge_occupancy = (
                lane_states.new_zeros(
                    batch_size,
                    self.num_modes,
                    0,
                )
            )

            route_edge_attention = (
                lane_states.new_zeros(
                    batch_size,
                    self.num_modes,
                    0,
                )
            )

            edge_embedding = (
                lane_states.new_zeros(
                    batch_size,
                    self.num_modes,
                    self.d_model,
                )
            )

            route_viability = (
                route_node_occupancy
                *
                node_viability[
                    :,
                    None,
                    :,
                ].to(
                    route_node_occupancy.dtype
                )
            ).sum(
                dim=-1
            )

        # --------------------------------------------------------------
        # Every mode now explicitly owns a route embedding.
        # --------------------------------------------------------------
        node_embedding = torch.einsum(
            "bkl,bld->bkd",
            route_node_occupancy,
            lane_states,
        )

        route_embedding = (
            self.route_embedding_fusion(
                torch.cat(
                    (
                        node_embedding,
                        edge_embedding,
                    ),
                    dim=-1,
                )
            )
        )

        mode_context = self.mode_norm(
            slot_context
            +
            route_embedding
        )

        # --------------------------------------------------------------
        # Terminal lane prediction is now ROUTE-CONSTRAINED.
        #
        # The old implementation classified every lane directly from a
        # generic mode query. V3 can strongly score a lane only if that
        # lane belongs to the route occupancy of this slot.
        # --------------------------------------------------------------
        goal_query = self.query_projection(
            mode_context
        )

        lane_logits = torch.einsum(
            "bkd,bld->bkl",
            goal_query,
            lane_key,
        ) / math.sqrt(
            self.d_model
        )

        lane_logits = (
            lane_logits
            +
            self.topology_bias
            *
            torch.log(
                route_node_occupancy.clamp_min(
                    eps
                )
            )
        )

        lane_logits = lane_logits.masked_fill(
            ~lane_mask_bool[
                :,
                None,
                :,
            ],
            torch.finfo(
                lane_logits.dtype
            ).min,
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
            goal_logits.float(),
            dim=-1,
        ).to(
            goal_logits.dtype
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

        # ==============================================================
        # V3 TERMINAL CLASS DIAGNOSTICS
        #
        # Continuous future geometry is now produced exclusively by
        # DifferentiableRouteProgress.  RouteGoalQueries is responsible
        # for route occupancy plus terminal lane/MAP_EXIT probability.
        # ==============================================================

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

        map_exit_probability = goal_prob[
            ...,
            -1
        ]

        route_lane_probability = self._masked_softmax(
            lane_logits,
            lane_mask[
                :,
                None,
                :,
            ].expand(
                batch_size,
                self.num_modes,
                num_lanes,
            ),
            dim=-1,
        )

        return {
            # Existing public outputs.
            "mode_context": mode_context,
            "goal_context": goal_context,
            "goal_logits": goal_logits,
            "goal_prob": goal_prob,
            "goal_lane_index": selected_lane,
            "goal_is_map_exit": goal_is_map_exit,

            # Differentiable V3 terminal geometry.
            "route_lane_probability": route_lane_probability,
            "map_exit_probability": map_exit_probability,

            # === V3 explicit route-set outputs ===
            "route_node_logits": route_node_logits,
            "route_node_occupancy": route_node_occupancy,
            "route_edge_logits": route_edge_logits,
            "route_edge_occupancy": route_edge_occupancy,
            "route_edge_attention": route_edge_attention,
            "route_embedding": route_embedding,
            "route_viability": route_viability,
        }
