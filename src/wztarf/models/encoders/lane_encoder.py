"""Encode lane polylines, select relevant lanes, and propagate lane-graph context."""

from __future__ import annotations

import math

import torch
from torch import nn

from wztarf.geometry.lanes import lane_edge_relation_features

class _LaneGraphLayer(nn.Module):
    """One typed permanent-lane graph message-passing layer."""

    def __init__(
        self,
        d_model: int,
        num_edge_types: int,
    ) -> None:
        super().__init__()

        self.edge_embedding = nn.Embedding(
            num_edge_types,
            d_model,
        )

        self.relation_encoder = nn.Sequential(
            nn.Linear(
                7,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                d_model,
            ),
        )

        self.message = nn.Sequential(
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

        self.update = nn.Sequential(
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

        self.norm = nn.LayerNorm(
            d_model
        )

    def forward(
        self,
        nodes: torch.Tensor,
        node_mask: torch.Tensor,
        lane_xy: torch.Tensor,
        lane_heading: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate valid typed edges independently within each batch item."""
        output = []

        for b in range(
            nodes.shape[0]
        ):
            h = nodes[b]
            aggregate = torch.zeros_like(
                h
            )

            count = torch.zeros(
                h.shape[0],
                1,
                dtype=h.dtype,
                device=h.device,
            )

            valid_edges = torch.nonzero(
                edge_mask[b].bool(),
                as_tuple=False,
            ).flatten()

            if valid_edges.numel() > 0:
                src = edge_index[
                    b,
                    0,
                    valid_edges,
                ].long()

                dst = edge_index[
                    b,
                    1,
                    valid_edges,
                ].long()

                etype = edge_type[
                    b,
                    valid_edges,
                ].long()

                valid_index = (
                    (src >= 0)
                    &
                    (dst >= 0)
                    &
                    (src < h.shape[0])
                    &
                    (dst < h.shape[0])
                )

                src = src[
                    valid_index
                ]
                dst = dst[
                    valid_index
                ]
                etype = etype[
                    valid_index
                ]

                if src.numel() > 0:
                    valid_nodes = (
                        node_mask[b, src]
                        &
                        node_mask[b, dst]
                    )

                    src = src[
                        valid_nodes
                    ]
                    dst = dst[
                        valid_nodes
                    ]
                    etype = etype[
                        valid_nodes
                    ]

                    edge_feature = self.edge_embedding(
                        etype
                    )

                    relation = lane_edge_relation_features(
                        lane_xy[b],
                        lane_heading[b],
                        src,
                        dst,
                    )
                    
                    relation_feature = self.relation_encoder(
                        relation
                    )
                    
                    message = self.message(
                        torch.cat(
                            (
                                h[src],
                                edge_feature,
                                relation_feature,
                            ),
                            dim=-1,
                        )
                    )

                    # AMP may produce FP16 messages while the graph state and
                    # accumulation buffer remain FP32. Accumulate in the graph
                    # state's dtype for numerical stability.
                    message = message.to(
                        dtype=aggregate.dtype
                    )

                    aggregate.index_add_(
                        0,
                        dst,
                        message,
                    )

                    count.index_add_(
                        0,
                        dst,
                        torch.ones(
                            dst.shape[0],
                            1,
                            dtype=h.dtype,
                            device=h.device,
                        ),
                    )

            aggregate = aggregate / count.clamp_min(
                1.0
            )

            updated = self.update(
                torch.cat(
                    (
                        h,
                        aggregate,
                    ),
                    dim=-1,
                )
            )

            h = self.norm(
                h
                +
                updated
            )

            h = (
                h
                *
                node_mask[b, :, None].to(
                    h.dtype
                )
            )

            output.append(
                h
            )

        return torch.stack(
            output,
            dim=0,
        )


class LaneEncoder(nn.Module):
    """Encode lane polylines and keep compact WZ-relevant graph context."""

    def __init__(
        self,
        input_dim: int = 8,
        d_model: int = 128,
        top_seed_lanes: int = 4,
        graph_layers: int = 2,
        num_edge_types: int = 16,
        lane_attr_dim: int = 10,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.top_seed_lanes = top_seed_lanes

        self.point_encoder = nn.Sequential(
            nn.Linear(
                input_dim,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                d_model,
            ),
        )

        self.point_attention = nn.Linear(
            d_model,
            1,
        )

        self.pool_projection = nn.Sequential(
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

        self.attr_encoder = nn.Sequential(
            nn.Linear(
                lane_attr_dim,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                d_model,
            ),
        )

        self.relevance = nn.Sequential(
            nn.Linear(
                3 * d_model,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                1,
            ),
        )

        self.graph = nn.ModuleList(
            [
                _LaneGraphLayer(
                    d_model,
                    num_edge_types,
                )
                for _ in range(
                    graph_layers
                )
            ]
        )

    def forward(
        self,
        lane_feat: torch.Tensor,
        lane_point_mask: torch.Tensor,
        lane_mask: torch.Tensor,
        lane_edge_index: torch.Tensor,
        lane_edge_type: torch.Tensor,
        lane_edge_mask: torch.Tensor,
        ego_context: torch.Tensor,
        wz_context: torch.Tensor,
        lane_attr: torch.Tensor | None = None,
        compact: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Encode lanes and return selected graph states."""
        if lane_feat.ndim != 4:
            raise ValueError(
                "lane_feat must have shape [B, L, P, F]."
            )

        batch_size, num_lanes, num_points, _ = lane_feat.shape

        point_mask = (
            lane_point_mask.bool()
            &
            lane_mask.bool()[:, :, None]
        )

        point_state = self.point_encoder(
            lane_feat
        )

        attention_logit = self.point_attention(
            point_state
        ).squeeze(-1)

        attention_logit = attention_logit.masked_fill(
            ~point_mask,
            torch.finfo(
                attention_logit.dtype
            ).min,
        )

        attention = torch.softmax(
            attention_logit,
            dim=-1,
        )

        attention = (
            attention
            *
            point_mask.to(
                attention.dtype
            )
        )

        attention = attention / (
            attention.sum(
                dim=-1,
                keepdim=True,
            )
            +
            1e-8
        )

        attention_pool = (
            attention[..., None]
            *
            point_state
        ).sum(
            dim=2
        )

        masked_point_state = point_state.masked_fill(
            ~point_mask[..., None],
            float("-inf"),
        )

        max_pool = masked_point_state.max(
            dim=2
        ).values

        has_point = point_mask.any(
            dim=-1
        )

        max_pool = torch.where(
            has_point[..., None],
            max_pool,
            torch.zeros_like(
                max_pool
            ),
        )

        lane_state = self.pool_projection(
            torch.cat(
                (
                    max_pool,
                    attention_pool,
                ),
                dim=-1,
            )
        )

        if lane_attr is not None:
            if lane_attr.shape[:2] != (
                batch_size,
                num_lanes,
            ):
                raise ValueError(
                    "lane_attr must begin with [B, L]."
                )

            lane_state = (
                lane_state
                +
                self.attr_encoder(
                    lane_attr
                )
            )

        valid_lane = (
            lane_mask.bool()
            &
            has_point
        )



        # ------------------------------------------------------------------
        # Representative lane position and heading used by explicit
        # geometric lane-edge relations.
        # ------------------------------------------------------------------
        
        center = lane_feat[
            ...,
            :2,
        ]
        
        center_mask = point_mask.to(
            center.dtype
        )
        
        lane_xy = (
            center
            *
            center_mask[..., None]
        ).sum(
            dim=2
        ) / (
            center_mask.sum(
                dim=2,
                keepdim=True,
            )
            +
            1e-8
        )
        
        segment = (
            center[:, :, 1:]
            -
            center[:, :, :-1]
        )
        
        segment_mask = (
            point_mask[:, :, 1:]
            &
            point_mask[:, :, :-1]
        )
        
        segment_length = torch.linalg.vector_norm(
            segment,
            dim=-1,
        )
        
        segment_direction = (
            segment
            /
            segment_length[
                ...,
                None,
            ].clamp_min(
                1e-8
            )
        )
        
        heading_sum = (
            segment_direction
            *
            segment_mask[
                ...,
                None,
            ].to(
                segment_direction.dtype
            )
        ).sum(
            dim=2
        )
        
        heading_norm = torch.linalg.vector_norm(
            heading_sum,
            dim=-1,
            keepdim=True,
        )
        
        lane_heading = (
            heading_sum
            /
            heading_norm.clamp_min(
                1e-8
            )
        )
        
        default_heading = torch.zeros_like(
            lane_heading
        )
        
        default_heading[
            ...,
            0,
        ] = 1.0
        
        lane_heading = torch.where(
            heading_norm > 1e-6,
            lane_heading,
            default_heading,
        )
        
        lane_heading = (
            lane_heading
            *
            valid_lane[
                ...,
                None,
            ].to(
                lane_heading.dtype
            )
        )

        
        
        ego = ego_context[:, None].expand(
            -1,
            num_lanes,
            -1,
        )

        wz = wz_context[:, None].expand(
            -1,
            num_lanes,
            -1,
        )

        relevance_logit = self.relevance(
            torch.cat(
                (
                    lane_state,
                    ego,
                    wz,
                ),
                dim=-1,
            )
        ).squeeze(-1)

        relevance_logit = relevance_logit.masked_fill(
            ~valid_lane,
            torch.finfo(
                attention_logit.dtype
            ).min,
        )
        if not compact:
            # Phase A pretraining keeps every valid represented lane.
            active_mask = valid_lane.clone()

        else:
            active_mask = torch.zeros_like(
                valid_lane
            )

            # Supervised forecasting keeps Top-k seeds plus one-hop neighbors.
            for b in range(
                batch_size
            ):
                valid_indices = torch.nonzero(
                    valid_lane[b],
                    as_tuple=False,
                ).flatten()

                if valid_indices.numel() == 0:
                    continue

                k = min(
                    self.top_seed_lanes,
                    int(
                        valid_indices.numel()
                    ),
                )

                local_scores = relevance_logit[
                    b,
                    valid_indices,
                ]

                selected_local = torch.topk(
                    local_scores,
                    k=k,
                ).indices

                seeds = valid_indices[
                    selected_local
                ]

                active_mask[
                    b,
                    seeds,
                ] = True

                valid_edges = torch.nonzero(
                    lane_edge_mask[b].bool(),
                    as_tuple=False,
                ).flatten()

                if valid_edges.numel() == 0:
                    continue

                src = lane_edge_index[
                    b,
                    0,
                    valid_edges,
                ].long()

                dst = lane_edge_index[
                    b,
                    1,
                    valid_edges,
                ].long()

                in_bounds = (
                    (src >= 0)
                    &
                    (dst >= 0)
                    &
                    (src < num_lanes)
                    &
                    (dst < num_lanes)
                )

                src = src[
                    in_bounds
                ]

                dst = dst[
                    in_bounds
                ]

                for seed in seeds:
                    connected = (
                        (src == seed)
                        |
                        (dst == seed)
                    )

                    neighbors = torch.cat(
                        (
                            src[
                                connected
                            ],
                            dst[
                                connected
                            ],
                        )
                    )

                    active_mask[
                        b,
                        neighbors,
                    ] = True

        active_mask &= valid_lane

        lane_state = (
            lane_state
            *
            active_mask[..., None].to(
                lane_state.dtype
            )
        )

        for graph_layer in self.graph:
            lane_state = graph_layer(
                lane_state,
                active_mask,
                lane_xy,
                lane_heading,
                lane_edge_index,
                lane_edge_type,
                lane_edge_mask,
            )

        pooled_score = relevance_logit.masked_fill(
            ~active_mask,
            torch.finfo(
                attention_logit.dtype
            ).min,
        )

        pooled_weight = torch.softmax(
            pooled_score,
            dim=1,
        )

        pooled_weight = (
            pooled_weight
            *
            active_mask.to(
                pooled_weight.dtype
            )
        )

        pooled_weight = pooled_weight / (
            pooled_weight.sum(
                dim=1,
                keepdim=True,
            )
            +
            1e-8
        )

        lane_context = (
            pooled_weight[..., None]
            *
            lane_state
        ).sum(
            dim=1
        )
            
        route_point_mask = (
            point_mask
            &
            active_mask[
                :,
                :,
                None,
            ]
        )
    
        lane_centerline = (
            center
            *
            route_point_mask[
                ...,
                None,
            ].to(
                center.dtype
            )
        )

        return {
            "lane_point_states": (
                point_state
                *
                point_mask[..., None].to(
                    point_state.dtype
                )
            ),
            "lane_states": lane_state,
            "lane_context": lane_context,
            "lane_relevance": pooled_weight,
            "lane_mask": active_mask,
            "lane_xy": lane_xy,
            "lane_heading": lane_heading,
            "lane_centerline": lane_centerline,
            "lane_point_mask": route_point_mask,
        }
