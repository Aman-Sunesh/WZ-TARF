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
        """Propagate typed lane edges for the full batch in one tensor path."""
        batch_size, num_lanes, d_model = nodes.shape
        num_edges = edge_index.shape[-1]

        src = edge_index[:, 0].long()
        dst = edge_index[:, 1].long()
        valid = edge_mask.bool().clone()
        valid &= (src >= 0) & (dst >= 0) & (src < num_lanes) & (dst < num_lanes)

        safe_src = src.clamp(0, max(num_lanes - 1, 0))
        safe_dst = dst.clamp(0, max(num_lanes - 1, 0))
        valid &= node_mask.gather(1, safe_src) & node_mask.gather(1, safe_dst)

        flat_nodes = nodes.reshape(batch_size * num_lanes, d_model)
        aggregate = torch.zeros_like(flat_nodes)
        count = torch.zeros(
            batch_size * num_lanes,
            1,
            dtype=nodes.dtype,
            device=nodes.device,
        )

        if bool(valid.any()):
            batch_index = torch.arange(batch_size, device=nodes.device)[:, None].expand(
                batch_size, num_edges
            )
            b = batch_index[valid]
            local_src = src[valid]
            local_dst = dst[valid]
            etype = edge_type.long()[valid]
            global_src = b * num_lanes + local_src
            global_dst = b * num_lanes + local_dst

            edge_feature = self.edge_embedding(etype)
            relation = lane_edge_relation_features(
                lane_xy.reshape(batch_size * num_lanes, 2),
                lane_heading.reshape(batch_size * num_lanes, 2),
                global_src,
                global_dst,
            )
            relation_feature = self.relation_encoder(relation)
            message = self.message(
                torch.cat(
                    (flat_nodes[global_src], edge_feature, relation_feature),
                    dim=-1,
                )
            ).to(dtype=aggregate.dtype)

            aggregate.index_add_(0, global_dst, message)
            count.index_add_(
                0,
                global_dst,
                torch.ones(
                    global_dst.shape[0],
                    1,
                    dtype=count.dtype,
                    device=count.device,
                ),
            )

        aggregate = (aggregate / count.clamp_min(1.0)).reshape(
            batch_size, num_lanes, d_model
        )
        updated = self.update(torch.cat((nodes, aggregate), dim=-1))
        output = self.norm(nodes + updated)
        return output * node_mask[..., None].to(output.dtype)


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
        encode_points: int = 48,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.top_seed_lanes = top_seed_lanes
        self.encode_points = int(encode_points)
        if self.encode_points < 2:
            raise ValueError("encode_points must be at least 2.")

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

    def _point_encoder_inputs(
        self,
        lane_feat: torch.Tensor,
        point_mask: torch.Tensor,
        *,
        compact: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the point subset used by the expensive point MLP.

        Canonical lane masks are packed.  In Phase B we sample at most
        ``encode_points`` per lane before the MLP; full geometry remains
        untouched for lane XY/heading and route-goal interpolation.
        """
        if not compact or lane_feat.shape[2] <= self.encode_points:
            return lane_feat, point_mask

        batch_size, num_lanes, num_points, feature_dim = lane_feat.shape
        sample_count = min(self.encode_points, num_points)
        count = point_mask.sum(dim=-1).long()
        slot = torch.arange(sample_count, device=lane_feat.device).view(1, 1, -1)

        last = (count - 1).clamp_min(0)[..., None]
        if sample_count == 1:
            uniform = torch.zeros_like(slot).expand(batch_size, num_lanes, -1)
        else:
            uniform = torch.round(
                slot.to(lane_feat.dtype)
                * last.to(lane_feat.dtype)
                / float(sample_count - 1)
            ).long()

        dense = slot.expand(batch_size, num_lanes, -1)
        index = torch.where(count[..., None] >= sample_count, uniform, dense)
        index = index.clamp(0, num_points - 1)
        sampled_mask = dense < count[..., None].clamp_max(sample_count)
        sampled_feat = lane_feat.gather(
            2,
            index[..., None].expand(-1, -1, -1, feature_dim),
        )
        sampled_feat = sampled_feat * sampled_mask[..., None].to(sampled_feat.dtype)
        return sampled_feat, sampled_mask

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
        return_point_states: bool = True,
    ) -> dict[str, torch.Tensor | None]:
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

        encoder_feat, encoder_point_mask = self._point_encoder_inputs(
            lane_feat,
            point_mask,
            compact=compact,
        )

        point_state = self.point_encoder(
            encoder_feat
        )

        attention_logit = self.point_attention(
            point_state
        ).squeeze(-1)

        attention_logit = attention_logit.masked_fill(
            ~encoder_point_mask,
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
            encoder_point_mask.to(
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
            ~encoder_point_mask[..., None],
            float("-inf"),
        )

        max_pool = masked_point_state.max(
            dim=2
        ).values

        has_point = encoder_point_mask.any(
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

        
        
        # ==========================================================
        # V3 ARCHITECTURE FREEZE: KEEP ALL VALID RETAINED LANES
        #
        # The dataset already caps the represented map at <= 74 lanes.
        # The former learned relevance MLP fed a discrete torch.topk,
        # so its parameters received no gradient while random
        # initialization could decide which lanes survived.
        #
        # V3 therefore performs no learned hard lane pruning.
        # ==========================================================
        active_mask = valid_lane.clone()

        lane_state = (
            lane_state
            *
            active_mask[
                ...,
                None,
            ].to(
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

        # Parameter-free deterministic pooling priority.
        # Ego coordinates are centered at the origin, so nearby lanes
        # receive larger pooling logits without introducing another
        # learned selector.
        pooled_score = (
            -torch.linalg.vector_norm(
                lane_xy.float(),
                dim=-1,
            )
        ).to(
            lane_state.dtype
        )

        pooled_score = pooled_score.masked_fill(
            ~active_mask,
            torch.finfo(
                pooled_score.dtype
            ).min,
        )

        # Preserve any legacy diagnostic return using this name.
        relevance_logit = pooled_score

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
                encoder_point_mask[..., None].to(
                    point_state.dtype
                )
                if return_point_states
                else None
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
