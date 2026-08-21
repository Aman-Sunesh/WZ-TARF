"""Adapt permanent lane connectivity using soft WorkZone-conditioned viability."""

from __future__ import annotations

import torch
from torch import nn

from wztarf.geometry.lanes import lane_edge_relation_features

class TemporaryTopologyAdapter(nn.Module):
    """Compute soft lane-node and lane-edge viability around the WorkZone."""

    def __init__(
        self,
        d_model: int = 128,
        num_edge_types: int = 16,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        self.node_viability = nn.Sequential(
            nn.Linear(
                2 * d_model,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                1,
            ),
        )

        self.edge_type_embedding = nn.Embedding(
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

        self.edge_viability = nn.Sequential(
            nn.Linear(
                5 * d_model,
                2 * d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                2 * d_model,
                1,
            ),
        )

        self.message_layers = nn.ModuleList(
            [
                nn.Sequential(
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
                for _ in range(
                    num_layers
                )
            ]
        )

        self.update_layers = nn.ModuleList(
            [
                nn.Sequential(
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
                for _ in range(
                    num_layers
                )
            ]
        )

        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(
                    d_model
                )
                for _ in range(
                    num_layers
                )
            ]
        )

    def forward(
        self,
        lane_states: torch.Tensor,
        lane_mask: torch.Tensor,
        lane_edge_index: torch.Tensor,
        lane_edge_type: torch.Tensor,
        lane_edge_mask: torch.Tensor,
        wz_context: torch.Tensor,
        *,
        lane_xy: torch.Tensor,
        lane_heading: torch.Tensor,
        topology_mode: str = "workzone",
    ) -> dict[str, torch.Tensor]:
        """Return WZ-adapted lane states and soft topology scores."""
        batch_size, num_lanes, _ = lane_states.shape
        num_edges = lane_edge_index.shape[-1]

        if lane_xy.shape != (
            batch_size,
            num_lanes,
            2,
        ):
            raise ValueError(
                "lane_xy must have shape [B, L, 2]."
            )
        
        if lane_heading.shape != (
            batch_size,
            num_lanes,
            2,
        ):
            raise ValueError(
                "lane_heading must have shape [B, L, 2]."
            )
    
        # ==============================================================
        # V3 EXPLICIT TOPOLOGY ABLATION
        #
        # workzone:
        #     learned WZ-conditioned temporary graph
        #
        # static:
        #     permanent map graph with valid nodes/edges assigned
        #     viability 1.0.  No WorkZone information participates.
        # ==============================================================
        topology_mode = str(
            topology_mode
        ).lower()

        if topology_mode not in {
            "workzone",
            "static",
        }:
            raise ValueError(
                "topology_mode must be 'workzone' or 'static'."
            )

        if topology_mode == "workzone":
            wz_nodes = wz_context[:, None].expand(
                -1,
                num_lanes,
                -1,
            )

            node_viability = torch.sigmoid(
                self.node_viability(
                    torch.cat(
                        (
                            lane_states,
                            wz_nodes,
                        ),
                        dim=-1,
                    )
                )
            ).squeeze(-1)

            node_viability = (
                node_viability
                *
                lane_mask.to(
                    node_viability.dtype
                )
            )

        else:
            node_viability = lane_mask.to(
                lane_states.dtype
            )

        # Build one flattened edge list for the entire batch and reuse it for
        # edge viability and every topology layer.  This replaces repeated
        # Python loops and repeated edge extraction/remapping.
        src = lane_edge_index[:, 0].long()
        dst = lane_edge_index[:, 1].long()
        valid = lane_edge_mask.bool().clone()
        valid &= (src >= 0) & (dst >= 0) & (src < num_lanes) & (dst < num_lanes)
        safe_src = src.clamp(0, max(num_lanes - 1, 0))
        safe_dst = dst.clamp(0, max(num_lanes - 1, 0))
        valid &= lane_mask.gather(1, safe_src) & lane_mask.gather(1, safe_dst)

        batch_grid = torch.arange(
            batch_size,
            device=lane_states.device,
        )[:, None].expand(batch_size, num_edges)
        edge_slot_grid = torch.arange(
            num_edges,
            device=lane_states.device,
        )[None, :].expand(batch_size, num_edges)

        edge_batch = batch_grid[valid]
        edge_slot = edge_slot_grid[valid]
        local_src = src[valid]
        local_dst = dst[valid]
        edge_type = lane_edge_type.long()[valid]
        global_src = edge_batch * num_lanes + local_src
        global_dst = edge_batch * num_lanes + local_dst

        flat_lane_states = lane_states.reshape(batch_size * num_lanes, self.d_model)
        flat_lane_xy = lane_xy.reshape(batch_size * num_lanes, 2)
        flat_lane_heading = lane_heading.reshape(batch_size * num_lanes, 2)

        edge_embedding = self.edge_type_embedding(edge_type)
        relation = lane_edge_relation_features(
            flat_lane_xy.float(),
            flat_lane_heading.float(),
            global_src,
            global_dst,
        )
        relation_embedding = self.relation_encoder(relation)
        if topology_mode == "workzone":
            edge_wz = wz_context[
                edge_batch
            ]

            edge_input = torch.cat(
                (
                    flat_lane_states[global_src],
                    flat_lane_states[global_dst],
                    relation_embedding,
                    edge_embedding,
                    edge_wz,
                ),
                dim=-1,
            )

            gate_valid = torch.sigmoid(
                self.edge_viability(
                    edge_input
                )
            ).squeeze(
                -1
            )

        else:
            # Permanent graph: every represented valid edge is viable.
            gate_valid = torch.ones(
                edge_batch.shape[0],
                dtype=lane_states.dtype,
                device=lane_states.device,
            )

        # === WZTARF V3 EFFECTIVE TEMPORARY EDGE VIABILITY ===
        #
        # The learned edge gate is explicitly conditioned on the WorkZone
        # context.  In V3 the conditioning context is formed only at the
        # topology interface from independently encoded WZ and worker streams.
        #
        # A transition is useful only if both incident lane nodes remain
        # viable.  Averaging source/destination viability keeps the initial
        # message scale comparable to V2 while making closures downstream of
        # an otherwise viable source suppress the transition as well.
        src_node_viability = node_viability[
            edge_batch,
            local_src,
        ]
        dst_node_viability = node_viability[
            edge_batch,
            local_dst,
        ]

        incident_viability = 0.5 * (
            src_node_viability
            +
            dst_node_viability
        )

        effective_gate_valid = (
            gate_valid.to(incident_viability.dtype)
            *
            incident_viability
        )

        edge_gate_raw = torch.zeros(
            batch_size,
            num_edges,
            dtype=lane_states.dtype,
            device=lane_states.device,
        )

        edge_gate = torch.zeros(
            batch_size,
            num_edges,
            dtype=lane_states.dtype,
            device=lane_states.device,
        )

        # Every [batch, edge_slot] pair is unique, so indexed assignment is
        # deterministic even when several edges share a source/destination.
        edge_gate_raw[
            edge_batch,
            edge_slot,
        ] = gate_valid.to(edge_gate_raw.dtype)

        edge_gate[
            edge_batch,
            edge_slot,
        ] = effective_gate_valid.to(edge_gate.dtype)

        h = lane_states
        flat_node_viability = node_viability.reshape(batch_size * num_lanes)

        for message_layer, update_layer, norm in zip(
            self.message_layers,
            self.update_layers,
            self.norms,
        ):
            flat_h = h.reshape(batch_size * num_lanes, self.d_model)
            message = message_layer(
                torch.cat((flat_h[global_src], edge_embedding), dim=-1)
            )
            # effective_gate_valid already incorporates both source and
            # destination node viability.
            message_gate = effective_gate_valid
            message = (message * message_gate[:, None]).to(flat_h.dtype)

            aggregate = torch.zeros_like(flat_h)
            count = torch.zeros(
                batch_size * num_lanes,
                1,
                dtype=flat_h.dtype,
                device=flat_h.device,
            )
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
            aggregate = aggregate / count.clamp_min(1.0)
            aggregate = aggregate.reshape(batch_size, num_lanes, self.d_model)

            update = update_layer(torch.cat((h, aggregate), dim=-1))
            h = norm(h + update)
            h = h * lane_mask[..., None].to(h.dtype)

        pooling_score = node_viability.masked_fill(
            ~lane_mask.bool(),
            torch.finfo(
                node_viability.dtype
            ).min,
        )

        pooling_weight = torch.softmax(
            pooling_score,
            dim=1,
        )

        pooling_weight = (
            pooling_weight
            *
            lane_mask.to(
                pooling_weight.dtype
            )
        )

        pooling_weight = pooling_weight / (
            pooling_weight.sum(
                dim=1,
                keepdim=True,
            )
            +
            1e-8
        )

        context = (
            pooling_weight[..., None]
            *
            h
        ).sum(
            dim=1
        )

        return {
            "lane_states": h,
            "lane_context": context,
            "node_viability": node_viability,
            "edge_viability": edge_gate,
            "edge_viability_raw": edge_gate_raw,
        }
