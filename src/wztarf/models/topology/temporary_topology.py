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

        edge_gate = torch.zeros(
            batch_size,
            num_edges,
            dtype=lane_states.dtype,
            device=lane_states.device,
        )

        h = lane_states

        # Edge viability is computed once from the initial lane encoding.
        for b in range(
            batch_size
        ):
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

            edge_type = lane_edge_type[
                b,
                valid_edges,
            ].long()

            valid = (
                (src >= 0)
                &
                (dst >= 0)
                &
                (src < num_lanes)
                &
                (dst < num_lanes)
            )

            src = src[
                valid
            ]
            dst = dst[
                valid
            ]
            edge_type = edge_type[
                valid
            ]
            valid_edges = valid_edges[
                valid
            ]

            if src.numel() == 0:
                continue

            node_valid = (
                lane_mask[b, src]
                &
                lane_mask[b, dst]
            )

            src = src[
                node_valid
            ]
            dst = dst[
                node_valid
            ]
            edge_type = edge_type[
                node_valid
            ]
            valid_edges = valid_edges[
                node_valid
            ]

            if src.numel() == 0:
                continue

            if int(
                edge_type.max().item()
            ) >= self.edge_type_embedding.num_embeddings:
                raise ValueError(
                    "lane_edge_type exceeds configured num_edge_types."
                )

            edge_embedding = self.edge_type_embedding(
                edge_type
            )

            relation = lane_edge_relation_features(
                lane_xy[b],
                lane_heading[b],
                src,
                dst,
            )
            
            relation_embedding = self.relation_encoder(
                relation
            )

            wz = wz_context[
                b
            ][None].expand(
                src.shape[0],
                -1,
            )

            edge_input = torch.cat(
                (
                    lane_states[
                        b,
                        src,
                    ],
                    lane_states[
                        b,
                        dst,
                    ],
                    relation_embedding,
                    edge_embedding,
                    wz,
                ),
                dim=-1,
)

            gate = torch.sigmoid(
                self.edge_viability(
                    edge_input
                )
            ).squeeze(-1)

            edge_gate[
                b,
                valid_edges,
            ] = gate.to(
                dtype=edge_gate.dtype
            )

        for message_layer, update_layer, norm in zip(
            self.message_layers,
            self.update_layers,
            self.norms,
        ):
            updated_batches = []

            for b in range(
                batch_size
            ):
                aggregate = torch.zeros_like(
                    h[b]
                )

                count = torch.zeros(
                    num_lanes,
                    1,
                    dtype=h.dtype,
                    device=h.device,
                )

                valid_edges = torch.nonzero(
                    lane_edge_mask[b].bool(),
                    as_tuple=False,
                ).flatten()

                if valid_edges.numel() > 0:
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

                    edge_type = lane_edge_type[
                        b,
                        valid_edges,
                    ].long()

                    valid = (
                        (src >= 0)
                        &
                        (dst >= 0)
                        &
                        (src < num_lanes)
                        &
                        (dst < num_lanes)
                    )

                    src = src[
                        valid
                    ]
                    dst = dst[
                        valid
                    ]
                    edge_type = edge_type[
                        valid
                    ]
                    valid_edges = valid_edges[
                        valid
                    ]

                    if src.numel() > 0:
                        node_valid = (
                            lane_mask[b, src]
                            &
                            lane_mask[b, dst]
                        )

                        src = src[
                            node_valid
                        ]
                        dst = dst[
                            node_valid
                        ]
                        edge_type = edge_type[
                            node_valid
                        ]
                        valid_edges = valid_edges[
                            node_valid
                        ]

                    if src.numel() > 0:
                        edge_embedding = self.edge_type_embedding(
                            edge_type
                        )

                        message = message_layer(
                            torch.cat(
                                (
                                    h[
                                        b,
                                        src,
                                    ],
                                    edge_embedding,
                                ),
                                dim=-1,
                            )
                        )

                        gate = (
                            edge_gate[
                                b,
                                valid_edges,
                            ]
                            *
                            node_viability[
                                b,
                                src,
                            ]
                        )

                        message = (
                            message
                            *
                            gate[:, None]
                        )

                        # AMP can produce FP16 messages while the graph
                        # accumulation buffer remains FP32. index_add_ requires
                        # source and destination to have identical dtypes.
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

                update = update_layer(
                    torch.cat(
                        (
                            h[b],
                            aggregate,
                        ),
                        dim=-1,
                    )
                )

                new_h = norm(
                    h[b]
                    +
                    update
                )

                new_h = (
                    new_h
                    *
                    lane_mask[
                        b,
                        :,
                        None,
                    ].to(
                        new_h.dtype
                    )
                )

                updated_batches.append(
                    new_h
                )

            h = torch.stack(
                updated_batches,
                dim=0,
            )

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
        }
