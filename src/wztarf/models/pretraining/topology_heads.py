"""Predict WorkZone-conditioned lane and edge topology pretraining targets."""

from __future__ import annotations

import torch
from torch import nn


class TopologyHeads(nn.Module):
    """Predict lane overlap, distance, and temporary edge compatibility."""

    def __init__(
        self,
        *,
        d_model: int = 128,
        num_edge_types: int = 16,
    ) -> None:
        super().__init__()

        self.edge_type_embedding = nn.Embedding(
            num_edge_types,
            d_model,
        )

        self.lane_head = nn.Sequential(
            nn.Linear(
                2 * d_model,
                d_model,
            ),
            nn.ReLU(),
        )

        self.overlap_head = nn.Linear(
            d_model,
            1,
        )

        self.distance_head = nn.Sequential(
            nn.Linear(
                d_model,
                1,
            ),
            nn.Softplus(),
        )

        self.edge_head = nn.Sequential(
            nn.Linear(
                4 * d_model,
                2 * d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                2 * d_model,
                1,
            ),
        )

    def forward(
        self,
        *,
        lane_states: torch.Tensor,
        lane_mask: torch.Tensor,
        wz_context: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return topology predictions in original lane/edge indexing."""
        batch_size, num_lanes, d_model = lane_states.shape
        num_edges = edge_index.shape[-1]

        wz = wz_context[
            :,
            None,
        ].expand(
            -1,
            num_lanes,
            -1,
        )

        lane_hidden = self.lane_head(
            torch.cat(
                (
                    lane_states,
                    wz,
                ),
                dim=-1,
            )
        )

        lane_overlap_pred = torch.sigmoid(
            self.overlap_head(
                lane_hidden
            )
        ).squeeze(-1)

        lane_distance_pred = self.distance_head(
            lane_hidden
        ).squeeze(-1)

        edge_logits = torch.zeros(
            batch_size,
            num_edges,
            dtype=lane_states.dtype,
            device=lane_states.device,
        )

        predicted_edge_mask = torch.zeros(
            batch_size,
            num_edges,
            dtype=torch.bool,
            device=lane_states.device,
        )

        for b in range(
            batch_size
        ):
            valid_edges = torch.nonzero(
                edge_mask[
                    b
                ].bool(),
                as_tuple=False,
            ).flatten()

            if valid_edges.numel() == 0:
                continue

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

            etype = etype[
                valid
            ]

            valid_edges = valid_edges[
                valid
            ]

            if src.numel() == 0:
                continue

            valid_nodes = (
                lane_mask[
                    b,
                    src,
                ]
                &
                lane_mask[
                    b,
                    dst,
                ]
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

            valid_edges = valid_edges[
                valid_nodes
            ]

            if src.numel() == 0:
                continue

            if int(
                etype.max().item()
            ) >= self.edge_type_embedding.num_embeddings:
                raise ValueError(
                    "lane_edge_type exceeds configured num_edge_types."
                )

            edge_feature = self.edge_type_embedding(
                etype
            )

            wz_edge = wz_context[
                b
            ][
                None
            ].expand(
                src.shape[0],
                -1,
            )

            logits = self.edge_head(
                torch.cat(
                    (
                        lane_states[
                            b,
                            src,
                        ],
                        lane_states[
                            b,
                            dst,
                        ],
                        edge_feature,
                        wz_edge,
                    ),
                    dim=-1,
                )
            ).squeeze(-1)

            edge_logits[
                b,
                valid_edges,
            ] = logits

            predicted_edge_mask[
                b,
                valid_edges,
            ] = True

        return {
            "lane_overlap_pred": lane_overlap_pred,
            "lane_distance_pred": lane_distance_pred,
            "edge_compat_logits": edge_logits,
            "topology_lane_mask": lane_mask.bool(),
            "topology_edge_mask": predicted_edge_mask,
        }
