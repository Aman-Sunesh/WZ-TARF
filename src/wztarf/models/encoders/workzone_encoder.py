"""Encode WZ boundary, polygon, sign, and worker geometry as typed tokens."""

from __future__ import annotations

import torch
from torch import nn

from wztarf.geometry.workzone import points_in_polygon


class WorkZoneEncoder(nn.Module):
    """Encode structured WorkZone geometry with typed self-attention."""

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        self.feature_encoder = nn.Sequential(
            nn.Linear(
                10,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                d_model,
            ),
        )

        # edge, polygon, sign, worker
        self.type_embedding = nn.Embedding(
            4,
            d_model,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )

        self.pool_score = nn.Linear(
            d_model,
            1,
        )

    def forward(
        self,
        wz_feat: torch.Tensor,
        worker_feat: torch.Tensor,
        ego_speed: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return typed WZ token states and a pooled WZ context."""
        if wz_feat.ndim != 3 or wz_feat.shape[1:] != (5, 3):
            raise ValueError(
                "wz_feat must have shape [B, 5, 3]."
            )

        if worker_feat.ndim != 3 or worker_feat.shape[-1] != 3:
            raise ValueError(
                "worker_feat must have shape [B, W, 3]."
            )

        batch_size = wz_feat.shape[0]

        corners = wz_feat[
            :,
            :4,
            :2,
        ]

        corner_valid = wz_feat[
            :,
            :4,
            2,
        ] > 0

        next_corner = torch.roll(
            corners,
            shifts=-1,
            dims=1,
        )

        next_valid = torch.roll(
            corner_valid,
            shifts=-1,
            dims=1,
        )

        edge_valid = (
            corner_valid
            &
            next_valid
        )

        edge_vector = (
            next_corner
            -
            corners
        )

        edge_length = torch.linalg.vector_norm(
            edge_vector,
            dim=-1,
        )

        tangent = edge_vector / (
            edge_length[..., None]
            +
            1e-8
        )

        midpoint = (
            corners
            +
            next_corner
        ) / 2.0

        midpoint_distance = torch.linalg.vector_norm(
            midpoint,
            dim=-1,
        )

        midpoint_sin = midpoint[..., 1] / (
            midpoint_distance
            +
            1e-8
        )

        midpoint_cos = midpoint[..., 0] / (
            midpoint_distance
            +
            1e-8
        )

        signed_area = 0.5 * (
            corners[..., 0]
            *
            next_corner[..., 1]
            -
            corners[..., 1]
            *
            next_corner[..., 0]
        ).sum(
            dim=1
        )

        ccw = (
            signed_area
            >=
            0
        )[:, None]

        outward_ccw = torch.stack(
            (
                tangent[..., 1],
                -tangent[..., 0],
            ),
            dim=-1,
        )

        outward_cw = torch.stack(
            (
                -tangent[..., 1],
                tangent[..., 0],
            ),
            dim=-1,
        )

        outward = torch.where(
            ccw[..., None],
            outward_ccw,
            outward_cw,
        )

        edge_features = torch.cat(
            (
                midpoint,
                midpoint_distance[..., None],
                midpoint_sin[..., None],
                midpoint_cos[..., None],
                tangent,
                outward,
                edge_length[..., None],
            ),
            dim=-1,
        )

        polygon_valid = corner_valid.all(
            dim=1
        )

        polygon_center = corners.mean(
            dim=1
        )

        polygon_distance = torch.linalg.vector_norm(
            polygon_center,
            dim=-1,
        )

        polygon_length = 0.5 * (
            edge_length[:, 0]
            +
            edge_length[:, 2]
        )

        polygon_width = 0.5 * (
            edge_length[:, 1]
            +
            edge_length[:, 3]
        )

        orientation = tangent[
            :,
            0,
        ]

        polygon_features = torch.stack(
            (
                polygon_center[:, 0],
                polygon_center[:, 1],
                polygon_distance,
                polygon_center[:, 1] / (polygon_distance + 1e-8),
                polygon_center[:, 0] / (polygon_distance + 1e-8),
                polygon_length,
                polygon_width,
                signed_area.abs(),
                orientation[:, 1],
                orientation[:, 0],
            ),
            dim=-1,
        )[:, None, :]

        sign_xy = wz_feat[
            :,
            4,
            :2,
        ]

        sign_valid = wz_feat[
            :,
            4,
            2,
        ] > 0

        sign_distance = torch.linalg.vector_norm(
            sign_xy,
            dim=-1,
        )

        sign_ahead = (
            sign_xy[:, 0]
            >
            0
        ).to(
            sign_xy.dtype
        )

        if ego_speed is None:
            time_to_passage = torch.zeros_like(
                sign_distance
            )
        else:
            safe_speed = ego_speed.clamp_min(
                0.1
            )

            time_to_passage = torch.where(
                sign_ahead.bool(),
                sign_xy[:, 0].clamp_min(0.0)
                /
                safe_speed,
                torch.zeros_like(
                    safe_speed
                ),
            )

        sign_features = torch.stack(
            (
                sign_xy[:, 0],
                sign_xy[:, 1],
                sign_distance,
                sign_xy[:, 1] / (sign_distance + 1e-8),
                sign_xy[:, 0] / (sign_distance + 1e-8),
                sign_ahead,
                time_to_passage,
                torch.zeros_like(sign_distance),
                torch.zeros_like(sign_distance),
                torch.zeros_like(sign_distance),
            ),
            dim=-1,
        )[:, None, :]

        worker_xy = worker_feat[
            ...,
            :2,
        ]

        worker_valid = worker_feat[
            ...,
            2,
        ] > 0

        worker_distance = torch.linalg.vector_norm(
            worker_xy,
            dim=-1,
        )

        worker_ahead = (
            worker_xy[..., 0]
            >
            0
        ).to(
            worker_xy.dtype
        )

        if ego_speed is None:
            worker_ttp = torch.zeros_like(
                worker_distance
            )
        else:
            worker_ttp = torch.where(
                worker_ahead.bool(),
                worker_xy[..., 0].clamp_min(0.0)
                /
                ego_speed[:, None].clamp_min(0.1),
                torch.zeros_like(
                    worker_distance
                ),
            )

        worker_inside = torch.zeros_like(
            worker_distance
        )

        for b in range(batch_size):
            if bool(
                polygon_valid[b]
            ):
                worker_inside[
                    b
                ] = points_in_polygon(
                    worker_xy[b],
                    corners[b],
                ).to(
                    worker_distance.dtype
                )

        worker_features = torch.stack(
            (
                worker_xy[..., 0],
                worker_xy[..., 1],
                worker_distance,
                worker_xy[..., 1] / (worker_distance + 1e-8),
                worker_xy[..., 0] / (worker_distance + 1e-8),
                worker_ahead,
                worker_ttp,
                worker_inside,
                torch.zeros_like(worker_distance),
                torch.zeros_like(worker_distance),
            ),
            dim=-1,
        )

        token_features = torch.cat(
            (
                edge_features,
                polygon_features,
                sign_features,
                worker_features,
            ),
            dim=1,
        )

        token_mask = torch.cat(
            (
                edge_valid,
                polygon_valid[:, None],
                sign_valid[:, None],
                worker_valid,
            ),
            dim=1,
        )

        num_workers = worker_feat.shape[1]

        type_id = torch.tensor(
            [
                0,
                0,
                0,
                0,
                1,
                2,
                *(
                    [3]
                    *
                    num_workers
                ),
            ],
            dtype=torch.long,
            device=wz_feat.device,
        )

        tokens = (
            self.feature_encoder(
                token_features
            )
            +
            self.type_embedding(
                type_id
            )[None]
        )

        original_valid = token_mask.any(
            dim=1
        )

        safe_mask = token_mask.clone()

        no_valid = ~original_valid

        if bool(
            no_valid.any()
        ):
            safe_mask[
                no_valid,
                0,
            ] = True

            tokens = tokens.clone()
            tokens[
                no_valid,
                0,
            ] = 0.0

        tokens = self.transformer(
            tokens,
            src_key_padding_mask=~safe_mask,
        )

        score = self.pool_score(
            tokens
        ).squeeze(-1)

        score = score.masked_fill(
            ~safe_mask,
            -1e9,
        )

        weight = torch.softmax(
            score,
            dim=1,
        )

        weight = (
            weight
            *
            safe_mask.to(
                weight.dtype
            )
        )

        weight = weight / (
            weight.sum(
                dim=1,
                keepdim=True,
            )
            +
            1e-8
        )

        context = (
            tokens
            *
            weight[..., None]
        ).sum(
            dim=1
        )

        context = (
            context
            *
            original_valid[:, None].to(
                context.dtype
            )
        )

        token_xy = torch.cat(
            (
                midpoint,
                polygon_center[:, None],
                sign_xy[:, None],
                worker_xy,
            ),
            dim=1,
        )

        return {
            "wz_tokens": tokens,
            "wz_token_mask": token_mask,
            "wz_token_xy": token_xy,
            "wz_context": context,
            "wz_valid": original_valid,
        }
