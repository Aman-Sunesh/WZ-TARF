"""Refine coarse trajectories using only nearby scene tokens."""

from __future__ import annotations

import torch
from torch import nn


class LocalRefiner(nn.Module):
    """Apply local cross-attention around each coarse trajectory mode."""

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        local_radius_m: float = 8.0,
    ) -> None:
        super().__init__()

        if local_radius_m <= 0:
            raise ValueError(
                "local_radius_m must be positive."
            )

        self.d_model = d_model
        self.local_radius_m = local_radius_m

        self.point_encoder = nn.Sequential(
            nn.Linear(
                2,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                d_model,
            ),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.delta_head = nn.Sequential(
            nn.Linear(
                2 * d_model,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                2,
            ),
        )

    def forward(
        self,
        coarse_xy: torch.Tensor,
        mode_context: torch.Tensor,
        scene_tokens: torch.Tensor,
        scene_xy: torch.Tensor,
        scene_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return residual correction `[B, K, T, 2]`.

        Scene tokens farther than `local_radius_m` from an entire coarse
        trajectory are excluded from that trajectory's attention.
        """
        if coarse_xy.ndim != 4 or coarse_xy.shape[-1] != 2:
            raise ValueError(
                "coarse_xy must have shape [B, K, T, 2]."
            )

        batch_size, num_modes, future_steps, _ = coarse_xy.shape

        if mode_context.shape[:2] != (
            batch_size,
            num_modes,
        ):
            raise ValueError(
                "mode_context must have shape [B, K, D]."
            )

        if scene_tokens.ndim != 3:
            raise ValueError(
                "scene_tokens must have shape [B, N, D]."
            )

        if scene_xy.shape != (
            batch_size,
            scene_tokens.shape[1],
            2,
        ):
            raise ValueError(
                "scene_xy must have shape [B, N, 2]."
            )

        if scene_mask.shape != (
            batch_size,
            scene_tokens.shape[1],
        ):
            raise ValueError(
                "scene_mask must have shape [B, N]."
            )

        # [B, K, T, N]
        distance = torch.linalg.vector_norm(
            coarse_xy[:, :, :, None, :]
            -
            scene_xy[:, None, None, :, :],
            dim=-1,
        )

        local = (
            distance.min(dim=2).values
            <=
            self.local_radius_m
        )

        local &= scene_mask[:, None, :].bool()

        # Fall back to all valid tokens if nothing is locally available.
        for b in range(batch_size):
            for k in range(num_modes):
                if not bool(local[b, k].any()):
                    local[b, k] = scene_mask[b].bool()

        point_query = (
            self.point_encoder(
                coarse_xy
            )
            +
            mode_context[:, :, None, :]
        )

        query = point_query.reshape(
            batch_size * num_modes,
            future_steps,
            self.d_model,
        )

        keys = scene_tokens[:, None].expand(
            batch_size,
            num_modes,
            scene_tokens.shape[1],
            self.d_model,
        ).reshape(
            batch_size * num_modes,
            scene_tokens.shape[1],
            self.d_model,
        )

        local_mask = local.reshape(
            batch_size * num_modes,
            scene_tokens.shape[1],
        )

        # If a sample truly contains no scene token, return zero correction
        # for that trajectory instead of sending an all-masked attention row.
        no_context = ~local_mask.any(
            dim=1
        )

        safe_mask = local_mask.clone()

        if bool(no_context.any()):
            safe_mask[
                no_context,
                0,
            ] = True

            keys = keys.clone()
            keys[
                no_context,
                0,
            ] = 0.0

        attended, _ = self.attention(
            query=query,
            key=keys,
            value=keys,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )

        attended = self.norm(
            query
            +
            attended
        )

        delta = self.delta_head(
            torch.cat(
                (
                    query,
                    attended,
                ),
                dim=-1,
            )
        )

        delta = delta.reshape(
            batch_size,
            num_modes,
            future_steps,
            2,
        )

        if bool(no_context.any()):
            flat = delta.reshape(
                batch_size * num_modes,
                future_steps,
                2,
            )

            flat[
                no_context
            ] = 0.0

            delta = flat.reshape_as(
                delta
            )

        return delta
