"""Compute WorkZone Geometry Violation Rate for Top-1 predictions."""

from __future__ import annotations

import torch

from wztarf.geometry.workzone import points_in_polygon


def per_sample_wz_gvr(
    pred_xy: torch.Tensor,
    mode_prob: torch.Tensor,
    wz_polygon: torch.Tensor,
    wz_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return WZ-geometry violation and validity flags per sample.

    A violation occurs when any ego reference point from the Top-1 trajectory
    enters the represented restricted WorkZone polygon.

    No distance threshold is applied.

    Returns:
        violation:
            Boolean tensor `[B]`.

        valid:
            Boolean tensor `[B]` indicating whether the WZ polygon is valid.
    """
    if pred_xy.ndim != 4 or pred_xy.shape[-1] != 2:
        raise ValueError(
            "pred_xy must have shape [B, K, T, 2]."
        )

    batch_size, num_modes = pred_xy.shape[:2]

    if mode_prob.shape != (
        batch_size,
        num_modes,
    ):
        raise ValueError(
            "mode_prob must have shape [B, K]."
        )

    if wz_polygon.ndim != 3 or wz_polygon.shape[-1] != 2:
        raise ValueError(
            "wz_polygon must have shape [B, P, 2]."
        )

    if wz_polygon.shape[0] != batch_size:
        raise ValueError(
            "wz_polygon batch size must match pred_xy."
        )

    if wz_valid is None:
        valid = torch.ones(
            batch_size,
            dtype=torch.bool,
            device=pred_xy.device,
        )
    else:
        if wz_valid.shape != (batch_size,):
            raise ValueError(
                "wz_valid must have shape [B]."
            )

        valid = wz_valid.bool()

    top_idx = mode_prob.argmax(
        dim=1
    )

    batch_idx = torch.arange(
        batch_size,
        device=pred_xy.device,
    )

    selected = pred_xy[
        batch_idx,
        top_idx,
    ]

    flags = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=pred_xy.device,
    )

    for b in range(batch_size):
        if not bool(valid[b]):
            continue

        flags[b] = points_in_polygon(
            selected[b],
            wz_polygon[b],
        ).any()

    return flags, valid


def wz_gvr(
    pred_xy: torch.Tensor,
    mode_prob: torch.Tensor,
    wz_polygon: torch.Tensor,
    wz_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return Top-1 WZ geometry violation rate over valid WZ samples."""
    flags, valid = per_sample_wz_gvr(
        pred_xy,
        mode_prob,
        wz_polygon,
        wz_valid,
    )

    if not bool(valid.any()):
        return torch.tensor(
            float("nan"),
            dtype=pred_xy.dtype,
            device=pred_xy.device,
        )

    return flags[
        valid
    ].float().mean()
