"""Compute the union of WorkZone geometry and worker-safety violations."""

from __future__ import annotations

import torch

from wztarf.metrics.safety.wsvr import (
    per_sample_worker_violation,
)
from wztarf.metrics.safety.wz_gvr import (
    per_sample_wz_gvr,
)


def wzvr(
    pred_xy: torch.Tensor,
    mode_prob: torch.Tensor,
    wz_polygon: torch.Tensor,
    worker_xy: torch.Tensor,
    worker_mask: torch.Tensor,
    wz_valid: torch.Tensor | None = None,
    worker_threshold_m: float = 2.0,
) -> torch.Tensor:
    """Return the Top-1 WorkZone violation union rate.

    A sample violates if either:

    - its Top-1 trajectory enters the restricted WorkZone polygon; or
    - when a valid represented worker exists, its Top-1 trajectory comes
      within `worker_threshold_m` of any worker.

    Samples without workers are still evaluated for geometry violations.
    Samples with no valid WZ geometry are excluded because WZVR cannot be
    completely evaluated for them.
    """
    geometry, valid_wz = per_sample_wz_gvr(
        pred_xy,
        mode_prob,
        wz_polygon,
        wz_valid,
    )

    worker, _ = per_sample_worker_violation(
        pred_xy,
        mode_prob,
        worker_xy,
        worker_mask,
        worker_threshold_m,
    )

    if not bool(valid_wz.any()):
        return torch.tensor(
            float("nan"),
            dtype=pred_xy.dtype,
            device=pred_xy.device,
        )

    combined = (
        geometry
        |
        worker
    )

    return combined[
        valid_wz
    ].float().mean()
