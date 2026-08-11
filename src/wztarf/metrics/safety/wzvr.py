"""Aggregate Work-Zone Violation Rate (WZVR)."""

import torch
from wztarf.metrics.safety.wz_gvr import per_sample_wz_gvr
from wztarf.metrics.safety.wsvr import per_sample_worker_violation


def wzvr(
    pred_xy: torch.Tensor,
    mode_prob: torch.Tensor,
    wz_polygon: torch.Tensor,
    worker_xy: torch.Tensor,
    worker_mask: torch.Tensor,
    wz_valid: torch.Tensor | None = None,
    worker_threshold_m: float = 2.0,
) -> torch.Tensor:
    """Return the union of Top-1 WZ geometry and worker-safety violations."""
    geometry = per_sample_wz_gvr(pred_xy, mode_prob, wz_polygon, wz_valid)
    worker, _ = per_sample_worker_violation(
        pred_xy, mode_prob, worker_xy, worker_mask, worker_threshold_m
    )
    return (geometry | worker).float().mean()
