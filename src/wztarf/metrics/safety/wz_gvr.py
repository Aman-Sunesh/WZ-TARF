"""Work-Zone Geometry Violation Rate (WZ-GVR)."""

import torch
from wztarf.geometry.workzone import points_in_polygon


def per_sample_wz_gvr(
    pred_xy: torch.Tensor,
    mode_prob: torch.Tensor,
    wz_polygon: torch.Tensor,
    wz_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return one Top-1 WorkZone-geometry violation flag per sample."""
    top_idx = mode_prob.argmax(dim=1)
    batch_idx = torch.arange(pred_xy.shape[0], device=pred_xy.device)
    selected = pred_xy[batch_idx, top_idx]
    flags = torch.zeros(pred_xy.shape[0], dtype=torch.bool, device=pred_xy.device)

    for b in range(pred_xy.shape[0]):
        if wz_valid is not None and not bool(wz_valid[b]):
            continue
        flags[b] = points_in_polygon(selected[b], wz_polygon[b]).any()
    return flags


def wz_gvr(
    pred_xy: torch.Tensor,
    mode_prob: torch.Tensor,
    wz_polygon: torch.Tensor,
    wz_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the fraction of Top-1 trajectories entering restricted WZ geometry."""
    return per_sample_wz_gvr(pred_xy, mode_prob, wz_polygon, wz_valid).float().mean()
