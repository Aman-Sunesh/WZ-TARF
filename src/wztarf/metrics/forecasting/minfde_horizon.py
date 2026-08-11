"""Best-of-K displacement error at an exact future step."""
import torch
from wztarf.metrics.common import displacement_error

def minfde_horizon(pred_xy: torch.Tensor, gt_xy: torch.Tensor, horizon_step: int) -> torch.Tensor:
    """Compute min displacement error at a 1-indexed future step."""
    if not 1 <= horizon_step <= pred_xy.shape[2]:
        raise ValueError("invalid horizon_step")
    de = displacement_error(pred_xy, gt_xy)[:, :, horizon_step - 1]
    return de.min(dim=1).values.mean()
