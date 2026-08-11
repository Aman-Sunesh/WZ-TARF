"""Best-of-K miss rate using a terminal-error threshold."""
import torch
from wztarf.metrics.common import fde_per_mode

def miss_rate(pred_xy: torch.Tensor, gt_xy: torch.Tensor, threshold_m: float = 2.0) -> torch.Tensor:
    best = fde_per_mode(pred_xy, gt_xy).min(dim=1).values
    return (best > threshold_m).float().mean()
