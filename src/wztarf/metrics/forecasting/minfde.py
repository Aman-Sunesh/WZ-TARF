"""Best-of-K final displacement error."""
import torch
from wztarf.metrics.common import fde_per_mode

def minfde(pred_xy: torch.Tensor, gt_xy: torch.Tensor) -> torch.Tensor:
    """Return batch-mean minFDE."""
    return fde_per_mode(pred_xy, gt_xy).min(dim=1).values.mean()
