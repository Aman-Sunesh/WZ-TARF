"""Best-of-K average displacement error."""
import torch
from wztarf.metrics.common import ade_per_mode

def minade(pred_xy: torch.Tensor, gt_xy: torch.Tensor) -> torch.Tensor:
    """Return batch-mean minADE."""
    return ade_per_mode(pred_xy, gt_xy).min(dim=1).values.mean()
