"""95th percentile of per-sample minADE."""
import torch
from wztarf.metrics.common import ade_per_mode

def p95_minade(pred_xy: torch.Tensor, gt_xy: torch.Tensor) -> torch.Tensor:
    return torch.quantile(ade_per_mode(pred_xy, gt_xy).min(dim=1).values, 0.95)
