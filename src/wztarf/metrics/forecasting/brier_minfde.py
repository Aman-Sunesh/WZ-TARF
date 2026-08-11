"""Brier-minFDE endpoint error plus probability penalty."""
import torch
from wztarf.metrics.common import fde_per_mode

def brier_minfde(pred_xy: torch.Tensor, gt_xy: torch.Tensor, mode_prob: torch.Tensor) -> torch.Tensor:
    fde = fde_per_mode(pred_xy, gt_xy)
    best_fde, idx = fde.min(dim=1)
    p = mode_prob.gather(1, idx[:, None]).squeeze(1)
    return (best_fde + (1.0 - p).square()).mean()
