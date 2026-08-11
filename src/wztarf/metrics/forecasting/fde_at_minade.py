"""FDE of the mode selected by minimum ADE."""
import torch
from wztarf.metrics.common import ade_per_mode, fde_per_mode

def fde_at_minade(pred_xy: torch.Tensor, gt_xy: torch.Tensor) -> torch.Tensor:
    ade, fde = ade_per_mode(pred_xy, gt_xy), fde_per_mode(pred_xy, gt_xy)
    idx = ade.argmin(dim=1, keepdim=True)
    return fde.gather(1, idx).squeeze(1).mean()
