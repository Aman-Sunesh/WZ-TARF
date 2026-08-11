"""ADE of the mode selected by minimum FDE."""
import torch
from wztarf.metrics.common import ade_per_mode, fde_per_mode

def ade_at_minfde(pred_xy: torch.Tensor, gt_xy: torch.Tensor) -> torch.Tensor:
    ade, fde = ade_per_mode(pred_xy, gt_xy), fde_per_mode(pred_xy, gt_xy)
    idx = fde.argmin(dim=1, keepdim=True)
    return ade.gather(1, idx).squeeze(1).mean()
