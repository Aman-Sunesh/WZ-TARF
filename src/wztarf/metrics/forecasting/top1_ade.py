"""ADE of the model's highest-probability trajectory."""
import torch
from wztarf.metrics.common import ade_per_mode

def top1_ade(pred_xy: torch.Tensor, gt_xy: torch.Tensor, mode_prob: torch.Tensor) -> torch.Tensor:
    ade = ade_per_mode(pred_xy, gt_xy)
    idx = mode_prob.argmax(dim=1, keepdim=True)
    return ade.gather(1, idx).squeeze(1).mean()
