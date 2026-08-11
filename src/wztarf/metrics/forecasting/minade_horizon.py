"""Best-of-K ADE over an exact evaluation horizon."""
import torch
from wztarf.metrics.common import displacement_error

def minade_horizon(pred_xy: torch.Tensor, gt_xy: torch.Tensor, horizon_steps: int) -> torch.Tensor:
    """Compute minADE over the first `horizon_steps` future points."""
    if not 1 <= horizon_steps <= pred_xy.shape[2]:
        raise ValueError("invalid horizon_steps")
    ade = displacement_error(pred_xy, gt_xy)[:, :, :horizon_steps].mean(dim=-1)
    return ade.min(dim=1).values.mean()
