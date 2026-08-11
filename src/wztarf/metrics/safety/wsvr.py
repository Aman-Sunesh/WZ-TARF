"""Worker Safety Violation Rate (WSVR@2m)."""

import torch


def per_sample_worker_violation(
    pred_xy: torch.Tensor,
    mode_prob: torch.Tensor,
    worker_xy: torch.Tensor,
    worker_mask: torch.Tensor,
    threshold_m: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return violation flags and a mask showing which samples contain workers."""
    top_idx = mode_prob.argmax(dim=1)
    batch_idx = torch.arange(pred_xy.shape[0], device=pred_xy.device)
    selected = pred_xy[batch_idx, top_idx]

    has_worker = worker_mask.any(dim=1)
    flags = torch.zeros(pred_xy.shape[0], dtype=torch.bool, device=pred_xy.device)

    for b in range(pred_xy.shape[0]):
        valid_workers = worker_xy[b][worker_mask[b]]
        if valid_workers.numel() == 0:
            continue
        flags[b] = torch.cdist(selected[b], valid_workers).min() < threshold_m

    return flags, has_worker


def wsvr(
    pred_xy: torch.Tensor,
    mode_prob: torch.Tensor,
    worker_xy: torch.Tensor,
    worker_mask: torch.Tensor,
    threshold_m: float = 2.0,
) -> torch.Tensor:
    """Return WSVR conditional on at least one valid represented worker."""
    flags, has_worker = per_sample_worker_violation(
        pred_xy, mode_prob, worker_xy, worker_mask, threshold_m
    )
    if not bool(has_worker.any()):
        return torch.tensor(float("nan"), device=pred_xy.device)
    return flags[has_worker].float().mean()
