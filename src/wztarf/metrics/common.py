"""Shared displacement-error helpers."""

import torch


def validate_prediction_shapes(pred_xy: torch.Tensor, gt_xy: torch.Tensor) -> None:
    """Validate `[B,K,T,2]` predictions and `[B,T,2]` ground truth."""
    if pred_xy.ndim != 4 or pred_xy.shape[-1] != 2:
        raise ValueError("pred_xy must have shape [B,K,T,2]")
    if gt_xy.ndim != 3 or gt_xy.shape[-1] != 2:
        raise ValueError("gt_xy must have shape [B,T,2]")
    if pred_xy.shape[0] != gt_xy.shape[0] or pred_xy.shape[2] != gt_xy.shape[1]:
        raise ValueError("batch size and horizon must match")


def displacement_error(pred_xy: torch.Tensor, gt_xy: torch.Tensor) -> torch.Tensor:
    """Return Euclidean displacement error `[B,K,T]`."""
    validate_prediction_shapes(pred_xy, gt_xy)
    return torch.linalg.vector_norm(pred_xy - gt_xy[:, None], dim=-1)


def ade_per_mode(pred_xy: torch.Tensor, gt_xy: torch.Tensor) -> torch.Tensor:
    """Return ADE per candidate mode as `[B,K]`."""
    return displacement_error(pred_xy, gt_xy).mean(dim=-1)


def fde_per_mode(pred_xy: torch.Tensor, gt_xy: torch.Tensor) -> torch.Tensor:
    """Return FDE per candidate mode as `[B,K]`."""
    return displacement_error(pred_xy, gt_xy)[:, :, -1]
