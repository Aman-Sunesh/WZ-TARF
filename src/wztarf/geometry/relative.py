"""Cartesian plus distance/direction relational features."""

import torch


def relative_relation(src_xy: torch.Tensor, dst_xy: torch.Tensor, eps: float = 1e-8):
    """Return Δx, Δy, distance, sin(bearing), and cos(bearing)."""
    delta = dst_xy - src_xy
    dx, dy = delta.unbind(dim=-1)
    rho = torch.linalg.vector_norm(delta, dim=-1)
    return torch.stack((dx, dy, rho, dy / (rho + eps), dx / (rho + eps)), dim=-1)
