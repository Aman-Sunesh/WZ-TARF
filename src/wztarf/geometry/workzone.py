"""Geometry for the restricted WorkZone polygon."""

import torch


def points_in_polygon(points: torch.Tensor, polygon: torch.Tensor) -> torch.Tensor:
    """Return a boolean point-in-polygon flag for each `[N,2]` input point."""
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError("points must have shape [N,2]")
    if polygon.ndim != 2 or polygon.shape[-1] != 2:
        raise ValueError("polygon must have shape [P,2]")

    x, y = points[:, 0], points[:, 1]
    px, py = polygon[:, 0], polygon[:, 1]
    inside = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    j = polygon.shape[0] - 1
    for i in range(polygon.shape[0]):
        crosses = ((py[i] > y) != (py[j] > y)) & (
            x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i] + 1e-12) + px[i]
        )
        inside ^= crosses
        j = i
    return inside
