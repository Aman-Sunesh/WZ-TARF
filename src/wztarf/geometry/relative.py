"""Construct distance, direction, and relative-heading geometric features."""

from __future__ import annotations

import torch


def relative_relation(
    src_xy: torch.Tensor,
    dst_xy: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return Cartesian displacement plus explicit distance and direction.

    For source position `src_xy` and destination position `dst_xy`, returns:

        [delta_x,
         delta_y,
         distance,
         sin(bearing),
         cos(bearing)]

    The input tensors may contain arbitrary leading dimensions as long as
    their final dimension is 2 and they are broadcast-compatible.

    Args:
        src_xy:
            Source XY positions `[..., 2]`.

        dst_xy:
            Destination XY positions `[..., 2]`.

        eps:
            Numerical stabilizer used when distance is zero.

    Returns:
        Relative feature tensor `[..., 5]`.
    """
    if src_xy.shape[-1] != 2:
        raise ValueError(
            "src_xy must end with dimension 2."
        )

    if dst_xy.shape[-1] != 2:
        raise ValueError(
            "dst_xy must end with dimension 2."
        )

    if eps <= 0:
        raise ValueError(
            "eps must be positive."
        )

    delta = (
        dst_xy
        -
        src_xy
    )

    dx = delta[..., 0]
    dy = delta[..., 1]

    distance = torch.linalg.vector_norm(
        delta,
        dim=-1,
    )

    sin_bearing = (
        dy
        /
        (distance + eps)
    )

    cos_bearing = (
        dx
        /
        (distance + eps)
    )

    return torch.stack(
        (
            dx,
            dy,
            distance,
            sin_bearing,
            cos_bearing,
        ),
        dim=-1,
    )


def relative_heading(
    src_yaw: torch.Tensor,
    dst_yaw: torch.Tensor,
) -> torch.Tensor:
    """Encode relative heading without angular wrap discontinuities.

    Returns:

        [sin(dst_yaw - src_yaw),
         cos(dst_yaw - src_yaw)]

    Args:
        src_yaw:
            Source heading in radians.

        dst_yaw:
            Destination heading in radians.

    Returns:
        Tensor with final dimension 2.
    """
    delta_yaw = (
        dst_yaw
        -
        src_yaw
    )

    return torch.stack(
        (
            torch.sin(delta_yaw),
            torch.cos(delta_yaw),
        ),
        dim=-1,
    )
