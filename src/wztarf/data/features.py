"""Construct temporal motion, control, gaze, and physical-time features."""

from __future__ import annotations
import math
import torch


def relative_history_time(
    history_steps: int = 10,
    fps: int = 5,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return physical history times ending at the prediction anchor.

    For 10 observations at 5 Hz:

        [-1.8, -1.6, ..., -0.2, 0.0]
    """
    if history_steps <= 0:
        raise ValueError("history_steps must be positive.")

    if fps <= 0:
        raise ValueError("fps must be positive.")

    return (
        torch.arange(
            -(history_steps - 1),
            1,
            dtype=dtype,
            device=device,
        )
        / float(fps)
    )


def future_time(
    future_steps: int = 25,
    fps: int = 5,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return physical future times beginning one frame after the anchor.

    For 25 future observations at 5 Hz:

        [0.2, 0.4, ..., 5.0]
    """
    if future_steps <= 0:
        raise ValueError("future_steps must be positive.")

    if fps <= 0:
        raise ValueError("fps must be positive.")

    return (
        torch.arange(
            1,
            future_steps + 1,
            dtype=dtype,
            device=device,
        )
        / float(fps)
    )


def _history_time_column(
    reference: torch.Tensor,
    fps: int,
) -> torch.Tensor:
    """Create a broadcast-compatible `[..., T, 1]` time column."""
    if reference.ndim < 2:
        raise ValueError(
            "Reference tensor must contain time and feature dimensions."
        )

    time_steps = reference.shape[-2]

    times = relative_history_time(
        history_steps=time_steps,
        fps=fps,
        dtype=reference.dtype,
        device=reference.device,
    )

    # [T] -> [1, ..., 1, T, 1]
    view_shape = (
        *([1] * (reference.ndim - 2)),
        time_steps,
        1,
    )

    times = times.reshape(view_shape)

    target_shape = (
        *reference.shape[:-1],
        1,
    )

    return times.expand(target_shape)


def _temporal_difference(
    values: torch.Tensor,
) -> torch.Tensor:
    """Return frame-to-frame differences with zero at the first timestep."""
    if values.ndim < 2:
        raise ValueError(
            "Expected tensor with time on the second-to-last dimension."
        )

    delta = torch.zeros_like(values)

    delta[..., 1:, :] = (
        values[..., 1:, :]
        -
        values[..., :-1, :]
    )

    return delta


def _wrapped_yaw_rate(
    yaw: torch.Tensor,
    fps: int,
) -> torch.Tensor:
    """Compute wrapped yaw rate in radians per second.

    `yaw` has shape `[..., T]`.
    """
    if yaw.ndim < 1:
        raise ValueError("Yaw tensor must have a time dimension.")

    yaw_rate = torch.zeros_like(yaw)

    raw_delta = (
        yaw[..., 1:]
        -
        yaw[..., :-1]
    )

    # atan2(sin Δψ, cos Δψ) gives a wrapped angle in [-π, π].
    wrapped_delta = torch.atan2(
        torch.sin(raw_delta),
        torch.cos(raw_delta),
    )

    yaw_rate[..., 1:] = wrapped_delta * float(fps)

    return yaw_rate


def build_motion_features(
    ego_hist: torch.Tensor,
    fps: int = 5,
) -> torch.Tensor:
    """Construct the ego-motion input used by the motion GRU.

    Expected source layout:

        ego_hist[..., T, 6]
        = [x, y, vx, vy, yaw, speed]

    Returned feature layout:

        [x,
         y,
         vx,
         vy,
         ax,
         ay,
         sin(yaw),
         cos(yaw),
         yaw_rate,
         speed,
         relative_time]

    Output shape:

        [..., T, 11]
    """
    if ego_hist.ndim < 2:
        raise ValueError(
            "ego_hist must have shape [..., T, 6]."
        )

    if ego_hist.shape[-1] < 6:
        raise ValueError(
            f"ego_hist requires at least 6 features, "
            f"got {ego_hist.shape[-1]}."
        )

    if fps <= 0:
        raise ValueError("fps must be positive.")

    x = ego_hist[..., 0:1]
    y = ego_hist[..., 1:2]

    velocity = ego_hist[..., 2:4]

    yaw = ego_hist[..., 4]

    speed = ego_hist[..., 5:6]

    # Acceleration = Δvelocity / Δt = Δvelocity * fps.
    acceleration = (
        _temporal_difference(velocity)
        *
        float(fps)
    )

    yaw_rate = _wrapped_yaw_rate(
        yaw,
        fps=fps,
    ).unsqueeze(-1)

    sin_yaw = torch.sin(yaw).unsqueeze(-1)
    cos_yaw = torch.cos(yaw).unsqueeze(-1)

    time = _history_time_column(
        ego_hist,
        fps=fps,
    )

    return torch.cat(
        (
            x,
            y,
            velocity,
            acceleration,
            sin_yaw,
            cos_yaw,
            yaw_rate,
            speed,
            time,
        ),
        dim=-1,
    )


def build_control_features(
    control_hist: torch.Tensor,
    control_mask: torch.Tensor,
    fps: int = 5,
) -> torch.Tensor:
    """Construct the control-stream input.

    Expected source layout:

        control_hist[..., T, 3]
        = [steering, throttle, brake]

        control_mask[..., T]

    Returned layout:

        [steering,
         throttle,
         brake,
         delta_steering,
         delta_throttle,
         delta_brake,
         validity_mask,
         relative_time]

    Output shape:

        [..., T, 8]

    The deltas are frame-to-frame changes, not per-second derivatives.
    """
    if control_hist.ndim < 2:
        raise ValueError(
            "control_hist must have shape [..., T, 3]."
        )

    if control_hist.shape[-1] != 3:
        raise ValueError(
            f"Expected 3 control features, "
            f"got {control_hist.shape[-1]}."
        )

    if control_mask.shape != control_hist.shape[:-1]:
        raise ValueError(
            "control_mask must match control_hist without "
            "the feature dimension."
        )

    if fps <= 0:
        raise ValueError("fps must be positive.")

    delta_control = _temporal_difference(
        control_hist
    )

    validity = (
        control_mask
        .to(dtype=control_hist.dtype)
        .unsqueeze(-1)
    )

    time = _history_time_column(
        control_hist,
        fps=fps,
    )

    return torch.cat(
        (
            control_hist,
            delta_control,
            validity,
            time,
        ),
        dim=-1,
    )


def build_gaze_features(
    gaze_feat: torch.Tensor,
    gaze_mask: torch.Tensor,
    fps: int = 5,
) -> torch.Tensor:
    """Construct the temporal gaze-stream input.

    Expected source layout:

        gaze_feat[..., T, 3]
        = [gaze_x, gaze_y, confidence]

        gaze_mask[..., T]

    Returned layout:

        [gaze_x,
         gaze_y,
         confidence,
         validity_mask,
         relative_time]

    Output shape:

        [..., T, 5]

    Image-space normalization, reliability statistics, and route-intent
    gating belong to the gaze encoder rather than this data utility.
    """
    if gaze_feat.ndim < 2:
        raise ValueError(
            "gaze_feat must have shape [..., T, 3]."
        )

    if gaze_feat.shape[-1] != 3:
        raise ValueError(
            f"Expected 3 gaze features, "
            f"got {gaze_feat.shape[-1]}."
        )

    if gaze_mask.shape != gaze_feat.shape[:-1]:
        raise ValueError(
            "gaze_mask must match gaze_feat without "
            "the feature dimension."
        )

    validity = (
        gaze_mask
        .to(dtype=gaze_feat.dtype)
        .unsqueeze(-1)
    )

    time = _history_time_column(
        gaze_feat,
        fps=fps,
    )

    return torch.cat(
        (
            gaze_feat,
            validity,
            time,
        ),
        dim=-1,
    )
