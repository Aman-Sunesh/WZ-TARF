"""A20 late-X action policy used by the best released WZ-TARF run.

The policy never sees future ground truth at inference.  It reads a compact
descriptor of the six predicted trajectories plus scene-state features, then
chooses one of five fixed forward shifts for the furthest-forward mode.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class ActionPolicy(nn.Module):
    """Five-way policy matching the saved A20 checkpoint architecture."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 192),
            nn.GELU(),
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 5),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def gate_features(
    pred_xy: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the exact 114-D A20 feature vector and far-mode index.

    Args:
        pred_xy: Six trajectories with shape ``[B, 6, 25, 2]``.
        state: Development state vector with shape ``[B, 78]``.
    """
    if pred_xy.ndim != 4 or pred_xy.shape[1] != 6 or pred_xy.shape[-1] != 2:
        raise ValueError("pred_xy must have shape [B, 6, T, 2]")
    if state.ndim != 2 or state.shape[0] != pred_xy.shape[0]:
        raise ValueError("state must have shape [B, D]")

    batch_size = pred_xy.shape[0]
    endpoint = pred_xy[:, :, -1]
    endpoint_x = endpoint[..., 0]
    endpoint_y = endpoint[..., 1]

    order = torch.argsort(endpoint_x, dim=1)
    sorted_x = torch.gather(endpoint_x, 1, order)
    sorted_y = torch.gather(endpoint_y, 1, order)
    gaps = sorted_x[:, 1:] - sorted_x[:, :-1]
    far_idx = order[:, -1]

    rows = torch.arange(batch_size, device=pred_xy.device)
    far_traj = pred_xy[rows, far_idx]

    # Positions at 1, 2, 3, 4 and 5 seconds for 5 Hz / 25-step data.
    sample_idx = torch.tensor([4, 9, 14, 19, 24], device=pred_xy.device)
    if pred_xy.shape[2] != 25:
        raise ValueError("A20 was trained for exactly 25 future steps")
    sampled = far_traj[:, sample_idx].reshape(batch_size, -1)

    last = far_traj[:, -6:]
    velocity = last[:, 1:] - last[:, :-1]
    mean_velocity = velocity.mean(dim=1)
    last_velocity = velocity[:, -1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    mean_acceleration = acceleration.mean(dim=1)

    support = torch.cat(
        (
            sorted_x,
            sorted_y,
            gaps,
            sorted_x[:, -1:] - sorted_x[:, :1],
            sorted_x.mean(dim=1, keepdim=True),
            sorted_x.std(dim=1, unbiased=False, keepdim=True),
            sampled,
            mean_velocity,
            last_velocity,
            mean_acceleration,
        ),
        dim=1,
    )
    return torch.cat((state.float(), support.float()), dim=1), far_idx


def apply_a20_policy(
    pred_xy: torch.Tensor,
    state: torch.Tensor,
    checkpoint: str | Path | dict,
    *,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the frozen A20 policy to a six-mode prediction set.

    Returns the corrected trajectories and the selected action index.
    """
    payload = (
        torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict)
        else checkpoint
    )
    required = {
        "state_dict", "feature_mean", "feature_std", "actions", "onset", "power"
    }
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"A20 checkpoint is missing: {sorted(missing)}")

    features, far_idx = gate_features(pred_xy, state)
    feature_mean = payload["feature_mean"].float()
    feature_std = payload["feature_std"].float().clamp_min(1e-4)
    if features.shape[1] != feature_mean.numel():
        raise ValueError(
            f"A20 feature mismatch: got {features.shape[1]}, "
            f"expected {feature_mean.numel()}"
        )

    run_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    policy = ActionPolicy(features.shape[1]).to(run_device)
    policy.load_state_dict(payload["state_dict"], strict=True)
    policy.eval()

    normalized = (features.float() - feature_mean) / feature_std
    with torch.inference_mode():
        action_idx = policy(normalized.to(run_device)).argmax(dim=1).cpu()

    actions = payload["actions"].float()
    shifts = actions[action_idx]
    steps = pred_xy.shape[2]
    tau = torch.arange(1, steps + 1, dtype=torch.float32) / float(steps)
    onset = float(payload["onset"])
    power = float(payload["power"])
    phi = ((tau - onset) / (1.0 - onset)).clamp(0.0, 1.0).pow(power)

    corrected = pred_xy.clone()
    rows = torch.arange(corrected.shape[0])
    corrected[rows, far_idx.cpu(), :, 0] += shifts[:, None] * phi[None]
    return corrected, action_idx
