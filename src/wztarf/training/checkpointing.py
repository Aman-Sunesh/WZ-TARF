"""Checkpoint save/resume helpers for WZ-TARF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class CheckpointState:
    """Metadata restored from a training checkpoint."""

    epoch: int
    global_step: int
    best_metric: float | None
    config: dict[str, Any]
    extra: dict[str, Any]
    path: Path


def _torch_load(path: Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _model_state(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, Mapping) and value and all(torch.is_tensor(v) for v in value.values()):
            return value

    if payload and all(torch.is_tensor(v) for v in payload.values()):
        return payload  # direct model.state_dict() checkpoint

    raise KeyError("Checkpoint contains no model state_dict")


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int,
    global_step: int,
    best_metric: float | None = None,
    config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a complete resumable training state."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "format_version": 1,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": None if best_metric is None else float(best_metric),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": dict(config or {}),
        "extra": dict(extra or {}),
    }

    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> CheckpointState:
    """Restore model and optional optimizer/scheduler/scaler state."""
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    payload = _torch_load(checkpoint_path, map_location)
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint must be mapping-like")

    model.load_state_dict(_model_state(payload), strict=strict)

    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])

    return CheckpointState(
        epoch=int(payload.get("epoch", 0)),
        global_step=int(payload.get("global_step", 0)),
        best_metric=(
            None if payload.get("best_metric") is None else float(payload["best_metric"])
        ),
        config=dict(payload.get("config") or {}),
        extra=dict(payload.get("extra") or {}),
        path=checkpoint_path.resolve(),
    )


def load_pretrained_backbone(
    path: str | Path,
    *,
    model: torch.nn.Module,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
):
    """Load only neural weights from a Phase-A or regular checkpoint.

    Optimizer/scheduler state is intentionally ignored.  A common ``backbone.``
    prefix is stripped when every parameter carries it.
    """
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    payload = _torch_load(checkpoint_path, map_location)
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint must be mapping-like")

    state = dict(_model_state(payload))
    if state and all(key.startswith("backbone.") for key in state):
        state = {key[len("backbone."):]: value for key, value in state.items()}

    # Phase-A checkpoints are saved from WZTARFPretrainingModel.
    # In that model the ordinary forecasting network lives under
    # ``backbone.*`` while training-only heads live at the top level
    # (future_encoder.*, reconstruction.*, topology.*, etc.).
    #
    # When such a checkpoint is loaded into the Phase-B WZTARF model,
    # extract only the forecasting backbone and remove the namespace.
    # Ordinary Phase-B/full-model checkpoints are left unchanged.
    if isinstance(state, dict):
        model_keys = set(model.state_dict().keys())
        raw_keys = set(state.keys())

        phase_a_backbone = {
            key[len("backbone."):]: value
            for key, value in state.items()
            if isinstance(key, str)
            and key.startswith("backbone.")
        }

        if phase_a_backbone:
            stripped_keys = set(
                phase_a_backbone.keys()
            )

            # A raw Phase-A checkpoint has no direct WZTARF key matches;
            # its WZTARF parameters are namespaced under backbone.*.
            # Require meaningful agreement with the target model before
            # performing the remap so unrelated checkpoints are not altered.
            direct_matches = raw_keys & model_keys
            stripped_matches = stripped_keys & model_keys

            if (
                not direct_matches
                and stripped_matches
            ):
                state = phase_a_backbone

    return model.load_state_dict(state, strict=strict)
