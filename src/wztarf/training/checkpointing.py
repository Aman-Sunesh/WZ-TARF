"""Save and restore complete training state for reproducible runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_FORMAT_VERSION = "wztarf_checkpoint_v1"


@dataclass
class CheckpointState:
    """Metadata restored from a training checkpoint."""

    epoch: int
    global_step: int
    best_metric: float | None
    config: dict[str, Any]
    extra: dict[str, Any]


def _state_dict_or_none(
    obj: Any,
) -> dict[str, Any] | None:
    """Return an object's state dictionary when the object is provided."""
    if obj is None:
        return None

    state_dict = getattr(
        obj,
        "state_dict",
        None,
    )

    if not callable(state_dict):
        raise TypeError(
            f"{type(obj).__name__} does not provide state_dict()."
        )

    return state_dict()


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int,
    global_step: int = 0,
    best_metric: float | None = None,
    config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Save complete model and optimization state.

    Args:
        path:
            Destination `.pt` or `.pth` checkpoint.

        model:
            Model whose parameters are saved.

        optimizer:
            Optional optimizer state.

        scheduler:
            Optional learning-rate scheduler state.

        scaler:
            Optional AMP GradScaler state.

        epoch:
            Last completed epoch.

        global_step:
            Number of optimization steps completed.

        best_metric:
            Best validation-selection metric observed so far.

        config:
            Run configuration stored with the checkpoint.

        extra:
            Additional metadata such as run ID or validation metrics.

    Returns:
        Resolved checkpoint path.
    """
    path = Path(path).expanduser()

    if path.suffix.lower() not in {
        ".pt",
        ".pth",
        ".ckpt",
    }:
        raise ValueError(
            "Checkpoint path must end in .pt, .pth, or .ckpt."
        )

    if epoch < 0:
        raise ValueError(
            "epoch cannot be negative."
        )

    if global_step < 0:
        raise ValueError(
            "global_step cannot be negative."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": (
            float(best_metric)
            if best_metric is not None
            else None
        ),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": _state_dict_or_none(
            optimizer
        ),
        "scheduler_state_dict": _state_dict_or_none(
            scheduler
        ),
        "scaler_state_dict": _state_dict_or_none(
            scaler
        ),
        "config": (
            dict(config)
            if config is not None
            else {}
        ),
        "extra": (
            dict(extra)
            if extra is not None
            else {}
        ),
    }

    # Write to a temporary file first. The final checkpoint path is replaced
    # only after torch.save finishes successfully.
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        payload,
        temporary_path,
    )

    temporary_path.replace(
        path
    )

    return path.resolve()


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    restore_optimizer: bool = True,
    restore_scheduler: bool = True,
    restore_scaler: bool = True,
) -> CheckpointState:
    """Load model parameters and optionally restore training state.

    Args:
        path:
            Saved checkpoint.

        model:
            Model receiving the saved parameters.

        optimizer:
            Optimizer to restore when available.

        scheduler:
            Scheduler to restore when available.

        scaler:
            AMP scaler to restore when available.

        strict:
            Passed to `model.load_state_dict()`.

        restore_optimizer:
            Restore optimizer state when both the checkpoint and optimizer
            contain one.

        restore_scheduler:
            Restore scheduler state when available.

        restore_scaler:
            Restore AMP scaler state when available.

    Returns:
        Restored checkpoint metadata.
    """
    path = Path(path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {path}"
        )

    payload = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    if not isinstance(
        payload,
        Mapping,
    ):
        raise TypeError(
            "Checkpoint must contain a dictionary."
        )

    version = payload.get(
        "format_version"
    )

    if (
        version is not None
        and version != CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError(
            f"Unsupported checkpoint format: {version}"
        )

    if "model_state_dict" not in payload:
        raise KeyError(
            "Checkpoint is missing 'model_state_dict'."
        )

    model.load_state_dict(
        payload["model_state_dict"],
        strict=strict,
    )

    if (
        restore_optimizer
        and optimizer is not None
        and payload.get("optimizer_state_dict") is not None
    ):
        optimizer.load_state_dict(
            payload["optimizer_state_dict"]
        )

    if (
        restore_scheduler
        and scheduler is not None
        and payload.get("scheduler_state_dict") is not None
    ):
        scheduler.load_state_dict(
            payload["scheduler_state_dict"]
        )

    if (
        restore_scaler
        and scaler is not None
        and payload.get("scaler_state_dict") is not None
    ):
        scaler.load_state_dict(
            payload["scaler_state_dict"]
        )

    return CheckpointState(
        epoch=int(
            payload.get(
                "epoch",
                0,
            )
        ),
        global_step=int(
            payload.get(
                "global_step",
                0,
            )
        ),
        best_metric=(
            float(
                payload["best_metric"]
            )
            if payload.get("best_metric") is not None
            else None
        ),
        config=dict(
            payload.get(
                "config",
                {},
            )
        ),
        extra=dict(
            payload.get(
                "extra",
                {},
            )
        ),
    )

def load_pretrained_backbone(
    path: str | Path,
    *,
    model: torch.nn.Module,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, list[str]]:
    """Load only the WZ-TARF backbone from a Phase A checkpoint."""
    path = Path(
        path
    ).expanduser()

    payload = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    if isinstance(
        payload,
        Mapping,
    ) and "model_state_dict" in payload:
        state_dict = payload[
            "model_state_dict"
        ]
    else:
        state_dict = payload

    if not isinstance(
        state_dict,
        Mapping,
    ):
        raise TypeError(
            "Pretraining checkpoint does not contain a state dictionary."
        )

    prefix = "backbone."

    backbone_state = {
        key[
            len(prefix):
        ]: value
        for key, value in state_dict.items()
        if key.startswith(
            prefix
        )
    }

    if not backbone_state:
        raise ValueError(
            "Checkpoint contains no 'backbone.' parameters."
        )

    incompatible = model.load_state_dict(
        backbone_state,
        strict=False,
    )

    missing = list(
        incompatible.missing_keys
    )

    unexpected = list(
        incompatible.unexpected_keys
    )

    if strict and (
        missing
        or unexpected
    ):
        raise RuntimeError(
            "Phase A backbone does not exactly match WZTARF. "
            f"Missing={missing}, unexpected={unexpected}"
        )

    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }

