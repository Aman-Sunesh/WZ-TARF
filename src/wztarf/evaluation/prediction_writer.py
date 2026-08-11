"""Save model predictions and evaluation inputs for reproducible re-scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


PREDICTION_FORMAT_VERSION = "wztarf_predictions_v1"


def _cpu_tensor(
    value: torch.Tensor | None,
) -> torch.Tensor | None:
    """Detach an optional tensor and move it to CPU."""
    if value is None:
        return None

    return (
        value
        .detach()
        .cpu()
        .contiguous()
    )


def _validate_prediction_artifact(
    pred_xy: torch.Tensor,
    mode_prob: torch.Tensor,
    gt_xy: torch.Tensor,
) -> None:
    """Validate canonical prediction artifact shapes."""
    if pred_xy.ndim != 4 or pred_xy.shape[-1] != 2:
        raise ValueError(
            "pred_xy must have shape [N, K, T, 2]."
        )

    if gt_xy.ndim != 3 or gt_xy.shape[-1] != 2:
        raise ValueError(
            "gt_xy must have shape [N, T, 2]."
        )

    if mode_prob.ndim != 2:
        raise ValueError(
            "mode_prob must have shape [N, K]."
        )

    if pred_xy.shape[0] != gt_xy.shape[0]:
        raise ValueError(
            "Prediction and GT sample counts do not match."
        )

    if pred_xy.shape[2] != gt_xy.shape[1]:
        raise ValueError(
            "Prediction and GT horizons do not match."
        )

    if pred_xy.shape[:2] != mode_prob.shape:
        raise ValueError(
            "mode_prob dimensions do not match pred_xy."
        )


def save_predictions(
    path: str | Path,
    *,
    pred_xy: torch.Tensor,
    mode_prob: torch.Tensor,
    gt_xy: torch.Tensor,
    metadata: Sequence[Mapping[str, Any]] | None = None,
    source_paths: Sequence[str | None] | None = None,
    wz_feat: torch.Tensor | None = None,
    worker_feat: torch.Tensor | None = None,
    fps: int = 5,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Save predictions and all inputs required for post-hoc metrics.

    The serialized artifact includes:

        pred_xy
            `[N, K, T, 2]`

        mode_prob
            `[N, K]`

        gt_xy
            `[N, T, 2]`

        wz_feat
            optional `[N, 5, 3]`

        wz_worker_feat
            optional `[N, W, 3]`

        metadata
            one dictionary per sample

        source_paths
            optional serialized-data paths

        fps
            dataset sampling rate

    Args:
        path:
            Destination `.pt` file.

        extra:
            Optional additional serializable information such as run ID,
            split name, checkpoint path, or configuration hash.

    Returns:
        Resolved path to the saved prediction artifact.
    """
    path = Path(path)

    if path.suffix.lower() != ".pt":
        raise ValueError(
            "Prediction artifacts must use the '.pt' extension."
        )

    _validate_prediction_artifact(
        pred_xy,
        mode_prob,
        gt_xy,
    )

    num_samples = int(
        pred_xy.shape[0]
    )

    if metadata is not None:
        if len(metadata) != num_samples:
            raise ValueError(
                "metadata length must equal the number of predictions."
            )

        metadata_payload = [
            dict(item)
            for item in metadata
        ]

    else:
        metadata_payload = [
            {}
            for _ in range(num_samples)
        ]

    if source_paths is not None:
        if len(source_paths) != num_samples:
            raise ValueError(
                "source_paths length must equal the number of predictions."
            )

        source_payload = [
            str(item)
            if item is not None
            else None
            for item in source_paths
        ]

    else:
        source_payload = [
            None
            for _ in range(num_samples)
        ]

    if wz_feat is not None:
        if wz_feat.shape[0] != num_samples:
            raise ValueError(
                "wz_feat sample count does not match predictions."
            )

    if worker_feat is not None:
        if worker_feat.shape[0] != num_samples:
            raise ValueError(
                "worker_feat sample count does not match predictions."
            )

    payload: dict[str, Any] = {
        "format_version": PREDICTION_FORMAT_VERSION,
        "pred_xy": _cpu_tensor(
            pred_xy
        ),
        "mode_prob": _cpu_tensor(
            mode_prob
        ),
        "gt_xy": _cpu_tensor(
            gt_xy
        ),
        "wz_feat": _cpu_tensor(
            wz_feat
        ),
        "wz_worker_feat": _cpu_tensor(
            worker_feat
        ),
        "metadata": metadata_payload,
        "source_paths": source_payload,
        "fps": int(fps),
        "num_samples": num_samples,
        "num_modes": int(
            pred_xy.shape[1]
        ),
        "future_steps": int(
            pred_xy.shape[2]
        ),
        "extra": (
            dict(extra)
            if extra is not None
            else {}
        ),
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        payload,
        path,
    )

    return path.resolve()


def load_predictions(
    path: str | Path,
) -> dict[str, Any]:
    """Load and validate a saved WZ-TARF prediction artifact."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Prediction artifact does not exist: {path}"
        )

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(payload, Mapping):
        raise TypeError(
            "Prediction artifact must contain a dictionary."
        )

    required = {
        "format_version",
        "pred_xy",
        "mode_prob",
        "gt_xy",
    }

    missing = sorted(
        required
        -
        set(payload.keys())
    )

    if missing:
        raise KeyError(
            f"Prediction artifact is missing fields: {missing}"
        )

    if payload["format_version"] != PREDICTION_FORMAT_VERSION:
        raise ValueError(
            "Unsupported prediction artifact version: "
            f"{payload['format_version']}"
        )

    _validate_prediction_artifact(
        payload["pred_xy"],
        payload["mode_prob"],
        payload["gt_xy"],
    )

    return dict(payload)
