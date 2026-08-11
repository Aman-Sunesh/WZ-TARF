"""Run WZ-TARF inference over a dataset and produce evaluation results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .metrics_runner import compute_all_metrics
from .prediction_writer import save_predictions


def _move_to_device(
    value: Any,
    device: torch.device,
) -> Any:
    """Recursively move tensors to the evaluation device.

    Metadata strings, dictionaries containing non-tensors, and other Python
    objects remain unchanged.
    """
    if isinstance(value, torch.Tensor):
        return value.to(
            device=device,
            non_blocking=True,
        )

    if isinstance(value, Mapping):
        return {
            key: _move_to_device(item, device)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _move_to_device(item, device)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _move_to_device(item, device)
            for item in value
        )

    return value


def _extract_model_outputs(
    output: Any,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Extract canonical predictions from a model output.

    Supported output formats:

    Dictionary:
        {
            "pred_xy": Tensor[B, K, T, 2],
            "mode_prob": Tensor[B, K],
            ... auxiliary outputs ...
        }

    Tuple:
        (
            pred_xy,
            mode_prob,
        )

    Returns:
        pred_xy:
            Multimodal trajectories `[B, K, T, 2]`.

        mode_prob:
            Mode probabilities `[B, K]`.

        auxiliary:
            Any additional dictionary outputs.
    """
    if isinstance(output, Mapping):
        if "pred_xy" not in output:
            raise KeyError(
                "Model output dictionary must contain 'pred_xy'."
            )

        if "mode_prob" not in output:
            raise KeyError(
                "Model output dictionary must contain 'mode_prob'."
            )

        pred_xy = output["pred_xy"]
        mode_prob = output["mode_prob"]

        auxiliary = {
            key: value
            for key, value in output.items()
            if key not in {"pred_xy", "mode_prob"}
        }

    elif isinstance(output, (tuple, list)):
        if len(output) < 2:
            raise ValueError(
                "Tuple/list model output must contain at least "
                "(pred_xy, mode_prob)."
            )

        pred_xy = output[0]
        mode_prob = output[1]
        auxiliary = {}

    else:
        raise TypeError(
            "Model output must be a mapping or tuple/list, "
            f"got {type(output).__name__}."
        )

    if not isinstance(pred_xy, torch.Tensor):
        raise TypeError(
            "'pred_xy' must be a torch.Tensor."
        )

    if not isinstance(mode_prob, torch.Tensor):
        raise TypeError(
            "'mode_prob' must be a torch.Tensor."
        )

    if pred_xy.ndim != 4 or pred_xy.shape[-1] != 2:
        raise ValueError(
            "'pred_xy' must have shape [B, K, T, 2], "
            f"got {tuple(pred_xy.shape)}."
        )

    if mode_prob.ndim != 2:
        raise ValueError(
            "'mode_prob' must have shape [B, K], "
            f"got {tuple(mode_prob.shape)}."
        )

    if pred_xy.shape[:2] != mode_prob.shape:
        raise ValueError(
            "Prediction mode dimensions do not match: "
            f"pred_xy={tuple(pred_xy.shape)}, "
            f"mode_prob={tuple(mode_prob.shape)}."
        )

    return (
        pred_xy,
        mode_prob,
        auxiliary,
    )


def _extract_metadata(
    batch: Mapping[str, Any],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Return one metadata dictionary per sample in the batch."""
    meta = batch.get("meta")

    if meta is None:
        return [
            {}
            for _ in range(batch_size)
        ]

    # Our collator intentionally preserves metadata dictionaries as a list.
    if isinstance(meta, list):
        if len(meta) != batch_size:
            raise ValueError(
                "Metadata list length does not match batch size."
            )

        return [
            dict(item)
            if isinstance(item, Mapping)
            else {"value": item}
            for item in meta
        ]

    # Fall back gracefully for custom DataLoader collators.
    if isinstance(meta, Mapping):
        result: list[dict[str, Any]] = []

        for index in range(batch_size):
            item: dict[str, Any] = {}

            for key, value in meta.items():
                if isinstance(value, torch.Tensor):
                    selected = value[index]

                    item[key] = (
                        selected.item()
                        if selected.numel() == 1
                        else selected.detach().cpu().tolist()
                    )

                elif isinstance(value, (list, tuple)):
                    item[key] = value[index]

                else:
                    item[key] = value

            result.append(item)

        return result

    return [
        {"value": meta}
        for _ in range(batch_size)
    ]


def _extract_source_paths(
    batch: Mapping[str, Any],
    batch_size: int,
) -> list[str | None]:
    """Return serialized source paths when the dataset provides them."""
    source = batch.get("source_path")

    if source is None:
        return [
            None
            for _ in range(batch_size)
        ]

    if isinstance(source, (list, tuple)):
        if len(source) != batch_size:
            raise ValueError(
                "source_path count does not match batch size."
            )

        return [
            str(path)
            if path is not None
            else None
            for path in source
        ]

    return [
        str(source)
        for _ in range(batch_size)
    ]


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    device: str | torch.device,
    output_dir: str | Path | None = None,
    prediction_filename: str = "predictions.pt",
    metrics_filename: str = "metrics.json",
    fps: int = 5,
    miss_threshold_m: float = 2.0,
    worker_threshold_m: float = 2.0,
    save_prediction_file: bool = True,
) -> dict[str, Any]:
    """Evaluate a model over a complete dataset split.

    Required batch fields:
        future_xy:
            `[B, T, 2]`.

    Safety metrics additionally use:
        wz_feat:
            `[B, 5, 3]`, where the first four rows represent
            polygon corners and the final column is validity.

        wz_worker_feat:
            `[B, W, 3]`, where the final column is validity.

    Args:
        model:
            Forecasting model.

        dataloader:
            Evaluation DataLoader.

        device:
            Device used for model inference.

        output_dir:
            Optional directory in which predictions and metrics are saved.

        fps:
            Dataset sampling frequency.

        miss_threshold_m:
            minFDE threshold used by MR.

        worker_threshold_m:
            Distance threshold used by WSVR.

        save_prediction_file:
            Whether to save the complete prediction artifact.

    Returns:
        Dictionary containing:
            metrics
            pred_xy
            mode_prob
            gt_xy
            metadata
            source_paths
    """
    device = torch.device(device)

    model = model.to(device)
    model.eval()

    pred_batches: list[torch.Tensor] = []
    probability_batches: list[torch.Tensor] = []
    gt_batches: list[torch.Tensor] = []

    wz_batches: list[torch.Tensor] = []
    worker_batches: list[torch.Tensor] = []

    metadata: list[dict[str, Any]] = []
    source_paths: list[str | None] = []

    saw_wz = False
    saw_workers = False

    for batch_index, batch in enumerate(dataloader):
        if not isinstance(batch, Mapping):
            raise TypeError(
                "Each DataLoader batch must be dictionary-like."
            )

        if "future_xy" not in batch:
            raise KeyError(
                f"Batch {batch_index} does not contain 'future_xy'."
            )

        batch_size = int(batch["future_xy"].shape[0])

        metadata.extend(
            _extract_metadata(
                batch,
                batch_size,
            )
        )

        source_paths.extend(
            _extract_source_paths(
                batch,
                batch_size,
            )
        )

        device_batch = _move_to_device(
            batch,
            device,
        )

        output = model(device_batch)

        pred_xy, mode_prob, _ = _extract_model_outputs(
            output
        )

        gt_xy = device_batch["future_xy"]

        if gt_xy.ndim != 3 or gt_xy.shape[-1] != 2:
            raise ValueError(
                "'future_xy' must have shape [B, T, 2]."
            )

        if pred_xy.shape[0] != gt_xy.shape[0]:
            raise ValueError(
                "Prediction batch size does not match ground truth."
            )

        if pred_xy.shape[2] != gt_xy.shape[1]:
            raise ValueError(
                "Prediction horizon does not match ground truth."
            )

        pred_batches.append(
            pred_xy.detach().cpu()
        )

        probability_batches.append(
            mode_prob.detach().cpu()
        )

        gt_batches.append(
            gt_xy.detach().cpu()
        )

        if "wz_feat" in device_batch:
            saw_wz = True

            wz_batches.append(
                device_batch["wz_feat"]
                .detach()
                .cpu()
            )

        if "wz_worker_feat" in device_batch:
            saw_workers = True

            worker_batches.append(
                device_batch["wz_worker_feat"]
                .detach()
                .cpu()
            )

    if not pred_batches:
        raise ValueError(
            "Evaluation DataLoader produced zero batches."
        )

    pred_xy = torch.cat(
        pred_batches,
        dim=0,
    )

    mode_prob = torch.cat(
        probability_batches,
        dim=0,
    )

    gt_xy = torch.cat(
        gt_batches,
        dim=0,
    )

    wz_feat = (
        torch.cat(wz_batches, dim=0)
        if saw_wz
        else None
    )

    worker_feat = (
        torch.cat(worker_batches, dim=0)
        if saw_workers
        else None
    )

    metrics = compute_all_metrics(
        pred_xy=pred_xy,
        gt_xy=gt_xy,
        mode_prob=mode_prob,
        wz_feat=wz_feat,
        worker_feat=worker_feat,
        fps=fps,
        miss_threshold_m=miss_threshold_m,
        worker_threshold_m=worker_threshold_m,
    )

    result: dict[str, Any] = {
        "metrics": metrics,
        "pred_xy": pred_xy,
        "mode_prob": mode_prob,
        "gt_xy": gt_xy,
        "metadata": metadata,
        "source_paths": source_paths,
    }

    if wz_feat is not None:
        result["wz_feat"] = wz_feat

    if worker_feat is not None:
        result["wz_worker_feat"] = worker_feat

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if save_prediction_file:
            save_predictions(
                output_dir / prediction_filename,
                pred_xy=pred_xy,
                mode_prob=mode_prob,
                gt_xy=gt_xy,
                metadata=metadata,
                source_paths=source_paths,
                wz_feat=wz_feat,
                worker_feat=worker_feat,
                fps=fps,
            )

        metrics_path = (
            output_dir
            /
            metrics_filename
        )

        metrics_path.write_text(
            json.dumps(
                metrics,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    return result


def _load_model_state(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a model checkpoint using common checkpoint conventions."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Checkpoint must be dictionary-like."
        )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        # Support checkpoints saved directly as model.state_dict().
        tensor_values = all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        )

        if not tensor_values:
            raise KeyError(
                "Could not find 'model_state_dict' or 'state_dict' "
                "inside checkpoint."
            )

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=strict,
    )

    return dict(checkpoint)


def evaluate_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    dataloader: DataLoader,
    *,
    device: str | torch.device,
    output_dir: str | Path | None = None,
    strict: bool = True,
    fps: int = 5,
    miss_threshold_m: float = 2.0,
    worker_threshold_m: float = 2.0,
    save_prediction_file: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint and evaluate it on one dataset split."""
    checkpoint = _load_model_state(
        model,
        checkpoint_path,
        map_location="cpu",
        strict=strict,
    )

    result = evaluate_model(
        model,
        dataloader,
        device=device,
        output_dir=output_dir,
        fps=fps,
        miss_threshold_m=miss_threshold_m,
        worker_threshold_m=worker_threshold_m,
        save_prediction_file=save_prediction_file,
    )

    result["checkpoint_path"] = str(
        Path(checkpoint_path).resolve()
    )

    result["checkpoint_metadata"] = {
        key: value
        for key, value in checkpoint.items()
        if key not in {
            "state_dict",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
        }
        and not isinstance(value, torch.Tensor)
    }

    return result
