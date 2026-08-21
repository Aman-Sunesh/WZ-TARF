"""Evaluate a trained checkpoint and generate final run reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from wztarf.data import (
    WorkZoneDataset,
    collate_workzone_batch,
    collate_workzone_fixed,
)
from wztarf.evaluation import evaluate_checkpoint
from wztarf.metrics.efficiency import (
    inference_latency_ms,
    parameter_count,
    peak_gpu_memory_mb,
    reset_peak_gpu_memory,
    throughput_samples_per_second,
)
from wztarf.models import WZTARF, WZTARFConfig
from wztarf.reporting import (
    append_experiment_row,
    environment_snapshot,
    write_markdown_report,
)
from wztarf.utils import load_yaml, seed_all


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    """Parse command-line evaluation options."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained WZ-TARF checkpoint."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "base.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--data-roots",
        type=str,
        required=True,
        help="Comma-separated processed dataset roots.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Examples: cuda, cuda:0, cpu. Defaults to CUDA when available.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
    )

    return parser.parse_args()


def _device_from_arg(
    requested: str | None,
) -> torch.device:
    """Resolve the requested inference device."""
    if requested is not None:
        device = torch.device(
            requested
        )
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but CUDA is unavailable."
        )

    return device


def _move_to_device(
    value: Any,
    device: torch.device,
) -> Any:
    """Recursively move tensors while preserving metadata."""
    if isinstance(value, torch.Tensor):
        return value.to(
            device=device,
            non_blocking=True,
        )

    if isinstance(value, dict):
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


def main() -> None:
    """Evaluate one checkpoint and write predictions, metrics, and reports."""
    args = _parse_args()

    config = load_yaml(
        args.config
    )

    seed = int(
        config["experiment"]["seed"]
    )

    seed_all(
        seed
    )
    torch.set_float32_matmul_precision("high")

    device = _device_from_arg(
        args.device
    )

    data_config = config["data"]
    evaluation_config = config["evaluation"]

    split = (
        args.split
        if args.split is not None
        else data_config.get(
            "test_split",
            "test",
        )
    )

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(
            evaluation_config.get(
                "batch_size",
                8,
            )
        )
    )

    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(
            data_config.get(
                "num_workers",
                0,
            )
        )
    )

    roots = [
        Path(item.strip())
        for item in args.data_roots.split(",")
        if item.strip()
    ]

    dataset = WorkZoneDataset(
        roots=roots,
        split=split,
        validate=bool(data_config.get("validate_samples", False)),
        include_source_path=True,
    )

    collate_fn = (
        collate_workzone_fixed
        if bool(data_config.get("fixed_collate", True))
        else collate_workzone_batch
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            bool(
                data_config.get(
                    "pin_memory",
                    True,
                )
            )
            and device.type == "cuda"
        ),
        collate_fn=collate_fn,
    )

    model = WZTARF(
        WZTARFConfig(
            **config["model"]
        )
    )

    run_id = (
        args.run_id
        if args.run_id is not None
        else args.checkpoint.stem
    )

    report_dir = (
        PROJECT_ROOT
        /
        "reports"
        /
        run_id
    )

    result = evaluate_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        dataloader=dataloader,
        device=device,
        output_dir=report_dir,
        fps=int(
            data_config["fps"]
        ),
        miss_threshold_m=float(
            evaluation_config[
                "miss_threshold_m"
            ]
        ),
        worker_threshold_m=float(
            evaluation_config[
                "worker_threshold_m"
            ]
        ),
    )

    # --------------------------------------------------------------
    # Efficiency measurement uses batch size 1.
    # --------------------------------------------------------------

    latency_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    latency_batch = next(
        iter(
            latency_loader
        )
    )

    latency_batch = _move_to_device(
        latency_batch,
        device,
    )

    model.to(
        device
    )

    model.eval()

    latency = inference_latency_ms(
        model,
        latency_batch,
        device=device,
        warmup=int(
            evaluation_config[
                "latency_warmup"
            ]
        ),
        iterations=int(
            evaluation_config[
                "latency_iterations"
            ]
        ),
    )

    if device.type == "cuda":
        reset_peak_gpu_memory(
            device
        )

        with torch.inference_mode():
            model(
                latency_batch
            )

        torch.cuda.synchronize(
            device
        )

        memory_mb = peak_gpu_memory_mb(
            device
        )

    else:
        memory_mb = float(
            "nan"
        )

    efficiency = {
        **latency,
        "throughput_samples_per_second": (
            throughput_samples_per_second(
                latency["mean_ms"],
                batch_size=1,
            )
        ),
        "parameter_count": parameter_count(
            model
        ),
        "peak_gpu_memory_mb": memory_mb,
    }

    environment = environment_snapshot(
        PROJECT_ROOT
    )

    write_markdown_report(
        report_dir,
        run_id=run_id,
        metrics=result["metrics"],
        config=config,
        efficiency=efficiency,
        environment=environment,
        checkpoint=args.checkpoint,
    )

    append_experiment_row(
        PROJECT_ROOT
        /
        "reports"
        /
        "all_experiments.csv",
        run_id=run_id,
        metrics=result["metrics"],
        config=config,
        summary={
            "checkpoint": str(
                args.checkpoint.resolve()
            ),
            "split": split,
            **efficiency,
        },
    )

    print(
        json.dumps(
            {
                "run_id": run_id,
                "metrics": result["metrics"],
                "efficiency": efficiency,
                "report_dir": str(
                    report_dir.resolve()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
