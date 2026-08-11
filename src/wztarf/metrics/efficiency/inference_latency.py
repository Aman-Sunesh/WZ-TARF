"""Measure synchronized model-forward inference latency."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch


@torch.inference_mode()
def inference_latency_ms(
    model: torch.nn.Module,
    batch: Any,
    *,
    device: str | torch.device,
    warmup: int = 20,
    iterations: int = 200,
) -> dict[str, float]:
    """Return mean, median, P90, and P95 model-forward latency in milliseconds.

    The batch should already have the desired deployment batch size. The
    primary WZ-TARF latency report uses batch size one.
    """
    if warmup < 0:
        raise ValueError(
            "warmup cannot be negative."
        )

    if iterations <= 0:
        raise ValueError(
            "iterations must be positive."
        )

    device = torch.device(
        device
    )

    use_cuda = (
        device.type
        ==
        "cuda"
    )

    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA latency requested but CUDA is unavailable."
        )

    model.eval()

    for _ in range(warmup):
        model(batch)

    if use_cuda:
        torch.cuda.synchronize(
            device
        )

    values: list[float] = []

    for _ in range(iterations):
        if use_cuda:
            torch.cuda.synchronize(
                device
            )

        start = time.perf_counter()

        model(batch)

        if use_cuda:
            torch.cuda.synchronize(
                device
            )

        elapsed_ms = (
            time.perf_counter()
            -
            start
        ) * 1000.0

        values.append(
            elapsed_ms
        )

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean_ms": float(
            x.mean()
        ),
        "median_ms": float(
            np.median(x)
        ),
        "p90_ms": float(
            np.quantile(x, 0.90)
        ),
        "p95_ms": float(
            np.quantile(x, 0.95)
        ),
    }
