"""Synchronized model-forward latency measurement."""

from __future__ import annotations

import statistics
import time

import torch


def _quantile(values: list[float], q: float) -> float:
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), q).item())


@torch.inference_mode()
def inference_latency_ms(
    model,
    batch,
    *,
    device: str | torch.device | None = None,
    warmup: int = 20,
    iterations: int = 200,
) -> dict[str, float]:
    """Return mean, median, P90, and P95 model-forward latency in milliseconds."""
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be >= 0 and iterations must be positive")

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)
    model.eval()

    for _ in range(warmup):
        model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    values: list[float] = []
    if device.type == "cuda":
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(batch)
            end.record()
            torch.cuda.synchronize(device)
            values.append(float(start.elapsed_time(end)))
    else:
        for _ in range(iterations):
            start = time.perf_counter()
            model(batch)
            values.append((time.perf_counter() - start) * 1000.0)

    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p90_ms": _quantile(values, 0.90),
        "p95_ms": _quantile(values, 0.95),
    }
