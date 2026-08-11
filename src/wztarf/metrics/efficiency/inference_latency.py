"""Synchronized model-forward latency measurement."""

import time
import numpy as np
import torch


@torch.inference_mode()
def inference_latency_ms(model, batch, warmup: int = 20, iterations: int = 200) -> dict[str, float]:
    """Return mean, median, P90, and P95 latency in milliseconds."""
    model.eval()

    for _ in range(warmup):
        model(batch)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    values = []
    for _ in range(iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        model(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000.0)

    x = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(x.mean()),
        "median_ms": float(np.median(x)),
        "p90_ms": float(np.quantile(x, 0.90)),
        "p95_ms": float(np.quantile(x, 0.95)),
    }
