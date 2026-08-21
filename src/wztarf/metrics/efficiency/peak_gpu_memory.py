"""Peak CUDA memory measurement."""

from __future__ import annotations

import torch


def reset_peak_gpu_memory(device: str | torch.device | None = None) -> None:
    """Reset CUDA peak allocated-memory statistics when CUDA is available."""
    if not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats(device)


def peak_gpu_memory_mb(device: str | torch.device | None = None) -> float:
    """Return peak allocated CUDA memory in MiB, or NaN on CPU."""
    if not torch.cuda.is_available():
        return float("nan")
    return torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
