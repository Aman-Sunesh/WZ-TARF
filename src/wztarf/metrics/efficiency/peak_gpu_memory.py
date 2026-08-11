"""Measure peak CUDA memory allocated during inference."""

from __future__ import annotations

import torch


def reset_peak_gpu_memory(
    device: str | torch.device | None = None,
) -> None:
    """Reset CUDA peak-memory statistics before an inference measurement."""
    if not torch.cuda.is_available():
        return

    if device is None:
        device = torch.cuda.current_device()

    torch.cuda.reset_peak_memory_stats(
        device
    )


def peak_gpu_memory_mb(
    device: str | torch.device | None = None,
) -> float:
    """Return peak CUDA memory allocated since the last reset, in MiB."""
    if not torch.cuda.is_available():
        return float("nan")

    if device is None:
        device = torch.cuda.current_device()

    return (
        torch.cuda.max_memory_allocated(
            device
        )
        /
        (1024.0 ** 2)
    )
