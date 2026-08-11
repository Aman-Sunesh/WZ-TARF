"""Peak CUDA memory measurement."""

import torch


def peak_gpu_memory_mb() -> float:
    """Return peak allocated CUDA memory in MiB, or NaN on CPU."""
    if not torch.cuda.is_available():
        return float("nan")
    return torch.cuda.max_memory_allocated() / (1024.0 ** 2)
