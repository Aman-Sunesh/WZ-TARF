"""Capture reproducibility-critical software and hardware metadata."""

import platform
import subprocess
import torch


def environment_snapshot() -> dict:
    """Return Python, PyTorch, CUDA, GPU, and Git metadata."""
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_commit = None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "git_commit": git_commit,
    }
