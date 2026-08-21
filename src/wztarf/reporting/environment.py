"""Capture reproducibility-critical software and hardware metadata."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

import torch


def environment_snapshot(project_root: str | Path | None = None) -> dict:
    """Return Python, PyTorch, CUDA, GPU, platform, and Git metadata."""
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    git_commit = None
    try:
        kwargs = {
            "text": True,
            "stderr": subprocess.DEVNULL,
        }
        if project_root is not None:
            kwargs["cwd"] = str(Path(project_root))
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            **kwargs,
        ).strip()
    except Exception:
        pass

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "git_commit": git_commit,
    }
