"""Capture software, hardware, and Git metadata needed to reproduce a run."""

from __future__ import annotations

import platform
import socket
import subprocess
from pathlib import Path
from typing import Any

import torch


def _run_git_command(
    args: list[str],
    *,
    cwd: Path | None,
) -> str | None:
    """Run a Git command and return stripped output when available."""
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return None


def environment_snapshot(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return software, hardware, CUDA, and Git information for a run.

    Args:
        project_root:
            Repository root used for Git metadata. When omitted, Git is
            queried from the current working directory.

    Returns:
        JSON-serializable environment dictionary.
    """
    cwd = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else None
    )

    cuda_available = torch.cuda.is_available()

    gpu_devices: list[dict[str, Any]] = []

    if cuda_available:
        for device_index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(
                device_index
            )

            gpu_devices.append(
                {
                    "index": device_index,
                    "name": properties.name,
                    "total_memory_mb": round(
                        properties.total_memory
                        /
                        (1024.0 ** 2),
                        2,
                    ),
                    "compute_capability": (
                        f"{properties.major}."
                        f"{properties.minor}"
                    ),
                }
            )

    git_commit = _run_git_command(
        ["rev-parse", "HEAD"],
        cwd=cwd,
    )

    git_branch = _run_git_command(
        ["rev-parse", "--abbrev-ref", "HEAD"],
        cwd=cwd,
    )

    git_status = _run_git_command(
        ["status", "--porcelain"],
        cwd=cwd,
    )

    git_dirty = (
        bool(git_status)
        if git_status is not None
        else None
    )

    cudnn_version = (
        torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None
    )

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),

        "pytorch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": cudnn_version,

        "gpu_count": (
            torch.cuda.device_count()
            if cuda_available
            else 0
        ),
        "gpus": gpu_devices,

        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
    }
