"""CLI entry point for Phase A self-supervised pretraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from wztarf.models import WZTARF, WZTARFConfig
from wztarf.utils import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    """Parse Phase A command-line options."""
    parser = argparse.ArgumentParser(
        description="Run WZ-TARF Phase A self-supervised pretraining."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "base.yaml",
    )

    parser.add_argument(
        "--data-roots",
        type=str,
        required=True,
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
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    """Validate Phase A model support before launching pretraining."""
    args = _parse_args()

    config = load_yaml(
        args.config
    )

    model = WZTARF(
        WZTARFConfig(
            **config["model"]
        )
    )

    if not callable(
        getattr(
            model,
            "pretraining_forward",
            None,
        )
    ):
        raise RuntimeError(
            "Phase A is not wired yet. WZTARF must first implement "
            "pretraining_forward(batch, mask_plan) together with the "
            "masked-reconstruction heads, future encoder/projection heads, "
            "and topology-reconstruction heads. Do not run pretraining "
            "until those model-side components are added."
        )

    raise RuntimeError(
        "The model now exposes pretraining_forward(), but scripts/pretrain.py "
        "still needs to be wired to the finalized Phase A model wrapper."
    )


if __name__ == "__main__":
    main()
