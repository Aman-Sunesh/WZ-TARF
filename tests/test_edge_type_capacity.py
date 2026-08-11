"""Validate configured lane-edge embedding capacity against real data."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from wztarf.data import WorkZoneDataset
from wztarf.data.dataset import validate_edge_type_capacity
from wztarf.utils import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_configured_edge_embedding_capacity() -> None:
    """Ensure configured edge embedding capacity fits serialized train data."""
    roots_value = os.environ.get(
        "WZTARF_DATA_ROOTS"
    )

    if not roots_value:
        pytest.skip(
            "Set WZTARF_DATA_ROOTS to run the real-data edge-capacity test."
        )

    roots = [
        Path(
            item.strip()
        )
        for item in roots_value.split(",")
        if item.strip()
    ]

    if not roots:
        pytest.skip(
            "WZTARF_DATA_ROOTS contains no dataset roots."
        )

    config = load_yaml(
        PROJECT_ROOT
        /
        "configs"
        /
        "base.yaml"
    )

    dataset = WorkZoneDataset(
        roots=roots,
        split=config[
            "data"
        ].get(
            "train_split",
            "train",
        ),
        validate=True,
    )

    validate_edge_type_capacity(
        dataset,
        num_edge_types=int(
            config[
                "model"
            ][
                "num_edge_types"
            ]
        ),
    )
