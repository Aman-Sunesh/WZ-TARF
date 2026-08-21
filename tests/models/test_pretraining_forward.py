"""Smoke test for the complete Phase A model path."""

from __future__ import annotations

import torch

from wztarf.models import WZTARFConfig
from wztarf.models.pretraining import WZTARFPretrainingModel
from wztarf.pretraining import build_mask_plan

from .test_wztarf_forward import _batch


def test_pretraining_forward() -> None:
    """Phase A should produce every objective input."""
    batch = _batch()

    model = WZTARFPretrainingModel(
        WZTARFConfig(
            d_model=32,
            motion_hidden=32,
            control_hidden=16,
            gaze_hidden=16,
            agent_hidden=16,
            num_edge_types=4,
        )
    )

    plan = build_mask_plan(
        batch
    )

    output = model.pretraining_forward(
        batch,
        plan,
    )

    assert set(
        output[
            "context_embeddings"
        ]
    ) == {
        1,
        3,
        5,
    }

    assert set(
        output[
            "future_embeddings"
        ]
    ) == {
        1,
        3,
        5,
    }

    assert output[
        "lane_overlap_pred"
    ].shape == (
        2,
        6,
    )

    assert output[
        "lane_distance_pred"
    ].shape == (
        2,
        6,
    )
