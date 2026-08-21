"""Integration test for one complete forward and backward pass."""

from __future__ import annotations

import torch

from wztarf.losses import (
    LossWeights,
    supervised_loss,
)
from wztarf.models import (
    WZTARF,
    WZTARFConfig,
)


def _batch() -> dict[str, torch.Tensor]:
    """Create a small valid synthetic WZ-TARF batch."""
    batch_size = 2
    history_steps = 10
    future_steps = 25
    agents = 3
    lanes = 6
    points = 8
    edges = 8

    ego_hist = torch.zeros(
        batch_size,
        history_steps,
        6,
    )

    ego_hist[
        ...,
        0,
    ] = torch.linspace(
        -1.8,
        0.0,
        history_steps,
    )

    ego_hist[
        ...,
        2,
    ] = 1.0

    ego_hist[
        ...,
        5,
    ] = 1.0

    future_xy = torch.zeros(
        batch_size,
        future_steps,
        2,
    )

    future_xy[
        ...,
        0,
    ] = torch.linspace(
        0.2,
        5.0,
        future_steps,
    )

    lane_feat = torch.zeros(
        batch_size,
        lanes,
        points,
        8,
    )

    x = torch.linspace(
        -2.0,
        8.0,
        points,
    )

    for lane_index in range(
        lanes
    ):
        lane_feat[
            :,
            lane_index,
            :,
            0,
        ] = x

        lane_feat[
            :,
            lane_index,
            :,
            1,
        ] = (
            lane_index
            -
            2
        ) * 4.0

        lane_feat[
            :,
            lane_index,
            :,
            5,
        ] = 1.75

        lane_feat[
            :,
            lane_index,
            :,
            7,
        ] = -1.75

    lane_point_mask = torch.ones(
        batch_size,
        lanes,
        points,
        dtype=torch.bool,
    )

    lane_mask = torch.ones(
        batch_size,
        lanes,
        dtype=torch.bool,
    )

    lane_edge_index = torch.full(
        (
            batch_size,
            2,
            edges,
        ),
        -1,
        dtype=torch.long,
    )

    lane_edge_type = torch.zeros(
        batch_size,
        edges,
        dtype=torch.long,
    )

    lane_edge_mask = torch.zeros(
        batch_size,
        edges,
        dtype=torch.bool,
    )

    for edge_index in range(
        lanes - 1
    ):
        lane_edge_index[
            :,
            0,
            edge_index,
        ] = edge_index

        lane_edge_index[
            :,
            1,
            edge_index,
        ] = edge_index + 1

        lane_edge_mask[
            :,
            edge_index,
        ] = True

    wz_feat = torch.zeros(
        batch_size,
        5,
        3,
    )

    wz_feat[
        :,
        :4,
        :2,
    ] = torch.tensor(
        [
            [20.0, -2.0],
            [24.0, -2.0],
            [24.0, 2.0],
            [20.0, 2.0],
        ]
    )

    wz_feat[
        :,
        :4,
        2,
    ] = 1.0

    wz_feat[
        :,
        4,
    ] = torch.tensor(
        [
            15.0,
            0.0,
            1.0,
        ]
    )

    return {
        "ego_hist": ego_hist,
        "future_xy": future_xy,
        "control_hist": torch.zeros(
            batch_size,
            history_steps,
            3,
        ),
        "control_mask": torch.ones(
            batch_size,
            history_steps,
            dtype=torch.bool,
        ),
        "gaze_feat": torch.zeros(
            batch_size,
            history_steps,
            3,
        ),
        "gaze_mask": torch.ones(
            batch_size,
            history_steps,
            dtype=torch.bool,
        ),
        "agent_hist": torch.zeros(
            batch_size,
            history_steps,
            agents,
            11,
        ),
        "agent_mask": torch.zeros(
            batch_size,
            history_steps,
            agents,
            dtype=torch.bool,
        ),
        "lane_feat": lane_feat,
        "lane_point_mask": lane_point_mask,
        "lane_mask": lane_mask,
        "lane_attr": torch.zeros(
            batch_size,
            lanes,
            10,
        ),
        "lane_edge_index": lane_edge_index,
        "lane_edge_type": lane_edge_type,
        "lane_edge_mask": lane_edge_mask,
        "wz_feat": wz_feat,
        "wz_worker_feat": torch.zeros(
            batch_size,
            2,
            3,
        ),
    }


def test_wztarf_forward_backward() -> None:
    """The complete forecasting graph should run and receive gradients."""
    batch = _batch()

    model = WZTARF(
        WZTARFConfig(
            d_model=32,
            motion_hidden=32,
            control_hidden=16,
            gaze_hidden=16,
            agent_hidden=16,
            num_modes=6,
            num_edge_types=4,
        )
    )

    output = model(
        batch
    )

    assert output[
        "pred_xy"
    ].shape == (
        2,
        6,
        25,
        2,
    )

    assert output[
        "mode_prob"
    ].shape == (
        2,
        6,
    )

    assert torch.allclose(
        output[
            "mode_prob"
        ].sum(
            dim=-1
        ),
        torch.ones(
            2
        ),
        atol=1e-5,
    )

    loss_output = supervised_loss(
        output,
        batch,
        weights=LossWeights(
            trajectory=1.0,
            endpoint=0.25,
            classification=1.0,
            behavior=0.0,
            ranking_quality=0.0,
            ranking_pairwise=0.0,
            lane=1.0,
            topology=0.0,
            topo_diversity=0.0,
            route_coverage=0.0,
            route=1.0,
            angle=0.0,
            dynamics=0.0,
            diversity=0.0,
            road=0.0,
            wz_geometry=0.0,
            worker=0.0,
            refinement=0.0,
        ),
    )

    assert torch.isfinite(
        loss_output.total
    )

    loss_output.total.backward()

    has_gradient = any(
        parameter.grad is not None
        and
        bool(
            torch.isfinite(
                parameter.grad
            ).all()
        )
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert has_gradient
