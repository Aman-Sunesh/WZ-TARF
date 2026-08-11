"""Run a minimal end-to-end smoke test of forecasting and safety metrics."""

from __future__ import annotations

import math

import torch

from wztarf.evaluation import compute_all_metrics


def main() -> None:
    """Verify that the complete metric runner accepts canonical tensors."""
    batch_size = 2
    num_modes = 6
    future_steps = 25

    pred_xy = torch.zeros(
        batch_size,
        num_modes,
        future_steps,
        2,
    )

    gt_xy = torch.zeros(
        batch_size,
        future_steps,
        2,
    )

    mode_prob = torch.full(
        (
            batch_size,
            num_modes,
        ),
        1.0 / num_modes,
    )

    # Restricted square around the origin.
    wz_feat = torch.zeros(
        batch_size,
        5,
        3,
    )

    corners = torch.tensor(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )

    wz_feat[
        :,
        :4,
        :2,
    ] = corners

    wz_feat[
        :,
        :4,
        2,
    ] = 1.0

    # Valid sign token.
    wz_feat[
        :,
        4,
    ] = torch.tensor(
        [
            5.0,
            0.0,
            1.0,
        ]
    )

    # No represented workers.
    workers = torch.zeros(
        batch_size,
        2,
        3,
    )

    metrics = compute_all_metrics(
        pred_xy=pred_xy,
        gt_xy=gt_xy,
        mode_prob=mode_prob,
        wz_feat=wz_feat,
        worker_feat=workers,
        fps=5,
    )

    assert metrics["minADE_6"] == 0.0
    assert metrics["minFDE_6"] == 0.0
    assert metrics["Top1_ADE"] == 0.0
    assert metrics["Top1_FDE"] == 0.0

    # Predictions are inside the restricted square.
    assert metrics["WZ_GVR"] == 1.0
    assert metrics["WZVR"] == 1.0

    # WSVR is undefined because neither sample has a represented worker.
    assert math.isnan(
        metrics["WSVR@2m"]
    )

    print(
        "Metric smoke test passed."
    )


if __name__ == "__main__":
    main()
