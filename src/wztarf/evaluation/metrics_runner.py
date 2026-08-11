"""Compute the complete forecasting and safety metric suite."""

from __future__ import annotations

import math

import torch

from wztarf.metrics.common import validate_prediction_shapes
from wztarf.metrics.forecasting import (
    ade_at_minfde,
    brier_minfde,
    fde_at_minade,
    minade,
    minade_horizon,
    minfde,
    minfde_horizon,
    miss_rate,
    p90_minade,
    p95_minade,
    top1_ade,
    top1_fde,
)
from wztarf.metrics.safety import (
    wsvr,
    wz_gvr,
    wzvr,
)


def _number(
    value: torch.Tensor,
) -> float:
    """Convert a scalar tensor into a Python float."""
    return float(
        value.detach().cpu().item()
    )


def compute_all_metrics(
    *,
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    mode_prob: torch.Tensor,
    wz_feat: torch.Tensor | None = None,
    worker_feat: torch.Tensor | None = None,
    fps: int = 5,
    miss_threshold_m: float = 2.0,
    worker_threshold_m: float = 2.0,
) -> dict[str, float]:
    """Compute the final K=6 WZ-TARF evaluation suite."""
    validate_prediction_shapes(
        pred_xy,
        gt_xy,
    )

    batch_size, num_modes, future_steps, _ = pred_xy.shape

    if num_modes != 6:
        raise ValueError(
            f"WZ-TARF evaluation expects K=6, got K={num_modes}."
        )

    if mode_prob.shape != (
        batch_size,
        num_modes,
    ):
        raise ValueError(
            "mode_prob must have shape [B, K]."
        )

    if not bool(
        torch.isfinite(
            mode_prob
        ).all()
    ):
        raise ValueError(
            "mode_prob contains non-finite values."
        )

    if bool(
        (mode_prob < 0).any()
    ):
        raise ValueError(
            "mode_prob cannot contain negative values."
        )

    probability_sum = mode_prob.sum(
        dim=-1
    )

    if not torch.allclose(
        probability_sum,
        torch.ones_like(
            probability_sum
        ),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError(
            "mode_prob must contain normalized probabilities."
        )

    horizon_steps = {
        second: second * fps
        for second in (
            1,
            3,
            5,
        )
    }

    if max(
        horizon_steps.values()
    ) > future_steps:
        raise ValueError(
            "Prediction horizon is shorter than 5 seconds."
        )

    metrics = {
        "minADE_6": _number(
            minade(
                pred_xy,
                gt_xy,
            )
        ),
        "minFDE_6": _number(
            minfde(
                pred_xy,
                gt_xy,
            )
        ),
        "P90_minADE_6": _number(
            p90_minade(
                pred_xy,
                gt_xy,
            )
        ),
        "P95_minADE_6": _number(
            p95_minade(
                pred_xy,
                gt_xy,
            )
        ),
        "Top1_ADE": _number(
            top1_ade(
                pred_xy,
                gt_xy,
                mode_prob,
            )
        ),
        "Top1_FDE": _number(
            top1_fde(
                pred_xy,
                gt_xy,
                mode_prob,
            )
        ),
        "MR_6@2m": _number(
            miss_rate(
                pred_xy,
                gt_xy,
                threshold_m=miss_threshold_m,
            )
        ),
        "Brier_minFDE_6": _number(
            brier_minfde(
                pred_xy,
                gt_xy,
                mode_prob,
            )
        ),
        "FDE@minADE_6": _number(
            fde_at_minade(
                pred_xy,
                gt_xy,
            )
        ),
        "ADE@minFDE_6": _number(
            ade_at_minfde(
                pred_xy,
                gt_xy,
            )
        ),
    }

    for second, steps in horizon_steps.items():
        metrics[
            f"minADE_6@{second}s"
        ] = _number(
            minade_horizon(
                pred_xy,
                gt_xy,
                steps,
            )
        )

        metrics[
            f"minFDE_6@{second}s"
        ] = _number(
            minfde_horizon(
                pred_xy,
                gt_xy,
                steps,
            )
        )

    metrics[
        "WZ_GVR"
    ] = float(
        "nan"
    )

    metrics[
        "WSVR@2m"
    ] = float(
        "nan"
    )

    metrics[
        "WZVR"
    ] = float(
        "nan"
    )

    if wz_feat is not None:
        polygon = wz_feat[
            :,
            :4,
            :2,
        ]

        wz_valid = (
            wz_feat[
                :,
                :4,
                2,
            ]
            >
            0
        ).all(
            dim=1
        )

        metrics[
            "WZ_GVR"
        ] = _number(
            wz_gvr(
                pred_xy,
                mode_prob,
                polygon,
                wz_valid,
            )
        )

        if worker_feat is not None:
            worker_xy = worker_feat[
                ...,
                :2,
            ]

            worker_mask = (
                worker_feat[
                    ...,
                    2,
                ]
                >
                0
            )

            metrics[
                "WSVR@2m"
            ] = _number(
                wsvr(
                    pred_xy,
                    mode_prob,
                    worker_xy,
                    worker_mask,
                    threshold_m=worker_threshold_m,
                )
            )

            metrics[
                "WZVR"
            ] = _number(
                wzvr(
                    pred_xy,
                    mode_prob,
                    polygon,
                    worker_xy,
                    worker_mask,
                    wz_valid=wz_valid,
                    worker_threshold_m=worker_threshold_m,
                )
            )

    return metrics
