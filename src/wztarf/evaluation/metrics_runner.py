"""Compute the complete forecasting and safety metric suite."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

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




# =====================================================================
# V3 GROUP-BALANCED FORECASTING METRICS
# =====================================================================

_GROUP_ALIASES = {
    "scenario": (
        "scenario_id",
        "scenario",
        "scenario_name",
    ),
    "workzone": (
        "wz_id",
        "workzone_id",
        "workzone",
        "wz",
        "wz_index",
    ),
    "participant": (
        "participant_id",
        "participant",
        "driver_id",
        "driver",
        "subject_id",
        "subject",
    ),
}


def _metadata_scalar(
    value,
) -> str | None:
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()

    text = str(value).strip()

    if not text:
        return None

    if text.lower() in {
        "none",
        "nan",
        "null",
    }:
        return None

    return text


def _metadata_group_value(
    metadata: Mapping[str, object],
    group: str,
) -> str | None:
    aliases = _GROUP_ALIASES[group]

    for key in aliases:
        if key in metadata:
            value = _metadata_scalar(
                metadata[key]
            )
            if value is not None:
                return value

    source = _metadata_scalar(
        metadata.get(
            "source_path"
        )
    )

    if source is None:
        return None

    if group == "scenario":
        match = re.search(
            r"(?i)(?:^|[\\/_.-])(S\d+(?:_[A-Za-z0-9]+)?)(?:[\\/_.-]|$)",
            source,
        )
        if match:
            return match.group(1)

    if group == "workzone":
        match = re.search(
            r"(?i)(?:workzone|wz)[_-]?(\d+)",
            source,
        )
        if match:
            return f"WZ{match.group(1)}"

    if group == "participant":
        match = re.search(
            r"(?i)(?:participant|subject|driver|person)[_-]?([A-Za-z0-9]+)",
            source,
        )
        if match:
            return match.group(1)

    return None


def compute_grouped_forecasting_metrics(
    *,
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    metadata: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """Compute equally weighted macro minADE/minFDE across dataset groups.

    minADE and minFDE retain their standard independent best-of-K mode
    definitions before samples are aggregated into groups.
    """
    validate_prediction_shapes(
        pred_xy,
        gt_xy,
    )

    num_samples = int(
        pred_xy.shape[0]
    )

    if len(metadata) != num_samples:
        raise ValueError(
            "metadata length must match prediction count for grouped metrics."
        )

    distance = torch.linalg.vector_norm(
        pred_xy.float()
        -
        gt_xy[
            :,
            None,
        ].float(),
        dim=-1,
    )

    sample_minade = distance.mean(
        dim=-1
    ).amin(
        dim=-1
    )

    sample_minfde = distance[
        :,
        :,
        -1,
    ].amin(
        dim=-1
    )

    result: dict[str, float] = {}

    labels: dict[str, list[str | None]] = {
        "scenario": [],
        "workzone": [],
        "participant": [],
        "scenario_x_workzone": [],
    }

    for item in metadata:
        scenario = _metadata_group_value(
            item,
            "scenario",
        )

        workzone = _metadata_group_value(
            item,
            "workzone",
        )

        participant = _metadata_group_value(
            item,
            "participant",
        )

        labels["scenario"].append(
            scenario
        )

        labels["workzone"].append(
            workzone
        )

        labels["participant"].append(
            participant
        )

        labels[
            "scenario_x_workzone"
        ].append(
            (
                f"{scenario}::{workzone}"
                if (
                    scenario is not None
                    and workzone is not None
                )
                else None
            )
        )

    for group_name, sample_labels in labels.items():
        groups: dict[str, list[int]] = {}

        for index, label in enumerate(
            sample_labels
        ):
            if label is None:
                continue

            groups.setdefault(
                label,
                [],
            ).append(
                index
            )

        covered = sum(
            len(indices)
            for indices in groups.values()
        )

        result[
            f"metadata_coverage_{group_name}"
        ] = (
            float(covered)
            /
            float(max(num_samples, 1))
        )

        result[
            f"group_count_{group_name}"
        ] = float(
            len(groups)
        )

        if not groups:
            result[
                f"macro_minADE_6_{group_name}"
            ] = float("nan")

            result[
                f"macro_minFDE_6_{group_name}"
            ] = float("nan")

            result[
                f"worst_group_minADE_6_{group_name}"
            ] = float("nan")

            result[
                f"worst_group_minFDE_6_{group_name}"
            ] = float("nan")

            continue

        group_ade = []
        group_fde = []

        for indices in groups.values():
            index_tensor = torch.tensor(
                indices,
                dtype=torch.long,
                device=sample_minade.device,
            )

            group_ade.append(
                sample_minade[
                    index_tensor
                ].mean()
            )

            group_fde.append(
                sample_minfde[
                    index_tensor
                ].mean()
            )

        ade_tensor = torch.stack(
            group_ade
        )

        fde_tensor = torch.stack(
            group_fde
        )

        result[
            f"macro_minADE_6_{group_name}"
        ] = _number(
            ade_tensor.mean()
        )

        result[
            f"macro_minFDE_6_{group_name}"
        ] = _number(
            fde_tensor.mean()
        )

        result[
            f"worst_group_minADE_6_{group_name}"
        ] = _number(
            ade_tensor.max()
        )

        result[
            f"worst_group_minFDE_6_{group_name}"
        ] = _number(
            fde_tensor.max()
        )

    return result


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
    # V3 SAFE MODE-PROB METRIC NORMALIZATION
    # Validation may produce BF16 softmax probabilities. Individual values
    # are valid probabilities, but BF16 rounding can make a six-mode row sum
    # slightly above or below 1. Convert to FP32 and renormalize only at the
    # metric boundary. This does not alter model predictions or training.
    if mode_prob is not None:
        mode_prob = mode_prob.float()

        if not bool(torch.isfinite(mode_prob).all().item()):
            raise ValueError(
                "mode_prob contains non-finite values."
            )

        if bool((mode_prob < 0.0).any().item()):
            raise ValueError(
                "mode_prob contains negative probabilities."
            )

        mode_prob_sum = mode_prob.sum(
            dim=-1,
            keepdim=True,
        )

        if bool((mode_prob_sum <= 0.0).any().item()):
            raise ValueError(
                "mode_prob contains a row with zero total probability."
            )

        mode_prob = (
            mode_prob
            /
            mode_prob_sum
        )

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
