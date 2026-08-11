"""Compute the complete WZ-TARF evaluation suite from saved predictions."""

from __future__ import annotations

import math

import torch

from wztarf.metrics.common import validate_prediction_shapes
from wztarf.metrics.forecasting.ade_at_minfde import ade_at_minfde
from wztarf.metrics.forecasting.brier_minfde import brier_minfde
from wztarf.metrics.forecasting.fde_at_minade import fde_at_minade
from wztarf.metrics.forecasting.minade import minade
from wztarf.metrics.forecasting.minade_horizon import minade_horizon
from wztarf.metrics.forecasting.minfde import minfde
from wztarf.metrics.forecasting.minfde_horizon import minfde_horizon
from wztarf.metrics.forecasting.miss_rate import miss_rate
from wztarf.metrics.forecasting.p90_minade import p90_minade
from wztarf.metrics.forecasting.p95_minade import p95_minade
from wztarf.metrics.forecasting.top1_ade import top1_ade
from wztarf.metrics.forecasting.top1_fde import top1_fde
from wztarf.metrics.safety.wsvr import wsvr
from wztarf.metrics.safety.wz_gvr import wz_gvr
from wztarf.metrics.safety.wzvr import wzvr


def _as_float(
    value: torch.Tensor | float | int,
) -> float:
    """Convert a scalar tensor or number to a plain Python float."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                "Metric output must be scalar."
            )

        return float(
            value.detach().cpu().item()
        )

    return float(value)


def _validate_probabilities(
    mode_prob: torch.Tensor,
    *,
    batch_size: int,
    num_modes: int,
    tolerance: float = 1e-4,
) -> None:
    """Validate that model mode scores are proper probabilities."""
    if mode_prob.ndim != 2:
        raise ValueError(
            "mode_prob must have shape [B, K]."
        )

    if tuple(mode_prob.shape) != (
        batch_size,
        num_modes,
    ):
        raise ValueError(
            "mode_prob shape does not match pred_xy. "
            f"Expected {(batch_size, num_modes)}, "
            f"got {tuple(mode_prob.shape)}."
        )

    if not torch.isfinite(mode_prob).all():
        raise ValueError(
            "mode_prob contains NaN or infinite values."
        )

    if (mode_prob < 0).any():
        raise ValueError(
            "mode_prob contains negative values. "
            "Pass probabilities, not raw logits."
        )

    row_sum = mode_prob.sum(dim=1)

    if not torch.allclose(
        row_sum,
        torch.ones_like(row_sum),
        atol=tolerance,
        rtol=tolerance,
    ):
        raise ValueError(
            "Each mode_prob row must sum to 1. "
            "Pass softmax probabilities rather than logits."
        )


def _seconds_to_steps(
    seconds: float,
    fps: int,
) -> int:
    """Convert a physical horizon to an exact number of future samples."""
    if seconds <= 0:
        raise ValueError(
            "Evaluation horizon must be positive."
        )

    if fps <= 0:
        raise ValueError(
            "fps must be positive."
        )

    raw_steps = (
        seconds
        *
        fps
    )

    rounded = int(
        round(raw_steps)
    )

    if not math.isclose(
        raw_steps,
        rounded,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError(
            f"{seconds} s is not exactly representable at {fps} Hz."
        )

    return rounded


def _extract_workzone_geometry(
    wz_feat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the four-corner WZ polygon and sample-validity mask.

    Expected:
        wz_feat: `[B, 5, 3]`

    Layout:
        rows 0:4:
            WorkZone polygon corners

        row 4:
            warning sign

        feature 0:2:
            XY position

        feature 2:
            validity
    """
    if wz_feat.ndim != 3:
        raise ValueError(
            "wz_feat must have shape [B, 5, 3]."
        )

    if wz_feat.shape[1] < 4:
        raise ValueError(
            "wz_feat must contain at least four polygon corners."
        )

    if wz_feat.shape[-1] < 3:
        raise ValueError(
            "wz_feat must contain XY plus validity."
        )

    polygon = (
        wz_feat[:, :4, :2]
        .float()
    )

    corner_valid = (
        wz_feat[:, :4, 2]
        >
        0
    )

    # WZ-GVR is valid only when all four polygon corners are represented.
    polygon_valid = (
        corner_valid
        .all(dim=1)
    )

    return (
        polygon,
        polygon_valid,
    )


def _extract_workers(
    worker_feat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract worker XY positions and validity masks.

    Expected:
        worker_feat: `[B, W, 3]`

    Feature layout:
        x
        y
        validity
    """
    if worker_feat.ndim != 3:
        raise ValueError(
            "worker_feat must have shape [B, W, 3]."
        )

    if worker_feat.shape[-1] < 3:
        raise ValueError(
            "worker_feat must contain XY plus validity."
        )

    worker_xy = (
        worker_feat[..., :2]
        .float()
    )

    worker_mask = (
        worker_feat[..., 2]
        >
        0
    )

    return (
        worker_xy,
        worker_mask,
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
    """Compute all final forecasting and WorkZone safety metrics.

    Args:
        pred_xy:
            Predicted trajectories `[N, K, T, 2]`.

        gt_xy:
            Ground-truth future `[N, T, 2]`.

        mode_prob:
            Normalized trajectory probabilities `[N, K]`.

        wz_feat:
            Optional WorkZone tokens `[N, 5, 3]`.

        worker_feat:
            Optional worker tokens `[N, W, 3]`.

        fps:
            Dataset sampling frequency.

        miss_threshold_m:
            Terminal threshold for MR.

        worker_threshold_m:
            Worker-clearance threshold for WSVR.

    Returns:
        Dictionary containing scalar metrics.

    Safety metrics are included only when the corresponding WorkZone data
    are supplied.
    """
    validate_prediction_shapes(
        pred_xy,
        gt_xy,
    )

    batch_size = pred_xy.shape[0]
    num_modes = pred_xy.shape[1]
    future_steps = pred_xy.shape[2]

    _validate_probabilities(
        mode_prob,
        batch_size=batch_size,
        num_modes=num_modes,
    )

    if miss_threshold_m <= 0:
        raise ValueError(
            "miss_threshold_m must be positive."
        )

    if worker_threshold_m <= 0:
        raise ValueError(
            "worker_threshold_m must be positive."
        )

    horizon_1s = _seconds_to_steps(
        1.0,
        fps,
    )

    horizon_3s = _seconds_to_steps(
        3.0,
        fps,
    )

    horizon_5s = _seconds_to_steps(
        5.0,
        fps,
    )

    if future_steps < horizon_5s:
        raise ValueError(
            "The final metric suite requires at least a 5 s "
            f"forecast horizon. Received {future_steps} steps "
            f"at {fps} Hz."
        )

    metrics: dict[str, float] = {
        # --------------------------------------------------------------
        # Best-of-K forecasting accuracy
        # --------------------------------------------------------------
        "minADE_6": _as_float(
            minade(
                pred_xy,
                gt_xy,
            )
        ),

        "minFDE_6": _as_float(
            minfde(
                pred_xy,
                gt_xy,
            )
        ),

        # --------------------------------------------------------------
        # Horizon-specific best-of-K accuracy
        # --------------------------------------------------------------
        "minADE_6@1s": _as_float(
            minade_horizon(
                pred_xy,
                gt_xy,
                horizon_1s,
            )
        ),

        "minADE_6@3s": _as_float(
            minade_horizon(
                pred_xy,
                gt_xy,
                horizon_3s,
            )
        ),

        "minADE_6@5s": _as_float(
            minade_horizon(
                pred_xy,
                gt_xy,
                horizon_5s,
            )
        ),

        "minFDE_6@1s": _as_float(
            minfde_horizon(
                pred_xy,
                gt_xy,
                horizon_1s,
            )
        ),

        "minFDE_6@3s": _as_float(
            minfde_horizon(
                pred_xy,
                gt_xy,
                horizon_3s,
            )
        ),

        "minFDE_6@5s": _as_float(
            minfde_horizon(
                pred_xy,
                gt_xy,
                horizon_5s,
            )
        ),

        # --------------------------------------------------------------
        # Tail robustness
        # --------------------------------------------------------------
        "P90_minADE_6": _as_float(
            p90_minade(
                pred_xy,
                gt_xy,
            )
        ),

        "P95_minADE_6": _as_float(
            p95_minade(
                pred_xy,
                gt_xy,
            )
        ),

        # --------------------------------------------------------------
        # Model-selected Top-1 trajectory
        # --------------------------------------------------------------
        "Top1_ADE": _as_float(
            top1_ade(
                pred_xy,
                gt_xy,
                mode_prob,
            )
        ),

        "Top1_FDE": _as_float(
            top1_fde(
                pred_xy,
                gt_xy,
                mode_prob,
            )
        ),

        # --------------------------------------------------------------
        # Coverage and calibration
        # --------------------------------------------------------------
        "MR_6@2m": _as_float(
            miss_rate(
                pred_xy,
                gt_xy,
                threshold_m=miss_threshold_m,
            )
        ),

        "Brier_minFDE_6": _as_float(
            brier_minfde(
                pred_xy,
                gt_xy,
                mode_prob,
            )
        ),

        # --------------------------------------------------------------
        # Path / endpoint temporal consistency
        # --------------------------------------------------------------
        "FDE@minADE": _as_float(
            fde_at_minade(
                pred_xy,
                gt_xy,
            )
        ),

        "ADE@minFDE": _as_float(
            ade_at_minfde(
                pred_xy,
                gt_xy,
            )
        ),

        # Useful report metadata.
        "num_samples": float(
            batch_size
        ),

        "num_modes": float(
            num_modes
        ),

        "future_steps": float(
            future_steps
        ),

        "fps": float(
            fps
        ),
    }

    # ------------------------------------------------------------------
    # WorkZone geometry safety
    # ------------------------------------------------------------------

    if wz_feat is not None:
        if wz_feat.shape[0] != batch_size:
            raise ValueError(
                "wz_feat batch size does not match predictions."
            )

        wz_polygon, wz_valid = _extract_workzone_geometry(
            wz_feat
        )

        metrics["WZ_GVR"] = _as_float(
            wz_gvr(
                pred_xy,
                mode_prob,
                wz_polygon,
                wz_valid,
            )
        )

        metrics["num_valid_wz_polygons"] = float(
            wz_valid.sum().item()
        )

    # ------------------------------------------------------------------
    # Worker safety
    # ------------------------------------------------------------------

    if worker_feat is not None:
        if worker_feat.shape[0] != batch_size:
            raise ValueError(
                "worker_feat batch size does not match predictions."
            )

        worker_xy, worker_mask = _extract_workers(
            worker_feat
        )

        worker_present = (
            worker_mask.any(dim=1)
        )

        metrics["WSVR@2m"] = _as_float(
            wsvr(
                pred_xy,
                mode_prob,
                worker_xy,
                worker_mask,
                threshold_m=worker_threshold_m,
            )
        )

        metrics["num_worker_present_samples"] = float(
            worker_present.sum().item()
        )

    # ------------------------------------------------------------------
    # Aggregate WZ violation
    # ------------------------------------------------------------------

    if (
        wz_feat is not None
        and
        worker_feat is not None
    ):
        wz_polygon, wz_valid = _extract_workzone_geometry(
            wz_feat
        )

        worker_xy, worker_mask = _extract_workers(
            worker_feat
        )

        metrics["WZVR"] = _as_float(
            wzvr(
                pred_xy,
                mode_prob,
                wz_polygon,
                worker_xy,
                worker_mask,
                wz_valid=wz_valid,
                worker_threshold_m=worker_threshold_m,
            )
        )

    return metrics
