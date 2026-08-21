"""Batch WorkZone samples while preserving variable-size scene masks."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from typing import Any

import torch
from torch.utils.data import default_collate


# Lane-level fields whose first dimension corresponds to the number of lanes.
_LANE_FIELDS = {
    "lane_mask",
    "lane_boundary_source",
    "lane_width_mask",
    "lane_cont_attr",
    "lane_type_id",
    "lane_mark_type",
    "lane_quality_flags",
    "lane_attr",
}


# Edge-level fields whose first dimension corresponds to the number of edges.
_EDGE_FIELDS = {
    "lane_edge_type",
    "lane_edge_mask",
}


def _pad_tensor(
    tensor: torch.Tensor,
    target_shape: tuple[int, ...],
    fill_value: int | float | bool = 0,
) -> torch.Tensor:
    """Pad a tensor to `target_shape` without changing existing values."""
    if tensor.ndim != len(target_shape):
        raise ValueError(
            f"Cannot pad tensor of shape {tuple(tensor.shape)} "
            f"to target shape {target_shape}."
        )

    if any(current > target for current, target in zip(tensor.shape, target_shape)):
        raise ValueError(
            f"Target shape {target_shape} is smaller than "
            f"tensor shape {tuple(tensor.shape)}."
        )

    output = torch.full(
        target_shape,
        fill_value=fill_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )

    slices = tuple(slice(0, size) for size in tensor.shape)
    output[slices] = tensor
    return output


def _all_same_shape(values: list[torch.Tensor]) -> bool:
    """Return True when all tensors have identical shapes."""
    first_shape = values[0].shape
    return all(value.shape == first_shape for value in values)


def _stack_or_pad_agent_field(
    key: str,
    values: list[torch.Tensor],
    max_agents: int,
) -> torch.Tensor:
    """Pad variable agent dimensions and stack the batch."""
    if key == "agent_hist":
        # [T, K, F]
        target = (
            values[0].shape[0],
            max_agents,
            values[0].shape[-1],
        )
    elif key == "agent_mask":
        # [T, K]
        target = (
            values[0].shape[0],
            max_agents,
        )
    else:
        raise KeyError(key)

    padded = [
        _pad_tensor(
            value,
            target,
            fill_value=False if value.dtype == torch.bool else 0,
        )
        for value in values
    ]
    return torch.stack(padded, dim=0)


def _stack_or_pad_lane_field(
    key: str,
    values: list[torch.Tensor],
    max_lanes: int,
    max_points: int,
) -> torch.Tensor:
    """Pad lane and lane-point dimensions before stacking."""
    first = values[0]

    if key == "lane_feat":
        # [L, P, F]
        target = (
            max_lanes,
            max_points,
            first.shape[-1],
        )

    elif key == "lane_point_mask":
        # [L, P]
        target = (
            max_lanes,
            max_points,
        )

    elif key in _LANE_FIELDS:
        # [L, ...]
        target = (
            max_lanes,
            *first.shape[1:],
        )

    else:
        raise KeyError(key)

    padded = [
        _pad_tensor(
            value,
            target,
            fill_value=False if value.dtype == torch.bool else 0,
        )
        for value in values
    ]
    return torch.stack(padded, dim=0)


def _stack_or_pad_edge_field(
    key: str,
    values: list[torch.Tensor],
    max_edges: int,
) -> torch.Tensor:
    """Pad graph-edge tensors before stacking."""
    if key == "lane_edge_index":
        # [2, E]
        target = (2, max_edges)

        # -1 explicitly identifies padded edge indices.
        padded = [
            _pad_tensor(value, target, fill_value=-1)
            for value in values
        ]

    elif key in _EDGE_FIELDS:
        # [E]
        target = (max_edges,)

        padded = [
            _pad_tensor(
                value,
                target,
                fill_value=False if value.dtype == torch.bool else 0,
            )
            for value in values
        ]

    else:
        raise KeyError(key)

    return torch.stack(padded, dim=0)


def collate_workzone_batch(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collate WorkZone samples into one batch.

    Fixed-size tensors are stacked directly.

    Variable-size structures are padded consistently:
        agent_hist:       [B, T, K_max, F]
        agent_mask:       [B, T, K_max]

        lane_feat:        [B, L_max, P_max, F]
        lane_point_mask:  [B, L_max, P_max]
        lane-level data:  [B, L_max, ...]

        lane_edge_index:  [B, 2, E_max]
        edge-level data:  [B, E_max]

    Metadata dictionaries and source paths remain Python lists so that
    identifiers and nested metadata are not modified by batching.
    """
    if not samples:
        raise ValueError("Cannot collate an empty sample list.")

    expected_keys = set(samples[0].keys())

    for index, sample in enumerate(samples[1:], start=1):
        sample_keys = set(sample.keys())

        if sample_keys != expected_keys:
            missing = sorted(expected_keys - sample_keys)
            extra = sorted(sample_keys - expected_keys)

            raise KeyError(
                f"Sample {index} has inconsistent keys. "
                f"Missing={missing}, extra={extra}"
            )

    batch: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Determine maximum variable dimensions for this batch.
    # ------------------------------------------------------------------

    max_agents = 0
    if "agent_hist" in expected_keys:
        max_agents = max(
            int(sample["agent_hist"].shape[1])
            for sample in samples
        )

    max_lanes = 0
    max_points = 0

    if "lane_feat" in expected_keys:
        max_lanes = max(
            int(sample["lane_feat"].shape[0])
            for sample in samples
        )

        max_points = max(
            int(sample["lane_feat"].shape[1])
            for sample in samples
        )

    max_edges = 0
    if "lane_edge_index" in expected_keys:
        max_edges = max(
            int(sample["lane_edge_index"].shape[1])
            for sample in samples
        )

    # ------------------------------------------------------------------
    # Collate each field.
    # ------------------------------------------------------------------

    for key in sorted(expected_keys):
        values = [sample[key] for sample in samples]

        # Keep arbitrary nested metadata untouched.
        if key in {"meta", "map_meta", "source_path"}:
            batch[key] = values
            continue

        first = values[0]

        if isinstance(first, torch.Tensor):
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise TypeError(
                    f"Field '{key}' mixes tensor and non-tensor values."
                )

            # Most final dataset tensors are already padded to fixed shape.
            if _all_same_shape(values):
                batch[key] = torch.stack(values, dim=0)
                continue

            if key in {"agent_hist", "agent_mask"}:
                batch[key] = _stack_or_pad_agent_field(
                    key,
                    values,
                    max_agents,
                )
                continue

            if key in {"lane_feat", "lane_point_mask"} or key in _LANE_FIELDS:
                batch[key] = _stack_or_pad_lane_field(
                    key,
                    values,
                    max_lanes,
                    max_points,
                )
                continue

            if key == "lane_edge_index" or key in _EDGE_FIELDS:
                batch[key] = _stack_or_pad_edge_field(
                    key,
                    values,
                    max_edges,
                )
                continue

            raise ValueError(
                f"Tensor field '{key}' has variable shapes "
                f"{[tuple(value.shape) for value in values]} "
                "but no explicit collation rule."
            )

        if isinstance(first, Mapping):
            # Meta-like dictionaries should not be recursively converted.
            batch[key] = values
            continue

        if isinstance(first, (str, PathLike)):
            batch[key] = values
            continue

        # Scalars, numbers, and other standard objects.
        batch[key] = default_collate(values)

    return batch


# Avoid importing os only for the isinstance check above.
try:
    from os import PathLike
except ImportError:  # pragma: no cover
    PathLike = str


def collate_workzone_fixed(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fast collate for the canonical self-contained final WorkZone samples.

    The July24 final dataset is already padded to fixed tensor capacities
    (20 agents, 74 lanes, 234 points/lane, 512 edges).  Re-discovering maxima
    and dispatching through variable-size padding rules every batch therefore
    adds Python overhead without changing the tensors.  This path performs
    direct stacking and leaves nested metadata untouched.

    Use ``collate_workzone_batch`` for legacy/variable-size samples.
    """
    if not samples:
        raise ValueError("Cannot collate an empty sample list.")

    first = samples[0]
    keys = first.keys()
    batch: dict[str, Any] = {}

    for key in keys:
        values = [sample[key] for sample in samples]
        value0 = values[0]

        if key in {"meta", "map_meta", "source_path"}:
            batch[key] = values
        elif torch.is_tensor(value0):
            # torch.stack itself validates shape consistency and is the only
            # operation needed for the canonical final tensors.
            batch[key] = torch.stack(values, dim=0)
        elif isinstance(value0, (Mapping, str, PathLike)):
            batch[key] = values
        else:
            batch[key] = default_collate(values)

    return batch
