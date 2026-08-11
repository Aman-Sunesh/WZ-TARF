"""Define and validate the canonical tensor schema used by WZ-TARF."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import torch


@dataclass(frozen=True)
class SequenceSpec:
    """Temporal and multimodal output specification."""

    fps: int = 5
    history_steps: int = 10
    future_steps: int = 25
    num_modes: int = 6

    @property
    def dt(self) -> float:
        """Seconds between adjacent observations."""
        return 1.0 / float(self.fps)

    @property
    def history_seconds(self) -> float:
        """Observed temporal span represented by the history."""
        return self.history_steps / float(self.fps)

    @property
    def future_seconds(self) -> float:
        """Forecast horizon in seconds."""
        return self.future_steps / float(self.fps)


DEFAULT_SEQUENCE_SPEC = SequenceSpec()


REQUIRED_FIELDS = frozenset(
    {
        "ego_hist",
        "future_xy",
        "control_hist",
        "control_mask",
        "gaze_feat",
        "gaze_mask",
        "agent_hist",
        "agent_mask",
        "lane_feat",
        "lane_point_mask",
        "lane_mask",
        "lane_edge_index",
        "lane_edge_type",
        "lane_edge_mask",
        "wz_feat",
        "wz_worker_feat",
        "meta",
        "map_meta",
    }
)


OPTIONAL_LANE_FIELDS = frozenset(
    {
        "lane_boundary_source",
        "lane_width_mask",
        "lane_cont_attr",
        "lane_type_id",
        "lane_mark_type",
        "lane_quality_flags",
        "lane_attr",
    }
)


class SampleSchemaError(ValueError):
    """Raised when a serialized WorkZone sample violates the expected schema."""


def _tensor(
    sample: Mapping[str, Any],
    key: str,
    errors: list[str],
) -> torch.Tensor | None:
    """Retrieve a tensor field while collecting validation errors."""
    value = sample.get(key)

    if not isinstance(value, torch.Tensor):
        errors.append(
            f"'{key}' must be a torch.Tensor, "
            f"got {type(value).__name__}."
        )
        return None

    return value


def validate_sample(
    sample: Mapping[str, Any],
    *,
    spec: SequenceSpec = DEFAULT_SEQUENCE_SPEC,
    source: str | None = None,
) -> None:
    """Validate one serialized final WorkZone sample.

    The validator checks structural consistency without requiring fixed
    maximum numbers of agents, lanes, lane points, or graph edges. Those
    dimensions may be padded dynamically by the batch collator.

    Expected core layouts:

        ego_hist           [T_hist, 6]
        future_xy          [T_future, 2]

        control_hist       [T_hist, 3]
        control_mask       [T_hist]

        gaze_feat          [T_hist, 3]
        gaze_mask          [T_hist]

        agent_hist         [T_hist, K_agents, 11]
        agent_mask         [T_hist, K_agents]

        lane_feat          [L, P, 8]
        lane_point_mask    [L, P]
        lane_mask          [L]

        lane_edge_index    [2, E]
        lane_edge_type     [E]
        lane_edge_mask     [E]

        wz_feat            [5, 3]
        wz_worker_feat     [W, 3]

        meta               dict-like
        map_meta           dict-like
    """
    if not isinstance(sample, Mapping):
        raise SampleSchemaError(
            f"Sample must be mapping-like, got {type(sample).__name__}."
        )

    errors: list[str] = []

    # ------------------------------------------------------------------
    # Required keys
    # ------------------------------------------------------------------

    missing = sorted(
        REQUIRED_FIELDS
        -
        set(sample.keys())
    )

    if missing:
        errors.append(
            f"Missing required fields: {missing}."
        )

    # ------------------------------------------------------------------
    # Core temporal tensors
    # ------------------------------------------------------------------

    ego_hist = _tensor(
        sample,
        "ego_hist",
        errors,
    )

    if ego_hist is not None:
        expected = (
            spec.history_steps,
            6,
        )

        if tuple(ego_hist.shape) != expected:
            errors.append(
                f"'ego_hist' must have shape {expected}, "
                f"got {tuple(ego_hist.shape)}."
            )

    future_xy = _tensor(
        sample,
        "future_xy",
        errors,
    )

    if future_xy is not None:
        expected = (
            spec.future_steps,
            2,
        )

        if tuple(future_xy.shape) != expected:
            errors.append(
                f"'future_xy' must have shape {expected}, "
                f"got {tuple(future_xy.shape)}."
            )

    control_hist = _tensor(
        sample,
        "control_hist",
        errors,
    )

    control_mask = _tensor(
        sample,
        "control_mask",
        errors,
    )

    if control_hist is not None:
        expected = (
            spec.history_steps,
            3,
        )

        if tuple(control_hist.shape) != expected:
            errors.append(
                f"'control_hist' must have shape {expected}, "
                f"got {tuple(control_hist.shape)}."
            )

    if control_mask is not None:
        expected = (
            spec.history_steps,
        )

        if tuple(control_mask.shape) != expected:
            errors.append(
                f"'control_mask' must have shape {expected}, "
                f"got {tuple(control_mask.shape)}."
            )

    gaze_feat = _tensor(
        sample,
        "gaze_feat",
        errors,
    )

    gaze_mask = _tensor(
        sample,
        "gaze_mask",
        errors,
    )

    if gaze_feat is not None:
        expected = (
            spec.history_steps,
            3,
        )

        if tuple(gaze_feat.shape) != expected:
            errors.append(
                f"'gaze_feat' must have shape {expected}, "
                f"got {tuple(gaze_feat.shape)}."
            )

    if gaze_mask is not None:
        expected = (
            spec.history_steps,
        )

        if tuple(gaze_mask.shape) != expected:
            errors.append(
                f"'gaze_mask' must have shape {expected}, "
                f"got {tuple(gaze_mask.shape)}."
            )

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------

    agent_hist = _tensor(
        sample,
        "agent_hist",
        errors,
    )

    agent_mask = _tensor(
        sample,
        "agent_mask",
        errors,
    )

    if agent_hist is not None:
        if agent_hist.ndim != 3:
            errors.append(
                "'agent_hist' must have shape [T, K, 11]."
            )
        else:
            if agent_hist.shape[0] != spec.history_steps:
                errors.append(
                    "'agent_hist' history length does not match "
                    f"{spec.history_steps}."
                )

            if agent_hist.shape[-1] != 11:
                errors.append(
                    "'agent_hist' must contain 11 features per agent."
                )

    if agent_mask is not None:
        if agent_mask.ndim != 2:
            errors.append(
                "'agent_mask' must have shape [T, K]."
            )

        elif agent_hist is not None and agent_hist.ndim == 3:
            if tuple(agent_mask.shape) != tuple(agent_hist.shape[:2]):
                errors.append(
                    "'agent_mask' must match the first two dimensions "
                    "of 'agent_hist'."
                )

    # ------------------------------------------------------------------
    # Lane geometry
    # ------------------------------------------------------------------

    lane_feat = _tensor(
        sample,
        "lane_feat",
        errors,
    )

    lane_point_mask = _tensor(
        sample,
        "lane_point_mask",
        errors,
    )

    lane_mask = _tensor(
        sample,
        "lane_mask",
        errors,
    )

    num_lanes: int | None = None

    if lane_feat is not None:
        if lane_feat.ndim != 3:
            errors.append(
                "'lane_feat' must have shape [L, P, 8]."
            )
        else:
            num_lanes = int(lane_feat.shape[0])

            if lane_feat.shape[-1] != 8:
                errors.append(
                    "'lane_feat' must contain 8 features per lane point."
                )

    if lane_point_mask is not None:
        if lane_point_mask.ndim != 2:
            errors.append(
                "'lane_point_mask' must have shape [L, P]."
            )

        elif lane_feat is not None and lane_feat.ndim == 3:
            if tuple(lane_point_mask.shape) != tuple(lane_feat.shape[:2]):
                errors.append(
                    "'lane_point_mask' must match the first two "
                    "dimensions of 'lane_feat'."
                )

    if lane_mask is not None:
        if lane_mask.ndim != 1:
            errors.append(
                "'lane_mask' must have shape [L]."
            )

        elif num_lanes is not None:
            if lane_mask.shape[0] != num_lanes:
                errors.append(
                    "'lane_mask' length must equal the number of lanes."
                )

    # Validate additional lane-level attributes when available.
    for key in OPTIONAL_LANE_FIELDS:
        if key not in sample:
            continue

        value = _tensor(
            sample,
            key,
            errors,
        )

        if (
            value is not None
            and num_lanes is not None
            and value.shape[0] != num_lanes
        ):
            errors.append(
                f"'{key}' first dimension must equal "
                f"the lane count ({num_lanes})."
            )

    # ------------------------------------------------------------------
    # Lane graph
    # ------------------------------------------------------------------

    edge_index = _tensor(
        sample,
        "lane_edge_index",
        errors,
    )

    edge_type = _tensor(
        sample,
        "lane_edge_type",
        errors,
    )

    edge_mask = _tensor(
        sample,
        "lane_edge_mask",
        errors,
    )

    num_edges: int | None = None

    if edge_index is not None:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            errors.append(
                "'lane_edge_index' must have shape [2, E]."
            )
        else:
            num_edges = int(edge_index.shape[1])

    if edge_type is not None:
        if edge_type.ndim != 1:
            errors.append(
                "'lane_edge_type' must have shape [E]."
            )
        elif num_edges is not None and edge_type.shape[0] != num_edges:
            errors.append(
                "'lane_edge_type' length must equal edge count."
            )

    if edge_mask is not None:
        if edge_mask.ndim != 1:
            errors.append(
                "'lane_edge_mask' must have shape [E]."
            )
        elif num_edges is not None and edge_mask.shape[0] != num_edges:
            errors.append(
                "'lane_edge_mask' length must equal edge count."
            )

    # ------------------------------------------------------------------
    # WorkZone geometry
    # ------------------------------------------------------------------

    wz_feat = _tensor(
        sample,
        "wz_feat",
        errors,
    )

    if wz_feat is not None:
        if tuple(wz_feat.shape) != (5, 3):
            errors.append(
                "'wz_feat' must have shape [5, 3] "
                "(four polygon corners plus warning sign)."
            )

    worker_feat = _tensor(
        sample,
        "wz_worker_feat",
        errors,
    )

    if worker_feat is not None:
        if worker_feat.ndim != 2 or worker_feat.shape[-1] != 3:
            errors.append(
                "'wz_worker_feat' must have shape [W, 3]."
            )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    meta = sample.get("meta")

    if not isinstance(meta, Mapping):
        errors.append(
            "'meta' must be dictionary-like."
        )

    map_meta = sample.get("map_meta")

    if not isinstance(map_meta, Mapping):
        errors.append(
            "'map_meta' must be dictionary-like."
        )

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    if errors:
        location = (
            f" in {source}"
            if source is not None
            else ""
        )

        message = (
            f"Invalid WorkZone sample{location}:\n- "
            +
            "\n- ".join(errors)
        )

        raise SampleSchemaError(message)
