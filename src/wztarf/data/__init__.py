"""Dataset loading, batching, schema validation, and feature construction."""

from .collate import collate_workzone_batch
from .dataset import WorkZoneDataset, discover_pt_files
from .features import (
    build_control_features,
    build_gaze_features,
    build_motion_features,
    future_time,
    relative_history_time,
)
from .targets import (
    SupervisedTargets,
    build_supervised_targets,
)
from .map_coverage import (
    GoalTarget,
    build_map_coverage_mask,
    build_road_reliability_mask,
    build_terminal_goal_target,
    distance_to_lane_union,
)
from .schema import (
    DEFAULT_SEQUENCE_SPEC,
    REQUIRED_FIELDS,
    SampleSchemaError,
    SequenceSpec,
    validate_sample,
)

__all__ = [
    "WorkZoneDataset",
    "discover_pt_files",
    "collate_workzone_batch",
    "SequenceSpec",
    "DEFAULT_SEQUENCE_SPEC",
    "REQUIRED_FIELDS",
    "SampleSchemaError",
    "validate_sample",
    "relative_history_time",
    "future_time",
    "build_motion_features",
    "build_control_features",
    "build_gaze_features",
    "GoalTarget",
    "build_map_coverage_mask",
    "build_road_reliability_mask",
    "build_terminal_goal_target",
    "distance_to_lane_union",
]
