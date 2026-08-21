"""Model configuration for WZ-TARF."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class WZTARFConfig:
    """Configuration controlling the WZ-TARF model structure."""

    d_model: int = 128

    motion_hidden: int = 128
    control_hidden: int = 64
    gaze_hidden: int = 64
    agent_hidden: int = 64

    num_modes: int = 6

    fps: int = 5
    history_steps: int = 10
    future_steps: int = 25

    top_seed_lanes: int = 4
    lane_encode_points: int = 48
    wz_lane_geometry_points: int | None = 64

    lane_graph_layers: int = 2
    topology_layers: int = 2

    # V3 explicit route-set controls.
    route_topology_bias: float = 1.0
    route_competition_strength: float = 0.5
    route_walk_steps: int = 10
    route_progress_scale_m: float = 8.0
    route_exit_extension_m: float = 60.0

    # Historical compatibility boundary.  Phase A/B and ProgressFix predate
    # the V3 dense hard-route progress repair modules; HEADONLY adds them.
    use_dense_progress_repair: bool = True

    # ==============================================================
    # V3 publication attribution controls.
    # ==============================================================
    topology_mode: str = "workzone"

    use_controls: bool = True
    use_gaze: bool = True
    use_workers: bool = True

    # Training-only sample-level auxiliary modality dropout.
    # Phase-A structured masking is not compounded because this dropout
    # is applied only when encode_scene receives no mask_plan.
    aux_dropout_controls: float = 0.10
    aux_dropout_gaze: float = 0.10
    aux_dropout_workers: float = 0.10

    num_edge_types: int = 16

    use_refiner: bool = False

    # Direct K=6 Cartesian trajectory bypass.
    #
    # When enabled, route/topology remains available for context and
    # auxiliary supervision, but route-progress no longer determines XY.
    use_direct_decoder: bool = False
    use_direct_anchor_calibration: bool = False
    use_direct_longitudinal_repair: bool = False
    refiner_radius_m: float = 8.0
    
    # V3 route-relative correction bounds.
    decoder_longitudinal_correction_m: float = 2.0
    decoder_lateral_correction_m: float = 1.25
    refiner_longitudinal_correction_m: float = 0.75
    refiner_lateral_correction_m: float = 0.45
