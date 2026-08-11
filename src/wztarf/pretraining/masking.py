"""Build structured cross-modal masks for WorkZone-aware pretraining."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class MaskingConfig:
    """Control the amount of structured masking applied during pretraining."""

    temporal_ratio: float = 0.25
    agent_ratio: float = 0.25
    lane_ratio: float = 0.25
    workzone_ratio: float = 0.25
    worker_ratio: float = 0.25
    primary_ratio: float = 0.60

    def __post_init__(self) -> None:
        """Validate all masking ratios."""
        for name, value in self.__dict__.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must lie in [0, 1], got {value}."
                )


@dataclass
class MaskPlan:
    """Boolean masks indicating which valid observations are hidden."""

    motion: torch.Tensor
    controls: torch.Tensor
    gaze: torch.Tensor
    agents: torch.Tensor
    lanes: torch.Tensor
    workzone: torch.Tensor
    workers: torch.Tensor
    primary_modality: list[str]


def _random_index(
    high: int,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> int:
    """Return one random integer in `[0, high)`."""
    if high <= 0:
        raise ValueError(
            "high must be positive."
        )

    return int(
        torch.randint(
            low=0,
            high=high,
            size=(1,),
            device=device,
            generator=generator,
        ).item()
    )


def _contiguous_mask_1d(
    valid: torch.Tensor,
    ratio: float,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Mask one contiguous span of valid positions in a one-dimensional stream."""
    valid = valid.bool()

    mask = torch.zeros_like(
        valid,
        dtype=torch.bool,
    )

    valid_indices = torch.nonzero(
        valid,
        as_tuple=False,
    ).flatten()

    count = int(
        valid_indices.numel()
    )

    if count == 0 or ratio <= 0:
        return mask

    span = max(
        1,
        int(
            round(
                count
                *
                ratio
            )
        ),
    )

    span = min(
        span,
        count,
    )

    max_start = (
        count
        -
        span
        +
        1
    )

    start = _random_index(
        max_start,
        device=valid.device,
        generator=generator,
    )

    selected = valid_indices[
        start:
        start + span
    ]

    mask[
        selected
    ] = True

    return mask


def _temporal_object_mask(
    valid: torch.Tensor,
    ratio: float,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Mask contiguous temporal spans independently for tracked objects.

    Args:
        valid:
            `[T, N]` validity mask, where N may represent agents.

    Returns:
        `[T, N]` mask.
    """
    if valid.ndim != 2:
        raise ValueError(
            "valid must have shape [T, N]."
        )

    output = torch.zeros_like(
        valid,
        dtype=torch.bool,
    )

    for object_index in range(
        valid.shape[1]
    ):
        output[
            :,
            object_index,
        ] = _contiguous_mask_1d(
            valid[
                :,
                object_index,
            ],
            ratio,
            generator=generator,
        )

    return output


def _lane_point_mask(
    valid: torch.Tensor,
    ratio: float,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Mask contiguous point segments independently within each lane polyline.

    Args:
        valid:
            `[L, P]`.

    Returns:
        `[L, P]`.
    """
    if valid.ndim != 2:
        raise ValueError(
            "Lane validity must have shape [L, P]."
        )

    output = torch.zeros_like(
        valid,
        dtype=torch.bool,
    )

    for lane_index in range(
        valid.shape[0]
    ):
        output[
            lane_index
        ] = _contiguous_mask_1d(
            valid[
                lane_index
            ],
            ratio,
            generator=generator,
        )

    return output


def _random_token_mask(
    valid: torch.Tensor,
    ratio: float,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Randomly mask valid non-temporal tokens."""
    valid = valid.bool()

    if ratio <= 0:
        return torch.zeros_like(
            valid,
            dtype=torch.bool,
        )

    random_value = torch.rand(
        valid.shape,
        dtype=torch.float32,
        device=valid.device,
        generator=generator,
    )

    return (
        valid
        &
        (random_value < ratio)
    )


def _validate_batch(
    batch: Mapping[str, Any],
) -> None:
    """Check that all tensors needed by the masking policy are present."""
    required = {
        "ego_hist",
        "control_mask",
        "gaze_mask",
        "agent_mask",
        "lane_point_mask",
        "lane_mask",
        "wz_feat",
        "wz_worker_feat",
    }

    missing = (
        required
        -
        set(batch.keys())
    )

    if missing:
        raise KeyError(
            f"Masking batch is missing fields: {sorted(missing)}"
        )


def build_mask_plan(
    batch: Mapping[str, Any],
    *,
    config: MaskingConfig | None = None,
    generator: torch.Generator | None = None,
) -> MaskPlan:
    """Construct complementary structured masks for a batched scene.

    Expected batched shapes:

        ego_hist
            `[B, T, F]`

        control_mask
            `[B, T]`

        gaze_mask
            `[B, T]`

        agent_mask
            `[B, T, A]`

        lane_point_mask
            `[B, L, P]`

        lane_mask
            `[B, L]`

        wz_feat
            `[B, 5, 3]`

        wz_worker_feat
            `[B, W, 3]`

    Each sample receives ordinary structured masking across all available
    modalities and one randomly selected primary modality receives stronger
    masking. This creates complementary cross-modal reconstruction tasks rather
    than treating masking as independent feature dropout.

    The resulting masks mark only valid observations.
    """
    _validate_batch(
        batch
    )

    if config is None:
        config = MaskingConfig()

    ego_hist = batch[
        "ego_hist"
    ]

    if ego_hist.ndim != 3:
        raise ValueError(
            "ego_hist must have batched shape [B, T, F]."
        )

    batch_size = ego_hist.shape[0]
    history_steps = ego_hist.shape[1]
    device = ego_hist.device

    control_valid = batch[
        "control_mask"
    ].bool()

    gaze_valid = batch[
        "gaze_mask"
    ].bool()

    agent_valid = batch[
        "agent_mask"
    ].bool()

    lane_valid = (
        batch[
            "lane_point_mask"
        ].bool()
        &
        batch[
            "lane_mask"
        ].bool().unsqueeze(-1)
    )

    wz_feat = batch[
        "wz_feat"
    ]

    worker_feat = batch[
        "wz_worker_feat"
    ]

    if wz_feat.ndim != 3 or wz_feat.shape[-1] < 3:
        raise ValueError(
            "wz_feat must have shape [B, N, >=3]."
        )

    if worker_feat.ndim != 3 or worker_feat.shape[-1] < 3:
        raise ValueError(
            "wz_worker_feat must have shape [B, W, >=3]."
        )

    wz_valid = (
        wz_feat[..., 2]
        >
        0
    )

    worker_valid = (
        worker_feat[..., 2]
        >
        0
    )

    motion_valid = torch.ones(
        batch_size,
        history_steps,
        dtype=torch.bool,
        device=device,
    )

    motion_mask = torch.zeros_like(
        motion_valid
    )

    control_mask = torch.zeros_like(
        control_valid
    )

    gaze_mask = torch.zeros_like(
        gaze_valid
    )

    agent_mask = torch.zeros_like(
        agent_valid
    )

    lanes_mask = torch.zeros_like(
        lane_valid
    )

    workzone_mask = torch.zeros_like(
        wz_valid
    )

    workers_mask = torch.zeros_like(
        worker_valid
    )

    primary_modalities: list[str] = []

    for batch_index in range(
        batch_size
    ):
        # --------------------------------------------------------------
        # Ordinary structured masking
        # --------------------------------------------------------------

        motion_mask[
            batch_index
        ] = _contiguous_mask_1d(
            motion_valid[
                batch_index
            ],
            config.temporal_ratio,
            generator=generator,
        )

        control_mask[
            batch_index
        ] = _contiguous_mask_1d(
            control_valid[
                batch_index
            ],
            config.temporal_ratio,
            generator=generator,
        )

        gaze_mask[
            batch_index
        ] = _contiguous_mask_1d(
            gaze_valid[
                batch_index
            ],
            config.temporal_ratio,
            generator=generator,
        )

        agent_mask[
            batch_index
        ] = _temporal_object_mask(
            agent_valid[
                batch_index
            ],
            config.agent_ratio,
            generator=generator,
        )

        lanes_mask[
            batch_index
        ] = _lane_point_mask(
            lane_valid[
                batch_index
            ],
            config.lane_ratio,
            generator=generator,
        )

        workzone_mask[
            batch_index
        ] = _random_token_mask(
            wz_valid[
                batch_index
            ],
            config.workzone_ratio,
            generator=generator,
        )

        workers_mask[
            batch_index
        ] = _random_token_mask(
            worker_valid[
                batch_index
            ],
            config.worker_ratio,
            generator=generator,
        )

        # --------------------------------------------------------------
        # Choose an available primary modality for stronger masking.
        # --------------------------------------------------------------

        available: list[str] = [
            "motion",
        ]

        if bool(
            control_valid[
                batch_index
            ].any()
        ):
            available.append(
                "controls"
            )

        if bool(
            gaze_valid[
                batch_index
            ].any()
        ):
            available.append(
                "gaze"
            )

        if bool(
            agent_valid[
                batch_index
            ].any()
        ):
            available.append(
                "agents"
            )

        if bool(
            lane_valid[
                batch_index
            ].any()
        ):
            available.append(
                "lanes"
            )

        if bool(
            wz_valid[
                batch_index
            ].any()
        ):
            available.append(
                "workzone"
            )

        if bool(
            worker_valid[
                batch_index
            ].any()
        ):
            available.append(
                "workers"
            )

        selected = available[
            _random_index(
                len(available),
                device=device,
                generator=generator,
            )
        ]

        primary_modalities.append(
            selected
        )

        # Stronger masking for the selected modality. OR is used so ordinary
        # masks are preserved.
        if selected == "motion":
            motion_mask[
                batch_index
            ] |= _contiguous_mask_1d(
                motion_valid[
                    batch_index
                ],
                config.primary_ratio,
                generator=generator,
            )

        elif selected == "controls":
            control_mask[
                batch_index
            ] |= _contiguous_mask_1d(
                control_valid[
                    batch_index
                ],
                config.primary_ratio,
                generator=generator,
            )

        elif selected == "gaze":
            gaze_mask[
                batch_index
            ] |= _contiguous_mask_1d(
                gaze_valid[
                    batch_index
                ],
                config.primary_ratio,
                generator=generator,
            )

        elif selected == "agents":
            agent_mask[
                batch_index
            ] |= _temporal_object_mask(
                agent_valid[
                    batch_index
                ],
                config.primary_ratio,
                generator=generator,
            )

        elif selected == "lanes":
            lanes_mask[
                batch_index
            ] |= _lane_point_mask(
                lane_valid[
                    batch_index
                ],
                config.primary_ratio,
                generator=generator,
            )

        elif selected == "workzone":
            workzone_mask[
                batch_index
            ] |= _random_token_mask(
                wz_valid[
                    batch_index
                ],
                config.primary_ratio,
                generator=generator,
            )

        elif selected == "workers":
            workers_mask[
                batch_index
            ] |= _random_token_mask(
                worker_valid[
                    batch_index
                ],
                config.primary_ratio,
                generator=generator,
            )

    return MaskPlan(
        motion=motion_mask,
        controls=control_mask,
        gaze=gaze_mask,
        agents=agent_mask,
        lanes=lanes_mask,
        workzone=workzone_mask,
        workers=workers_mask,
        primary_modality=primary_modalities,
    )
