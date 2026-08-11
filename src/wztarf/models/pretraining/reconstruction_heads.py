"""Decode masked structured modalities from the masked scene representation."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


class ReconstructionHeads(nn.Module):
    """Provide small modality-specific heads for Phase A reconstruction."""

    def __init__(
        self,
        *,
        d_model: int = 128,
        control_dim: int = 64,
    ) -> None:
        super().__init__()

        self.motion = nn.Linear(
            d_model,
            11,
        )

        self.controls = nn.Linear(
            control_dim,
            8,
        )

        # Gaze XY is intentionally excluded. Reconstruct confidence,
        # validity, and relative time instead.
        self.gaze = nn.Linear(
            d_model,
            3,
        )

        self.agents = nn.Linear(
            d_model,
            11,
        )

        self.lanes = nn.Linear(
            d_model,
            8,
        )

        self.workzone = nn.Linear(
            d_model,
            2,
        )

        self.workers = nn.Linear(
            d_model,
            2,
        )

    def forward(
        self,
        *,
        scene: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
        motion_target: torch.Tensor,
        control_target: torch.Tensor,
        gaze_target: torch.Tensor,
    ) -> dict[str, Any]:
        """Return predictions, targets, validity masks, and loss types."""
        batch_size = motion_target.shape[0]

        wz_count = batch[
            "wz_feat"
        ].shape[1]

        worker_count = batch[
            "wz_worker_feat"
        ].shape[1]

        wz_state = scene[
            "wz_context"
        ][
            :,
            None,
        ].expand(
            batch_size,
            wz_count,
            -1,
        )

        worker_state = scene[
            "wz_context"
        ][
            :,
            None,
        ].expand(
            batch_size,
            worker_count,
            -1,
        )

        predictions = {
            "motion": self.motion(
                scene[
                    "ego_states"
                ]
            ),
            "controls": self.controls(
                scene[
                    "control_states"
                ]
            ),
            "gaze": self.gaze(
                scene[
                    "gaze_states"
                ]
            ),
            "agents": self.agents(
                scene[
                    "agent_temporal_states"
                ]
            ),
            "lanes": self.lanes(
                scene[
                    "lane_point_states"
                ]
            ),
            "workzone": self.workzone(
                wz_state
            ),
            "workers": self.workers(
                worker_state
            ),
        }

        targets = {
            "motion": motion_target,
            "controls": control_target,
            "gaze": gaze_target[
                ...,
                2:5,
            ],
            "agents": batch[
                "agent_hist"
            ],
            "lanes": batch[
                "lane_feat"
            ],
            "workzone": batch[
                "wz_feat"
            ][
                ...,
                :2,
            ],
            "workers": batch[
                "wz_worker_feat"
            ][
                ...,
                :2,
            ],
        }

        valid_masks = {
            "motion": torch.ones(
                motion_target.shape[:2],
                dtype=torch.bool,
                device=motion_target.device,
            ),
            "controls": batch[
                "control_mask"
            ].bool(),
            "gaze": batch[
                "gaze_mask"
            ].bool(),
            "agents": batch[
                "agent_mask"
            ].bool(),
            "lanes": (
                batch[
                    "lane_point_mask"
                ].bool()
                &
                batch[
                    "lane_mask"
                ].bool()[
                    :,
                    :,
                    None,
                ]
            ),
            "workzone": (
                batch[
                    "wz_feat"
                ][
                    ...,
                    2,
                ]
                >
                0
            ),
            "workers": (
                batch[
                    "wz_worker_feat"
                ][
                    ...,
                    2,
                ]
                >
                0
            ),
        }

        return {
            "reconstruction_predictions": predictions,
            "reconstruction_targets": targets,
            "reconstruction_valid_masks": valid_masks,
            "reconstruction_loss_types": {
                name: "huber"
                for name in predictions
            },
        }
