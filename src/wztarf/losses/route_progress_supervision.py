"""Dense hard-route progress supervision for WZ-TARF V3."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _project_future_to_routes(
    route_walk_xy: torch.Tensor,
    route_walk_s: torch.Tensor,
    future_xy: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """Project every GT future point onto every candidate hard route.

    Returns:
        distance:
            [B,K,T] Euclidean point-to-route distance.

        projected_s:
            [B,K,T] longitudinal coordinate on that SAME route.

    Route geometry and GT are detached deliberately: this objective trains
    progress, not route geometry.
    """

    route_xy = (
        route_walk_xy
        .detach()
        .float()
    )

    route_s = (
        route_walk_s
        .detach()
        .float()
    )

    gt = (
        future_xy
        .detach()
        .float()
    )

    if (
        route_xy.ndim != 4
        or route_xy.shape[-1] != 2
    ):
        raise ValueError(
            "route_walk_xy must be [B,K,P,2]."
        )

    if (
        route_s.shape
        != route_xy.shape[:-1]
    ):
        raise ValueError(
            "route_walk_s must be [B,K,P]."
        )

    if (
        gt.ndim != 3
        or gt.shape[-1] != 2
    ):
        raise ValueError(
            "future_xy must be [B,T,2]."
        )

    a = route_xy[
        :,
        :,
        :-1,
    ]

    b = route_xy[
        :,
        :,
        1:,
    ]

    sa = route_s[
        :,
        :,
        :-1,
    ]

    sb = route_s[
        :,
        :,
        1:,
    ]

    ab = b - a

    length_sq = (
        ab.square()
        .sum(dim=-1)
    )

    valid_segment = (
        length_sq > 1.0e-8
    )

    p = gt[
        :,
        None,
        :,
        None,
        :,
    ]

    alpha = (
        (
            (
                p
                -
                a[
                    :,
                    :,
                    None,
                    :,
                    :,
                ]
            )
            *
            ab[
                :,
                :,
                None,
                :,
                :,
            ]
        )
        .sum(dim=-1)
        /
        length_sq
        .clamp_min(1.0e-8)[
            :,
            :,
            None,
            :,
        ]
    ).clamp(
        0.0,
        1.0,
    )

    closest = (
        a[
            :,
            :,
            None,
            :,
            :,
        ]
        +
        alpha[..., None]
        *
        ab[
            :,
            :,
            None,
            :,
            :,
        ]
    )

    segment_distance = (
        torch.linalg.vector_norm(
            p - closest,
            dim=-1,
        )
    )

    segment_distance = torch.where(
        valid_segment[
            :,
            :,
            None,
            :,
        ],
        segment_distance,
        torch.full_like(
            segment_distance,
            float("inf"),
        ),
    )

    interpolated_s = (
        sa[
            :,
            :,
            None,
            :,
        ]
        +
        alpha
        *
        (
            sb
            -
            sa
        )[
            :,
            :,
            None,
            :,
        ]
    )

    segment_index = (
        segment_distance.argmin(
            dim=-1
        )
    )

    best_distance = (
        segment_distance.gather(
            -1,
            segment_index[..., None],
        )
        .squeeze(-1)
    )

    best_s = (
        interpolated_s.gather(
            -1,
            segment_index[..., None],
        )
        .squeeze(-1)
    )

    return (
        best_distance,
        best_s,
    )


def route_progress_supervision_loss(
    *,
    route_progress: torch.Tensor,
    route_progress_sequence: torch.Tensor | None,
    dense_route_guide: torch.Tensor | None,
    route_walk_xy: torch.Tensor,
    route_walk_s: torch.Tensor,
    future_xy: torch.Tensor,
    fps: int,
    scale_m: float = 5.0,
    guide_weight: float = 0.50,
) -> torch.Tensor:
    """Train dense progress on the single best physical route.

    Why hard assignment?

    The forensic audit showed that the previous plausible-route mixture
    improved scalar s(t) fit but still left route/progress identity weak.

    minADE/minFDE need at least one physically correct mode.  Therefore this
    objective assigns each sample to the candidate route with the lowest
    detached full-future geometric cost:

        mean perpendicular distance
        + 0.25 * final perpendicular distance

    On that same route we supervise:

        1. all 25 longitudinal progress coordinates s(t)
        2. the actual dense route-guide XY trajectory

    The direct guide term makes the training objective agree with the
    quantity that regressed in the previous experiment.
    """

    if fps <= 0:
        raise ValueError(
            "fps must be positive."
        )

    if scale_m <= 0:
        raise ValueError(
            "scale_m must be positive."
        )

    if guide_weight < 0:
        raise ValueError(
            "guide_weight cannot be negative."
        )

    distance, projected_s = (
        _project_future_to_routes(
            route_walk_xy,
            route_walk_s,
            future_xy,
        )
    )

    geometry_cost = (
        distance.mean(dim=-1)
        +
        0.25
        *
        distance[
            :,
            :,
            -1
        ]
    )

    best_route = (
        geometry_cost
        .argmin(dim=1)
        .detach()
    )

    batch_size = future_xy.shape[0]

    batch_index = torch.arange(
        batch_size,
        device=future_xy.device,
    )

    # Projection can move backwards by tiny amounts because of polyline
    # discretization. The model is intentionally monotonic.
    target_sequence = (
        torch.cummax(
            projected_s,
            dim=-1,
        ).values
        .detach()
    )

    if route_progress_sequence is not None:

        prediction = (
            route_progress_sequence.float()
        )

        if (
            prediction.shape
            != target_sequence.shape
        ):
            raise ValueError(
                "route_progress_sequence must be [B,K,T] "
                "and match projected targets."
            )

        selected_prediction = prediction[
            batch_index,
            best_route,
        ]

        selected_target = target_sequence[
            batch_index,
            best_route,
        ]

        # Gradually emphasize the late horizon without ignoring early motion.
        time_weight = torch.linspace(
            0.5,
            2.0,
            prediction.shape[-1],
            dtype=prediction.dtype,
            device=prediction.device,
        )

        progress_error = (
            F.smooth_l1_loss(
                selected_prediction
                /
                float(scale_m),
                selected_target
                /
                float(scale_m),
                beta=0.2,
                reduction="none",
            )
        )

        progress_loss = (
            (
                progress_error
                *
                time_weight[
                    None,
                    :,
                ]
            ).sum(dim=-1)
            /
            time_weight.sum()
        ).mean()

    else:
        # Legacy fallback.
        pred = route_progress.float()

        horizon_index = torch.tensor(
            [
                min(
                    fps - 1,
                    future_xy.shape[1] - 1,
                ),
                min(
                    3 * fps - 1,
                    future_xy.shape[1] - 1,
                ),
                min(
                    5 * fps - 1,
                    future_xy.shape[1] - 1,
                ),
            ],
            dtype=torch.long,
            device=pred.device,
        )

        target = target_sequence.index_select(
            2,
            horizon_index,
        )

        selected_prediction = pred[
            batch_index,
            best_route,
        ]

        selected_target = target[
            batch_index,
            best_route,
        ]

        horizon_weight = torch.tensor(
            [
                0.5,
                1.0,
                2.0,
            ],
            dtype=pred.dtype,
            device=pred.device,
        )

        progress_error = (
            F.smooth_l1_loss(
                selected_prediction
                /
                float(scale_m),
                selected_target
                /
                float(scale_m),
                beta=0.2,
                reduction="none",
            )
        )

        progress_loss = (
            (
                progress_error
                *
                horizon_weight[
                    None,
                    :,
                ]
            ).sum(dim=-1)
            /
            horizon_weight.sum()
        ).mean()

    # ------------------------------------------------------------------
    # DIRECT GUIDE SUPERVISION
    #
    # The guide can move only along its physical route.  Comparing it
    # directly to GT therefore produces the along-route gradient needed to
    # choose the correct s(t), while unavoidable lateral distance remains a
    # geometric floor.
    # ------------------------------------------------------------------

    guide_loss = (
        progress_loss.new_zeros(())
    )

    if (
        dense_route_guide is not None
        and guide_weight > 0.0
    ):

        guide = (
            dense_route_guide.float()
        )

        if (
            guide.ndim != 4
            or
            guide.shape[-1] != 2
        ):
            raise ValueError(
                "dense_route_guide must be [B,K,T,2]."
            )

        if (
            guide.shape[0]
            != batch_size
            or
            guide.shape[2]
            != future_xy.shape[1]
        ):
            raise ValueError(
                "dense_route_guide shape does not match GT."
            )

        selected_guide = guide[
            batch_index,
            best_route,
        ]

        gt = future_xy.float()

        guide_error = (
            F.smooth_l1_loss(
                selected_guide,
                gt,
                beta=0.5,
                reduction="none",
            )
            .mean(dim=-1)
        )

        time_weight = torch.linspace(
            0.5,
            2.0,
            guide_error.shape[-1],
            dtype=guide_error.dtype,
            device=guide_error.device,
        )

        guide_loss = (
            (
                guide_error
                *
                time_weight[
                    None,
                    :,
                ]
            ).sum(dim=-1)
            /
            time_weight.sum()
        ).mean()

    loss = (
        progress_loss
        +
        float(guide_weight)
        *
        guide_loss
    )

    if not bool(
        torch.isfinite(loss)
    ):
        raise RuntimeError(
            "route_progress_supervision_loss became nonfinite."
        )

    return loss
