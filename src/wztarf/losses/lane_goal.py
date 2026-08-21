"""Supervise terminal retained-lane or MAP_EXIT goal classification."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def lane_goal_loss(
    goal_logits: torch.Tensor,
    goal_target: torch.Tensor,
    goal_valid: torch.Tensor,
    winner_idx: torch.Tensor,
) -> torch.Tensor:
    """Compute terminal-goal classification for the selected route mode.

    Args:
        goal_logits:
            Goal-class logits `[B, K, C]`.

        goal_target:
            Integer target `[B]`. The MAP_EXIT class is simply one of the
            valid class indices.

        goal_valid:
            Boolean mask `[B]`. False indicates ambiguous in-map supervision
            and removes that sample from this loss.

        winner_idx:
            WTA mode index `[B]`.

    Returns:
        Scalar cross-entropy loss.
    """
    if goal_logits.ndim != 3:
        raise ValueError(
            "goal_logits must have shape [B, K, C]."
        )

    batch_size = goal_logits.shape[0]

    if goal_target.shape != (batch_size,):
        raise ValueError(
            "goal_target must have shape [B]."
        )

    if goal_valid.shape != (batch_size,):
        raise ValueError(
            "goal_valid must have shape [B]."
        )

    if winner_idx.shape != (batch_size,):
        raise ValueError(
            "winner_idx must have shape [B]."
        )

    goal_valid = goal_valid.bool()

    if not bool(goal_valid.any()):
        return goal_logits.new_zeros(())

    batch_idx = torch.arange(
        batch_size,
        device=goal_logits.device,
    )

    selected_logits = goal_logits[
        batch_idx,
        winner_idx,
    ]

    # === WZTARF FINITE LANE-GOAL GUARD V1 ===
    valid_logits = selected_logits[goal_valid]
    valid_targets = goal_target[goal_valid].long()

    # Cross-entropy is undefined when a supervised row contains NaN/Inf
    # logits (for example, an all-masked goal row). Such rows must not
    # poison the complete batch loss.
    # Negative infinity is allowed for masked non-target classes.
    # A supervised row is unusable only when:
    #   - it contains NaN,
    #   - it contains +Inf,
    #   - it has no finite class at all, or
    #   - the ground-truth target itself is masked/non-finite.
    has_nan = torch.isnan(
        valid_logits
    ).any(dim=-1)

    has_pos_inf = torch.isposinf(
        valid_logits
    ).any(dim=-1)

    has_finite_class = torch.isfinite(
        valid_logits
    ).any(dim=-1)

    target_logits = valid_logits.gather(
        1,
        valid_targets.unsqueeze(1),
    ).squeeze(1)

    target_is_finite = torch.isfinite(
        target_logits
    )

    # Invalid lane classes in RouteGoalQueries are represented by
    # torch.finfo(dtype).min rather than -Inf.  This is technically
    # finite, so explicitly detect the exact masking sentinel.
    mask_sentinel = torch.finfo(
        valid_logits.dtype
    ).min

    target_is_unmasked = (
        target_logits
        !=
        mask_sentinel
    )

    usable_rows = (
        ~has_nan
        &
        ~has_pos_inf
        &
        has_finite_class
        &
        target_is_finite
        &
        target_is_unmasked
    )

    if not bool(usable_rows.all()):
        bad_rows = int(
            (~usable_rows).sum().item()
        )

        masked_target_rows = int(
            (~target_is_unmasked).sum().item()
        )

        nonfinite_target_rows = int(
            (~target_is_finite).sum().item()
        )

        all_masked_rows = int(
            (~has_finite_class).sum().item()
        )

        nan_rows = int(
            has_nan.sum().item()
        )

        pos_inf_rows = int(
            has_pos_inf.sum().item()
        )

        print(
            f"[Phase B][LANE-GOAL-WARN] "
            f"discarding {bad_rows}/"
            f"{valid_logits.shape[0]} rows | "
            f"masked_target={masked_target_rows} | "
            f"nonfinite_target={nonfinite_target_rows} | "
            f"all_masked={all_masked_rows} | "
            f"nan={nan_rows} | "
            f"pos_inf={pos_inf_rows}",
            flush=True,
        )

    if not bool(usable_rows.any()):
        return goal_logits.new_zeros(())

    valid_logits = valid_logits[
        usable_rows
    ]

    valid_targets = valid_targets[
        usable_rows
    ]

    # === WZTARF ROBUST LANE CE FP64 V1 ===
    #
    # Invalid lane classes are represented upstream by
    # torch.finfo(dtype).min.  Differences involving values near the
    # float32 minimum can overflow inside CE/log-softmax even though
    # every individual input value is technically finite.
    #
    # This classification tensor is tiny, so evaluate this one
    # objective in FP64.  Masked lane classes remain effectively
    # impossible while MAP_EXIT and valid lane classes are unchanged.
    masked_classes = (
        valid_logits
        ==
        mask_sentinel
    )

    ce_logits = valid_logits.double()

    ce_logits = ce_logits.masked_fill(
        masked_classes,
        -1.0e300,
    )

    lane_ce = F.cross_entropy(
        ce_logits,
        valid_targets,
    )

    if not bool(torch.isfinite(lane_ce)):
        finite_values = ce_logits[
            torch.isfinite(ce_logits)
        ]

        if finite_values.numel() > 0:
            finite_min = float(
                finite_values.min().detach()
            )
            finite_max = float(
                finite_values.max().detach()
            )
        else:
            finite_min = float("nan")
            finite_max = float("nan")

        print(
            f"[Phase B][LANE-CE-NONFINITE] "
            f"loss={float(lane_ce.detach())} | "
            f"logit_min={finite_min:.6g} | "
            f"logit_max={finite_max:.6g} | "
            f"rows={ce_logits.shape[0]} | "
            f"classes={ce_logits.shape[1]}",
            flush=True,
        )

    return lane_ce
