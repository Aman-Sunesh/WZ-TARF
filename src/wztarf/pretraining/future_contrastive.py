"""Align observed context with future behavior while suppressing false negatives."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

import torch
import torch.nn.functional as F


def build_false_negative_mask(
    sequence_ids: Sequence[Hashable],
    anchor_time_s: torch.Tensor,
    *,
    exclusion_seconds: float = 5.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the allowed-pair mask used by contrastive pretraining.

    Dense sliding windows from the same drive can share much of their future.
    Such samples should not be treated as negatives simply because they occupy
    different positions in the batch.

    Args:
        sequence_ids:
            Sequence, drive, or scene identifier for each batch sample.

        anchor_time_s:
            Prediction-anchor time for each sample `[B]`, in seconds.

        exclusion_seconds:
            Samples from the same sequence whose anchors are closer than this
            value are excluded from the negative set.

        device:
            Device of the returned mask. Defaults to the device of
            `anchor_time_s`.

    Returns:
        Boolean `[B, B]` mask.

        `True` means the future sample in the column is allowed to participate
        in the contrastive denominator for the context sample in the row.

        The diagonal is always True because each sample must retain its own
        positive pair.
    """
    if exclusion_seconds < 0:
        raise ValueError(
            "exclusion_seconds cannot be negative."
        )

    if anchor_time_s.ndim != 1:
        raise ValueError(
            "anchor_time_s must have shape [B]."
        )

    batch_size = anchor_time_s.shape[0]

    if len(sequence_ids) != batch_size:
        raise ValueError(
            "sequence_ids length must match anchor_time_s."
        )

    if device is None:
        device = anchor_time_s.device

    times = anchor_time_s.to(
        device=device,
        dtype=torch.float32,
    )

    allowed = torch.ones(
        batch_size,
        batch_size,
        dtype=torch.bool,
        device=device,
    )

    for row in range(batch_size):
        for column in range(batch_size):
            if row == column:
                continue

            same_sequence = (
                sequence_ids[row]
                ==
                sequence_ids[column]
            )

            if not same_sequence:
                continue

            temporal_distance = torch.abs(
                times[row]
                -
                times[column]
            )

            if float(temporal_distance.item()) < exclusion_seconds:
                allowed[
                    row,
                    column,
                ] = False

    # Always preserve the true positive.
    allowed.fill_diagonal_(
        True
    )

    return allowed


def _contrastive_direction(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    temperature: float,
    allowed_mask: torch.Tensor | None,
    negative_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Compute one directional InfoNCE objective."""
    if query.ndim != 2:
        raise ValueError(
            "query must have shape [B, D]."
        )

    if key.shape != query.shape:
        raise ValueError(
            "query and key must have identical shapes."
        )

    batch_size = query.shape[0]

    query = F.normalize(
        query,
        dim=-1,
    )

    key = F.normalize(
        key,
        dim=-1,
    )

    logits = (
        query
        @
        key.transpose(0, 1)
    ) / temperature

    if allowed_mask is not None:
        if allowed_mask.shape != (
            batch_size,
            batch_size,
        ):
            raise ValueError(
                "allowed_mask must have shape [B, B]."
            )

        allowed_mask = allowed_mask.bool()

        # A sample's positive pair must never be removed.
        positive_mask = torch.eye(
            batch_size,
            dtype=torch.bool,
            device=logits.device,
        )

        allowed_mask = (
            allowed_mask
            |
            positive_mask
        )

    if negative_weights is not None:
        if negative_weights.shape != (
            batch_size,
            batch_size,
        ):
            raise ValueError(
                "negative_weights must have shape [B, B]."
            )

        weights = negative_weights.to(
            device=logits.device,
            dtype=logits.dtype,
        )

        if (weights <= 0).any():
            raise ValueError(
                "negative_weights must be strictly positive."
            )

        # Positive-pair weight remains one. Weighting is only intended to
        # strengthen or weaken selected negatives.
        weights = weights.clone()

        diagonal = torch.arange(
            batch_size,
            device=logits.device,
        )

        weights[
            diagonal,
            diagonal,
        ] = 1.0

        logits = (
            logits
            +
            torch.log(weights)
        )

    if allowed_mask is not None:
        logits = logits.masked_fill(
            ~allowed_mask,
            float("-inf"),
        )

    target = torch.arange(
        batch_size,
        device=logits.device,
    )

    return F.cross_entropy(
        logits,
        target,
    )


def future_contrastive_loss(
    context_embeddings: Mapping[int, torch.Tensor],
    future_embeddings: Mapping[int, torch.Tensor],
    *,
    allowed_mask: torch.Tensor | None = None,
    horizon_weights: Mapping[int, float] | None = None,
    negative_weights: torch.Tensor | None = None,
    temperature: float = 0.1,
    symmetric: bool = False,
) -> torch.Tensor:
    """Compute horizon-aware context-to-future contrastive alignment.

    Expected horizon keys are normally:

        1
        3
        5

    representing 1 s, 3 s, and 5 s embeddings.

    Args:
        context_embeddings:
            Mapping from horizon in seconds to context embedding `[B, D]`.

        future_embeddings:
            Mapping from horizon in seconds to future embedding `[B, D]`.

        allowed_mask:
            Optional `[B, B]` mask produced by
            `build_false_negative_mask()`.

        horizon_weights:
            Optional scalar weight for each horizon.

        negative_weights:
            Optional `[B, B]` positive weights for behaviorally meaningful
            hard negatives. Values larger than one strengthen those negatives.
            False-negative exclusion is still controlled by `allowed_mask`.

        temperature:
            InfoNCE softmax temperature.

        symmetric:
            When True, average context-to-future and future-to-context losses.
            The initial WZ-TARF configuration can keep this False unless a
            symmetric objective is explicitly evaluated.

    Returns:
        Scalar weighted contrastive loss.
    """
    if temperature <= 0:
        raise ValueError(
            "temperature must be positive."
        )

    context_keys = set(
        context_embeddings.keys()
    )

    future_keys = set(
        future_embeddings.keys()
    )

    if context_keys != future_keys:
        raise ValueError(
            "Context and future embeddings must contain the same horizons."
        )

    if not context_keys:
        raise ValueError(
            "At least one contrastive horizon is required."
        )

    if horizon_weights is None:
        horizon_weights = {
            horizon: 1.0
            for horizon in context_keys
        }

    missing_weights = (
        context_keys
        -
        set(horizon_weights.keys())
    )

    if missing_weights:
        raise ValueError(
            f"Missing horizon weights for: {sorted(missing_weights)}"
        )

    total: torch.Tensor | None = None
    total_weight = 0.0

    for horizon in sorted(
        context_keys
    ):
        weight = float(
            horizon_weights[horizon]
        )

        if weight < 0:
            raise ValueError(
                "Horizon weights cannot be negative."
            )

        if weight == 0:
            continue

        context = context_embeddings[
            horizon
        ]

        future = future_embeddings[
            horizon
        ]

        loss = _contrastive_direction(
            context,
            future,
            temperature=temperature,
            allowed_mask=allowed_mask,
            negative_weights=negative_weights,
        )

        if symmetric:
            reverse_mask = (
                allowed_mask.transpose(0, 1)
                if allowed_mask is not None
                else None
            )

            reverse_weights = (
                negative_weights.transpose(0, 1)
                if negative_weights is not None
                else None
            )

            reverse_loss = _contrastive_direction(
                future,
                context,
                temperature=temperature,
                allowed_mask=reverse_mask,
                negative_weights=reverse_weights,
            )

            loss = (
                loss
                +
                reverse_loss
            ) / 2.0

        weighted = (
            weight
            *
            loss
        )

        total = (
            weighted
            if total is None
            else total + weighted
        )

        total_weight += weight

    if total is None or total_weight <= 0:
        # This path is only possible when every horizon weight is zero.
        reference = next(
            iter(
                context_embeddings.values()
            )
        )

        return reference.sum() * 0.0

    return total / total_weight
