"""Compute reconstruction losses only at positions hidden by the mask plan."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F


def _expand_mask(
    mask: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Expand a token-level mask across trailing feature dimensions."""
    mask = mask.bool()

    if mask.ndim > target.ndim:
        raise ValueError(
            "Mask cannot have more dimensions than its target."
        )

    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(
            -1
        )

    try:
        return mask.expand_as(
            target
        )

    except RuntimeError as error:
        raise ValueError(
            f"Mask shape is incompatible with target shape "
            f"{tuple(target.shape)}."
        ) from error


def _masked_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_type: str,
    huber_beta: float,
) -> torch.Tensor:
    """Compute regression loss only at masked elements."""
    if prediction.shape != target.shape:
        raise ValueError(
            "Regression prediction and target must have identical shapes."
        )

    expanded_mask = _expand_mask(
        mask,
        target,
    )

    if not bool(
        expanded_mask.any()
    ):
        return prediction.sum() * 0.0

    if loss_type == "huber":
        element_loss = F.smooth_l1_loss(
            prediction,
            target,
            beta=huber_beta,
            reduction="none",
        )

    elif loss_type == "mse":
        element_loss = F.mse_loss(
            prediction,
            target,
            reduction="none",
        )

    elif loss_type == "l1":
        element_loss = F.l1_loss(
            prediction,
            target,
            reduction="none",
        )

    else:
        raise ValueError(
            f"Unsupported regression loss type: {loss_type}"
        )

    return element_loss[
        expanded_mask
    ].mean()


def _masked_classification_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute cross-entropy only at masked categorical positions."""
    if prediction.ndim != target.ndim + 1:
        raise ValueError(
            "Classification prediction must have one class dimension "
            "beyond the target."
        )

    if prediction.shape[:-1] != target.shape:
        raise ValueError(
            "Classification prediction leading dimensions must match target."
        )

    if mask.shape != target.shape:
        raise ValueError(
            "Classification mask must have the same shape as target."
        )

    mask = mask.bool()

    if not bool(
        mask.any()
    ):
        return prediction.sum() * 0.0

    num_classes = prediction.shape[-1]

    flat_prediction = prediction.reshape(
        -1,
        num_classes,
    )

    flat_target = target.long().reshape(
        -1
    )

    flat_mask = mask.reshape(
        -1
    )

    element_loss = F.cross_entropy(
        flat_prediction,
        flat_target,
        reduction="none",
    )

    return element_loss[
        flat_mask
    ].mean()


def masked_reconstruction_loss(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    *,
    modality_weights: Mapping[str, float] | None = None,
    loss_types: Mapping[str, str] | None = None,
    valid_masks: Mapping[str, torch.Tensor] | None = None,
    huber_beta: float = 1.0,
) -> torch.Tensor:
    """Combine reconstruction losses across masked scene modalities.

    Args:
        predictions:
            Reconstruction-head outputs keyed by modality.

        targets:
            Original unmasked targets using the same keys.

        masks:
            Boolean masks identifying positions that were intentionally hidden.

        modality_weights:
            Optional loss weight per modality.

        loss_types:
            Optional loss type per modality. Supported values:

                "huber"
                "mse"
                "l1"
                "cross_entropy"

            Huber is the default.

        valid_masks:
            Optional additional masks indicating genuinely valid source data.
            These are intersected with the intentional masking plan.

        huber_beta:
            Smooth-L1 transition point.

    Returns:
        Scalar weighted masked-reconstruction loss.

    Exact image-plane gaze XY reconstruction does not need to be used. A gaze
    reconstruction head can instead provide a coarse or latent target while
    still using this same objective.
    """
    if not predictions:
        raise ValueError(
            "At least one reconstruction prediction is required."
        )

    prediction_keys = set(
        predictions.keys()
    )

    missing_targets = (
        prediction_keys
        -
        set(targets.keys())
    )

    missing_masks = (
        prediction_keys
        -
        set(masks.keys())
    )

    if missing_targets:
        raise KeyError(
            f"Missing reconstruction targets: {sorted(missing_targets)}"
        )

    if missing_masks:
        raise KeyError(
            f"Missing reconstruction masks: {sorted(missing_masks)}"
        )

    modality_weights = (
        dict(modality_weights)
        if modality_weights is not None
        else {}
    )

    loss_types = (
        dict(loss_types)
        if loss_types is not None
        else {}
    )

    total: torch.Tensor | None = None
    total_weight = 0.0

    for modality in sorted(
        prediction_keys
    ):
        prediction = predictions[
            modality
        ]

        target = targets[
            modality
        ]

        mask = masks[
            modality
        ].bool()

        if (
            valid_masks is not None
            and
            modality in valid_masks
        ):
            valid = valid_masks[
                modality
            ].bool()

            if valid.shape != mask.shape:
                raise ValueError(
                    f"Valid mask for '{modality}' has shape "
                    f"{tuple(valid.shape)}, expected {tuple(mask.shape)}."
                )

            mask = (
                mask
                &
                valid
            )

        weight = float(
            modality_weights.get(
                modality,
                1.0,
            )
        )

        if weight < 0:
            raise ValueError(
                f"Negative reconstruction weight for '{modality}'."
            )

        if weight == 0:
            continue

        loss_type = loss_types.get(
            modality,
            "huber",
        )

        if loss_type == "cross_entropy":
            component = _masked_classification_loss(
                prediction,
                target,
                mask,
            )

        else:
            component = _masked_regression_loss(
                prediction,
                target,
                mask,
                loss_type=loss_type,
                huber_beta=huber_beta,
            )

        weighted = (
            weight
            *
            component
        )

        total = (
            weighted
            if total is None
            else total + weighted
        )

        total_weight += weight

    if total is None or total_weight <= 0:
        reference = next(
            iter(
                predictions.values()
            )
        )

        return reference.sum() * 0.0

    return total / total_weight
