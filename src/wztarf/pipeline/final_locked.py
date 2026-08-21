"""Frozen, publication-final WZ-TARF postprocessing chain.

This module intentionally does not alter the canonical fresh-training pipeline.

Locked chain:
    exact A3:F1
      -> historical X fixed12
      -> historical X endpoint-zero
      -> A20 x 2.0
      -> EXP_011 Y fixed12
      -> EXP_011 Y endpoint-zero

Final frozen TEST:
    minADE6 = 1.037688971 m
    minFDE6 = 2.009524822 m
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from wztarf.pipeline.late import (
    EndpointZeroCalibrator,
    apply_endpoint_zero,
    apply_fixed12,
)
from wztarf.postprocess.action_policy import (
    ActionPolicy,
    gate_features,
)


LOCKED_A20_SCALE = 2.0
REFERENCE_TEST_ADE = 1.037688971
REFERENCE_TEST_FDE = 2.009524822


def torch_load(path: str | Path) -> Any:
    try:
        return torch.load(
            Path(path),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            Path(path),
            map_location="cpu",
        )


def default_artifact_root() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "artifacts"
        / "final_wz_locked"
    )


def normalize_endpoint_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Make preserved legacy endpoint payloads compatible with release code."""

    out = dict(payload)
    state = dict(out["state_dict"])

    rename = {
        "mode_embedding.weight":
            "mode_embedding",
        "desc_norm.weight":
            "descriptor_norm.weight",
        "desc_norm.bias":
            "descriptor_norm.bias",
    }

    for old, new in rename.items():
        if old in state and new not in state:
            state[new] = state.pop(old)

    out["state_dict"] = state

    if "n_basis" not in out:
        out["n_basis"] = int(
            out["basis"].shape[0]
        )

    if "cap" not in out:
        out["cap"] = 1.5

    return out


def y_descriptor(
    pred: torch.Tensor,
) -> torch.Tensor:
    """Exact 9-D Y analogue of the historical X trajectory descriptor."""

    if pred.ndim != 4:
        raise ValueError(
            f"Expected [B,6,T,2], got {tuple(pred.shape)}"
        )

    y = pred[..., 1]

    idx = torch.tensor(
        [
            4,
            9,
            14,
            19,
            pred.shape[2] - 1,
        ],
        device=pred.device,
    )

    sampled = y[:, :, idx]
    mean = y.mean(-1, keepdim=True)
    std = y.std(
        -1,
        unbiased=False,
        keepdim=True,
    )

    early = (
        y[:, :, 4:5]
        - y[:, :, 0:1]
    ) / 4.0

    late = (
        y[:, :, -1:]
        - y[:, :, -6:-5]
    )

    descriptor = torch.cat(
        (
            sampled,
            mean,
            std,
            early,
            late,
        ),
        dim=-1,
    )

    if descriptor.shape[-1] != 9:
        raise RuntimeError(
            "Y trajectory descriptor must be 9-D"
        )

    return descriptor


def apply_y_fixed12(
    pred: torch.Tensor,
    payload: Mapping[str, Any],
) -> torch.Tensor:
    """Mode-specific Y correction y'=y+a*tau+b*tau^2."""

    a = payload["a"]
    b = payload["b"]

    tau = (
        torch.arange(
            1,
            pred.shape[2] + 1,
            device=pred.device,
            dtype=pred.dtype,
        )
        / float(pred.shape[2])
    )

    out = pred.clone()

    out[..., 1] += (
        a.to(
            pred.device,
            pred.dtype,
        )[None, :, None]
        * tau[None, None]
    )

    out[..., 1] += (
        b.to(
            pred.device,
            pred.dtype,
        )[None, :, None]
        * tau.square()[None, None]
    )

    return out


def apply_y_endpoint_zero(
    pred: torch.Tensor,
    state: torch.Tensor,
    payload: Mapping[str, Any],
    *,
    device: torch.device,
    batch_size: int = 2048,
) -> torch.Tensor:
    """State-conditioned endpoint-zero Y residual.

    The basis is exactly zero at t=T, so this stage cannot alter FDE.
    """

    payload = normalize_endpoint_payload(
        payload
    )

    model = EndpointZeroCalibrator(
        int(payload["n_basis"]),
        float(payload["cap"]),
    ).to(device)

    model.load_state_dict(
        payload["state_dict"],
        strict=True,
    )

    model.eval()

    state_mean = payload[
        "state_mean"
    ].float()

    state_std = payload[
        "state_std"
    ].float().clamp_min(
        1.0e-4
    )

    basis = payload[
        "basis"
    ].float().to(device)

    parts: list[torch.Tensor] = []

    with torch.inference_mode():

        for start in range(
            0,
            pred.shape[0],
            batch_size,
        ):
            p = pred[
                start:start + batch_size
            ].to(device)

            s = (
                (
                    state[
                        start:start + batch_size
                    ].float()
                    - state_mean
                )
                / state_std
            ).to(device)

            coeff = model(
                s,
                y_descriptor(p),
            )

            dy = torch.einsum(
                "bkc,ct->bkt",
                coeff,
                basis,
            )

            corrected = p.clone()
            corrected[..., 1] += dy

            parts.append(
                corrected.cpu()
            )

    return torch.cat(
        parts,
        dim=0,
    )


def apply_a20_scaled(
    pred: torch.Tensor,
    state: torch.Tensor,
    payload: Mapping[str, Any],
    *,
    scale: float = LOCKED_A20_SCALE,
    device: torch.device,
) -> torch.Tensor:
    """Apply the exact frozen A20 policy with the locked action scale."""

    features, far_idx = gate_features(
        pred,
        state,
    )

    mean = payload[
        "feature_mean"
    ].float()

    std = payload[
        "feature_std"
    ].float().clamp_min(
        1.0e-4
    )

    policy = ActionPolicy(
        features.shape[1]
    ).to(device)

    policy.load_state_dict(
        payload["state_dict"],
        strict=True,
    )

    policy.eval()

    with torch.inference_mode():
        action = policy(
            (
                (
                    features.float()
                    - mean
                )
                / std
            ).to(device)
        ).argmax(1).cpu()

    shifts = (
        payload["actions"]
        .float()[action]
        * float(scale)
    )

    tau = (
        torch.arange(
            1,
            pred.shape[2] + 1,
            dtype=pred.dtype,
            device=pred.device,
        )
        / float(pred.shape[2])
    )

    onset = float(
        payload["onset"]
    )

    power = float(
        payload["power"]
    )

    phi = (
        (
            (tau - onset)
            / (1.0 - onset)
        )
        .clamp(0.0, 1.0)
        .pow(power)
    )

    out = pred.clone()

    rows = torch.arange(
        out.shape[0],
        device=out.device,
    )

    out[
        rows,
        far_idx.to(out.device),
        :,
        0,
    ] += (
        shifts.to(
            out.device,
            out.dtype,
        )[:, None]
        * phi[None]
    )

    return out


def load_locked_artifacts(
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:

    root = (
        Path(artifact_root)
        if artifact_root is not None
        else default_artifact_root()
    )

    return {
        "root": root,
        "a3":
            torch_load(
                root / "a3f1_e01.pt"
            ),
        "fixed12":
            torch_load(
                root / "fixed12.pt"
            ),
        "x_endpoint":
            normalize_endpoint_payload(
                torch_load(
                    root
                    / "endpoint_zero.pt"
                )
            ),
        "a20":
            torch_load(
                root / "a20_policy.pt"
            ),
        "y_fixed12":
            torch_load(
                root / "y_fixed12.pt"
            ),
        "y_endpoint":
            normalize_endpoint_payload(
                torch_load(
                    root
                    / "y_endpoint_zero.pt"
                )
            ),
    }


def apply_locked_postprocess(
    pred: torch.Tensor,
    state: torch.Tensor,
    *,
    artifacts: Mapping[str, Any],
    device: torch.device,
    return_stages: bool = False,
):
    """Apply the exact permanently frozen final postprocessing chain."""

    stages: dict[
        str,
        torch.Tensor,
    ] = {}

    current = apply_fixed12(
        pred,
        artifacts["fixed12"]["a"],
        artifacts["fixed12"]["b"],
    )

    stages["x_fixed12"] = current

    current = apply_endpoint_zero(
        current,
        state,
        artifacts["x_endpoint"],
        device=device,
    )

    stages[
        "x_endpoint_zero"
    ] = current

    current = apply_a20_scaled(
        current,
        state,
        artifacts["a20"],
        scale=LOCKED_A20_SCALE,
        device=device,
    )

    stages[
        "a20_scale_2"
    ] = current

    current = apply_y_fixed12(
        current,
        artifacts["y_fixed12"],
    )

    stages[
        "y_fixed12"
    ] = current

    before_endpoint = (
        current[..., -1, :]
        .clone()
    )

    current = apply_y_endpoint_zero(
        current,
        state,
        artifacts["y_endpoint"],
        device=device,
    )

    # Mathematical invariant of endpoint-zero stage.
    torch.testing.assert_close(
        current[..., -1, :],
        before_endpoint,
        atol=1.0e-6,
        rtol=0.0,
    )

    stages[
        "y_endpoint_zero_final"
    ] = current

    if return_stages:
        return current, stages

    return current
