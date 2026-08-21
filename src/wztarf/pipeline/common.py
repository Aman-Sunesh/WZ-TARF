"""Shared utilities for the canonical fresh WZ-TARF end-to-end recipe."""
from __future__ import annotations

import json
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from wztarf.data import collate_workzone_batch, collate_workzone_fixed
from wztarf.data.dataset import WorkZoneDataset
from wztarf.models import WZTARF, WZTARFConfig
from wztarf.utils import make_generator, seed_worker


FINAL_AUX_WEIGHTS: dict[str, float] = {
    "trajectory": 0.0,  # exact K6 metric is back-propagated in a separate pass
    "endpoint": 3.0,
    "classification": 0.0,
    "behavior": 0.11558221350887386,
    "ranking_quality": 0.005829607193944534,
    "ranking_pairwise": 0.023109797634287072,
    "lane": 0.000012713086178219568,
    "topology": 0.5206833499872551,
    "topo_diversity": 0.02206599181895725,
    "route_coverage": 0.0005637230484035371,
    "route": 0.00004808674279155135,
    "angle": 0.8137651314874784,
    "dynamics": 0.0030495131855796834,
    "diversity": 0.0,
    "road": 0.00021592040659237168,
    "wz_geometry": 0.00013631355459813984,
    "worker": 0.07130411278942961,
    "refinement": 0.008190362664186988,
    "route_progress_supervision": 0.0,
}


def resolve_device(requested: str | None) -> torch.device:
    device = torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def parse_roots(raw: str | Sequence[str | Path]) -> list[Path]:
    values = raw.split(",") if isinstance(raw, str) else raw
    roots = [Path(value).expanduser() for value in values if str(value).strip()]
    if not roots:
        raise ValueError("At least one processed dataset root is required")
    return roots


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    fixed_collate: bool = True,
) -> DataLoader:
    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_workzone_fixed if fixed_collate else collate_workzone_batch,
        worker_init_fn=seed_worker,
        generator=make_generator(seed) if shuffle else None,
        **kwargs,
    )


def extract_model_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "backbone_state_dict", "state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, Mapping) and value and all(torch.is_tensor(v) for v in value.values()):
                return dict(value)
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return dict(payload)
    raise RuntimeError("Could not extract a model state_dict from checkpoint")


def torch_load(path: str | Path) -> Any:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


def load_model_from_checkpoint(
    checkpoint: str | Path,
    config: Mapping[str, Any],
    *,
    direct: bool,
    anchor: bool,
    repair: bool,
    strict: bool = False,
) -> WZTARF:
    model_cfg = dict(config["model"])
    model_cfg.update(
        use_direct_decoder=direct,
        use_direct_anchor_calibration=anchor,
        use_direct_longitudinal_repair=repair,
    )
    if direct:
        # Late metric stages were deterministic: no sample-level stream dropout.
        model_cfg.update(
            aux_dropout_controls=0.0,
            aux_dropout_gaze=0.0,
            aux_dropout_workers=0.0,
        )
    model = WZTARF(WZTARFConfig(**model_cfg))
    state = extract_model_state(torch_load(checkpoint))
    result = model.load_state_dict(state, strict=strict)
    return model, result


def save_stage_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    config: Mapping[str, Any],
    stage: str,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "stage": stage,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": dict(config),
        "extra": dict(extra or {}),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def exact_metric_parts(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, torch.Tensor]:
    diff = pred - gt[:, None]
    dist = torch.linalg.vector_norm(diff, dim=-1)
    ade_mode = dist.mean(dim=-1)
    fde_mode = dist[:, :, -1]
    ade_idx = ade_mode.argmin(dim=1)
    fde_idx = fde_mode.argmin(dim=1)
    row = torch.arange(pred.shape[0], device=pred.device)
    return {
        "ADE": ade_mode[row, ade_idx],
        "FDE": fde_mode[row, fde_idx],
        "ade_mode": ade_mode,
        "fde_mode": fde_mode,
        "ade_idx": ade_idx,
        "fde_idx": fde_idx,
        "XADE": diff[..., 0].abs().mean(-1)[row, ade_idx],
        "XFDE": diff[:, :, -1, 0].abs()[row, fde_idx],
        "YADE": diff[..., 1].abs().mean(-1)[row, ade_idx],
        "YFDE": diff[:, :, -1, 1].abs()[row, fde_idx],
    }


def exact_means(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    parts = exact_metric_parts(pred, gt)
    return {key: float(parts[key].float().mean()) for key in ("ADE", "FDE", "XADE", "XFDE", "YADE", "YFDE")}


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {key: 0.0 for key in ("ADE", "FDE", "XADE", "XFDE", "YADE", "YFDE")}
    count = 0
    for batch_cpu in loader:
        batch = move_to_device(batch_cpu, device)
        output = model(batch)
        parts = exact_metric_parts(output["pred_xy"].float(), batch["future_xy"].float())
        n = int(batch["future_xy"].shape[0])
        count += n
        for key in totals:
            totals[key] += float(parts[key].sum())
    return {key: value / max(count, 1) for key, value in totals.items()}


def temporal_summary(x: torch.Tensor) -> torch.Tensor:
    """Six causal statistics per channel: last, mean, std, span, mean-diff, last-diff."""
    if x.ndim != 3:
        raise ValueError(f"Expected [B,T,C], got {tuple(x.shape)}")
    delta = x[:, 1:] - x[:, :-1]
    return torch.cat(
        (
            x[:, -1],
            x.mean(dim=1),
            x.std(dim=1, unbiased=False),
            x[:, -1] - x[:, 0],
            delta.mean(dim=1),
            delta[:, -1],
        ),
        dim=1,
    )


def build_causal_state(batch: Mapping[str, Any]) -> torch.Tensor:
    """Build the exact 78-D causal late-policy state; explicit wz_feat is excluded."""
    ego = temporal_summary(batch["ego_hist"].float())          # 6*6 = 36
    control = temporal_summary(batch["control_hist"].float())  # 3*6 = 18
    gaze = temporal_summary(batch["gaze_feat"].float())        # 3*6 = 18
    workers = batch["wz_worker_feat"].float().reshape(ego.shape[0], -1)  # 2*3 = 6
    state = torch.cat((ego, control, gaze, workers), dim=1)
    if state.shape[1] != 78:
        raise RuntimeError(f"Late causal state must be 78-D, got {state.shape[1]}")
    return torch.nan_to_num(state)


def participant_of_sample(sample: Mapping[str, Any], path: Path | None = None) -> str:
    meta = sample.get("meta", {})
    if isinstance(meta, Mapping):
        for key in ("participant", "participant_id", "subject"):
            if meta.get(key) is not None:
                return str(meta[key])
    candidates = [str(path)] if path is not None else []
    candidates.extend(str(sample.get(key, "")) for key in ("participant", "participant_id", "subject"))
    for text in candidates:
        match = re.search(r"(?<![A-Za-z0-9])P(\d+)(?!\d)", text, flags=re.IGNORECASE)
        if match:
            return f"P{int(match.group(1))}"
    raise RuntimeError("Could not determine participant identity for internal DEV/HOLD split")


def internal_dev_hold_indices(
    dataset: WorkZoneDataset,
    hold_participants: Sequence[str] = ("P7", "P18", "P28"),
) -> tuple[list[int], list[int]]:
    hold_set = {str(x).upper() for x in hold_participants}
    dev: list[int] = []
    hold: list[int] = []
    for index, path in enumerate(dataset.files):
        sample = torch.load(path, map_location="cpu", weights_only=False)
        participant = participant_of_sample(sample, path).upper()
        (hold if participant in hold_set else dev).append(index)
    if not dev or not hold:
        raise RuntimeError(f"Invalid internal split: DEV={len(dev)} HOLD={len(hold)}")
    return dev, hold


def canonical_aux_weights(config: Mapping[str, Any]) -> dict[str, float]:
    """Frozen A3 auxiliary vector with the scientifically required No-WZ zeros."""
    values = dict(FINAL_AUX_WEIGHTS)
    static = str(config["model"].get("topology_mode", "workzone")).lower() == "static"
    if static:
        values["topology"] = 0.0
        values["wz_geometry"] = 0.0
    if not bool(config["model"].get("use_workers", True)):
        values["worker"] = 0.0
    return values


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def seed_stage(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
