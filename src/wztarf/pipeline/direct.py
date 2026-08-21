"""Dense-progress head-only, Direct-K6, AnchorCal, native-K64, and A3:F1 stages.

These stages train fresh neural parameters.  No released best-run checkpoint is
consulted by this module.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wztarf.losses.supervised import LossWeights, supervised_loss
from wztarf.models import WZTARF, WZTARFConfig

from .common import (
    canonical_aux_weights,
    exact_metric_parts,
    evaluate_model,
    extract_model_state,
    move_to_device,
    save_stage_checkpoint,
    seed_stage,
    torch_load,
    write_json,
)


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=bool(enabled and device.type == "cuda"),
    )


@torch.no_grad()
def _evaluate_model_stage(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
) -> dict[str, float]:
    """Historical Trainer.validate metric path for stage selection.

    Target/Anchor used BF16 autocast when use_amp=True; HEADONLY was FP32.
    """
    model.eval()
    totals = {key: 0.0 for key in ("ADE", "FDE", "XADE", "XFDE", "YADE", "YFDE")}
    count = 0
    for batch_cpu in loader:
        batch = move_to_device(batch_cpu, device)
        with _autocast(device, use_amp):
            output = model(batch)
        parts = exact_metric_parts(output["pred_xy"].float(), batch["future_xy"].float())
        n = int(batch["future_xy"].shape[0])
        count += n
        for key in totals:
            totals[key] += float(parts[key].sum())
    return {key: value / max(count, 1) for key, value in totals.items()}


def _zero_training_dropout(model: nn.Module) -> None:
    """Match the recovered late-stage recipe without switching trainable RNNs to eval mode."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        elif isinstance(module, nn.MultiheadAttention):
            module.dropout = 0.0


def _headonly_weights() -> LossWeights:
    """Exact REPRO_HEADONLY_FRESH supervised budget."""
    return LossWeights(
        trajectory=1.0,
        endpoint=1.0,
        route_progress_supervision=1.0,
        classification=0.0,
        behavior=0.0,
        ranking_quality=0.0,
        ranking_pairwise=0.0,
        lane=0.0,
        topology=0.0,
        topo_diversity=0.0,
        route_coverage=0.0,
        route=0.0,
        angle=0.0,
        dynamics=0.0,
        diversity=0.0,
        road=0.0,
        wz_geometry=0.0,
        worker=0.0,
        refinement=0.0,
    )


def _make_epoch_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: Any,
    epochs: int,
):
    """Mirror the historical scripts/train.py epoch-level scheduler."""
    if not scheduler_cfg:
        return None
    if isinstance(scheduler_cfg, str):
        scheduler_type = scheduler_cfg.lower()
        scheduler_cfg = {"type": scheduler_type}
    else:
        scheduler_type = str(scheduler_cfg.get("type", "none")).lower()
    if scheduler_type == "none":
        return None
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(epochs), 1),
            eta_min=float(scheduler_cfg.get("eta_min", 0.0)),
        )
    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def train_dense_progress_headonly(
    *,
    progressfix_checkpoint: str | Path,
    config: Mapping[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: str | Path,
) -> Path:
    """Exact historical ProgressFix -> REPRO_HEADONLY_FRESH transition.

    Historical WZ lineage:
      ProgressFix: 247 tensors / 2,243,801 parameters
      HEADONLY:    +10 fresh dense-progress tensors / +54,681 parameters

    The inherited 247 tensors are loaded bit-for-bit and frozen.  Only the
    three newly introduced dense-progress modules are trainable.

    For configurations without the explicit historical ProgressFix stage
    (currently the compatibility path used by No-WZ), the pre-existing
    257->257 behavior remains available and is gated by the config.
    """
    cfg = config["canonical_pipeline"]["dense_progress_headonly"]
    progressfix_cfg = config["canonical_pipeline"].get("progressfix", {})
    require_historical_transition = bool(progressfix_cfg.get("enabled", False))

    seed = int(cfg.get("seed", config["experiment"].get("seed", 2023)))
    seed_stage(seed)
    torch.set_float32_matmul_precision("high")

    model_cfg = dict(config["model"])
    model_cfg.update(
        use_dense_progress_repair=True,
        use_direct_decoder=False,
        use_direct_anchor_calibration=False,
        use_direct_longitudinal_repair=False,
        aux_dropout_controls=float(cfg.get("aux_dropout_controls", 0.1)),
        aux_dropout_gaze=float(cfg.get("aux_dropout_gaze", 0.1)),
        aux_dropout_workers=float(cfg.get("aux_dropout_workers", 0.1)),
    )

    # Construct AFTER seed_stage exactly at the historical HEADONLY boundary.
    # This is where the 10 dense-progress tensors are born.
    model = WZTARF(WZTARFConfig(**model_cfg))
    constructed_state = model.state_dict()
    out_tensors, out_params = _state_fingerprint(constructed_state)
    if (out_tensors, out_params) != (257, 2298482):
        raise RuntimeError(
            "HEADONLY constructed model fingerprint mismatch: "
            f"got {out_tensors}/{out_params}; expected 257/2298482"
        )

    source_state = extract_model_state(torch_load(progressfix_checkpoint))
    in_tensors, in_params = _state_fingerprint(source_state)

    dense_prefixes = (
        "route_progress.hard_route_geometry_encoder.",
        "route_progress.dense_progress_fusion.",
        "route_progress.dense_progress_residual_head.",
    )

    if require_historical_transition:
        if (in_tensors, in_params) != (247, 2243801):
            raise RuntimeError(
                "Historical HEADONLY must consume fresh ProgressFix "
                "247/2243801; got "
                f"{in_tensors}/{in_params} from {progressfix_checkpoint}. "
                "Refusing to collapse/skip ProgressFix."
            )

        result = model.load_state_dict(source_state, strict=False)
        missing = sorted(result.missing_keys)
        unexpected = sorted(result.unexpected_keys)
        if unexpected:
            raise RuntimeError(
                "ProgressFix -> HEADONLY unexpected tensors: " + repr(unexpected)
            )
        if len(missing) != 10:
            raise RuntimeError(
                "ProgressFix -> HEADONLY must introduce exactly 10 tensors; "
                f"got {len(missing)}: {missing}"
            )
        bad_missing = [
            key for key in missing
            if not any(key.startswith(prefix) for prefix in dense_prefixes)
        ]
        if bad_missing:
            raise RuntimeError(
                "ProgressFix -> HEADONLY missing unexpected keys: "
                + repr(bad_missing)
            )
        missing_params = sum(int(constructed_state[key].numel()) for key in missing)
        if missing_params != 54681:
            raise RuntimeError(
                "ProgressFix -> HEADONLY fresh parameter count mismatch: "
                f"got {missing_params}; expected 54681"
            )

        # Strong inheritance proof: every source tensor must survive loading
        # bit-for-bit. This catches accidental reconstruction/reinitialization.
        loaded_state = model.state_dict()
        inherited_bad = [
            key for key, value in source_state.items()
            if key not in loaded_state or not torch.equal(value.cpu(), loaded_state[key].detach().cpu())
        ]
        if inherited_bad:
            raise RuntimeError(
                "ProgressFix inheritance is not bit-identical for: "
                + repr(inherited_bad[:20])
            )
        print(
            "[HEADONLY] historical boundary verified: "
            "247 inherited tensors bit-identical; "
            "10 fresh tensors / 54,681 params added.",
            flush=True,
        )
    else:
        # Compatibility path: preserve V2's old 257 -> 257 transition for
        # configurations that have no historical ProgressFix stage.
        if (in_tensors, in_params) != (257, 2298482):
            raise RuntimeError(
                "HEADONLY compatibility input fingerprint mismatch: "
                f"got {in_tensors}/{in_params}; expected 257/2298482"
            )
        model.load_state_dict(source_state, strict=True)
        missing = []

    model.to(device)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dense_modules = (
        model.route_progress.hard_route_geometry_encoder,
        model.route_progress.dense_progress_fusion,
        model.route_progress.dense_progress_residual_head,
    )
    if any(module is None for module in dense_modules):
        raise RuntimeError("HEADONLY dense-progress modules were not constructed")
    for module in dense_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_params = sum(int(p.numel()) for p in trainable)
    frozen_params = sum(int(p.numel()) for p in model.parameters() if not p.requires_grad)
    if trainable_params != 54681:
        raise RuntimeError(
            f"HEADONLY trainable parameter mismatch: {trainable_params}; expected 54681"
        )
    if frozen_params != 2243801:
        raise RuntimeError(
            f"HEADONLY frozen parameter mismatch: {frozen_params}; expected 2243801"
        )

    # Historical standalone train.py passed model.parameters() to AdamW.
    # Frozen tensors have no gradients, so only the 54,681 trainable params move.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 2.0e-4)),
        weight_decay=float(cfg.get("weight_decay", 1.0e-4)),
    )
    epochs = int(cfg.get("epochs", 3))
    scheduler = _make_epoch_scheduler(optimizer, cfg.get("scheduler"), epochs)
    fde_w = float(cfg.get("composite_fde_weight", 0.25))
    grad_clip = float(cfg.get("grad_clip_norm", 5.0))
    use_amp = bool(cfg.get("use_amp", False))
    beta_assign = float(cfg.get("beta_assign", 1.0))
    weights = _headonly_weights()

    history: list[dict[str, Any]] = []
    best_state = None
    best_score = float("inf")
    best_epoch = 0
    best_val = None

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        nonfinite_grad = 0
        nonfinite_loss = 0
        for batch_cpu in train_loader:
            batch = move_to_device(batch_cpu, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, use_amp):
                output = model(batch)
                loss = supervised_loss(
                    output,
                    batch,
                    weights=weights,
                    beta_assign=beta_assign,
                    classification_temperature=float(config["training"].get("classification_temperature", 1.0)),
                    fps=int(config["data"].get("fps", 5)),
                    goal_association_tolerance_m=float(config["training"].get("goal_association_tolerance_m", 0.25)),
                    road_gt_tolerance_m=float(config["training"].get("road_gt_tolerance_m", 0.25)),
                ).total
            if not bool(torch.isfinite(loss)):
                nonfinite_loss += 1
                optimizer.zero_grad(set_to_none=True)
                if nonfinite_loss >= 8:
                    raise FloatingPointError(
                        f"HEADONLY had {nonfinite_loss} non-finite losses in epoch {epoch}"
                    )
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, grad_clip, error_if_nonfinite=False
            )
            if not bool(torch.isfinite(grad_norm)):
                nonfinite_grad += 1
                optimizer.zero_grad(set_to_none=True)
                if nonfinite_grad >= 8:
                    raise FloatingPointError(
                        f"HEADONLY had {nonfinite_grad} non-finite gradients in epoch {epoch}"
                    )
                continue
            optimizer.step()
            n = int(batch["future_xy"].shape[0])
            running += float(loss.detach()) * n
            seen += n

        val = _evaluate_model_stage(model, val_loader, device, use_amp=use_amp)
        score = float(val["ADE"] + fde_w * val["FDE"])
        history.append({
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "val": val,
            "J_val": score,
            "nonfinite_grad": nonfinite_grad,
            "nonfinite_loss": nonfinite_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
        })
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_val = dict(val)
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        print(
            f"[HEADONLY E{epoch:02d}] VAL={val['ADE']:.4f}/{val['FDE']:.4f} "
            f"J={score:.4f} lr={optimizer.param_groups[0]['lr']:.7g}",
            flush=True,
        )
        if scheduler is not None:
            scheduler.step()

    if best_state is None:
        raise RuntimeError("No HEADONLY checkpoint selected")

    model.load_state_dict(best_state, strict=True)
    final_tensors, final_params = _state_fingerprint(model.state_dict())
    if (final_tensors, final_params) != (257, 2298482):
        raise RuntimeError(
            f"HEADONLY output fingerprint mismatch: {final_tensors}/{final_params}"
        )

    out_dir = Path(out_dir)
    checkpoint = save_stage_checkpoint(
        out_dir / "dense_progress_headonly.pt",
        model=model,
        config=config,
        stage="dense_progress_headonly",
        extra={
            "selected_epoch": best_epoch,
            "selected_validation": best_val,
            "selected_J_val": best_score,
            "history": history,
            "progressfix_input_fingerprint": {
                "tensors": in_tensors,
                "params": in_params,
            },
            "fresh_headonly_tensors": missing,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
            "structural_fingerprint": {
                "tensors": final_tensors,
                "params": final_params,
            },
            "historical_reference": {
                "selected_epoch": 3,
                "ADE": 2.5623061656951904,
                "FDE": 5.340317726135254,
                "J_val": 3.897385597229004,
            },
        },
    )
    write_json(
        out_dir / "dense_progress_headonly_history.json",
        {
            "history": history,
            "selected_epoch": best_epoch,
            "selected_validation": best_val,
        },
    )
    return checkpoint


def _direct_target_weights() -> LossWeights:
    """Recovered TARGET08_16 supervised budget.

    The non-zero coefficients are identifiable from the archived TARGET08_16
    validation summaries.  This is the Direct-K6 stage that actually fed the
    successful fresh AnchorCal reproduction; it is not the later
    generator/repair diagnostic branch.
    """
    return LossWeights(
        trajectory=1.0,
        endpoint=1.5,
        classification=0.25,
        behavior=0.0,
        ranking_quality=0.0,
        ranking_pairwise=0.0,
        lane=0.0,
        topology=0.0,
        topo_diversity=0.0,
        route_coverage=0.0,
        route=0.0,
        angle=0.0,
        dynamics=0.05,
        diversity=0.05,
        road=0.0,
        wz_geometry=0.0,
        worker=0.0,
        refinement=0.0,
        route_progress_supervision=0.0,
    )


def _anchor_calibration_weights() -> LossWeights:
    return LossWeights(
        trajectory=1.0,
        endpoint=1.5,
        classification=0.0,
        behavior=0.0,
        ranking_quality=0.0,
        ranking_pairwise=0.0,
        lane=0.0,
        topology=0.0,
        topo_diversity=0.0,
        route_coverage=0.0,
        route=0.0,
        angle=0.0,
        dynamics=0.0,
        diversity=0.0,
        road=0.0,
        wz_geometry=0.0,
        worker=0.0,
        refinement=0.0,
        route_progress_supervision=0.0,
    )


def _state_fingerprint(state: Mapping[str, torch.Tensor]) -> tuple[int, int]:
    return len(state), sum(int(value.numel()) for value in state.values())


def build_direct_target(
    headonly_checkpoint: str | Path,
    config: Mapping[str, Any],
) -> WZTARF:
    """Build the historical TARGET08_16 Direct-K6 model from a non-direct parent.

    The recovered successful AnchorCal input had exactly 314 tensors and
    2,822,517 parameters: Direct-K6 present, anchor and longitudinal repair
    absent.  TARGET08_16 created the 57 Direct decoder tensors fresh.
    """
    model_cfg = dict(config["model"])
    model_cfg.update(
        use_direct_decoder=True,
        use_dense_progress_repair=True,
        use_direct_anchor_calibration=False,
        use_direct_longitudinal_repair=False,
    )
    model = WZTARF(WZTARFConfig(**model_cfg))
    state = extract_model_state(torch_load(headonly_checkpoint))
    if any(key.startswith("direct_trajectory_decoder.") for key in state):
        raise RuntimeError(
            "TARGET08_16 requires a non-direct parent checkpoint; Direct decoder tensors were already present."
        )
    result = model.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"HEADONLY -> TARGET08_16 had unexpected tensors: {unexpected}")
    missing = list(result.missing_keys)
    bad_missing = [key for key in missing if not key.startswith("direct_trajectory_decoder.")]
    if bad_missing:
        raise RuntimeError(f"HEADONLY -> TARGET08_16 unexpected missing tensors: {bad_missing}")
    if len(missing) != 57:
        raise RuntimeError(f"TARGET08_16 expected exactly 57 fresh Direct tensors; got {len(missing)}")
    tensors, params = _state_fingerprint(model.state_dict())
    if (tensors, params) != (314, 2822517):
        raise RuntimeError(
            f"TARGET08_16 structural fingerprint mismatch: got {tensors} tensors/{params} params; "
            "expected 314/2822517"
        )
    return model


def train_direct_target(
    *,
    headonly_checkpoint: str | Path,
    config: Mapping[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: str | Path,
) -> Path:
    """Fresh historical TARGET08_16 Direct-K6 stage.

    This is full-model supervised optimization (not decoder-only training),
    with a fresh 57-tensor Direct-K6 decoder and the archived epoch scheduler/J_val
    selection ADE + 0.25*FDE.
    """
    cfg = config["canonical_pipeline"]["direct_target"]
    seed = int(cfg.get("seed", 2023))
    seed_stage(seed)
    torch.set_float32_matmul_precision("high")
    model = build_direct_target(headonly_checkpoint, config).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 2.0e-4)),
        weight_decay=float(cfg.get("weight_decay", 1.0e-4)),
    )
    epochs = int(cfg.get("epochs", 18))
    scheduler = _make_epoch_scheduler(optimizer, cfg.get("scheduler"), epochs)
    patience = int(cfg.get("patience", 5))
    fde_w = float(cfg.get("composite_fde_weight", 0.25))
    grad_clip = float(cfg.get("grad_clip_norm", 5.0))
    use_amp = bool(cfg.get("use_amp", True))
    beta_assign = float(cfg.get("beta_assign", 1.0))
    weights = _direct_target_weights()

    history: list[dict[str, Any]] = []
    best_state = None
    best_score = float("inf")
    best_epoch = 0
    best_val = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        nonfinite_grad = 0
        nonfinite_loss = 0
        for batch_cpu in train_loader:
            batch = move_to_device(batch_cpu, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, use_amp):
                output = model(batch)
                loss = supervised_loss(
                    output,
                    batch,
                    weights=weights,
                    beta_assign=beta_assign,
                    classification_temperature=float(config["training"].get("classification_temperature", 1.0)),
                    fps=int(config["data"].get("fps", 5)),
                    goal_association_tolerance_m=float(config["training"].get("goal_association_tolerance_m", 0.25)),
                    road_gt_tolerance_m=float(config["training"].get("road_gt_tolerance_m", 0.25)),
                ).total
            if not bool(torch.isfinite(loss)):
                nonfinite_loss += 1
                optimizer.zero_grad(set_to_none=True)
                if nonfinite_loss >= 8:
                    raise FloatingPointError(f"TARGET08_16 had {nonfinite_loss} non-finite losses in epoch {epoch}")
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip, error_if_nonfinite=False
            )
            if not bool(torch.isfinite(grad_norm)):
                nonfinite_grad += 1
                optimizer.zero_grad(set_to_none=True)
                if nonfinite_grad >= 8:
                    raise FloatingPointError(f"TARGET08_16 had {nonfinite_grad} non-finite gradients in epoch {epoch}")
                continue
            optimizer.step()
            n = int(batch["future_xy"].shape[0])
            running += float(loss.detach()) * n
            seen += n

        val = _evaluate_model_stage(model, val_loader, device, use_amp=use_amp)
        score = float(val["ADE"] + fde_w * val["FDE"])
        record = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "val": val,
            "J_val": score,
            "nonfinite_grad": nonfinite_grad,
            "nonfinite_loss": nonfinite_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_val = dict(val)
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        print(
            f"[TARGET08_16 E{epoch:02d}] VAL={val['ADE']:.4f}/{val['FDE']:.4f} "
            f"J={score:.4f} bestJ={best_score:.4f} lr={optimizer.param_groups[0]['lr']:.7g}",
            flush=True,
        )
        if scheduler is not None:
            scheduler.step()
        if patience > 0 and bad_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("No TARGET08_16 checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    out_dir = Path(out_dir)
    checkpoint = save_stage_checkpoint(
        out_dir / "direct_target08_16.pt",
        model=model,
        config=config,
        stage="direct_k6_target08_16",
        extra={
            "selected_epoch": best_epoch,
            "selected_validation": best_val,
            "selected_J_val": best_score,
            "history": history,
            "structural_fingerprint": {"tensors": 314, "params": 2822517},
            "historical_fingerprint_reference": {
                "selected_epoch": 6,
                "ADE": 1.893672,
                "FDE": 3.278257,
                "J_val": 2.713235914707184,
            },
        },
    )
    write_json(
        out_dir / "direct_target08_16_history.json",
        {"history": history, "selected_epoch": best_epoch, "selected_validation": best_val},
    )
    return checkpoint


def train_anchor_calibration(
    *,
    direct_target_checkpoint: str | Path,
    config: Mapping[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: str | Path,
) -> Path:
    """Exact recovered AnchorCal transition: 314 tensors -> fresh four-tensor head.

    The input has no anchor head.  AnchorCal constructs it fresh, freezes the
    complete existing predictor, and trains only the 16,770 anchor parameters.
    nn.Dropout/MHA probabilities are zeroed, while the model stays in train
    mode so the historical config-driven whole-modality dropout remains active.
    """
    cfg = config["canonical_pipeline"]["anchor_calibration"]
    seed_stage(int(cfg.get("seed", 2023)))
    source_state = extract_model_state(torch_load(direct_target_checkpoint))
    tensors, params = _state_fingerprint(source_state)
    if (tensors, params) != (314, 2822517):
        raise RuntimeError(
            f"AnchorCal input must be historical Direct-K6 fingerprint 314/2822517; got {tensors}/{params}"
        )
    if any("anchor_correction_head" in key for key in source_state):
        raise RuntimeError("AnchorCal input already contains anchor-correction tensors")

    model_cfg = dict(config["model"])
    model_cfg.update(
        use_direct_decoder=True,
        use_dense_progress_repair=True,
        use_direct_anchor_calibration=True,
        use_direct_longitudinal_repair=False,
    )
    model = WZTARF(WZTARFConfig(**model_cfg))
    result = model.load_state_dict(source_state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Direct-K6 -> AnchorCal unexpected tensors: {unexpected}")
    if len(missing) != 4 or any("anchor_correction_head" not in key for key in missing):
        raise RuntimeError(f"AnchorCal expected exactly four fresh anchor tensors; got {missing}")
    tensors, params = _state_fingerprint(model.state_dict())
    if (tensors, params) != (318, 2839287):
        raise RuntimeError(f"AnchorCal model fingerprint mismatch: got {tensors}/{params}; expected 318/2839287")
    model.to(device)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    decoder = model.direct_trajectory_decoder
    if decoder is None or decoder.anchor_correction_head is None:
        raise RuntimeError("Anchor calibration head unavailable")
    for parameter in decoder.anchor_correction_head.parameters():
        parameter.requires_grad_(True)
    _zero_training_dropout(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in trainable) != 16770:
        raise RuntimeError("Anchor-only parameter count mismatch")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg.get("learning_rate", 5.0e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    epochs = int(cfg.get("epochs", 10))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=float(cfg.get("eta_min", 5.0e-5))
    )
    patience = int(cfg.get("patience", 3))
    fde_w = float(cfg.get("composite_fde_weight", 0.5))
    grad_clip = float(cfg.get("grad_clip_norm", 5.0))
    use_amp = bool(cfg.get("use_amp", True))
    beta_assign = float(cfg.get("beta_assign", 1.0))
    weights = _anchor_calibration_weights()
    history: list[dict[str, Any]] = []
    best_state = None
    best_score = float("inf")
    best_epoch = 0
    best_val = None
    bad = 0

    for epoch in range(1, epochs + 1):
        # Historical train.py calls model.train() here.  Do not change this to
        # eval(): cuDNN GRUs must remain train-mode and aux stream dropout is
        # part of the archived recipe.
        model.train()
        running = 0.0
        seen = 0
        for batch_cpu in train_loader:
            batch = move_to_device(batch_cpu, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, use_amp):
                output = model(batch)
                loss = supervised_loss(
                    output,
                    batch,
                    weights=weights,
                    beta_assign=beta_assign,
                    classification_temperature=float(config["training"].get("classification_temperature", 1.0)),
                    fps=int(config["data"].get("fps", 5)),
                    goal_association_tolerance_m=float(config["training"].get("goal_association_tolerance_m", 0.25)),
                    road_gt_tolerance_m=float(config["training"].get("road_gt_tolerance_m", 0.25)),
                ).total
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite anchor loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
            n = int(batch["future_xy"].shape[0])
            running += float(loss.detach()) * n
            seen += n
        val = _evaluate_model_stage(model, val_loader, device, use_amp=use_amp)
        score = float(val["ADE"] + fde_w * val["FDE"])
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1), "val": val, "J_val": score})
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_val = dict(val)
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"[ANCHOR E{epoch:02d}] VAL={val['ADE']:.4f}/{val['FDE']:.4f} J={score:.4f}", flush=True)
        scheduler.step()
        if patience > 0 and bad >= patience:
            break

    if best_state is None:
        raise RuntimeError("No anchor checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    out_dir = Path(out_dir)
    checkpoint = save_stage_checkpoint(
        out_dir / "anchor_calibrated.pt",
        model=model,
        config=config,
        stage="anchor_calibration_dual_winner",
        extra={
            "selected_epoch": best_epoch,
            "selected_validation": best_val,
            "history": history,
            "trainable_parameters": 16770,
            "fresh_anchor_tensors": 4,
            "input_structural_fingerprint": {"tensors": 314, "params": 2822517},
        },
    )
    write_json(
        out_dir / "anchor_calibration_history.json",
        {"history": history, "selected_epoch": best_epoch, "selected_validation": best_val},
    )
    return checkpoint


class NativeK64Distribution(nn.Module):
    """Historical native K64 adaptation wrapper around the trained Direct-K6 decoder."""

    def __init__(self, source_decoder: nn.Module) -> None:
        super().__init__()
        self.decoder = copy.deepcopy(source_decoder)
        self.decoder.num_modes = 64
        old = source_decoder.mode_embedding.detach()
        new = torch.empty(64, old.shape[1], dtype=old.dtype, device=old.device)
        new[:6] = old
        for index in range(6, 64):
            parent = index % 6
            new[index] = old[parent] + 0.06 * torch.randn_like(old[parent])
        self.decoder.mode_embedding = nn.Parameter(new)
        self.confidence_bias = nn.Parameter(torch.full((64,), -6.0, dtype=old.dtype, device=old.device))
        with torch.no_grad():
            self.confidence_bias[:6].zero_()
        dim = int(old.shape[1])
        self.uncertainty_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2)
        )
        nn.init.zeros_(self.uncertainty_head[-1].weight)
        nn.init.constant_(self.uncertainty_head[-1].bias, 0.5)

    def forward(self, decoder_kwargs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        decoded = self.decoder(**decoder_kwargs)
        hidden = decoded["direct_hidden"]
        mode_hidden = 0.5 * hidden.mean(dim=2) + 0.5 * hidden[:, :, -1]
        raw = self.uncertainty_head(mode_hidden)
        sigma_s = (0.20 + F.softplus(raw[..., 0])).clamp(max=15.0)
        sigma_d = (0.10 + F.softplus(raw[..., 1])).clamp(max=6.0)
        return {
            "pred_xy": decoded["pred_xy"],
            "logits": decoded["mode_logits"] + self.confidence_bias[None],
            "sigma_s": sigma_s,
            "sigma_d": sigma_d,
        }


def _trajectory_frame(pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    first = pred[:, :, :1]
    delta = torch.cat((first, pred[:, :, 1:] - pred[:, :, :-1]), dim=2)
    length = torch.linalg.vector_norm(delta, dim=-1)
    tangent = delta / length[..., None].clamp_min(1.0e-4)
    fallback = torch.zeros_like(tangent)
    fallback[..., 0] = 1.0
    tangent = torch.where(length[..., None] > 1.0e-4, tangent, fallback)
    normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)
    return tangent, normal


def _mixture_nll(output: Mapping[str, torch.Tensor], gt: torch.Tensor) -> torch.Tensor:
    pred = output["pred_xy"].float()
    tangent, normal = _trajectory_frame(pred)
    error = gt[:, None] - pred
    es = (error * tangent).sum(-1)
    ed = (error * normal).sum(-1)
    ss = output["sigma_s"].float()[:, :, None]
    sd = output["sigma_d"].float()[:, :, None]
    component = (
        0.5 * (es / ss).square() + torch.log(ss)
        + 0.5 * (ed / sd).square() + torch.log(sd)
    ).mean(-1)
    return -torch.logsumexp(F.log_softmax(output["logits"].float(), dim=1) - component, dim=1).mean()


def _k64_loss(output: Mapping[str, torch.Tensor], gt: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred = output["pred_xy"].float()
    diff = pred - gt[:, None]
    distance = torch.linalg.vector_norm(diff, dim=-1)
    ade_mode = distance.mean(-1)
    fde_mode = distance[:, :, -1]
    xade_mode = diff[..., 0].abs().mean(-1)
    xfde_mode = diff[:, :, -1, 0].abs()
    hard = (
        2.0 * ade_mode.min(1).values
        + 1.5 * fde_mode.min(1).values
        + xade_mode.gather(1, ade_mode.argmin(1)[:, None]).squeeze(1)
        + xfde_mode.gather(1, fde_mode.argmin(1)[:, None]).squeeze(1)
    ).mean()
    mode_cost = 2.0 * ade_mode + 1.5 * fde_mode + xade_mode + xfde_mode
    tau = 0.75
    soft = (-tau * torch.logsumexp(-mode_cost / tau, dim=1)).mean()
    target = F.softmax(-mode_cost.detach() / 0.80, dim=1)
    confidence = -(target * F.log_softmax(output["logits"].float(), dim=1)).sum(1).mean()
    nll = _mixture_nll(output, gt)
    endpoint = pred[:, :, -1]
    pair = torch.cdist(endpoint, endpoint)
    upper = torch.triu(torch.ones(64, 64, dtype=torch.bool, device=pred.device), diagonal=1)
    diversity = torch.exp(-pair[:, upper] / 2.0).mean()
    total = hard + 0.10 * soft + 0.20 * confidence + 0.05 * nll + 0.02 * diversity
    return total, {"hard": hard, "soft": soft, "confidence": confidence, "nll": nll, "diversity": diversity}


def train_native_k64_adaptation(
    *,
    anchor_checkpoint: str | Path,
    config: Mapping[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: str | Path,
) -> Path:
    """Run the historical 2-warmup + 4-joint native-K64 representation adaptation.

    K64 is an intermediate training device only.  The returned checkpoint is the
    updated native K6 backbone; the 64-mode distribution is not deployed.
    """
    cfg = config["canonical_pipeline"]["native_k64"]
    seed = int(cfg.get("seed", 20260816))
    seed_stage(seed)

    model_cfg = dict(config["model"])
    model_cfg.update(
        use_direct_decoder=True,
        use_dense_progress_repair=True,
        use_direct_anchor_calibration=True,
        use_direct_longitudinal_repair=False,
        aux_dropout_controls=0.0,
        aux_dropout_gaze=0.0,
        aux_dropout_workers=0.0,
    )
    model = WZTARF(WZTARFConfig(**model_cfg)).to(device)
    model.load_state_dict(extract_model_state(torch_load(anchor_checkpoint)), strict=True)
    # Exact historical K64 source zeroed every nn.Dropout and MHA dropout and
    # forced whole-modality auxiliary dropout to zero before any K64 training.
    _zero_training_dropout(model)
    source_decoder = model.direct_trajectory_decoder
    if source_decoder is None or int(source_decoder.num_modes) != 6:
        raise RuntimeError("Native-K64 stage requires a Direct K=6 decoder")

    capture: dict[str, Any] = {}
    def pre_hook(module, positional, kwargs):
        capture["kwargs"] = kwargs
    handle = source_decoder.register_forward_pre_hook(pre_hook, with_kwargs=True)

    # Historical FINAL K64 retrain restored the backbone, re-seeded globally,
    # then constructed the K64 distribution.  Query perturbations and the
    # uncertainty-head initialization therefore consume the same global RNG.
    seed_stage(seed)
    distribution = NativeK64Distribution(source_decoder).to(device)
    warmup_epochs = int(cfg.get("warmup_epochs", 2))
    joint_epochs = int(cfg.get("joint_epochs", 4))
    grad_accum = int(cfg.get("grad_accum", 8))
    grad_clip = float(cfg.get("grad_clip_norm", 5.0))
    use_amp = bool(cfg.get("use_amp", True))
    wd = float(cfg.get("weight_decay", 1.0e-4))

    def set_trainability(joint: bool) -> None:
        for name, p in model.named_parameters():
            p.requires_grad_(bool(joint and not name.startswith("direct_trajectory_decoder.")))
        model.train() if joint else model.eval()
        # The native K6 decoder is only a hook/capture point in this stage.
        # Keep it deterministic exactly as in the historical K64 trainer.
        source_decoder.eval()

    def optimizer_for(joint: bool) -> torch.optim.Optimizer:
        special = [distribution.decoder.mode_embedding, distribution.confidence_bias]
        special_ids = {id(p) for p in special}
        uncertainty = list(distribution.uncertainty_head.parameters())
        special_ids.update(id(p) for p in uncertainty)
        shared = [p for p in distribution.parameters() if id(p) not in special_ids]
        groups: list[dict[str, Any]] = [
            {"params": special + uncertainty, "lr": float(cfg.get("new_query_lr", 2.0e-4))},
            {"params": shared, "lr": float(cfg.get("shared_decoder_lr", 5.0e-5))},
        ]
        if joint:
            upstream = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("direct_trajectory_decoder.")]
            groups.append({"params": upstream, "lr": float(cfg.get("backbone_lr", 1.0e-5))})
        return torch.optim.AdamW(groups, weight_decay=wd)

    history: list[dict[str, Any]] = []
    global_epoch = 0
    for phase, epochs, joint in (("WARMUP", warmup_epochs, False), ("JOINT", joint_epochs, True)):
        set_trainability(joint)
        optimizer = optimizer_for(joint)
        for local_epoch in range(1, epochs + 1):
            global_epoch += 1
            distribution.train()
            optimizer.zero_grad(set_to_none=True)
            running = 0.0
            seen = 0
            for batch_index, batch_cpu in enumerate(train_loader, 1):
                batch = move_to_device(batch_cpu, device)
                capture.clear()
                with _autocast(device, use_amp):
                    if joint:
                        _ = model(batch)
                    else:
                        with torch.no_grad():
                            _ = model(batch)
                    if "kwargs" not in capture:
                        raise RuntimeError("Direct decoder pre-hook did not fire during K64 stage")
                    output = distribution(capture["kwargs"])
                    loss, pieces = _k64_loss(output, batch["future_xy"].float())
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                (loss / grad_accum).backward()
                if batch_index % grad_accum == 0 or batch_index == len(train_loader):
                    params = list(distribution.parameters()) + ([p for p in model.parameters() if p.requires_grad] if joint else [])
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                n = int(batch["future_xy"].shape[0])
                running += float(loss.detach()) * n
                seen += n
            record = {"phase": phase, "epoch": local_epoch, "global_epoch": global_epoch, "train_loss": running / max(seen, 1)}
            history.append(record)
            print(f"[K64 {phase} E{local_epoch:02d}] loss={record['train_loss']:.5f}", flush=True)

    handle.remove()
    # Historical full retrain evaluates the adapted native K6 after the locked
    # 2+4 schedule.  This is a fingerprint only; it does not select weights.
    final_native_k6_val = evaluate_model(model, val_loader, device)
    print(
        f"[K64 NATIVE-K6 FINAL VAL] {final_native_k6_val['ADE']:.4f}/{final_native_k6_val['FDE']:.4f}",
        flush=True,
    )
    # Deployment stays exact K=6. Only the jointly adapted native backbone survives.
    out_dir = Path(out_dir)
    checkpoint = save_stage_checkpoint(
        out_dir / "k64_adapted_k6_backbone.pt",
        model=model,
        config=config,
        stage="native_k64_intermediate_adaptation",
        extra={
            "warmup_epochs": warmup_epochs,
            "joint_epochs": joint_epochs,
            "history": history,
            "deployment_num_modes": 6,
            "k64_is_training_only": True,
            "final_native_k6_validation": final_native_k6_val,
        },
    )
    torch.save(
        {
            "distribution_state_dict": {k: v.detach().cpu() for k, v in distribution.state_dict().items()},
            "backbone_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "history": history,
            "final_native_k6_validation": final_native_k6_val,
        },
        out_dir / "native_k64_training_artifact.pt",
    )
    write_json(
        out_dir / "native_k64_history.json",
        {"history": history, "final_native_k6_validation": final_native_k6_val},
    )
    return checkpoint


def train_a3f1_one_epoch(
    *,
    k64_backbone_checkpoint: str | Path,
    config: Mapping[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: str | Path,
) -> Path:
    """Canonical clean A3:F1 recipe: exactly one epoch, never TEST-selected."""
    cfg = config["canonical_pipeline"]["a3f1"]
    seed = int(cfg.get("seed", 20260816))
    seed_stage(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    model_cfg = dict(config["model"])
    model_cfg.update(
        use_direct_decoder=True,
        use_dense_progress_repair=True,
        use_direct_anchor_calibration=True,
        use_direct_longitudinal_repair=False,
        aux_dropout_controls=0.0,
        aux_dropout_gaze=0.0,
        aux_dropout_workers=0.0,
    )
    model = WZTARF(WZTARFConfig(**model_cfg)).to(device)
    model.load_state_dict(extract_model_state(torch_load(k64_backbone_checkpoint)), strict=True)
    for p in model.parameters():
        p.requires_grad_(True)
    # Historical A3:F1 zeroed nn.Dropout and MultiheadAttention dropout while
    # keeping the full model in train mode for the two backward passes.
    _zero_training_dropout(model)
    # The historical objective runner re-seeded immediately before resetting
    # the already-constructed backbone. Re-seed after constructor/load so the
    # epoch-1 optimizer/data stochastic state starts identically.
    seed_stage(seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 5.0e-5)),
        weight_decay=float(cfg.get("weight_decay", 1.0e-5)),
    )
    grad_clip = float(cfg.get("grad_clip_norm", 5.0))
    aux_weights = LossWeights(**canonical_aux_weights(config))
    ade_w = float(cfg.get("ade_weight", 3.0))
    fde_w = float(cfg.get("fde_weight", 1.0))

    model.train()
    running_metric = 0.0
    running_aux = 0.0
    seen = 0
    skipped = 0
    for batch_cpu in train_loader:
        batch = move_to_device(batch_cpu, device)
        optimizer.zero_grad(set_to_none=True)

        # Pass 1: exact benchmark objective, independent minADE and minFDE winners.
        with torch.autocast(device_type=device.type, enabled=False):
            pred = model(batch)["pred_xy"].float()
            parts = exact_metric_parts(pred, batch["future_xy"].float())
            metric = (ade_w * parts["ADE"].mean() + fde_w * parts["FDE"].mean()) / (ade_w + fde_w)
        metric.backward()

        # Pass 2: structured auxiliaries, with legacy trajectory regression exactly zero.
        with torch.autocast(device_type=device.type, enabled=False):
            output = model(batch)
            aux = supervised_loss(
                output,
                batch,
                weights=aux_weights,
                beta_assign=float(cfg.get("beta_assign", 1.0)),
                classification_temperature=float(config["training"].get("classification_temperature", 1.0)),
                fps=int(config["data"].get("fps", 5)),
                goal_association_tolerance_m=float(config["training"].get("goal_association_tolerance_m", 0.25)),
                road_gt_tolerance_m=float(config["training"].get("road_gt_tolerance_m", 0.25)),
            ).total
        aux.backward()

        bad = any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        if bad:
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()
        n = int(batch["future_xy"].shape[0])
        seen += n
        running_metric += float(metric.detach()) * n
        running_aux += float(aux.detach()) * n

    val = evaluate_model(model, val_loader, device)
    print(
        f"[A3:F1 E01] metric={running_metric/max(seen,1):.5f} "
        f"aux={running_aux/max(seen,1):.5f} skipped={skipped} "
        f"VAL={val['ADE']:.4f}/{val['FDE']:.4f}",
        flush=True,
    )
    out_dir = Path(out_dir)
    checkpoint = save_stage_checkpoint(
        out_dir / "a3f1_e01.pt",
        model=model,
        config=config,
        stage="A3F1_exact_one_epoch",
        extra={
            "objective": {"ADE": ade_w, "FDE": fde_w},
            "auxiliary_weights": canonical_aux_weights(config),
            "skipped_nonfinite_batches": skipped,
            "validation": val,
            "test_selection": False,
        },
    )
    write_json(
        out_dir / "a3f1_result.json",
        {"validation": val, "objective": {"ADE": ade_w, "FDE": fde_w}, "test_opened": False},
    )
    return checkpoint
