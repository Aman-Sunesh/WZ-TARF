"""Frozen-recipe longitudinal calibration and A20 policy training."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wztarf.models import WZTARF, WZTARFConfig
from wztarf.postprocess.action_policy import ActionPolicy, gate_features

from .common import (
    build_causal_state,
    exact_means,
    exact_metric_parts,
    extract_model_state,
    move_to_device,
    seed_stage,
    torch_load,
    write_json,
)


@torch.no_grad()
def cache_predictions(
    *,
    model_checkpoint: str | Path,
    config: Mapping[str, Any],
    loader: DataLoader,
    device: torch.device,
    label: str,
) -> dict[str, torch.Tensor]:
    model_cfg = dict(config["model"])
    model_cfg.update(
        use_direct_decoder=True,
        use_direct_anchor_calibration=True,
        use_direct_longitudinal_repair=False,
        aux_dropout_controls=0.0,
        aux_dropout_gaze=0.0,
        aux_dropout_workers=0.0,
    )
    model = WZTARF(WZTARFConfig(**model_cfg)).to(device)
    model.load_state_dict(extract_model_state(torch_load(model_checkpoint)), strict=True)
    model.eval()
    pred_parts: list[torch.Tensor] = []
    gt_parts: list[torch.Tensor] = []
    state_parts: list[torch.Tensor] = []
    for index, batch_cpu in enumerate(loader, 1):
        batch = move_to_device(batch_cpu, device)
        pred_parts.append(model(batch)["pred_xy"].float().cpu())
        gt_parts.append(batch["future_xy"].float().cpu())
        state_parts.append(build_causal_state(batch).float().cpu())
        if index == 1 or index == len(loader) or index % 100 == 0:
            print(f"[CACHE {label}] {index}/{len(loader)}", flush=True)
    return {
        "pred": torch.cat(pred_parts, dim=0),
        "gt": torch.cat(gt_parts, dim=0),
        "state": torch.cat(state_parts, dim=0),
    }


def apply_fixed12(pred: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if pred.ndim != 4 or pred.shape[1] != 6:
        raise ValueError("fixed12 expects [B,6,T,2]")
    if a.numel() != 6 or b.numel() != 6:
        raise ValueError("fixed12 requires six a and six b scalars")
    tau = torch.arange(1, pred.shape[2] + 1, device=pred.device, dtype=pred.dtype) / float(pred.shape[2])
    out = pred.clone()
    out[..., 0] += a.to(pred.device, pred.dtype)[None, :, None] * tau[None, None]
    out[..., 0] += b.to(pred.device, pred.dtype)[None, :, None] * tau.square()[None, None]
    return out


def _smooth_min(value: torch.Tensor, temperature: float) -> torch.Tensor:
    return -temperature * torch.logsumexp(-value / temperature, dim=1)


def _fit_fixed12(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    lambda_f: float,
    lambda_r: float,
    steps: int,
    lr: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    base = pred.to(device)
    target = gt.to(device)
    a = nn.Parameter(torch.zeros(6, device=device))
    b = nn.Parameter(torch.zeros(6, device=device))
    optimizer = torch.optim.Adam((a, b), lr=lr)
    last = float("nan")
    for _ in range(steps):
        calibrated = apply_fixed12(base, a, b)
        diff = torch.linalg.vector_norm(calibrated - target[:, None], dim=-1)
        ade = diff.mean(-1)
        fde = diff[:, :, -1]
        reg = a.square().mean() + 2.0 * b.square().mean() + 0.25 * (a + b).square().mean()
        loss = _smooth_min(ade, 0.08).mean() + lambda_f * _smooth_min(fde, 0.15).mean() + lambda_r * reg
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last = float(loss.detach())
    return a.detach().cpu(), b.detach().cpu(), last


def train_fixed12(
    *,
    cache: Mapping[str, torch.Tensor],
    dev_indices: Sequence[int],
    hold_indices: Sequence[int],
    config: Mapping[str, Any],
    device: torch.device,
    out_dir: str | Path,
) -> tuple[Path, torch.Tensor]:
    cfg = config["canonical_pipeline"]["fixed12"]
    dev = torch.as_tensor(dev_indices, dtype=torch.long)
    hold = torch.as_tensor(hold_indices, dtype=torch.long)
    pred = cache["pred"]
    gt = cache["gt"]
    candidates: list[dict[str, Any]] = []
    for lambda_f in cfg.get("lambda_f_screen", [0.20, 0.30, 0.40, 0.50]):
        for lambda_r in cfg.get("lambda_r_screen", [0.001, 0.010]):
            a, b, loss = _fit_fixed12(
                pred[dev], gt[dev],
                lambda_f=float(lambda_f), lambda_r=float(lambda_r),
                steps=int(cfg.get("screen_steps", 500)), lr=float(cfg.get("learning_rate", 0.03)), device=device,
            )
            hold_pred = apply_fixed12(pred[hold], a, b)
            metrics = exact_means(hold_pred, gt[hold])
            score = metrics["ADE"] + 0.5 * metrics["FDE"]
            candidates.append({"lambda_f": float(lambda_f), "lambda_r": float(lambda_r), "hold": metrics, "score": score, "a": a, "b": b, "loss": loss})
            print(f"[FIXED12 F={lambda_f} R={lambda_r}] HOLD={metrics['ADE']:.4f}/{metrics['FDE']:.4f}", flush=True)
    winner = min(candidates, key=lambda row: row["score"])
    final_a, final_b, final_loss = _fit_fixed12(
        pred, gt,
        lambda_f=winner["lambda_f"], lambda_r=winner["lambda_r"],
        steps=int(cfg.get("refit_steps", 650)), lr=float(cfg.get("learning_rate", 0.03)), device=device,
    )
    full_pred = apply_fixed12(pred, final_a, final_b)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fixed12.pt"
    torch.save(
        {
            "a": final_a,
            "b": final_b,
            "formula": "x'=x+a[k]*tau+b[k]*tau^2",
            "selected_lambda_f": winner["lambda_f"],
            "selected_lambda_r": winner["lambda_r"],
            "screen_hold": winner["hold"],
            "final_loss": final_loss,
            "test_opened": False,
        },
        path,
    )
    serial = [{k: v for k, v in row.items() if k not in {"a", "b"}} for row in candidates]
    write_json(out_dir / "fixed12_screen.json", {"candidates": serial, "winner": {k: v for k, v in winner.items() if k not in {"a", "b"}}})
    return path, full_pred


def endpoint_zero_basis(n_basis: int, steps: int = 25, *, device: torch.device | None = None) -> torch.Tensor:
    tau = torch.arange(1, steps + 1, dtype=torch.float32, device=device) / float(steps)
    degree = n_basis + 1
    rows = []
    for i in range(1, degree):
        phi = tau.pow(i) * (1.0 - tau).pow(degree - i)
        phi = phi / phi.abs().max().clamp_min(1.0e-8)
        rows.append(phi)
    basis = torch.stack(rows, dim=0)
    if basis.shape != (n_basis, steps):
        raise RuntimeError("Endpoint-zero basis construction failed")
    if float(basis[:, -1].abs().max()) > 1.0e-7:
        raise RuntimeError("Endpoint-zero basis is not zero at the final step")
    return basis


def trajectory_descriptor(pred: torch.Tensor) -> torch.Tensor:
    x = pred[..., 0]
    idx = torch.tensor([4, 9, 14, 19, pred.shape[2] - 1], device=pred.device)
    sampled = x[:, :, idx]
    mean_x = x.mean(-1, keepdim=True)
    std_x = x.std(-1, unbiased=False, keepdim=True)
    early_speed = (x[:, :, 4:5] - x[:, :, 0:1]) / 4.0
    late_delta = x[:, :, -1:] - x[:, :, -6:-5]
    descriptor = torch.cat((sampled, mean_x, std_x, early_speed, late_delta), dim=-1)
    if descriptor.shape[-1] != 9:
        raise RuntimeError("Trajectory descriptor must be 9-D")
    return descriptor


class EndpointZeroCalibrator(nn.Module):
    """78-D state + mode identity + 9-D trajectory descriptor -> endpoint-zero X basis."""
    def __init__(self, n_basis: int, cap: float) -> None:
        super().__init__()
        self.n_basis = int(n_basis)
        self.cap = float(cap)
        self.state_encoder = nn.Sequential(nn.Linear(78, 48), nn.GELU(), nn.Linear(48, 48), nn.GELU())
        self.mode_embedding = nn.Parameter(torch.randn(6, 8) * 0.02)
        self.descriptor_norm = nn.LayerNorm(9)
        self.head = nn.Sequential(nn.Linear(65, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, self.n_basis))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, state: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        encoded = self.state_encoder(state)[:, None].expand(-1, 6, -1)
        mode = self.mode_embedding[None].expand(state.shape[0], -1, -1)
        desc = self.descriptor_norm(descriptor)
        raw = self.head(torch.cat((encoded, mode, desc), dim=-1))
        return self.cap * torch.tanh(raw)


def apply_endpoint_zero(
    pred: torch.Tensor,
    state: torch.Tensor,
    payload: Mapping[str, Any],
    *,
    device: torch.device,
    batch_size: int = 2048,
) -> torch.Tensor:
    model = EndpointZeroCalibrator(int(payload["n_basis"]), float(payload["cap"])).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    state_mean = payload["state_mean"].float()
    state_std = payload["state_std"].float().clamp_min(1.0e-4)
    basis = payload["basis"].float().to(device)
    parts = []
    with torch.inference_mode():
        for start in range(0, pred.shape[0], batch_size):
            p = pred[start:start + batch_size].to(device)
            s = ((state[start:start + batch_size].float() - state_mean) / state_std).to(device)
            coeff = model(s, trajectory_descriptor(p))
            dx = torch.einsum("bkc,ct->bkt", coeff, basis)
            out = p.clone()
            out[..., 0] += dx
            parts.append(out.cpu())
    return torch.cat(parts, dim=0)


def train_endpoint_zero(
    *,
    fixed_pred: torch.Tensor,
    gt: torch.Tensor,
    state: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    out_dir: str | Path,
) -> tuple[Path, torch.Tensor]:
    cfg = config["canonical_pipeline"]["endpoint_zero"]
    seed = int(cfg.get("seed", 20260819))
    seed_stage(seed)
    n_basis = int(cfg.get("n_basis", 7))
    cap = float(cfg.get("cap", 1.5))
    epochs = int(cfg.get("epochs", 19))
    batch_size = int(cfg.get("batch_size", 512))
    state_mean = state.float().mean(0)
    state_std = state.float().std(0, unbiased=False).clamp_min(1.0e-4)
    basis = endpoint_zero_basis(n_basis, fixed_pred.shape[2])
    model = EndpointZeroCalibrator(n_basis, cap).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 7.5e-4)), weight_decay=float(cfg.get("weight_decay", 1.0e-4)))
    grad_clip = float(cfg.get("grad_clip_norm", 5.0))
    dense_weight = float(cfg.get("dense_weight", 0.30))
    tau_soft = float(cfg.get("soft_ade_temperature", 0.075))
    q_temp = float(cfg.get("responsibility_temperature", 0.20))
    coeff_reg_weight = float(cfg.get("coefficient_regularization", 0.0125))
    temporal = torch.arange(1, fixed_pred.shape[2] + 1, dtype=torch.float32) / float(fixed_pred.shape[2])
    time_weight = 0.5 + temporal
    time_weight = time_weight / time_weight.mean()

    with torch.no_grad():
        base_dist = torch.linalg.vector_norm(fixed_pred - gt[:, None], dim=-1)
        q_a = F.softmax(-base_dist.mean(-1) / q_temp, dim=1)

    n = fixed_pred.shape[0]
    for epoch in range(1, epochs + 1):
        model.train()
        gen = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(n, generator=gen)
        running = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            ids = order[start:start + batch_size]
            p = fixed_pred[ids].to(device)
            y = gt[ids].to(device)
            s = ((state[ids].float() - state_mean) / state_std).to(device)
            q = q_a[ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            coeff = model(s, trajectory_descriptor(p))
            dx = torch.einsum("bkc,ct->bkt", coeff, basis.to(device))
            corrected = p.clone()
            corrected[..., 0] += dx
            dist = torch.linalg.vector_norm(corrected - y[:, None], dim=-1)
            ade_mode = dist.mean(-1)
            soft_a = (-tau_soft * torch.logsumexp(-ade_mode / tau_soft, dim=1)).mean()
            dense = F.smooth_l1_loss(corrected[..., 0], y[:, None, :, 0].expand_as(corrected[..., 0]), reduction="none")
            dense = (dense * q[:, :, None] * time_weight.to(device)[None, None]).sum(dim=(1, 2)) / (q.sum(1) * time_weight.sum().to(device)).clamp_min(1.0e-6)
            loss = soft_a + dense_weight * dense.mean() + coeff_reg_weight * coeff.square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite endpoint-zero loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            bn = len(ids)
            running += float(loss.detach()) * bn
            seen += bn
        print(f"[ENDPOINT-ZERO E{epoch:02d}] loss={running/max(seen,1):.5f}", flush=True)

    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "state_mean": state_mean,
        "state_std": state_std,
        "basis": basis,
        "n_basis": n_basis,
        "cap": cap,
        "selected_epoch": epochs,
        "dense_weight": dense_weight,
        "test_opened": False,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "endpoint_zero.pt"
    torch.save(payload, path)
    corrected = apply_endpoint_zero(fixed_pred, state, payload, device=device)
    before_fde = exact_means(fixed_pred, gt)["FDE"]
    after_fde = exact_means(corrected, gt)["FDE"]
    if abs(after_fde - before_fde) > 1.0e-5:
        raise RuntimeError(f"Endpoint-zero stage changed FDE: {before_fde} -> {after_fde}")
    return path, corrected


def _apply_actions_table(pred: torch.Tensor, gt: torch.Tensor, far_idx: torch.Tensor, actions: torch.Tensor, onset: float, power: float) -> tuple[torch.Tensor, torch.Tensor]:
    steps = pred.shape[2]
    tau = torch.arange(1, steps + 1, dtype=torch.float32) / float(steps)
    phi = ((tau - onset) / (1.0 - onset)).clamp(0.0, 1.0).pow(power)
    all_a = []
    all_f = []
    rows = torch.arange(pred.shape[0])
    for amount in actions:
        candidate = pred.clone()
        candidate[rows, far_idx, :, 0] += float(amount) * phi[None]
        parts = exact_metric_parts(candidate, gt)
        all_a.append(parts["ADE"].float())
        all_f.append(parts["FDE"].float())
    return torch.stack(all_a, 1), torch.stack(all_f, 1)


def train_a20(
    *,
    endpoint_pred: torch.Tensor,
    gt: torch.Tensor,
    state: torch.Tensor,
    dev_indices: Sequence[int],
    hold_indices: Sequence[int],
    config: Mapping[str, Any],
    device: torch.device,
    out_dir: str | Path,
) -> Path:
    cfg = config["canonical_pipeline"]["a20"]
    seed = int(cfg.get("seed", 20260838))
    seed_stage(seed)
    actions = torch.tensor(cfg.get("actions", [0.0, 0.25, 0.50, 0.75, 1.0]), dtype=torch.float32)
    onset = float(cfg.get("onset", 0.75))
    power = float(cfg.get("power", 3.0))
    margin = float(cfg.get("margin", 0.05))
    gain_alpha = float(cfg.get("gain_alpha", 5.0))
    epochs = int(cfg.get("epochs", 22))
    batch_size = int(cfg.get("batch_size", 512))

    dev = torch.as_tensor(dev_indices, dtype=torch.long)
    hold = torch.as_tensor(hold_indices, dtype=torch.long)
    features, far_idx = gate_features(endpoint_pred, state)
    if features.shape[1] != 114:
        raise RuntimeError(f"A20 features must be 114-D, got {features.shape[1]}")
    dev_feat = features[dev].float()
    hold_feat = features[hold].float()
    feat_mean = dev_feat.mean(0)
    feat_std = dev_feat.std(0, unbiased=False).clamp_min(1.0e-4)
    xdev = (dev_feat - feat_mean) / feat_std
    xhold = (hold_feat - feat_mean) / feat_std

    dev_a, dev_f = _apply_actions_table(endpoint_pred[dev], gt[dev], far_idx[dev], actions, onset, power)
    hold_a, hold_f = _apply_actions_table(endpoint_pred[hold], gt[hold], far_idx[hold], actions, onset, power)
    dev_best_f = dev_f.min(1, keepdim=True).values
    regret = dev_f - dev_best_f
    labels = regret.argmin(1)
    gain = (dev_f[:, 0] - dev_best_f[:, 0]).clamp_min(0.0)
    hold_base_a = float(hold_a[:, 0].mean())
    hold_base_f = float(hold_f[:, 0].mean())

    model = ActionPolicy(114).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 5.0e-4)), weight_decay=float(cfg.get("weight_decay", 1.0e-4)))
    grad_clip = float(cfg.get("grad_clip_norm", 5.0))
    best: dict[str, Any] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(dev), generator=torch.Generator().manual_seed(seed + epoch))
        for start in range(0, len(dev), batch_size):
            ids = order[start:start + batch_size]
            logits = model(xdev[ids].to(device))
            r = regret[ids].to(device)
            label = labels[ids].to(device)
            g = gain[ids].to(device)
            best_logit = logits.gather(1, label[:, None])
            diff = best_logit - logits
            pair = F.softplus(margin - diff)
            numer = (pair * r).sum(1)
            denom = r.sum(1).clamp_min(1.0e-6)
            per_sample = numer / denom
            valid = r.sum(1) > 1.0e-8
            if not bool(valid.any()):
                continue
            loss = per_sample[valid]
            loss = loss * (1.0 + gain_alpha * g[valid].clamp(max=1.0))
            loss = loss.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            action = model(xhold.to(device)).argmax(1).cpu()
        selected_a = hold_a.gather(1, action[:, None]).squeeze(1)
        selected_f = hold_f.gather(1, action[:, None]).squeeze(1)
        a = float(selected_a.mean())
        f = float(selected_f.mean())
        penalty = 1000.0 * max(0.0, a - hold_base_a)
        score = f + penalty
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "epoch": epoch,
                "ADE": a,
                "FDE": f,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        print(f"[A20 E{epoch:02d}] HOLD={a:.4f}/{f:.4f}", flush=True)

    assert best is not None
    payload = {
        "state_dict": best["state_dict"],
        "feature_mean": feat_mean,
        "feature_std": feat_std,
        "config": {"id": 20, "name": "PAIRWISE_M005_GAIN", "kind": "pairwise", "margin": margin, "gain_alpha": gain_alpha},
        "selected_epoch": best["epoch"],
        "hold_result": {"ADE": best["ADE"], "FDE": best["FDE"]},
        "actions": actions,
        "onset": onset,
        "power": power,
        "test_opened": False,
        "refit": False,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "a20_policy.pt"
    torch.save(payload, path)
    write_json(out_dir / "a20_development_result.json", {"selected_epoch": best["epoch"], "hold": payload["hold_result"], "config": payload["config"], "test_opened": False})
    return path


def apply_final_postprocess(
    pred: torch.Tensor,
    state: torch.Tensor,
    *,
    fixed12_payload: Mapping[str, Any],
    endpoint_payload: Mapping[str, Any],
    a20_payload: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor:
    current = apply_fixed12(pred, fixed12_payload["a"], fixed12_payload["b"])
    current = apply_endpoint_zero(current, state, endpoint_payload, device=device)
    features, far_idx = gate_features(current, state)
    mean = a20_payload["feature_mean"].float()
    std = a20_payload["feature_std"].float().clamp_min(1.0e-4)
    policy = ActionPolicy(features.shape[1]).to(device)
    policy.load_state_dict(a20_payload["state_dict"], strict=True)
    policy.eval()
    with torch.inference_mode():
        action = policy(((features.float() - mean) / std).to(device)).argmax(1).cpu()
    actions = a20_payload["actions"].float()
    shifts = actions[action]
    tau = torch.arange(1, current.shape[2] + 1, dtype=torch.float32) / float(current.shape[2])
    onset = float(a20_payload["onset"])
    phi = ((tau - onset) / (1.0 - onset)).clamp(0.0, 1.0).pow(float(a20_payload["power"]))
    out = current.clone()
    rows = torch.arange(out.shape[0])
    out[rows, far_idx, :, 0] += shifts[:, None] * phi[None]
    return out


def save_final_bundle(
    *,
    a3_checkpoint: str | Path,
    fixed12_path: str | Path,
    endpoint_path: str | Path,
    a20_path: str | Path,
    config: Mapping[str, Any],
    out_dir: str | Path,
) -> Path:
    bundle = {
        "format_version": 1,
        "stage": "canonical_wztarf_final",
        "model_state_dict": extract_model_state(torch_load(a3_checkpoint)),
        "fixed12": torch_load(fixed12_path),
        "endpoint_zero": torch_load(endpoint_path),
        "a20": torch_load(a20_path),
        "config": dict(config),
        "pipeline": [
            "phase_a", "phase_b", "progressfix", "dense_progress_headonly", "direct_k6_target08_16",
            "anchor_calibration_dual_winner", "native_k64_intermediate_adaptation",
            "A3F1_exact_one_epoch", "fixed12", "endpoint_zero", "A20",
        ],
        "test_selection": False,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "final_pipeline_bundle.pt"
    torch.save(bundle, path)
    return path
