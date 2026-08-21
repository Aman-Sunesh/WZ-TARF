"""Train the complete canonical WZ-TARF recipe from fresh data.

TEST is never loaded unless --open-test is supplied, and even then it is opened
only after the neural backbone, fixed12 calibration, endpoint-zero calibrator,
and A20 policy have all been frozen.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset

from wztarf.data.dataset import WorkZoneDataset
from wztarf.pipeline.common import (
    exact_means,
    extract_model_state,
    internal_dev_hold_indices,
    make_loader,
    parse_roots,
    resolve_device,
    torch_load,
    write_json,
)
from wztarf.pipeline.direct import (
    train_a3f1_one_epoch,
    train_anchor_calibration,
    train_dense_progress_headonly,
    train_direct_target,
    train_native_k64_adaptation,
)
from wztarf.pipeline.late import (
    apply_final_postprocess,
    cache_predictions,
    save_final_bundle,
    train_a20,
    train_endpoint_zero,
    train_fixed12,
)
from wztarf.utils import load_yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fresh full WZ-TARF end-to-end training")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "wz.yaml")
    parser.add_argument("--data-roots", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--phase-b-checkpoint",
        type=Path,
        default=None,
        help=(
            "Use an explicitly supplied fresh Phase-B checkpoint and skip Phase A/B. "
            "This is for downstream reproduction/verification and never searches historical artifacts."
        ),
    )
    parser.add_argument(
        "--stop-after-headonly",
        action="store_true",
        help="Stop after exact REPRO_HEADONLY_FRESH and write its VAL fingerprint.",
    )
    parser.add_argument(
        "--stop-after-target",
        action="store_true",
        help="Stop after the fresh TARGET08_16 stage so its VAL fingerprint can be checked cheaply.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse checkpoints produced by THIS run-id only; never uses artifacts/best_wz.",
    )
    parser.add_argument(
        "--open-test",
        action="store_true",
        help="Evaluate official TEST once, after the complete recipe is frozen.",
    )
    return parser.parse_args()


def run_subprocess(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def checkpoint_fingerprint(path: Path) -> tuple[int, int]:
    state = extract_model_state(torch_load(path))
    return len(state), sum(int(v.numel()) for v in state.values())


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if "canonical_pipeline" not in config:
        raise KeyError("Config is missing canonical_pipeline; use the patched wz/no_wz configs")
    roots = parse_roots(args.data_roots)
    device = resolve_device(args.device)
    num_workers = args.num_workers if args.num_workers is not None else int(config["data"].get("num_workers", 0))
    pin_memory = bool(config["data"].get("pin_memory", True)) and device.type == "cuda"
    seed = int(config["experiment"].get("seed", 2023))

    progressfix_cfg = config["canonical_pipeline"].get("progressfix", {})
    use_historical_progressfix = bool(progressfix_cfg.get("enabled", False))
    phaseab_config_path = (
        ROOT / str(progressfix_cfg["phaseab_runtime_config"])
        if use_historical_progressfix
        else args.config
    )
    progressfix_runtime_config = (
        ROOT / str(progressfix_cfg["runtime_config"])
        if use_historical_progressfix
        else None
    )
    progressfix_trainer = (
        ROOT / str(progressfix_cfg["historical_trainer"])
        if use_historical_progressfix
        else None
    )
    if use_historical_progressfix:
        for required in (phaseab_config_path, progressfix_runtime_config, progressfix_trainer):
            if not required.is_file():
                raise FileNotFoundError(
                    f"Historical ProgressFix runtime asset missing: {required}"
                )

    run_root = ROOT / "checkpoints" / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    stage_dir = run_root / "canonical_stages"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase A/B, or an explicitly supplied fresh Phase-B checkpoint.
    # ------------------------------------------------------------------
    if args.phase_b_checkpoint is not None:
        phase_b = args.phase_b_checkpoint.expanduser().resolve()
        if not phase_b.is_file():
            raise FileNotFoundError(f"Explicit Phase-B checkpoint missing: {phase_b}")
        print(f"[PHASE-B OVERRIDE] using fresh checkpoint: {phase_b}", flush=True)
        print("[PHASE-B OVERRIDE] Phase A/B training skipped; downstream stages remain fresh.", flush=True)
    else:
        phase_a_id = f"{args.run_id}_phaseA"
        phase_a = ROOT / "checkpoints" / phase_a_id / "pretrain_best.pt"
        if not (args.resume_existing and phase_a.is_file()):
            run_subprocess([
                sys.executable, str(ROOT / "scripts" / "pretrain.py"),
                "--config", str(phaseab_config_path),
                "--data-roots", args.data_roots,
                "--run-id", phase_a_id,
                *( ["--device", args.device] if args.device else [] ),
            ])
        if not phase_a.is_file():
            raise FileNotFoundError(f"Phase-A best checkpoint missing: {phase_a}")

        phase_b_id = f"{args.run_id}_phaseB"
        phase_b = ROOT / "checkpoints" / phase_b_id / "best_composite.pt"
        if not (args.resume_existing and phase_b.is_file()):
            cmd = [
                sys.executable, str(progressfix_trainer if use_historical_progressfix else ROOT / "scripts" / "train.py"),
                "--config", str(phaseab_config_path),
                "--data-roots", args.data_roots,
                "--run-id", phase_b_id,
                "--pretrained", str(phase_a),
            ]
            if args.device:
                cmd += ["--device", args.device]
            if args.num_workers is not None:
                cmd += ["--num-workers", str(args.num_workers)]
            run_subprocess(cmd)
        if not phase_b.is_file():
            raise FileNotFoundError(f"Phase-B best checkpoint missing: {phase_b}")

    # ------------------------------------------------------------------
    # Historical missing bridge: Phase-B (247) -> ProgressFix (247).
    # ProgressFix is freshly trained; no preserved historical learned
    # checkpoint is loaded anywhere in this canonical run.
    # ------------------------------------------------------------------
    if use_historical_progressfix:
        phase_b_fp = checkpoint_fingerprint(Path(phase_b))
        if phase_b_fp != (247, 2243801):
            raise RuntimeError(
                "Historical Phase-B architecture must be 247/2243801 "
                f"before ProgressFix; got {phase_b_fp}"
            )

        progressfix_id = f"{args.run_id}_progressfix"
        progressfix = ROOT / "checkpoints" / progressfix_id / "best_composite.pt"
        if not (args.resume_existing and progressfix.is_file()):
            cmd = [
                sys.executable, str(progressfix_trainer),
                "--config", str(progressfix_runtime_config),
                "--data-roots", args.data_roots,
                "--run-id", progressfix_id,
                "--initialize", str(phase_b),
            ]
            if args.device:
                cmd += ["--device", args.device]
            if args.num_workers is not None:
                cmd += ["--num-workers", str(args.num_workers)]
            run_subprocess(cmd)

        if not progressfix.is_file():
            raise FileNotFoundError(
                f"Fresh ProgressFix checkpoint missing: {progressfix}"
            )
        progressfix_fp = checkpoint_fingerprint(progressfix)
        if progressfix_fp != (247, 2243801):
            raise RuntimeError(
                "Fresh ProgressFix output fingerprint mismatch: "
                f"got {progressfix_fp}; expected (247, 2243801)"
            )
        print(
            f"[PROGRESSFIX] fresh checkpoint={progressfix} "
            f"fingerprint={progressfix_fp}",
            flush=True,
        )
    else:
        # Compatibility path for configurations without the recovered WZ
        # ProgressFix stage (e.g. existing No-WZ workflows).
        progressfix = phase_b

    train_ds = WorkZoneDataset(roots=roots, split=config["data"].get("train_split", "train"), validate=False)
    val_ds = WorkZoneDataset(roots=roots, split=config["data"].get("val_split", "val"), validate=False)
    dev_indices, hold_indices = internal_dev_hold_indices(
        train_ds,
        tuple(config["canonical_pipeline"].get("hold_participants", ["P7", "P18", "P28"])),
    )
    print(
        f"[DATA] train={len(train_ds)} val={len(val_ds)} "
        f"internal_DEV={len(dev_indices)} internal_HOLD={len(hold_indices)}",
        flush=True,
    )

    def loader(dataset, *, batch_size: int, shuffle: bool, local_seed: int):
        return make_loader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=local_seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            fixed_collate=bool(config["data"].get("fixed_collate", True)),
        )

    val_loader = loader(val_ds, batch_size=int(config["evaluation"].get("batch_size", 8)), shuffle=False, local_seed=seed)

    # Exact historical bridge 1: ProgressFix (247) -> HEADONLY (+10 fresh tensors).
    headonly = stage_dir / "dense_progress_headonly.pt"
    if not (args.resume_existing and headonly.is_file()):
        c = config["canonical_pipeline"]["dense_progress_headonly"]
        headonly = train_dense_progress_headonly(
            progressfix_checkpoint=progressfix,
            config=config,
            train_loader=loader(
                train_ds,
                batch_size=int(c.get("batch_size", 8)),
                shuffle=True,
                local_seed=int(c.get("seed", 2023)),
            ),
            val_loader=val_loader,
            device=device,
            out_dir=stage_dir,
        )

    head_payload = torch.load(headonly, map_location="cpu", weights_only=False)
    head_extra = head_payload.get("extra", {})
    head_summary = {
        "run_id": args.run_id,
        "stage": "dense_progress_headonly",
        "selected_epoch": head_extra.get("selected_epoch"),
        "selected_validation": head_extra.get("selected_validation"),
        "historical_reference": {
            "selected_epoch": 3,
            "ADE": 2.5623061656951904,
            "FDE": 5.340317726135254,
            "J_val": 3.897385597229004,
        },
    }
    write_json(run_root / "HEADONLY_RESULT.json", head_summary)
    if args.stop_after_headonly:
        print("\n" + json.dumps(head_summary, indent=2), flush=True)
        return

    # Exact historical bridge 2: HEADONLY -> TARGET08_16.
    # TARGET creates the 57 Direct-K6 tensors fresh; no anchor/repair exists yet.
    direct_target = stage_dir / "direct_target08_16.pt"
    if not (args.resume_existing and direct_target.is_file()):
        c = config["canonical_pipeline"]["direct_target"]
        direct_target = train_direct_target(
            headonly_checkpoint=headonly,
            config=config,
            train_loader=loader(
                train_ds,
                batch_size=int(c.get("batch_size", 8)),
                shuffle=True,
                local_seed=int(c.get("seed", 2023)),
            ),
            val_loader=val_loader,
            device=device,
            out_dir=stage_dir,
        )

    target_payload = torch.load(direct_target, map_location="cpu", weights_only=False)
    target_extra = target_payload.get("extra", {})
    target_summary = {
        "run_id": args.run_id,
        "stage": "direct_k6_target08_16",
        "selected_epoch": target_extra.get("selected_epoch"),
        "selected_validation": target_extra.get("selected_validation"),
        "historical_reference": {
            "selected_epoch": 6,
            "ADE": 1.893672,
            "FDE": 3.278257,
            "J_val": 2.713235914707184,
        },
    }
    write_json(run_root / "TARGET08_16_RESULT.json", target_summary)
    if args.stop_after_target:
        print("\n" + json.dumps(target_summary, indent=2), flush=True)
        return

    # AnchorCal: construct the four anchor tensors fresh from TARGET08_16.
    anchor = stage_dir / "anchor_calibrated.pt"
    if not (args.resume_existing and anchor.is_file()):
        c = config["canonical_pipeline"]["anchor_calibration"]
        anchor = train_anchor_calibration(
            direct_target_checkpoint=direct_target,
            config=config,
            train_loader=loader(
                train_ds,
                batch_size=int(c.get("batch_size", 8)),
                shuffle=True,
                local_seed=int(c.get("seed", 2023)),
            ),
            val_loader=val_loader,
            device=device,
            out_dir=stage_dir,
        )

    # Native K64 intermediate adaptation. Training batch is historically 4.
    k64 = stage_dir / "k64_adapted_k6_backbone.pt"
    if not (args.resume_existing and k64.is_file()):
        c = config["canonical_pipeline"]["native_k64"]
        k64 = train_native_k64_adaptation(
            anchor_checkpoint=anchor,
            config=config,
            train_loader=loader(train_ds, batch_size=int(c.get("batch_size", 4)), shuffle=True, local_seed=int(c.get("seed", 20260816))),
            val_loader=val_loader,
            device=device,
            out_dir=stage_dir,
        )

    dev_train_ds = Subset(train_ds, dev_indices)

    # Historical A3:F1 winning transformation is encoded as exactly one epoch;
    # there is no five-epoch TEST sweep in this clean reproduction.
    a3 = stage_dir / "a3f1_e01.pt"
    if not (args.resume_existing and a3.is_file()):
        c = config["canonical_pipeline"]["a3f1"]
        a3 = train_a3f1_one_epoch(
            k64_backbone_checkpoint=k64,
            config=config,
            train_loader=loader(
                dev_train_ds,
                batch_size=int(c.get("batch_size", 8)),
                shuffle=True,
                local_seed=int(c.get("epoch1_loader_seed", int(c.get("seed", 20260816)) + 1009)),
            ),
            val_loader=val_loader,
            device=device,
            out_dir=stage_dir,
        )

    # Cache TRAIN predictions exactly once for the tiny late calibration stages.
    cache_loader = loader(
        train_ds,
        batch_size=int(config["canonical_pipeline"].get("cache_batch_size", 32)),
        shuffle=False,
        local_seed=seed,
    )
    cache_file = stage_dir / "train_a3_cache.pt"
    if args.resume_existing and cache_file.is_file():
        train_cache = torch.load(cache_file, map_location="cpu", weights_only=False)
    else:
        train_cache = cache_predictions(
            model_checkpoint=a3, config=config, loader=cache_loader, device=device, label="TRAIN-A3"
        )
        torch.save(train_cache, cache_file)

    fixed12_path = stage_dir / "fixed12.pt"
    fixed_pred_file = stage_dir / "train_fixed12_pred.pt"
    if args.resume_existing and fixed12_path.is_file() and fixed_pred_file.is_file():
        fixed_pred = torch.load(fixed_pred_file, map_location="cpu", weights_only=False)
    else:
        fixed12_path, fixed_pred = train_fixed12(
            cache=train_cache,
            dev_indices=dev_indices,
            hold_indices=hold_indices,
            config=config,
            device=device,
            out_dir=stage_dir,
        )
        torch.save(fixed_pred, fixed_pred_file)

    endpoint_path = stage_dir / "endpoint_zero.pt"
    endpoint_pred_file = stage_dir / "train_endpoint_zero_pred.pt"
    if args.resume_existing and endpoint_path.is_file() and endpoint_pred_file.is_file():
        endpoint_pred = torch.load(endpoint_pred_file, map_location="cpu", weights_only=False)
    else:
        endpoint_path, endpoint_pred = train_endpoint_zero(
            fixed_pred=fixed_pred,
            gt=train_cache["gt"],
            state=train_cache["state"],
            config=config,
            device=device,
            out_dir=stage_dir,
        )
        torch.save(endpoint_pred, endpoint_pred_file)

    a20_path = stage_dir / "a20_policy.pt"
    if not (args.resume_existing and a20_path.is_file()):
        a20_path = train_a20(
            endpoint_pred=endpoint_pred,
            gt=train_cache["gt"],
            state=train_cache["state"],
            dev_indices=dev_indices,
            hold_indices=hold_indices,
            config=config,
            device=device,
            out_dir=stage_dir,
        )

    bundle = save_final_bundle(
        a3_checkpoint=a3,
        fixed12_path=fixed12_path,
        endpoint_path=endpoint_path,
        a20_path=a20_path,
        config=config,
        out_dir=run_root,
    )

    fixed12_payload = torch.load(fixed12_path, map_location="cpu", weights_only=False)
    endpoint_payload = torch.load(endpoint_path, map_location="cpu", weights_only=False)
    a20_payload = torch.load(a20_path, map_location="cpu", weights_only=False)

    # Official VAL may be reported after freezing; it is not used to choose the late recipe.
    val_cache = cache_predictions(
        model_checkpoint=a3,
        config=config,
        loader=val_loader,
        device=device,
        label="FINAL-VAL",
    )
    val_final_pred = apply_final_postprocess(
        val_cache["pred"], val_cache["state"],
        fixed12_payload=fixed12_payload,
        endpoint_payload=endpoint_payload,
        a20_payload=a20_payload,
        device=device,
    )
    result: dict[str, object] = {
        "run_id": args.run_id,
        "variant": config["experiment"]["name"],
        "pipeline_bundle": str(bundle),
        "validation": exact_means(val_final_pred, val_cache["gt"]),
        "test_opened": False,
        "test_selection": False,
        "historical_reference_only": {"minADE6": 1.0413196086883545, "minFDE6": 2.013657808303833},
    }

    if args.open_test:
        # This is the first and only place the script constructs the TEST dataset.
        test_ds = WorkZoneDataset(roots=roots, split=config["data"].get("test_split", "test"), validate=False)
        test_loader = loader(
            test_ds,
            batch_size=int(config["evaluation"].get("batch_size", 8)),
            shuffle=False,
            local_seed=seed,
        )
        test_cache = cache_predictions(
            model_checkpoint=a3,
            config=config,
            loader=test_loader,
            device=device,
            label="FINAL-TEST",
        )
        test_final = apply_final_postprocess(
            test_cache["pred"], test_cache["state"],
            fixed12_payload=fixed12_payload,
            endpoint_payload=endpoint_payload,
            a20_payload=a20_payload,
            device=device,
        )
        result["test"] = exact_means(test_final, test_cache["gt"])
        result["test_opened"] = True

    write_json(run_root / "FINAL_PIPELINE_RESULT.json", result)
    print("\n" + json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
