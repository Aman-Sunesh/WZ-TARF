"""Train WZ-TARF from scratch or from an optional Phase-A checkpoint.

This is the public training entry point.  Experimental repair/sweep logic lives
outside the release path so the command stays readable and reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wztarf.data import WorkZoneDataset, collate_workzone_batch, collate_workzone_fixed
from wztarf.data.dataset import validate_edge_type_capacity
from wztarf.losses import LossWeights
from wztarf.models import WZTARF, WZTARFConfig
from wztarf.reporting import RunLogger
from wztarf.training import Trainer, load_pretrained_backbone
from wztarf.utils import load_yaml, make_generator, seed_all, seed_worker


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WZ-TARF.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "wz.yaml")
    parser.add_argument(
        "--data-roots",
        required=True,
        help="Comma-separated processed dataset roots, e.g. lap1,lap2.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=None,
        help="Optional Phase-A checkpoint. Omit this for supervised training from scratch.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume optimizer/model state from a supervised checkpoint.",
    )
    return parser.parse_args()


def resolve_device(requested: str | None) -> torch.device:
    device = torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def parse_roots(raw: str) -> list[Path]:
    roots = [Path(item.strip()).expanduser() for item in raw.split(",") if item.strip()]
    if not roots:
        raise ValueError("--data-roots must contain at least one path.")
    return roots


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_cfg: dict,
    epochs: int,
):
    cfg = training_cfg.get("scheduler")
    if not cfg or str(cfg.get("type", "none")).lower() == "none":
        return None
    kind = str(cfg.get("type")).lower()
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(cfg.get("eta_min", 0.0)),
        )
    raise ValueError(f"Unsupported scheduler type: {kind}")


def make_loader(
    dataset: WorkZoneDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    collate_fn,
    seed: int,
    prefetch_factor: int,
) -> DataLoader:
    kwargs = {}
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=prefetch_factor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=make_generator(seed) if shuffle else None,
        **kwargs,
    )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed = int(config["experiment"]["seed"])
    seed_all(seed)
    torch.set_float32_matmul_precision("high")

    device = resolve_device(args.device)
    roots = parse_roots(args.data_roots)
    data_cfg = config["data"]
    train_cfg = config["training"]

    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(data_cfg.get("num_workers", 0))
    )
    validate = bool(data_cfg.get("validate_samples", False))

    train_ds = WorkZoneDataset(
        roots=roots,
        split=data_cfg.get("train_split", "train"),
        validate=validate,
    )
    val_ds = WorkZoneDataset(
        roots=roots,
        split=data_cfg.get("val_split", "val"),
        validate=validate,
        include_source_path=True,
    )

    edge_samples = int(data_cfg.get("edge_validation_samples", 0))
    if edge_samples > 0:
        validate_edge_type_capacity(
            train_ds,
            num_edge_types=int(config["model"]["num_edge_types"]),
            max_samples=edge_samples,
        )

    collate_fn = (
        collate_workzone_fixed
        if bool(data_cfg.get("fixed_collate", True))
        else collate_workzone_batch
    )
    pin_memory = bool(data_cfg.get("pin_memory", True)) and device.type == "cuda"
    batch_size = int(train_cfg["batch_size"])

    train_loader = make_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        seed=seed,
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
    )
    val_loader = make_loader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        seed=seed,
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
    )

    model = WZTARF(WZTARFConfig(**config["model"]))
    if args.pretrained is not None:
        result = load_pretrained_backbone(
            args.pretrained,
            model=model,
            map_location="cpu",
            strict=False,
        )
        print(
            "[pretrained] missing=",
            result.missing_keys,
            "unexpected=",
            result.unexpected_keys,
            flush=True,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    epochs = int(train_cfg["epochs"])
    scheduler = build_scheduler(optimizer, train_cfg, epochs)

    run_id = args.run_id or str(config["experiment"]["name"])
    logger = RunLogger(project_root=PROJECT_ROOT, run_id=run_id)
    logger.save_config(config)
    logger.save_environment()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_weights=LossWeights(**config["loss_weights"]),
        device=device,
        checkpoint_dir=PROJECT_ROOT / "checkpoints" / run_id,
        logger=logger,
        config=config,
        beta_assign=float(train_cfg.get("beta_assign", 0.25)),
        classification_temperature=float(train_cfg.get("classification_temperature", 1.0)),
        fps=int(data_cfg["fps"]),
        goal_association_tolerance_m=float(train_cfg.get("goal_association_tolerance_m", 0.25)),
        road_gt_tolerance_m=float(train_cfg.get("road_gt_tolerance_m", 0.25)),
        grad_clip_norm=train_cfg.get("grad_clip_norm", 5.0),
        use_amp=bool(train_cfg.get("use_amp", True)),
        composite_fde_weight=float(train_cfg.get("composite_fde_weight", 0.25)),
    )

    print(
        f"[train] variant={config['experiment']['name']} train={len(train_ds)} "
        f"val={len(val_ds)} device={device} workers={num_workers}",
        flush=True,
    )
    summary = trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        selection_metric=str(train_cfg.get("selection_metric", "J_val")),
        selection_mode=str(train_cfg.get("selection_mode", "min")),
        patience=train_cfg.get("patience"),
        resume_from=args.resume,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
