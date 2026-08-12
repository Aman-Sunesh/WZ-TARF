"""Train with Phase B supervised fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wztarf.data import (
    WorkZoneDataset,
    collate_workzone_batch,
)
from wztarf.data.dataset import validate_edge_type_capacity
from wztarf.losses import LossWeights
from wztarf.models import (
    WZTARF,
    WZTARFConfig,
)
from wztarf.reporting import RunLogger
from wztarf.training import (
    Trainer,
    load_pretrained_backbone,
)
from wztarf.utils import (
    load_yaml,
    make_generator,
    seed_all,
    seed_worker,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    """Parse command-line training arguments."""
    parser = argparse.ArgumentParser(
        description="Train WZ-TARF."
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "base.yaml",
    )

    parser.add_argument(
        "--data-roots",
        type=str,
        required=True,
        help="Comma-separated processed dataset roots.",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--pretrained",
        type=Path,
        default=None,
        help="Phase A checkpoint used to initialize the forecasting backbone.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
    )

    return parser.parse_args()


def _resolve_device(
    requested: str | None,
) -> torch.device:
    """Choose CUDA when available unless a device was explicitly supplied."""
    if requested is not None:
        device = torch.device(
            requested
        )
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but CUDA is unavailable."
        )

    return device

def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
    epochs: int,
):
    """Construct the configured epoch-level learning-rate scheduler."""
    scheduler_config = config.get(
        "scheduler"
    )

    if not scheduler_config:
        return None

    scheduler_type = str(
        scheduler_config.get(
            "type",
            "none",
        )
    ).lower()

    if scheduler_type == "none":
        return None

    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(
                scheduler_config.get(
                    "eta_min",
                    0.0,
                )
            ),
        )

    raise ValueError(
        f"Unsupported scheduler type: {scheduler_type}"
    )


def main() -> None:
    """Construct datasets, model, optimizer, and run supervised training."""
    args = _parse_args()

    config = load_yaml(
        args.config
    )

    seed = int(
        config["experiment"]["seed"]
    )

    seed_all(
        seed
    )

    device = _resolve_device(
        args.device
    )

    roots = [
        Path(item.strip())
        for item in args.data_roots.split(",")
        if item.strip()
    ]

    if not roots:
        raise ValueError(
            "--data-roots must contain at least one dataset root."
        )

    data_config = config["data"]
    training_config = config["training"]

    validate_samples = bool(
        data_config.get(
            "validate_samples",
            False,
        )
    )

    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(
            data_config.get(
                "num_workers",
                0,
            )
        )
    )

    train_dataset = WorkZoneDataset(
        roots=roots,
        split=data_config.get(
            "train_split",
            "train",
        ),
        validate=validate_samples,
    )

    val_dataset = WorkZoneDataset(
        roots=roots,
        split=data_config.get(
            "val_split",
            "val",
        ),
        validate=validate_samples,
    )

    edge_validation_samples = int(
        data_config.get(
            "edge_validation_samples",
            64,
        )
    )

    if edge_validation_samples > 0:
        validate_edge_type_capacity(
            train_dataset,
            num_edge_types=int(
                config[
                    "model"
                ][
                    "num_edge_types"
                ]
            ),
            max_samples=edge_validation_samples,
        )

    pin_memory = (
        bool(
            data_config.get(
                "pin_memory",
                True,
            )
        )
        and
        device.type == "cuda"
    )

    generator = make_generator(
        seed
    )

    loader_worker_kwargs = {}

    if num_workers > 0:
        loader_worker_kwargs = {
            "persistent_workers": True,
            "prefetch_factor": int(
                data_config.get(
                    "prefetch_factor",
                    2,
                )
            ),
        }

    print(
        f"[data] train={len(train_dataset)} "
        f"val={len(val_dataset)} "
        f"workers={num_workers} "
        f"validate_samples={validate_samples}",
        flush=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(
            training_config[
                "batch_size"
            ]
        ),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_workzone_batch,
        worker_init_fn=seed_worker,
        generator=generator,
        **loader_worker_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(
            training_config[
                "batch_size"
            ]
        ),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_workzone_batch,
        worker_init_fn=seed_worker,
        **loader_worker_kwargs,
    )

    model = WZTARF(
        WZTARFConfig(
            **config["model"]
        )
    )

    if (
        args.resume is not None
        and args.pretrained is not None
    ):
        raise ValueError(
            "--resume and --pretrained cannot be used together."
        )

    if args.pretrained is not None:
        load_pretrained_backbone(
            args.pretrained,
            model=model,
            map_location="cpu",
            strict=True,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            training_config[
                "learning_rate"
            ]
        ),
        weight_decay=float(
            training_config[
                "weight_decay"
            ]
        ),
    )

    epochs = int(
        training_config[
            "epochs"
        ]
    )

    scheduler = _build_scheduler(
        optimizer,
        training_config,
        epochs,
    )

    run_id = (
        args.run_id
        if args.run_id is not None
        else str(
            config["experiment"][
                "name"
            ]
        )
    )

    logger = RunLogger(
        project_root=PROJECT_ROOT,
        run_id=run_id,
    )

    logger.save_config(
        config
    )

    logger.save_environment()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_weights=LossWeights(
            **config["loss_weights"]
        ),
        device=device,
        checkpoint_dir=(
            PROJECT_ROOT
            /
            "checkpoints"
            /
            run_id
        ),
        logger=logger,
        config=config,
        beta_assign=float(
            training_config[
                "beta_assign"
            ]
        ),
        classification_temperature=float(
            training_config[
                "classification_temperature"
            ]
        ),
        fps=int(
            data_config[
                "fps"
            ]
        ),
        goal_association_tolerance_m=float(
            training_config.get(
                "goal_association_tolerance_m",
                0.25,
            )
        ),
        road_gt_tolerance_m=float(
            training_config.get(
                "road_gt_tolerance_m",
                0.25,
            )
        ),
        grad_clip_norm=training_config.get(
            "grad_clip_norm"
        ),
        use_amp=bool(
            training_config.get(
                "use_amp",
                True,
            )
        ),
    )

    summary = trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        selection_metric=str(
            training_config.get(
                "selection_metric",
                "minADE_6",
            )
        ),
        selection_mode=str(
            training_config.get(
                "selection_mode",
                "min",
            )
        ),
        patience=training_config.get(
            "patience"
        ),
        resume_from=args.resume,
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
