"""Run Phase A WorkZone-aware self-supervised pretraining."""

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
from wztarf.models import WZTARFConfig
from wztarf.models.pretraining import WZTARFPretrainingModel
from wztarf.pretraining import MaskingConfig
from wztarf.reporting import RunLogger
from wztarf.training import (
    Pretrainer,
    PretrainingWeights,
)
from wztarf.utils import (
    load_yaml,
    make_generator,
    seed_all,
    seed_worker,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse Phase A arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "base.yaml",
    )

    parser.add_argument(
        "--data-roots",
        required=True,
    )

    parser.add_argument(
        "--run-id",
        default=None,
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    """Run complete Phase A pretraining."""
    args = parse_args()

    config = load_yaml(
        args.config
    )

    seed = int(
        config[
            "experiment"
        ][
            "seed"
        ]
    )

    seed_all(
        seed
    )

    device = torch.device(
        args.device
        if args.device is not None
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    roots = [
        Path(
            value.strip()
        )
        for value in args.data_roots.split(
            ","
        )
        if value.strip()
    ]

    data_config = config[
        "data"
    ]

    pretrain_config = config[
        "pretraining"
    ]

    train_dataset = WorkZoneDataset(
        roots=roots,
        split=data_config.get(
            "train_split",
            "train",
        ),
        validate=True,
        include_source_path=True,
    )

    val_dataset = WorkZoneDataset(
        roots=roots,
        split=data_config.get(
            "val_split",
            "val",
        ),
        validate=True,
        include_source_path=True,
    )

    num_workers = int(
        data_config.get(
            "num_workers",
            0,
        )
    )

    generator = make_generator(
        seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(
            pretrain_config[
                "batch_size"
            ]
        ),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_workzone_batch,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=(
            num_workers > 0
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(
            pretrain_config[
                "batch_size"
            ]
        ),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_workzone_batch,
        worker_init_fn=seed_worker,
        persistent_workers=(
            num_workers > 0
        ),
    )

    model = WZTARFPretrainingModel(
        WZTARFConfig(
            **config[
                "model"
            ]
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            pretrain_config[
                "learning_rate"
            ]
        ),
        weight_decay=float(
            pretrain_config[
                "weight_decay"
            ]
        ),
    )

    run_id = (
        args.run_id
        or
        (
            str(
                config[
                    "experiment"
                ][
                    "name"
                ]
            )
            +
            "_pretrain"
        )
    )

    logger = RunLogger(
        PROJECT_ROOT,
        run_id,
    )

    logger.save_config(
        config
    )

    logger.save_environment()

    pretrainer = Pretrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=(
            PROJECT_ROOT
            /
            "checkpoints"
            /
            run_id
        ),
        weights=PretrainingWeights(
            **pretrain_config[
                "weights"
            ]
        ),
        masking_config=MaskingConfig(
            **pretrain_config[
                "masking"
            ]
        ),
        logger=logger,
        config=config,
        grad_clip_norm=pretrain_config.get(
            "grad_clip_norm",
            5.0,
        ),
        use_amp=bool(
            pretrain_config.get(
                "use_amp",
                True,
            )
        ),
        fac_temperature=float(
            pretrain_config.get(
                "fac_temperature",
                0.1,
            )
        ),
        fac_exclusion_seconds=float(
            pretrain_config.get(
                "fac_exclusion_seconds",
                5.0,
            )
        ),
        fac_symmetric=bool(
            pretrain_config.get(
                "fac_symmetric",
                False,
            )
        ),
        fps=int(
            data_config[
                "fps"
            ]
        ),
        mask_seed=seed,
    )

    summary = pretrainer.fit(
        train_loader,
        val_loader,
        epochs=int(
            pretrain_config[
                "epochs"
            ]
        ),
        patience=pretrain_config.get(
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
