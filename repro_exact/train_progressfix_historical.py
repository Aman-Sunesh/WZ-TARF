"""Train with Phase B supervised fine-tuning."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wztarf.data import (
    WorkZoneDataset,
    collate_workzone_batch,
    collate_workzone_fixed,
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
        "--initialize",
        type=Path,
        default=None,
        help=(
            "Full supervised checkpoint used only to initialize model "
            "weights; optimizer/scheduler state is NOT restored."
        ),
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

    # Enable TensorFloat-32/high-performance matmul kernels where available.
    # This does not force lower-precision storage; AMP remains controlled by
    # the training configuration below.
    torch.set_float32_matmul_precision("high")

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
        include_source_path=True,
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

    collate_fn = (
        collate_workzone_fixed
        if bool(data_config.get("fixed_collate", True))
        else collate_workzone_batch
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
        collate_fn=collate_fn,
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
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        **loader_worker_kwargs,
    )

    model = WZTARF(
        WZTARFConfig(
            **config["model"]
        )
    )

    selected_initializers = sum(
        item is not None
        for item in (
            args.resume,
            args.pretrained,
            args.initialize,
        )
    )

    if selected_initializers > 1:
        raise ValueError(
            "--resume, --pretrained, and --initialize are mutually exclusive."
        )

    # ==========================================================
    # DENSE_PROGRESS_FULL_INITIALIZE
    #
    # Load a complete supervised model checkpoint without restoring
    # optimizer/scheduler/epoch state. This lets the dense progress repair
    # start from the strongest existing V3 Phase-B model.
    # ==========================================================

    if args.initialize is not None:

        raw_initialize = torch.load(
            args.initialize,
            map_location="cpu",
        )

        initialize_state = raw_initialize

        if isinstance(
            raw_initialize,
            dict,
        ):
            for checkpoint_key in (
                "model_state_dict",
                "model",
                "state_dict",
            ):
                candidate = raw_initialize.get(
                    checkpoint_key
                )

                if (
                    isinstance(candidate, dict)
                    and candidate
                    and all(
                        torch.is_tensor(value)
                        for value in candidate.values()
                    )
                ):
                    initialize_state = candidate
                    break

        if not isinstance(
            initialize_state,
            dict,
        ):
            raise RuntimeError(
                "Could not extract model state from --initialize checkpoint."
            )

        for prefix in (
            "model.",
            "module.",
            "net.",
        ):
            if (
                initialize_state
                and
                all(
                    key.startswith(prefix)
                    for key in initialize_state
                )
            ):
                initialize_state = {
                    key[
                        len(prefix):
                    ]: value
                    for key, value
                    in initialize_state.items()
                }

        initialize_result = model.load_state_dict(
            initialize_state,
            strict=False,
        )

        allowed_missing_prefixes = (
            "route_progress.hard_route_geometry_encoder.",
            "route_progress.dense_progress_fusion.",
            "route_progress.dense_progress_residual_head.",
            "direct_trajectory_decoder.",
        )

        bad_missing = [
            key
            for key in initialize_result.missing_keys
            if not key.startswith(
                allowed_missing_prefixes
            )
        ]

        if bad_missing:
            raise RuntimeError(
                "Unexpected missing keys during --initialize: "
                +
                repr(
                    bad_missing
                )
            )

        if initialize_result.unexpected_keys:
            raise RuntimeError(
                "Unexpected checkpoint keys during --initialize: "
                +
                repr(
                    initialize_result.unexpected_keys
                )
            )

        print(
            "[initialize] loaded full supervised checkpoint:",
            args.initialize,
        )

        print(
            "[initialize] new dense-progress parameters initialized fresh:",
            len(
                initialize_result.missing_keys
            ),
        )

    if args.pretrained is not None:
        load_pretrained_backbone(
            args.pretrained,
            model=model,
            map_location="cpu",
            # Dense-progress V3 adds new Phase-B-only parameters.
            # Existing pretrained keys must still match by name/shape.
            strict=False,
        )

    # ==========================================================
    # DENSE_PROGRESS_HEAD_ONLY
    # ==========================================================

    if bool(
        training_config.get(
            "train_dense_progress_only",
            False,
        )
    ):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        dense_modules = (
            model.route_progress.hard_route_geometry_encoder,
            model.route_progress.dense_progress_fusion,
            model.route_progress.dense_progress_residual_head,
        )

        for module in dense_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)

        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        frozen_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        )

        print(
            "[dense-progress-head-only] trainable parameters:",
            trainable_parameters,
        )

        print(
            "[dense-progress-head-only] frozen parameters:",
            frozen_parameters,
        )

    # ==========================================================
    # DIRECT_ANCHOR_CALIBRATION_ONLY
    #
    # Freeze the complete existing predictor and train only the new
    # zero-initialized absolute anchor correction head.
    #
    # This makes the experiment a strict repair of an existing Direct-K6
    # checkpoint rather than a destructive full-model retraining.
    # ==========================================================

    if bool(
        training_config.get(
            "train_direct_calibration_only",
            False,
        )
    ):
        if bool(
            training_config.get(
                "train_dense_progress_only",
                False,
            )
        ):
            raise ValueError(
                "train_direct_calibration_only and "
                "train_dense_progress_only cannot both be enabled."
            )

        for parameter in model.parameters():
            parameter.requires_grad_(
                False
            )

        direct_decoder = (
            model.direct_trajectory_decoder
        )

        if direct_decoder is None:
            raise RuntimeError(
                "Direct calibration training requires "
                "use_direct_decoder=True."
            )

        calibration_head = getattr(
            direct_decoder,
            "anchor_correction_head",
            None,
        )

        if calibration_head is None:
            raise RuntimeError(
                "Direct calibration training requires "
                "model.use_direct_anchor_calibration=True."
            )

        for parameter in calibration_head.parameters():
            parameter.requires_grad_(
                True
            )

        # ==========================================================
        # DETERMINISTIC FROZEN BACKBONE
        #
        # requires_grad=False freezes parameters, but model.train()
        # still activates Dropout / MultiheadAttention dropout.
        #
        # In calibration-only training that makes the supposedly
        # frozen predictor stochastic underneath the small correction
        # head.  Permanently set dropout probability to zero for this
        # run.  model.train() may still toggle module.training, but
        # with p=0 the frozen backbone remains deterministic.
        #
        # Training-only whole-modality dropout is controlled separately
        # through the runtime config and must also be zero.
        # ==========================================================

        deterministic_dropout_modules = 0
        deterministic_mha_modules = 0

        for module in model.modules():
            if isinstance(
                module,
                torch.nn.Dropout,
            ):
                module.p = 0.0
                deterministic_dropout_modules += 1

            if isinstance(
                module,
                torch.nn.MultiheadAttention,
            ):
                module.dropout = 0.0
                deterministic_mha_modules += 1

        print(
            "[direct-anchor-calibration-only] "
            "deterministic frozen backbone enabled: "
            f"Dropout modules={deterministic_dropout_modules}, "
            f"MultiheadAttention modules={deterministic_mha_modules}",
            flush=True,
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        frozen_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        )

        print(
            "[direct-anchor-calibration-only] "
            "trainable parameters:",
            trainable_parameters,
            flush=True,
        )

        print(
            "[direct-anchor-calibration-only] "
            "frozen parameters:",
            frozen_parameters,
            flush=True,
        )

    # ==========================================================
    # DIRECT_LONGITUDINAL_REPAIR_ONLY
    #
    # Freeze the complete existing predictor and train only:
    #   1. temporal longitudinal GRU
    #   2. longitudinal dx output head
    #   3. existing absolute anchor calibration head
    #
    # The Direct base decoder, scene encoders, topology, route system,
    # dynamics prior, and lateral trajectory production remain frozen.
    # ==========================================================

    if bool(
        training_config.get(
            "train_direct_longitudinal_only",
            False,
        )
    ):

        if bool(
            training_config.get(
                "train_dense_progress_only",
                False,
            )
        ):
            raise ValueError(
                "train_direct_longitudinal_only and "
                "train_dense_progress_only cannot both be enabled."
            )

        if bool(
            training_config.get(
                "train_direct_calibration_only",
                False,
            )
        ):
            raise ValueError(
                "train_direct_longitudinal_only and "
                "train_direct_calibration_only cannot both be enabled."
            )

        for modality_dropout_key in (
            "aux_dropout_controls",
            "aux_dropout_gaze",
            "aux_dropout_workers",
        ):
            if float(
                config["model"].get(
                    modality_dropout_key,
                    0.0,
                )
            ) != 0.0:
                raise ValueError(
                    "Direct longitudinal repair requires "
                    f"{modality_dropout_key}=0.0 for a deterministic "
                    "frozen backbone."
                )

        for parameter in model.parameters():
            parameter.requires_grad_(
                False
            )

        direct_decoder = (
            model.direct_trajectory_decoder
        )

        if direct_decoder is None:
            raise RuntimeError(
                "Direct longitudinal repair requires "
                "use_direct_decoder=True."
            )

        repair_gru = getattr(
            direct_decoder,
            "longitudinal_repair_gru",
            None,
        )

        repair_head = getattr(
            direct_decoder,
            "longitudinal_repair_head",
            None,
        )

        if (
            repair_gru is None
            or
            repair_head is None
        ):
            raise RuntimeError(
                "Direct longitudinal repair requires "
                "model.use_direct_longitudinal_repair=True."
            )

        for parameter in repair_gru.parameters():
            parameter.requires_grad_(
                True
            )

        for parameter in repair_head.parameters():
            parameter.requires_grad_(
                True
            )

        # Allow the existing 1/3/5 s absolute calibrator to co-adapt
        # as the underlying longitudinal progression changes.
        calibration_head = getattr(
            direct_decoder,
            "anchor_correction_head",
            None,
        )

        if calibration_head is not None:
            for parameter in calibration_head.parameters():
                parameter.requires_grad_(
                    True
                )

        # Frozen means functionally frozen: remove dropout noise.
        deterministic_dropout_modules = 0
        deterministic_mha_modules = 0

        for module in model.modules():

            if isinstance(
                module,
                torch.nn.Dropout,
            ):
                module.p = 0.0
                deterministic_dropout_modules += 1

            if isinstance(
                module,
                torch.nn.MultiheadAttention,
            ):
                module.dropout = 0.0
                deterministic_mha_modules += 1

        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        frozen_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        )

        print(
            "[direct-longitudinal-repair-only] "
            "deterministic frozen backbone enabled: "
            f"Dropout modules={deterministic_dropout_modules}, "
            f"MultiheadAttention modules={deterministic_mha_modules}",
            flush=True,
        )

        print(
            "[direct-longitudinal-repair-only] "
            "trainable parameters:",
            trainable_parameters,
            flush=True,
        )

        print(
            "[direct-longitudinal-repair-only] "
            "frozen parameters:",
            frozen_parameters,
            flush=True,
        )

    # ==========================================================
    # DIRECT_GENERATOR_REPAIR_ONLY
    #
    # Frozen:
    #   scene encoders
    #   map/topology stack
    #   route stack
    #   dynamics anchor
    #
    # Trainable:
    #   COMPLETE DirectTrajectoryDecoder
    #
    # This includes:
    #   Direct cross-attention
    #   Direct FFNs
    #   mode embeddings
    #   delta_head
    #   temporal longitudinal repair
    #   anchor calibration
    #   mode scoring
    #
    # We are intentionally allowing the actual Cartesian trajectory
    # representation to reorganize rather than repairing a frozen one.
    # ==========================================================

    if bool(
        training_config.get(
            "train_direct_generator_repair",
            False,
        )
    ):

        conflicting_modes = (
            "train_dense_progress_only",
            "train_direct_calibration_only",
            "train_direct_longitudinal_only",
        )

        enabled_conflicts = [
            key
            for key in conflicting_modes
            if bool(
                training_config.get(
                    key,
                    False,
                )
            )
        ]

        if enabled_conflicts:
            raise ValueError(
                "train_direct_generator_repair conflicts with: "
                +
                repr(
                    enabled_conflicts
                )
            )

        for modality_dropout_key in (
            "aux_dropout_controls",
            "aux_dropout_gaze",
            "aux_dropout_workers",
        ):
            if float(
                config["model"].get(
                    modality_dropout_key,
                    0.0,
                )
            ) != 0.0:
                raise ValueError(
                    "Direct generator repair requires "
                    f"{modality_dropout_key}=0.0."
                )

        # Freeze the complete model first.
        for parameter in model.parameters():
            parameter.requires_grad_(
                False
            )

        direct_decoder = (
            model.direct_trajectory_decoder
        )

        if direct_decoder is None:
            raise RuntimeError(
                "Direct generator repair requires "
                "use_direct_decoder=True."
            )

        # ----------------------------------------------------------
        # KEY CHANGE:
        # Train the complete Direct decoder, not another tiny head.
        # ----------------------------------------------------------
        for parameter in direct_decoder.parameters():
            parameter.requires_grad_(
                True
            )

        # Disable stochastic training noise. We are fine-tuning from a
        # strong checkpoint and want clean gradients.
        deterministic_dropout_modules = 0
        deterministic_mha_modules = 0

        for module in model.modules():

            if isinstance(
                module,
                torch.nn.Dropout,
            ):
                module.p = 0.0
                deterministic_dropout_modules += 1

            if isinstance(
                module,
                torch.nn.MultiheadAttention,
            ):
                module.dropout = 0.0
                deterministic_mha_modules += 1

        trainable_parameters = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        frozen_parameters = sum(
            p.numel()
            for p in model.parameters()
            if not p.requires_grad
        )

        print(
            "[direct-generator-repair] "
            "deterministic training enabled: "
            f"Dropout modules={deterministic_dropout_modules}, "
            f"MultiheadAttention modules={deterministic_mha_modules}",
            flush=True,
        )

        print(
            "[direct-generator-repair] "
            "trainable parameters:",
            trainable_parameters,
            flush=True,
        )

        print(
            "[direct-generator-repair] "
            "frozen parameters:",
            frozen_parameters,
            flush=True,
        )

        print(
            "[direct-generator-repair] "
            "trainable Direct modules:",
            flush=True,
        )

        for name, parameter in direct_decoder.named_parameters():
            if parameter.requires_grad:
                print(
                    "    ",
                    name,
                    tuple(parameter.shape),
                    flush=True,
                )

    # ==========================================================
    # DIRECT_SURGICAL_REPAIR_ONLY
    #
    # Preserve almost the entire pretrained WZ-TARF representation.
    #
    # Train ONLY:
    #   - second / final Direct cross-attention block
    #   - second / final Direct FFN block
    #   - delta_head
    #   - temporal longitudinal repair GRU/head
    #   - anchor correction head
    #
    # Differential LR:
    #   pretrained final block : 1e-5
    #   pretrained delta head  : 3e-5
    #   fresh temporal branch  : 2e-4
    #   calibrated anchor head : 5e-5
    # ==========================================================

    if bool(
        training_config.get(
            "train_direct_surgical_repair",
            False,
        )
    ):

        conflicting_modes = (
            "train_dense_progress_only",
            "train_direct_calibration_only",
            "train_direct_longitudinal_only",
            "train_direct_generator_repair",
        )

        enabled_conflicts = [
            key
            for key in conflicting_modes
            if bool(
                training_config.get(
                    key,
                    False,
                )
            )
        ]

        if enabled_conflicts:
            raise ValueError(
                "train_direct_surgical_repair conflicts with: "
                + repr(enabled_conflicts)
            )

        for key in (
            "aux_dropout_controls",
            "aux_dropout_gaze",
            "aux_dropout_workers",
        ):
            if float(
                config["model"].get(
                    key,
                    0.0,
                )
            ) != 0.0:
                raise ValueError(
                    "Surgical Direct repair requires "
                    f"{key}=0.0."
                )

        # ----------------------------------------------------------
        # Freeze EVERYTHING first.
        # ----------------------------------------------------------

        for parameter in model.parameters():
            parameter.requires_grad_(False)

        dd = model.direct_trajectory_decoder

        if dd is None:
            raise RuntimeError(
                "Surgical repair requires use_direct_decoder=True."
            )

        if len(dd.cross_attention) < 2:
            raise RuntimeError(
                "Expected at least two Direct attention blocks."
            )

        if len(dd.ffn) < 2:
            raise RuntimeError(
                "Expected at least two Direct FFN blocks."
            )

        if dd.longitudinal_repair_gru is None:
            raise RuntimeError(
                "Temporal longitudinal GRU is missing."
            )

        if dd.longitudinal_repair_head is None:
            raise RuntimeError(
                "Temporal longitudinal head is missing."
            )

        if dd.anchor_correction_head is None:
            raise RuntimeError(
                "Anchor correction head is missing."
            )


        # ----------------------------------------------------------
        # Group 1:
        # LAST PRETRAINED REPRESENTATION BLOCK ONLY
        # ----------------------------------------------------------

        last_block_modules = (
            dd.cross_attention[1],
            dd.cross_norm[1],
            dd.ffn[1],
            dd.ffn_norm[1],
        )

        last_block_parameters = [
            parameter
            for module in last_block_modules
            for parameter in module.parameters()
        ]

        for parameter in last_block_parameters:
            parameter.requires_grad_(True)


        # ----------------------------------------------------------
        # Group 2:
        # ACTUAL CARTESIAN DELTA PRODUCER
        # ----------------------------------------------------------

        delta_parameters = list(
            dd.delta_head.parameters()
        )

        for parameter in delta_parameters:
            parameter.requires_grad_(True)


        # ----------------------------------------------------------
        # Group 3:
        # FRESH TEMPORAL LONGITUDINAL BRANCH
        # ----------------------------------------------------------

        temporal_parameters = (
            list(
                dd.longitudinal_repair_gru.parameters()
            )
            +
            list(
                dd.longitudinal_repair_head.parameters()
            )
        )

        for parameter in temporal_parameters:
            parameter.requires_grad_(True)


        # ----------------------------------------------------------
        # Group 4:
        # EXISTING ABSOLUTE 1/3/5 S CALIBRATOR
        # ----------------------------------------------------------

        calibration_parameters = list(
            dd.anchor_correction_head.parameters()
        )

        for parameter in calibration_parameters:
            parameter.requires_grad_(True)


        # ----------------------------------------------------------
        # Ensure every optimizer parameter appears exactly once.
        # ----------------------------------------------------------

        all_group_parameters = (
            last_block_parameters
            +
            delta_parameters
            +
            temporal_parameters
            +
            calibration_parameters
        )

        parameter_ids = [
            id(parameter)
            for parameter in all_group_parameters
        ]

        if len(parameter_ids) != len(set(parameter_ids)):
            raise RuntimeError(
                "Duplicate parameters detected across surgical LR groups."
            )


        # ----------------------------------------------------------
        # Differential learning rates.
        # ----------------------------------------------------------

        lr_last = float(
            training_config.get(
                "direct_last_block_lr",
                1.0e-5,
            )
        )

        lr_delta = float(
            training_config.get(
                "direct_delta_lr",
                3.0e-5,
            )
        )

        lr_temporal = float(
            training_config.get(
                "direct_temporal_lr",
                2.0e-4,
            )
        )

        lr_calibration = float(
            training_config.get(
                "direct_calibration_lr",
                5.0e-5,
            )
        )

        optimizer_parameter_groups = [
            {
                "params": last_block_parameters,
                "lr": lr_last,
            },
            {
                "params": delta_parameters,
                "lr": lr_delta,
            },
            {
                "params": temporal_parameters,
                "lr": lr_temporal,
            },
            {
                "params": calibration_parameters,
                "lr": lr_calibration,
            },
        ]


        # ----------------------------------------------------------
        # Functionally deterministic pretrained backbone.
        # ----------------------------------------------------------

        deterministic_dropout_modules = 0
        deterministic_mha_modules = 0

        for module in model.modules():

            if isinstance(
                module,
                torch.nn.Dropout,
            ):
                module.p = 0.0
                deterministic_dropout_modules += 1

            if isinstance(
                module,
                torch.nn.MultiheadAttention,
            ):
                module.dropout = 0.0
                deterministic_mha_modules += 1


        grouped_count = sum(
            parameter.numel()
            for parameter in all_group_parameters
        )

        trainable_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        frozen_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        )

        if grouped_count != trainable_count:
            raise RuntimeError(
                "Optimizer/trainable parameter mismatch: "
                f"grouped={grouped_count}, "
                f"trainable={trainable_count}"
            )


        print(
            "[direct-surgical-repair] deterministic training: "
            f"Dropout={deterministic_dropout_modules}, "
            f"MHA={deterministic_mha_modules}",
            flush=True,
        )

        print(
            "[direct-surgical-repair] last block params:",
            sum(
                p.numel()
                for p in last_block_parameters
            ),
            "lr=",
            lr_last,
            flush=True,
        )

        print(
            "[direct-surgical-repair] delta head params:",
            sum(
                p.numel()
                for p in delta_parameters
            ),
            "lr=",
            lr_delta,
            flush=True,
        )

        print(
            "[direct-surgical-repair] temporal params:",
            sum(
                p.numel()
                for p in temporal_parameters
            ),
            "lr=",
            lr_temporal,
            flush=True,
        )

        print(
            "[direct-surgical-repair] calibration params:",
            sum(
                p.numel()
                for p in calibration_parameters
            ),
            "lr=",
            lr_calibration,
            flush=True,
        )

        print(
            "[direct-surgical-repair] TOTAL TRAINABLE:",
            trainable_count,
            flush=True,
        )

        print(
            "[direct-surgical-repair] TOTAL FROZEN:",
            frozen_count,
            flush=True,
        )

    optimizer = torch.optim.AdamW(
        (
            optimizer_parameter_groups
            if bool(
                training_config.get(
                    "train_direct_surgical_repair",
                    False,
                )
            )
            else model.parameters()
        ),
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
        composite_fde_weight=float(
            training_config.get(
                "composite_fde_weight",
                0.25,
            )
        ),
    )


    if os.environ.get("WZTARF_GRAD_CALIBRATE", "0") == "1":
        calibration_batches = int(
            os.environ.get(
                "WZTARF_GRAD_CALIBRATE_BATCHES",
                "16",
            )
        )

        print(
            "[GRAD-CAL] Running read-only gradient calibration. "
            "NO optimizer steps will be performed.",
            flush=True,
        )

        calibration = trainer.calibrate_loss_gradients(
            train_loader,
            max_batches=calibration_batches,
            target_endpoint_gradient_ratio=0.5,
        )

        print(
            "=== WZTARF GRADIENT CALIBRATION RESULT ===",
            flush=True,
        )

        print(
            json.dumps(
                calibration,
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )

        return

    summary = trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        selection_metric=str(
            training_config.get(
                "selection_metric",
                "J_val",
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
