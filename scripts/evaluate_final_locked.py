"""Fresh dataset evaluation of the frozen WZ-TARF final artifact chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wztarf.data.dataset import WorkZoneDataset
from wztarf.models import (
    WZTARF,
    WZTARFConfig,
)
from wztarf.pipeline.common import (
    build_causal_state,
    exact_means,
    make_loader,
    move_to_device,
    parse_roots,
    resolve_device,
)
from wztarf.pipeline.final_locked import (
    REFERENCE_TEST_ADE,
    REFERENCE_TEST_FDE,
    apply_locked_postprocess,
    default_artifact_root,
    load_locked_artifacts,
)
from wztarf.utils import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def build_locked_a3_model(
    *,
    payload,
    config,
    device,
):
    """Load the preserved historical A3 checkpoint into current release code."""

    model_cfg = dict(
        config["model"]
    )

    model_cfg.update(
        use_direct_decoder=True,
        use_direct_anchor_calibration=True,
        use_direct_longitudinal_repair=False,
        aux_dropout_controls=0.0,
        aux_dropout_gaze=0.0,
        aux_dropout_workers=0.0,
    )

    model = WZTARF(
        WZTARFConfig(
            **model_cfg
        )
    )

    if (
        not isinstance(payload, dict)
        or
        "model_state_dict"
        not in payload
    ):
        raise RuntimeError(
            "Historical A3 artifact is missing "
            "model_state_dict"
        )

    result = model.load_state_dict(
        payload[
            "model_state_dict"
        ],
        strict=False,
    )

    if result.unexpected_keys:
        raise RuntimeError(
            "Unexpected historical A3 tensors: "
            f"{result.unexpected_keys}"
        )

    # Historical exact checkpoint stored the native direct
    # decoder separately.
    if (
        "native_decoder_state_dict"
        in payload
    ):
        if (
            model.direct_trajectory_decoder
            is None
        ):
            raise RuntimeError(
                "Current model did not construct "
                "direct_trajectory_decoder"
            )

        model.direct_trajectory_decoder.load_state_dict(
            payload[
                "native_decoder_state_dict"
            ],
            strict=True,
        )

    bad_missing = [
        key
        for key in result.missing_keys
        if not key.startswith(
            "direct_trajectory_decoder."
        )
    ]

    if bad_missing:
        raise RuntimeError(
            "Unexpected missing tensors while "
            "loading historical A3: "
            f"{bad_missing}"
        )

    model.to(device)
    model.eval()

    return model


def main() -> None:

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data-roots",
        required=True,
    )

    ap.add_argument(
        "--split",
        choices=(
            "train",
            "val",
            "test",
        ),
        default="test",
    )

    ap.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "configs"
            / "wz.yaml"
        ),
    )

    ap.add_argument(
        "--device",
        default=None,
    )

    ap.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = ap.parse_args()

    artifact_root = (
        args.artifact_root
        if args.artifact_root is not None
        else default_artifact_root()
    )

    config = load_yaml(
        args.config
    )

    device = resolve_device(
        args.device
    )

    roots = parse_roots(
        args.data_roots
    )

    artifacts = load_locked_artifacts(
        artifact_root
    )

    model = build_locked_a3_model(
        payload=artifacts["a3"],
        config=config,
        device=device,
    )

    dataset = WorkZoneDataset(
        roots=roots,
        split=args.split,
        validate=False,
    )

    loader = make_loader(
        dataset,
        batch_size=int(
            config[
                "evaluation"
            ].get(
                "batch_size",
                8,
            )
        ),
        shuffle=False,
        seed=int(
            config[
                "experiment"
            ].get(
                "seed",
                2023,
            )
        ),
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        fixed_collate=bool(
            config[
                "data"
            ].get(
                "fixed_collate",
                True,
            )
        ),
    )

    preds = []
    gts = []
    states = []

    with torch.inference_mode():

        for index, batch_cpu in enumerate(
            loader,
            1,
        ):
            batch = move_to_device(
                batch_cpu,
                device,
            )

            output = model(
                batch
            )

            preds.append(
                output[
                    "pred_xy"
                ].float().cpu()
            )

            gts.append(
                batch[
                    "future_xy"
                ].float().cpu()
            )

            states.append(
                build_causal_state(
                    batch
                ).float().cpu()
            )

            if (
                index == 1
                or index == len(loader)
                or index % 100 == 0
            ):
                print(
                    f"[EVAL {args.split}] "
                    f"{index}/{len(loader)}",
                    flush=True,
                )

    pred = torch.cat(
        preds,
        dim=0,
    )

    gt = torch.cat(
        gts,
        dim=0,
    )

    state = torch.cat(
        states,
        dim=0,
    )

    raw = exact_means(
        pred,
        gt,
    )

    final = apply_locked_postprocess(
        pred,
        state,
        artifacts=artifacts,
        device=device,
    )

    final_m = exact_means(
        final,
        gt,
    )

    result = {
        "split":
            args.split,
        "samples":
            len(dataset),
        "raw_a3":
            raw,
        "final":
            final_m,
        "reference_test": {
            "ADE":
                REFERENCE_TEST_ADE,
            "FDE":
                REFERENCE_TEST_FDE,
        },
    }

    if args.split == "test":

        dade = abs(
            float(final_m["ADE"])
            - REFERENCE_TEST_ADE
        )

        dfde = abs(
            float(final_m["FDE"])
            - REFERENCE_TEST_FDE
        )

        result[
            "reference_difference"
        ] = {
            "ADE":
                dade,
            "FDE":
                dfde,
        }

        # Current CUDA kernels can have tiny numerical variation.
        result[
            "reference_gate_pass"
        ] = (
            dade <= 0.003
            and
            dfde <= 0.005
        )

    text = json.dumps(
        result,
        indent=2,
    )

    print()
    print(text)

    if args.output is not None:

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            text,
            encoding="utf-8",
        )

    if (
        args.split == "test"
        and not result[
            "reference_gate_pass"
        ]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
