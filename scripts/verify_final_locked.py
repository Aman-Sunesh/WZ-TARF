"""Verify the permanently frozen WZ-TARF result from the archived A3 TEST cache.

This is a benchmark/artifact reproducibility check.
It does NOT use a saved final-predictions tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from wztarf.pipeline.common import exact_means
from wztarf.pipeline.final_locked import (
    REFERENCE_TEST_ADE,
    REFERENCE_TEST_FDE,
    apply_locked_postprocess,
    default_artifact_root,
    load_locked_artifacts,
    torch_load,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest().upper()


def main() -> None:

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
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

    device = torch.device(
        args.device
    )

    # ---------------------------------------------------------
    # Hash integrity.
    # ---------------------------------------------------------

    hash_path = (
        artifact_root
        / "ARTIFACT_SHA256.json"
    )

    expected_hashes = json.loads(
        hash_path.read_text(
            encoding="utf-8-sig"
        )
    )

    for relative, expected in (
        expected_hashes.items()
    ):

        path = (
            artifact_root
            / relative
        )

        actual = sha256(
            path
        )

        if actual != expected:
            raise RuntimeError(
                "Artifact hash mismatch: "
                f"{relative}\n"
                f"expected={expected}\n"
                f"actual={actual}"
            )

    # ---------------------------------------------------------
    # Start at A3 predictions, NOT saved final predictions.
    # ---------------------------------------------------------

    cache = torch_load(
        artifact_root
        / "historical_test_a3_cache.pt"
    )

    pred = cache["pred"]
    gt = cache["gt"]
    state = cache["state"]

    artifacts = load_locked_artifacts(
        artifact_root
    )

    raw = exact_means(
        pred,
        gt,
    )

    final, stages = (
        apply_locked_postprocess(
            pred,
            state,
            artifacts=artifacts,
            device=device,
            return_stages=True,
        )
    )

    metrics = {
        "raw_a3":
            raw,
    }

    for name, value in (
        stages.items()
    ):
        metrics[name] = (
            exact_means(
                value,
                gt,
            )
        )

    final_m = metrics[
        "y_endpoint_zero_final"
    ]

    dade = abs(
        float(final_m["ADE"])
        - REFERENCE_TEST_ADE
    )

    dfde = abs(
        float(final_m["FDE"])
        - REFERENCE_TEST_FDE
    )

    passed = (
        dade <= 0.0002
        and
        dfde <= 0.0010
    )

    result = {
        "reference": {
            "ADE":
                REFERENCE_TEST_ADE,
            "FDE":
                REFERENCE_TEST_FDE,
        },
        "metrics":
            metrics,
        "absolute_difference": {
            "ADE":
                dade,
            "FDE":
                dfde,
        },
        "used_saved_final_predictions":
            False,
        "PASS":
            passed,
    }

    print()
    print("=" * 92)
    print(
        "WZ-TARF FINAL LOCKED "
        "ARTIFACT REPRODUCTION"
    )
    print("=" * 92)

    order = [
        "raw_a3",
        "x_fixed12",
        "x_endpoint_zero",
        "a20_scale_2",
        "y_fixed12",
        "y_endpoint_zero_final",
    ]

    for name in order:
        m = metrics[name]

        print(
            f"{name:<24} "
            f"{m['ADE']:.9f} / "
            f"{m['FDE']:.9f}"
        )

    print()
    print(
        "REFERENCE                "
        f"{REFERENCE_TEST_ADE:.9f} / "
        f"{REFERENCE_TEST_FDE:.9f}"
    )

    print(
        "ABS DIFF                 "
        f"{dade:.9f} / "
        f"{dfde:.9f}"
    )

    print()
    print(
        "FINAL REPRODUCTION = "
        + (
            "PASS"
            if passed
            else "FAIL"
        )
    )

    print("=" * 92)

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            json.dumps(
                result,
                indent=2,
            ),
            encoding="utf-8",
        )

    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
