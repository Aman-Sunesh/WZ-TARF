"""Evaluate a fully frozen canonical WZ-TARF final_pipeline_bundle.pt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wztarf.data.dataset import WorkZoneDataset
from wztarf.pipeline.common import exact_means, make_loader, parse_roots, resolve_device
from wztarf.pipeline.late import apply_final_postprocess, cache_predictions

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data-roots", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON metrics path")
    args = parser.parse_args()

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    if bundle.get("stage") != "canonical_wztarf_final":
        raise RuntimeError("Not a canonical final WZ-TARF bundle")
    config = bundle["config"]
    device = resolve_device(args.device)
    roots = parse_roots(args.data_roots)
    dataset = WorkZoneDataset(roots=roots, split=args.split, validate=False)
    loader = make_loader(
        dataset,
        batch_size=int(config["evaluation"].get("batch_size", 8)),
        shuffle=False,
        seed=int(config["experiment"].get("seed", 2023)),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        fixed_collate=bool(config["data"].get("fixed_collate", True)),
    )
    # cache_predictions accepts any mapping with model_state_dict, so the bundle itself is valid.
    temporary = args.bundle
    cache = cache_predictions(
        model_checkpoint=temporary,
        config=config,
        loader=loader,
        device=device,
        label=args.split.upper(),
    )
    pred = apply_final_postprocess(
        cache["pred"], cache["state"],
        fixed12_payload=bundle["fixed12"],
        endpoint_payload=bundle["endpoint_zero"],
        a20_payload=bundle["a20"],
        device=device,
    )
    result = {"split": args.split, **exact_means(pred, cache["gt"])}
    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"RESULT: {args.output.resolve()}")
    print(text)


if __name__ == "__main__":
    main()
