"""Verify the processed dataset used by the released experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from wztarf.data.dataset import WorkZoneDataset, discover_pt_files
from wztarf.data.schema import validate_sample

EXPECTED_COUNTS = {"train": 22540, "val": 2119, "test": 2233}
EXPECTED_PARTICIPANTS = {
    "train": {
        "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11",
        "P13", "P15", "P16", "P17", "P18", "P19", "P21", "P22",
        "P23", "P24", "P25", "P26", "P27", "P28", "P29", "P30",
        "P32", "P33",
    },
    "val": {"P14", "P20", "P34"},
    "test": {"P2", "P12", "P31"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-roots", required=True, help="Comma-separated processed roots.")
    parser.add_argument(
        "--check-participants",
        action="store_true",
        help="Load every sample and verify participant-disjoint split membership.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def participant_from_meta(meta: object) -> str | None:
    if isinstance(meta, dict):
        value = meta.get("participant") or meta.get("participant_id")
        if value is not None:
            return str(value)
    return None


def normalized_manifest_hash(files: list[Path], roots: list[Path]) -> str:
    """Hash relative sample names without recording machine-specific paths."""
    rows: list[str] = []
    resolved = [root.expanduser().resolve() for root in roots]
    for path in files:
        p = path.resolve()
        relative = None
        for index, root in enumerate(resolved):
            try:
                relative = f"root{index}/" + p.relative_to(root).as_posix()
                break
            except ValueError:
                continue
        rows.append(relative or p.name)
    return hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest()


def main() -> None:
    args = parse_args()
    roots = [Path(x.strip()) for x in args.data_roots.split(",") if x.strip()]
    if not roots:
        raise ValueError("No dataset roots supplied.")

    report: dict[str, object] = {"splits": {}}
    for split in ("train", "val", "test"):
        files = discover_pt_files(roots, split=split)
        if len(files) != EXPECTED_COUNTS[split]:
            raise RuntimeError(
                f"{split}: found {len(files)} samples; expected {EXPECTED_COUNTS[split]}."
            )

        dataset = WorkZoneDataset(files=files, validate=False)
        first = dataset[0]
        last = dataset[len(dataset) - 1]
        validate_sample(first, source=str(files[0]))
        validate_sample(last, source=str(files[-1]))

        split_report: dict[str, object] = {
            "count": len(files),
            "path_manifest_sha256": normalized_manifest_hash(files, roots),
            "first_file": files[0].name,
            "last_file": files[-1].name,
        }

        if args.check_participants:
            participants: set[str] = set()
            for i in range(len(dataset)):
                sample = dataset[i]
                participant = participant_from_meta(sample.get("meta"))
                if participant is None:
                    # Fallback for common Pxx file naming.
                    match = re.search(r"(?:^|[_-])(P\d+)(?:[_-]|$)", files[i].stem)
                    participant = match.group(1) if match else None
                if participant is None:
                    raise RuntimeError(f"Could not identify participant for {files[i]}")
                participants.add(participant)
            if participants != EXPECTED_PARTICIPANTS[split]:
                raise RuntimeError(
                    f"{split}: participant set mismatch. "
                    f"got={sorted(participants)} expected={sorted(EXPECTED_PARTICIPANTS[split])}"
                )
            split_report["participants"] = sorted(participants)

        report["splits"][split] = split_report

    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
