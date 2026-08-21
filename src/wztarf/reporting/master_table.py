"""Maintain one master CSV row per experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


def append_experiment_row(
    path: str | Path,
    *,
    run_id: str,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> Path:
    """Append one experiment summary, expanding the CSV header when needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    row: dict[str, Any] = {"run_id": run_id}
    row.update({f"metric.{k}": v for k, v in metrics.items()})
    row.update({f"summary.{k}": v for k, v in summary.items()})
    # Preserve complete config without creating hundreds of CSV columns.
    row["config_json"] = json.dumps(dict(config), sort_keys=True, default=str)

    rows: list[dict[str, Any]] = []
    existing_fields: list[str] = []
    if destination.exists() and destination.stat().st_size:
        with destination.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            existing_fields = list(reader.fieldnames or [])
            rows.extend(reader)

    fields = existing_fields[:]
    for key in row:
        if key not in fields:
            fields.append(key)
    for old in rows:
        for key in old:
            if key not in fields:
                fields.append(key)

    rows.append({key: row.get(key, "") for key in fields})
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            writer.writerow({key: item.get(key, "") for key in fields})
    return destination
