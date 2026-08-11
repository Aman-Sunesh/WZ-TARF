"""Maintain one flattened summary row per experiment in the master CSV."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _flatten_mapping(
    mapping: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten nested mappings using dotted keys.

    Example:

        {
            "model": {
                "d_model": 128
            }
        }

    becomes:

        {
            "model.d_model": 128
        }
    """
    flattened: dict[str, Any] = {}

    for key, value in mapping.items():
        full_key = (
            f"{prefix}.{key}"
            if prefix
            else str(key)
        )

        if isinstance(value, Mapping):
            flattened.update(
                _flatten_mapping(
                    value,
                    prefix=full_key,
                )
            )
        else:
            flattened[full_key] = value

    return flattened


def _csv_value(
    value: Any,
) -> Any:
    """Convert complex Python values into stable CSV-compatible values."""
    if value is None:
        return ""

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if isinstance(
        value,
        (list, tuple, dict),
    ):
        return json.dumps(
            value,
            sort_keys=True,
            default=str,
        )

    return str(value)


def append_experiment_row(
    path: str | Path,
    *,
    run_id: str,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    summary: Mapping[str, Any] | None = None,
    replace_existing: bool = True,
) -> Path:
    """Add or update one experiment in the master CSV table.

    Args:
        path:
            Usually `reports/all_experiments.csv`.

        run_id:
            Unique experiment identifier.

        metrics:
            Final evaluation metrics.

        config:
            Optional run configuration. Nested values are flattened using
            dotted names such as `model.d_model`.

        summary:
            Optional high-level fields such as best epoch, checkpoint,
            training duration, latency, or parameter count.

        replace_existing:
            Replace an existing row with the same run ID instead of creating
            a duplicate.

    Returns:
        Resolved CSV path.
    """
    if not run_id:
        raise ValueError(
            "run_id cannot be empty."
        )

    path = Path(path).expanduser()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row: dict[str, Any] = {
        "run_id": run_id,
    }

    if summary is not None:
        row.update(
            {
                f"summary.{key}": value
                for key, value in _flatten_mapping(
                    summary
                ).items()
            }
        )

    row.update(
        {
            f"metric.{key}": value
            for key, value in _flatten_mapping(
                metrics
            ).items()
        }
    )

    if config is not None:
        row.update(
            {
                f"config.{key}": value
                for key, value in _flatten_mapping(
                    config
                ).items()
            }
        )

    row = {
        key: _csv_value(value)
        for key, value in row.items()
    }

    existing_rows: list[dict[str, Any]] = []
    existing_fields: list[str] = []

    if path.is_file():
        with path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            reader = csv.DictReader(
                handle
            )

            existing_fields = list(
                reader.fieldnames or []
            )

            existing_rows = [
                dict(existing)
                for existing in reader
            ]

    existing_index: int | None = None

    for index, existing in enumerate(
        existing_rows
    ):
        if existing.get("run_id") == run_id:
            existing_index = index
            break

    if existing_index is not None:
        if not replace_existing:
            raise ValueError(
                f"Run '{run_id}' already exists in {path}."
            )

        # Preserve old columns while replacing supplied values.
        updated = dict(
            existing_rows[existing_index]
        )

        updated.update(
            row
        )

        existing_rows[
            existing_index
        ] = updated

    else:
        existing_rows.append(
            row
        )

    fieldnames = list(
        existing_fields
    )

    if "run_id" not in fieldnames:
        fieldnames.insert(
            0,
            "run_id",
        )

    for key in row:
        if key not in fieldnames:
            fieldnames.append(
                key
            )

    # Also account for columns present in older rows if the original CSV
    # happened to have no explicit header information available.
    for existing in existing_rows:
        for key in existing:
            if key not in fieldnames:
                fieldnames.append(
                    key
                )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for existing in existing_rows:
            writer.writerow(
                {
                    field: existing.get(
                        field,
                        "",
                    )
                    for field in fieldnames
                }
            )

    return path.resolve()
