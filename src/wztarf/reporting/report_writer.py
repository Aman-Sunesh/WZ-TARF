"""Generate the complete human-readable and machine-readable run report."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from wztarf.utils.io import (
    save_json,
    save_yaml,
)


def _format_value(
    value: Any,
) -> str:
    """Format a report value for readable Markdown."""
    if value is None:
        return "N/A"

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, float):
        if value != value:
            return "NaN"

        return f"{value:.6f}"

    return str(value)


def _mapping_table(
    values: Mapping[str, Any],
) -> list[str]:
    """Convert a flat mapping into Markdown table rows."""
    lines = [
        "| Field | Value |",
        "|---|---:|",
    ]

    for key, value in values.items():
        lines.append(
            f"| {key} | {_format_value(value)} |"
        )

    return lines


def _write_metrics_csv(
    path: Path,
    metrics: Mapping[str, Any],
) -> None:
    """Write one metric/value pair per CSV row."""
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "metric",
                "value",
            ]
        )

        for name, value in metrics.items():
            writer.writerow(
                [
                    name,
                    value,
                ]
            )


def write_markdown_report(
    report_dir: str | Path,
    *,
    run_id: str,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    efficiency: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    best_epoch: int | None = None,
    checkpoint: str | Path | None = None,
    training_duration_hours: float | None = None,
) -> dict[str, Path]:
    """Write all standard report artifacts for one completed experiment.

    Args:
        report_dir:
            Destination directory, usually `reports/<run_id>`.

        run_id:
            Unique experiment identifier.

        metrics:
            Final forecasting and safety metrics.

        config:
            Full run configuration.

        efficiency:
            Optional latency, throughput, memory, and parameter metrics.

        environment:
            Optional software/hardware snapshot.

        best_epoch:
            Selected checkpoint epoch.

        checkpoint:
            Selected checkpoint path.

        training_duration_hours:
            Total training duration.

    Returns:
        Dictionary containing paths to the generated report artifacts.
    """
    if not run_id:
        raise ValueError(
            "run_id cannot be empty."
        )

    report_dir = Path(
        report_dir
    ).expanduser()

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        report_dir
        /
        "report.md"
    )

    results_path = (
        report_dir
        /
        "results.json"
    )

    metrics_path = (
        report_dir
        /
        "metrics.csv"
    )

    config_path = (
        report_dir
        /
        "config.yaml"
    )

    results: dict[str, Any] = {
        "run_id": run_id,
        "best_epoch": best_epoch,
        "checkpoint": (
            str(checkpoint)
            if checkpoint is not None
            else None
        ),
        "training_duration_hours": training_duration_hours,
        "metrics": dict(metrics),
        "efficiency": (
            dict(efficiency)
            if efficiency is not None
            else {}
        ),
        "environment": (
            dict(environment)
            if environment is not None
            else {}
        ),
    }

    save_json(
        results_path,
        results,
    )

    save_yaml(
        config_path,
        dict(config),
    )

    _write_metrics_csv(
        metrics_path,
        metrics,
    )

    lines: list[str] = [
        f"# WZ-TARF Experiment Report: `{run_id}`",
        "",
        "## Run Summary",
        "",
        f"- **Run ID:** `{run_id}`",
        (
            f"- **Best epoch:** {best_epoch}"
            if best_epoch is not None
            else "- **Best epoch:** N/A"
        ),
        (
            f"- **Checkpoint:** `{checkpoint}`"
            if checkpoint is not None
            else "- **Checkpoint:** N/A"
        ),
        (
            "- **Training duration:** "
            f"{training_duration_hours:.3f} h"
            if training_duration_hours is not None
            else "- **Training duration:** N/A"
        ),
        "",
        "## Forecasting and Safety Metrics",
        "",
    ]

    lines.extend(
        _mapping_table(
            metrics
        )
    )

    if efficiency:
        lines.extend(
            [
                "",
                "## Efficiency",
                "",
            ]
        )

        lines.extend(
            _mapping_table(
                efficiency
            )
        )

    if environment:
        lines.extend(
            [
                "",
                "## Environment",
                "",
            ]
        )

        # Keep nested GPU metadata readable instead of forcing it into
        # an awkward Markdown table.
        for key, value in environment.items():
            if isinstance(
                value,
                (dict, list, tuple),
            ):
                continue

            lines.append(
                f"- **{key}:** "
                f"{_format_value(value)}"
            )

        complex_environment = {
            key: value
            for key, value in environment.items()
            if isinstance(
                value,
                (dict, list, tuple),
            )
        }

        if complex_environment:
            lines.extend(
                [
                    "",
                    "```yaml",
                    yaml.safe_dump(
                        complex_environment,
                        sort_keys=False,
                        default_flow_style=False,
                    ).rstrip(),
                    "```",
                ]
            )

    lines.extend(
        [
            "",
            "## Configuration",
            "",
            "```yaml",
            yaml.safe_dump(
                dict(config),
                sort_keys=False,
                default_flow_style=False,
            ).rstrip(),
            "```",
            "",
        ]
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return {
        "report": report_path.resolve(),
        "results": results_path.resolve(),
        "metrics": metrics_path.resolve(),
        "config": config_path.resolve(),
    }
