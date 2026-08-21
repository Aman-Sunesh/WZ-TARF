"""Generate a compact formatted Markdown report after each experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml


def write_markdown_report(
    output_dir: str | Path,
    *,
    run_id: str,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    efficiency: Mapping[str, Any],
    environment: Mapping[str, Any],
    checkpoint: str | Path,
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# WZ-TARF Report — {run_id}",
        "",
        f"Checkpoint: `{Path(checkpoint)}`",
        "",
        "## Forecasting and safety metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value} |")

    lines += ["", "## Efficiency", "", "| Metric | Value |", "|---|---:|"]
    for key, value in efficiency.items():
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(dict(environment), indent=2, default=str),
        "```",
        "",
        "## Configuration",
        "",
        "```yaml",
        yaml.safe_dump(dict(config), sort_keys=False).rstrip(),
        "```",
        "",
    ]

    path = directory / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (directory / "results.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "checkpoint": str(checkpoint),
                "metrics": dict(metrics),
                "efficiency": dict(efficiency),
                "environment": dict(environment),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (directory / "config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False),
        encoding="utf-8",
    )
    return path
