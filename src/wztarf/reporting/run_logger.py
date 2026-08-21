"""Per-run text, metric, configuration, and environment logger."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from .environment import environment_snapshot


class RunLogger:
    """Own the filesystem layout for one experiment run."""

    def __init__(self, project_root: Path, run_id: str) -> None:
        self.project_root = Path(project_root)
        self.run_id = str(run_id)
        self.run_dir = self.project_root / "logs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.train_log = self.run_dir / "train.log"
        self.metrics_jsonl = self.run_dir / "metrics.jsonl"

    def write_json(self, filename: str, payload: Mapping[str, Any]) -> None:
        (self.run_dir / filename).write_text(
            json.dumps(dict(payload), indent=2, default=str),
            encoding="utf-8",
        )

    def log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with self.train_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        split: str,
        epoch: int | None = None,
        step: int | None = None,
    ) -> None:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "split": split,
            "epoch": epoch,
            "step": step,
            "metrics": {
                key: (float(value) if hasattr(value, "__float__") else value)
                for key, value in metrics.items()
            },
        }
        with self.metrics_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def save_config(self, config: Mapping[str, Any]) -> None:
        (self.run_dir / "config.yaml").write_text(
            yaml.safe_dump(dict(config), sort_keys=False),
            encoding="utf-8",
        )

    def save_environment(self) -> None:
        self.write_json(
            "environment.json",
            environment_snapshot(self.project_root),
        )

    def save_run_summary(self, summary: Mapping[str, Any]) -> None:
        self.write_json("run_summary.json", summary)
