"""Record training messages, metrics, configuration, and run metadata."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from wztarf.reporting.environment import environment_snapshot
from wztarf.utils.io import (
    save_json,
    save_yaml,
)


def _jsonable(
    value: Any,
) -> Any:
    """Convert common experiment objects into JSON-serializable values."""
    if value is None:
        return None

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
        torch.Tensor,
    ):
        value = (
            value
            .detach()
            .cpu()
        )

        if value.numel() == 1:
            return value.item()

        return value.tolist()

    if is_dataclass(value):
        return _jsonable(
            asdict(value)
        )

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            _jsonable(item)
            for item in value
        ]

    # NumPy scalar objects and similar objects commonly implement `.item()`.
    item_method = getattr(
        value,
        "item",
        None,
    )

    if callable(item_method):
        try:
            return item_method()
        except Exception:
            pass

    return str(value)


def _utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(
        timezone.utc
    ).isoformat()


class RunLogger:
    """Own all log artifacts produced during one experiment run."""

    def __init__(
        self,
        project_root: str | Path,
        run_id: str,
    ) -> None:
        """Create the run's log directory.

        Args:
            project_root:
                WZ-TARF repository root.

            run_id:
                Unique run identifier used as the log-directory name.
        """
        if not run_id:
            raise ValueError(
                "run_id cannot be empty."
            )

        # Prevent accidental nested/path-traversal run identifiers.
        if Path(run_id).name != run_id:
            raise ValueError(
                "run_id must be a single directory name."
            )

        self.project_root = (
            Path(project_root)
            .expanduser()
            .resolve()
        )

        self.run_id = run_id

        self.run_dir = (
            self.project_root
            /
            "logs"
            /
            run_id
        )

        self.run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.train_log_path = (
            self.run_dir
            /
            "train.log"
        )

        self.metrics_path = (
            self.run_dir
            /
            "metrics.jsonl"
        )

        self.config_path = (
            self.run_dir
            /
            "config.yaml"
        )

        self.environment_path = (
            self.run_dir
            /
            "environment.json"
        )

        self.summary_path = (
            self.run_dir
            /
            "run_summary.json"
        )

    def log(
        self,
        message: str,
    ) -> None:
        """Append one timestamped human-readable message to `train.log`."""
        timestamp = _utc_timestamp()

        line = (
            f"[{timestamp}] "
            f"{message}\n"
        )

        with self.train_log_path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                line
            )

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        split: str | None = None,
        epoch: int | None = None,
        step: int | None = None,
    ) -> None:
        """Append one machine-readable metric record to `metrics.jsonl`.

        Each line is an independent JSON object, which keeps the file robust
        during long-running training jobs.
        """
        record: dict[str, Any] = {
            "timestamp": _utc_timestamp(),
            "run_id": self.run_id,
        }

        if split is not None:
            record["split"] = split

        if epoch is not None:
            record["epoch"] = int(
                epoch
            )

        if step is not None:
            record["step"] = int(
                step
            )

        record["metrics"] = _jsonable(
            dict(metrics)
        )

        with self.metrics_path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    record,
                    sort_keys=True,
                )
            )

            handle.write(
                "\n"
            )

    def save_config(
        self,
        config: Mapping[str, Any],
    ) -> Path:
        """Save the complete reproducibility configuration as YAML."""
        return save_yaml(
            self.config_path,
            _jsonable(
                dict(config)
            ),
        )

    def save_environment(
        self,
        environment: Mapping[str, Any] | None = None,
    ) -> Path:
        """Save the software/hardware/Git environment for this run."""
        if environment is None:
            environment = environment_snapshot(
                self.project_root
            )

        return save_json(
            self.environment_path,
            _jsonable(
                dict(environment)
            ),
        )

    def save_run_summary(
        self,
        summary: Mapping[str, Any],
    ) -> Path:
        """Save the final training/validation summary for the run."""
        payload = {
            "run_id": self.run_id,
            **_jsonable(
                dict(summary)
            ),
        }

        return save_json(
            self.summary_path,
            payload,
        )

    def write_json(
        self,
        filename: str,
        payload: Mapping[str, Any],
    ) -> Path:
        """Write an additional JSON artifact inside this run directory."""
        destination = Path(
            filename
        )

        if (
            destination.name
            !=
            filename
        ):
            raise ValueError(
                "filename must not contain directories."
            )

        if destination.suffix.lower() != ".json":
            raise ValueError(
                "write_json requires a '.json' filename."
            )

        return save_json(
            self.run_dir
            /
            destination,
            _jsonable(
                dict(payload)
            ),
        )
