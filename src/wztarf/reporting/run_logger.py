"""Per-run log and metadata writer."""

import json
from pathlib import Path


class RunLogger:
    """Own the filesystem layout for one experiment run."""

    def __init__(self, project_root: Path, run_id: str) -> None:
        self.run_dir = Path(project_root) / "logs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, filename: str, payload: dict) -> None:
        """Write a JSON artifact into this run's log directory."""
        (self.run_dir / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
