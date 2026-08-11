"""Provide small helpers for loading and saving configuration and JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_directory(
    path: str | Path,
) -> Path:
    """Create a directory if necessary and return its resolved path."""
    directory = Path(path).expanduser()

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory.resolve()


def load_yaml(
    path: str | Path,
) -> dict[str, Any]:
    """Load a YAML file and return its top-level mapping."""
    path = Path(path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            f"YAML file does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = yaml.safe_load(handle)

    # An empty YAML file is interpreted as an empty configuration.
    if data is None:
        return {}

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a YAML mapping in {path}, "
            f"got {type(data).__name__}."
        )

    return data


def save_yaml(
    path: str | Path,
    data: dict[str, Any],
) -> Path:
    """Save a dictionary as a readable YAML file."""
    if not isinstance(data, dict):
        raise TypeError(
            "save_yaml expects a dictionary."
        )

    path = Path(path).expanduser()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(
            data,
            handle,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )

    return path.resolve()


def load_json(
    path: str | Path,
) -> dict[str, Any]:
    """Load a JSON file and return its top-level mapping."""
    path = Path(path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            f"JSON file does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a JSON object in {path}, "
            f"got {type(data).__name__}."
        )

    return data


def save_json(
    path: str | Path,
    data: dict[str, Any],
    *,
    indent: int = 2,
) -> Path:
    """Save a dictionary as a formatted JSON file."""
    if not isinstance(data, dict):
        raise TypeError(
            "save_json expects a dictionary."
        )

    if indent < 0:
        raise ValueError(
            "indent cannot be negative."
        )

    path = Path(path).expanduser()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            data,
            handle,
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
        )

        # Keep generated text files POSIX-friendly.
        handle.write("\n")

    return path.resolve()
