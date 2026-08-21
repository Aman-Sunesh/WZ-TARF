"""Fail fast on common release mistakes before committing to GitHub."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_PY_LINES = 2000
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".toml", ".txt", ".json"}
SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", "checkpoints", "logs", "reports", "outputs"}


def iter_release_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        yield path


def main() -> None:
    errors: list[str] = []
    for path in iter_release_files():
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix == ".py":
            lines = len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
            if lines > MAX_PY_LINES:
                errors.append(f"Python file exceeds {MAX_PY_LINES} lines: {rel} ({lines})")
        if path.suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"[A-Za-z]:\\Users\\|[A-Za-z]:/Users/", text):
                errors.append(f"Machine-specific absolute path: {rel}")

    if errors:
        raise SystemExit("\n".join(errors))

    subprocess.run(["python", str(ROOT / "scripts" / "verify_best.py")], check=True, cwd=ROOT)
    subprocess.run(["python", "-m", "compileall", "-q", "src", "scripts", "tests"], check=True, cwd=ROOT)
    subprocess.run(["python", "-m", "pytest", "-q"], check=True, cwd=ROOT)
    print("Release verification: PASS")


if __name__ == "__main__":
    main()
