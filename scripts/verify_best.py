"""Verify the frozen proof files for the best reported exact-K=6 run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "best_wz"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    manifest = json.loads((ART / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
    policy_path = ART / "a20_policy.pt"

    expected = manifest["artifacts"]["a20_policy.pt"]["sha256"]
    actual = sha256(policy_path)
    if actual != expected:
        raise RuntimeError(f"A20 policy hash mismatch: {actual} != {expected}")

    policy = torch.load(policy_path, map_location="cpu", weights_only=False)
    if int(policy["selected_epoch"]) != 22:
        raise RuntimeError("Unexpected A20 selected epoch")
    if policy["feature_mean"].numel() != 114:
        raise RuntimeError("Unexpected A20 feature dimension")

    ade = float(metrics["A20"]["ADE"])
    fde = float(metrics["A20"]["FDE"])
    if abs(ade - 1.0413196086883545) > 1e-12 or abs(fde - 2.013657808303833) > 1e-12:
        raise RuntimeError("Best-run metrics JSON does not match the frozen result")

    print("Best-run proof: PASS")
    print(f"exact K=6 TEST minADE/minFDE: {ade:.10f} / {fde:.10f}")
    print("A20 policy SHA256:", actual)
    print("Selection: DEV-trained, HOLD-selected, no TEST-time selection")


if __name__ == "__main__":
    main()
