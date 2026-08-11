"""Dataset wrapper for final serialized WorkZone samples."""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class WorkZoneDataset(Dataset):
    """Load one `.pt` WorkZone sample per item."""

    def __init__(self, files: list[Path]) -> None:
        self.files = list(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return torch.load(self.files[index], map_location="cpu", weights_only=False)
