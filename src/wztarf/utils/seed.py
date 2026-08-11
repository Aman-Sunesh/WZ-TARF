"""Reproducibility helpers."""

import random
import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, CPU Torch, and CUDA Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
