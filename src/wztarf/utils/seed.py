"""Provide reproducible random seeding for training and DataLoader workers."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_all(
    seed: int,
    *,
    deterministic: bool = False,
) -> None:
    """Seed Python, NumPy, and PyTorch random-number generators.

    Args:
        seed:
            Non-negative random seed.

        deterministic:
            When True, request deterministic PyTorch algorithms where
            supported. This can reduce performance and may reject operations
            for which PyTorch has no deterministic implementation.

    Notes:
        CUDA generators are seeded when CUDA is available. Deterministic mode
        is optional because the primary training configuration should not
        silently trade speed for determinism.
    """
    if not isinstance(seed, int):
        raise TypeError(
            "seed must be an integer."
        )

    if seed < 0:
        raise ValueError(
            "seed must be non-negative."
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(
            True
        )

        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    else:
        # Do not leave deterministic settings enabled if this function is
        # called again for a normal performance-oriented run.
        torch.use_deterministic_algorithms(
            False
        )

        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = False


def seed_worker(
    worker_id: int,
) -> None:
    """Seed Python and NumPy inside a PyTorch DataLoader worker.

    PyTorch assigns each worker its own initial seed. We derive the NumPy and
    Python seeds from that value so data preprocessing remains reproducible
    when `num_workers > 0`.
    """
    del worker_id

    worker_seed = (
        torch.initial_seed()
        %
        (2**32)
    )

    np.random.seed(
        worker_seed
    )

    random.seed(
        worker_seed
    )


def make_generator(
    seed: int,
) -> torch.Generator:
    """Create a seeded PyTorch generator for deterministic DataLoader shuffling."""
    if not isinstance(seed, int):
        raise TypeError(
            "seed must be an integer."
        )

    if seed < 0:
        raise ValueError(
            "seed must be non-negative."
        )

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    return generator
