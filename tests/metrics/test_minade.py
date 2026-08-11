"""Unit tests for best-of-K average displacement error."""

import torch

from wztarf.metrics.forecasting.minade import minade


def test_minade_zero_when_one_mode_matches() -> None:
    """minADE is zero when at least one candidate matches all GT points."""
    pred = torch.ones(
        1,
        6,
        25,
        2,
    )

    pred[
        :,
        0,
    ] = 0.0

    gt = torch.zeros(
        1,
        25,
        2,
    )

    assert torch.isclose(
        minade(
            pred,
            gt,
        ),
        torch.tensor(
            0.0
        ),
    )


def test_minade_uses_best_mode() -> None:
    """minADE selects the lowest-ADE trajectory rather than averaging modes."""
    pred = torch.zeros(
        1,
        2,
        4,
        2,
    )

    pred[
        :,
        0,
        :,
        0,
    ] = 2.0

    pred[
        :,
        1,
        :,
        0,
    ] = 1.0

    gt = torch.zeros(
        1,
        4,
        2,
    )

    assert torch.isclose(
        minade(
            pred,
            gt,
        ),
        torch.tensor(
            1.0
        ),
    )
