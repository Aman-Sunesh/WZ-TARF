"""Unit tests for best-of-K final displacement error."""

import torch

from wztarf.metrics.forecasting.minfde import minfde


def test_minfde_zero_when_one_mode_matches_endpoint() -> None:
    """minFDE is zero when one mode reaches the correct final point."""
    pred = torch.ones(
        1,
        6,
        25,
        2,
    )

    pred[
        :,
        0,
        -1,
    ] = 0.0

    gt = torch.zeros(
        1,
        25,
        2,
    )

    assert torch.isclose(
        minfde(
            pred,
            gt,
        ),
        torch.tensor(
            0.0
        ),
    )


def test_minfde_only_uses_terminal_point() -> None:
    """Earlier trajectory errors do not affect FDE."""
    pred = torch.full(
        (
            1,
            2,
            5,
            2,
        ),
        100.0,
    )

    pred[
        0,
        0,
        -1,
    ] = torch.tensor(
        [
            2.0,
            0.0,
        ]
    )

    pred[
        0,
        1,
        -1,
    ] = torch.tensor(
        [
            1.0,
            0.0,
        ]
    )

    gt = torch.zeros(
        1,
        5,
        2,
    )

    assert torch.isclose(
        minfde(
            pred,
            gt,
        ),
        torch.tensor(
            1.0
        ),
    )
