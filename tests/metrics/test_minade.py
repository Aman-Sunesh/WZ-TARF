"""Unit test for minADE."""

import torch
from wztarf.metrics.forecasting.minade import minade


def test_minade_zero_when_one_mode_matches():
    pred = torch.ones(1, 6, 25, 2)
    pred[:, 0] = 0
    gt = torch.zeros(1, 25, 2)
    assert torch.isclose(minade(pred, gt), torch.tensor(0.0))
