"""Unit test for minFDE."""

import torch
from wztarf.metrics.forecasting.minfde import minfde


def test_minfde_zero_when_one_mode_matches_endpoint():
    pred = torch.ones(1, 6, 25, 2)
    pred[:, 0, -1] = 0
    gt = torch.zeros(1, 25, 2)
    assert torch.isclose(minfde(pred, gt), torch.tensor(0.0))
