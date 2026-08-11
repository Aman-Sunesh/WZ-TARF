"""Unit tests for Top-1 selection."""

import torch
from wztarf.metrics.forecasting.top1_ade import top1_ade
from wztarf.metrics.forecasting.top1_fde import top1_fde


def test_top1_uses_probability_argmax():
    pred = torch.zeros(1, 2, 3, 2)
    pred[:, 1] = 1.0
    gt = torch.zeros(1, 3, 2)
    prob = torch.tensor([[0.9, 0.1]])
    assert torch.isclose(top1_ade(pred, gt, prob), torch.tensor(0.0))
    assert torch.isclose(top1_fde(pred, gt, prob), torch.tensor(0.0))
