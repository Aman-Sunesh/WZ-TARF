"""Minimal smoke test for the implemented forecasting metrics."""

import torch
from wztarf.metrics.forecasting.minade import minade
from wztarf.metrics.forecasting.minfde import minfde


def main() -> None:
    pred = torch.zeros(2, 6, 25, 2)
    gt = torch.zeros(2, 25, 2)
    assert float(minade(pred, gt)) == 0.0
    assert float(minfde(pred, gt)) == 0.0
    print("Metric smoke test passed.")


if __name__ == "__main__":
    main()
