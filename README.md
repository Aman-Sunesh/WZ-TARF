# WZ-TARF

**WorkZone-Conditioned Topology-Adaptive Route Forecasting**

This repository deliberately separates data handling, geometry, model
components, pretraining, losses, metrics, evaluation, logging, and reporting.

Canonical prediction shapes:
- `pred_xy`: `[B, K, T, 2]`
- `gt_xy`: `[B, T, 2]`
- `mode_prob`: `[B, K]`
- `K = 6`
- history: 10 steps at 5 Hz
- future: 25 steps at 5 Hz

The first scaffold implements the metric layer and creates documented modules
for the rest of the architecture.
