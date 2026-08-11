# WZ-TARF

**WorkZone-Conditioned Topology-Adaptive Route Forecaster**

WZ-TARF is a multimodal trajectory-forecasting framework designed for roadway WorkZone scenarios. The model combines ego motion, driver controls, gaze, sparse surrounding agents, lane topology, and explicit WorkZone geometry to generate six route-conditioned future trajectories.

## Forecasting setup

Canonical tensor shapes:

```text
ego history:     10 steps at 5 Hz
future horizon:  25 steps at 5 Hz

pred_xy:         [B, K, T, 2]
gt_xy:           [B, T, 2]
mode_prob:       [B, K]

K = 6
T = 25
```

The history covers 2 seconds and the prediction horizon covers 5 seconds.

## Architecture

The model is organized around role-specific components:

```text
ego motion ───────────┐
controls ─────────────┤
gaze ─────────────────┤
agents ───────────────┤
lanes ────────────────┼─> horizon-aware fusion
WorkZone geometry ────┘          │
                                 ▼
                     temporary lane topology
                                 │
                                 ▼
                      six graph-route queries
                                 │
                  ┌──────────────┴──────────────┐
                  ▼                             ▼
          dynamics anchor              route-conditioned
                                           decoder
                                              │
                                              ▼
                                      optional refiner
                                              │
                                              ▼
                                    safety-aware scoring
```

Major model components are separated under:

```text
src/wztarf/models/
├── encoders/
├── fusion/
├── topology/
├── route/
├── decoders/
├── scoring/
└── wztarf.py
```

## Repository structure

```text
WZ-TARF/
├── checkpoints/
├── configs/
├── docs/
├── logs/
├── outputs/
├── reports/
├── scripts/
├── src/wztarf/
│   ├── data/
│   ├── evaluation/
│   ├── geometry/
│   ├── losses/
│   ├── metrics/
│   ├── models/
│   ├── pretraining/
│   ├── reporting/
│   ├── training/
│   └── utils/
├── tests/
├── pyproject.toml
└── requirements.txt
```

The repository deliberately separates data handling, geometry, neural-network components, self-supervised objectives, supervised losses, metrics, training, evaluation, and reporting.

## Installation

From the repository root:

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

## Supervised training

Dataset roots are supplied at runtime rather than hard-coded into the repository.

Example:

```bash
python scripts/train.py \
  --data-roots "/path/to/processed_final_lap1,/path/to/processed_final_lap2" \
  --run-id wztarf_v1
```

Training writes:

```text
checkpoints/<run_id>/
├── best.pt
└── last.pt

logs/<run_id>/
├── train.log
├── metrics.jsonl
├── config.yaml
├── environment.json
└── run_summary.json
```

## Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/wztarf_v1/best.pt \
  --data-roots "/path/to/processed_final_lap1,/path/to/processed_final_lap2" \
  --run-id wztarf_v1_test
```

Evaluation writes prediction artifacts, metrics, efficiency measurements, and formatted reports under:

```text
reports/<run_id>/
```

## Metrics

Forecasting metrics include:

```text
minADE6
minFDE6
minADE6@1s / 3s / 5s
minFDE6@1s / 3s / 5s
P90 / P95 minADE6
Top-1 ADE / FDE
MR6@2m
Brier-minFDE6
FDE@minADE
ADE@minFDE
```

WorkZone safety metrics include:

```text
WZ-GVR
WSVR@2m
WZVR
```

Efficiency reporting includes inference latency, throughput, trainable parameter count, and peak GPU memory.

## Testing

Run:

```bash
pytest
```

or the lightweight metric smoke test:

```bash
python scripts/smoke_metrics.py
```

## Pretraining status

Phase A objectives and the pretraining optimizer loop are implemented under:

```text
src/wztarf/pretraining/
src/wztarf/training/pretrainer.py
```

The remaining Phase A integration step is the model-side training-only pretraining heads and `pretraining_forward()` path. Phase A should not be launched until those heads are added.
