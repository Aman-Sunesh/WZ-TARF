# WZ-TARF

**WorkZone-Conditioned Topology-Adaptive Route Forecasting**

WZ-TARF is a multimodal trajectory prediction model designed for driving through road WorkZones.

Given the recent motion of the ego vehicle together with scene information such as lane geometry, nearby agents, vehicle controls, driver gaze, workers, and temporary WorkZone structure, WZ-TARF predicts **6 possible trajectories for the next 5 seconds**.

## Overview

Most trajectory prediction methods are designed around normal road geometry. WorkZones are different: lanes may temporarily shift, merge, close, or change their normal connectivity.

WZ-TARF is designed to explicitly account for these temporary changes.

The model combines:

- ego-vehicle motion history
- nearby-agent motion
- lane geometry and lane connectivity
- vehicle controls
- driver gaze
- WorkZone geometry
- worker information

The final model produces **K = 6** possible future trajectories.

## Forecasting setup

| Setting | Value |
|---|---:|
| Observation history | 2 s |
| Prediction horizon | 5 s |
| Sampling rate | 5 Hz |
| History frames | 10 |
| Future frames | 25 |
| Predicted trajectories | 6 |

### Dataset split

| Split | Samples | Participants |
|---|---:|---:|
| Train | 22,540 | 27 |
| Validation | 2,119 | 3 |
| Test | 2,233 | 3 |

The participant sets are disjoint across train, validation, and test.

## Results

The final locked WZ-TARF model achieves:

| Model | minADE₆ ↓ | minFDE₆ ↓ |
|---|---:|---:|
| **WZ-TARF** | **1.0377 m** | **2.0095 m** |

Exact result:

```text
minADE6 = 1.037688971
minFDE6 = 2.009524822
```

`minADE₆` measures the average trajectory error of the best of the six predictions.

`minFDE₆` measures the final-position error at 5 seconds of the best of the six predictions.

### Performance by horizon

| Horizon | minADE₆ ↓ | minFDE₆ ↓ |
|---|---:|---:|
| 1 s | 0.3018 | 0.2716 |
| 2 s | 0.3839 | 0.4445 |
| 3 s | 0.5343 | 0.8539 |
| 4 s | 0.7546 | 1.3911 |
| 5 s | **1.0377** | **2.0095** |

## Model pipeline

The complete WZ-TARF training pipeline is:

```textPhase A
  Representation pretraining
        ↓
Phase B
  Base trajectory predictor
        ↓
ProgressFix
  Trajectory-progress refinement
        ↓
Dense-progress HEADONLY
  Dense progress head refinement
        ↓
Direct-K6 trajectory generator
        ↓
Anchor calibration
        ↓
Native-K64 intermediate adaptation
        ↓
A3:F1 K=6 refinement
        ↓
X fixed12
  Longitudinal calibration
        ↓
X endpoint-zero
  Longitudinal trajectory-shape correction
        ↓
A20
  Late-horizon refinement
        ↓
Final WZ-TARF predictor
```

The K64 stage is used only as an intermediate training step.  
**All reported final results use exactly 6 predicted trajectories.**

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/Aman-Sunesh/WZ-TARF.git
cd WZ-TARF

python -m venv .venv
```

Activate the environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install the package:

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

## Dataset

The dataset itself is not included in this repository.

WZ-TARF expects the processed trajectory samples as serialized `.pt` files.

You can verify a processed dataset with:

```bash
python scripts/verify_dataset.py \
  --data-roots "/path/to/processed_final_lap1,/path/to/processed_final_lap2"
```

## Training

### Full WZ-TARF

```bash
python scripts/train_full_pipeline.py \
  --config configs/wz.yaml \
  --data-roots "/path/to/processed_final_lap1,/path/to/processed_final_lap2" \
  --run-id wz_tarf \
  --device cuda
```

### No-WorkZone baseline

```bash
python scripts/train_full_pipeline.py \
  --config configs/no_wz.yaml \
  --data-roots "/path/to/processed_final_lap1,/path/to/processed_final_lap2" \
  --run-id no_wz \
  --device cuda
```

The No-WZ configuration uses the same overall training pipeline but replaces WorkZone-conditioned topology with the static road topology.

## Test evaluation

TEST evaluation is intentionally separated from model development.

To evaluate a freshly trained canonical pipeline on TEST after all development and model selection are complete:

```bash
python scripts/train_full_pipeline.py \
  --config configs/wz.yaml \
  --data-roots "/path/to/processed_final_lap1,/path/to/processed_final_lap2" \
  --run-id wz_tarf \
  --device cuda \
  --resume-existing \
  --open-test
```

A freshly trained canonical bundle can also be evaluated with:

```bash
python scripts/evaluate_final_pipeline.py \
  --bundle checkpoints/wz_tarf/final_pipeline_bundle.pt \
  --data-roots "/path/to/processed_final_lap1,/path/to/processed_final_lap2" \
  --split test \
  --device cuda
```

## Repository structure

```text
WZ-TARF/
├── configs/          # experiment configurations
├── docs/             # additional documentation
├── scripts/          # training, evaluation, and verification scripts
├── src/wztarf/       # main WZ-TARF implementation
├── tests/            # unit tests
├── README.md
├── requirements.txt
└── pyproject.toml
```

The public repository contains the main implementation and reproduction code while leaving out old experimental sweeps, temporary outputs, and debugging logs.

## Main source modules

```text
src/wztarf/
├── data/             dataset loading and batching
├── models/           WZ-TARF model architecture
├── losses/           training objectives
├── training/         training utilities
├── evaluation/       trajectory metrics
├── pipeline/         full training/refinement pipeline
└── postprocess/      final trajectory calibration
```

## Reproducibility

The repository separates:

- **training from scratch**
- **evaluation of frozen models**
- **historical research artifacts**

Fresh neural-network training can vary slightly because of GPU and optimization nondeterminism.

The reported final locked result is:

```text
1.037688971 minADE6
2.009524822 minFDE6
```
