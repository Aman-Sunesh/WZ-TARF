# WZ-TARF

**WorkZone-Conditioned Topology-Adaptive Route Forecaster**

WZ-TARF predicts six possible 5-second ego trajectories in road work zones. The model uses ego motion, controls, gaze, nearby agents, lane geometry, and temporary WorkZone structure.

This repository has two goals:

1. provide a clean, readable path for **WZ vs No-WZ training from scratch**; and
2. preserve the exact evidence for the strongest research run without mixing experimental scripts into the public code path.

## Forecasting setup

- history: 10 frames at 5 Hz = 2 seconds
- future: 25 frames at 5 Hz = 5 seconds
- output modes: K = 6
- main metrics: exact `minADE6` and `minFDE6`

Canonical final split sizes:

| Split | Samples | Participants |
|---|---:|---:|
| Train | 22,540 | 27 |
| Validation | 2,119 | 3 |
| Test | 2,233 | 3 |

## Best frozen result

The strongest saved exact-K=6 official TEST result is:

| Model | minADE6 | minFDE6 |
|---|---:|---:|
| WZ-TARF + frozen A20 policy | **1.0413** | **2.0137** |

The exact values are `1.0413196087 / 2.0136578083`. The A20 policy was trained on development data, selected using the internal holdout, and applied to TEST without refitting or TEST-time selection.

Proof files are under `artifacts/best_wz/`. Run:

```bash
python scripts/verify_best.py
```

This verifies the saved policy hash, selected epoch, feature dimension, and frozen metrics JSON. See `docs/REPRODUCIBILITY.md` for the distinction between proof reproduction and fresh retraining.

## Installation

Python 3.10+ is required.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -e ".[dev]"
```

## Verify the processed dataset

The code expects the final serialized `.pt` samples. Dataset roots are passed at runtime; no machine-specific path is stored in the repository.

```bash
python scripts/verify_dataset.py \
  --data-roots "/path/to/processed_final_lap1,/path/to/processed_final_lap2"
```

For the slower participant-disjoint audit:

```bash
python scripts/verify_dataset.py \
  --data-roots "/path/to/processed_final_lap1,/path/to/processed_final_lap2" \
  --check-participants
```

## Train the complete fresh pipeline

The canonical entry point now trains the **entire** research recipe. It does not load `artifacts/best_wz` or any historical intermediate checkpoint.

```text
Phase A -> Phase B -> Direct-K6 generator -> strip longitudinal repair
        -> anchor-only calibration -> native-K64 intermediate adaptation (2+4)
        -> A3:F1 x 1 epoch -> fixed12 -> endpoint-zero -> A20 -> frozen bundle
```

WZ only:

```bash
python scripts/train_full_pipeline.py \
  --config configs/wz.yaml \
  --data-roots "/path/to/processed_final_lap1,/path/to/processed_final_lap2" \
  --run-id canonical_fresh_wz \
  --device cuda \
  --open-test
```

No-WZ only uses the identical stage graph:

```bash
python scripts/train_full_pipeline.py \
  --config configs/no_wz.yaml \
  --data-roots "/path/to/processed_final_lap1,/path/to/processed_final_lap2" \
  --run-id canonical_fresh_no_wz \
  --device cuda \
  --open-test
```

For the paper comparison, run both sequentially:

```bash
python scripts/train_comparison.py \
  --data-roots "/path/to/processed_final_lap1,/path/to/processed_final_lap2" \
  --device cuda \
  --open-test
```

`configs/no_wz.yaml` removes explicit WZ polygon/topology conditioning and keeps the separately observed worker stream. The late A3 auxiliary pass also forces `topology=0` and `wz_geometry=0` for No-WZ, while every other canonical stage/hyperparameter is shared.

`--open-test` is deliberately optional. Without it, TEST is never constructed. For a single-condition `train_full_pipeline.py` run, TEST is constructed only after that final bundle is frozen. For `train_comparison.py`, **both WZ and No-WZ are frozen before either TEST result is revealed**, then each frozen bundle is evaluated once. `--resume-existing` resumes only artifacts produced under the requested run ID; it never searches `artifacts/best_wz`.

The historical WZ target on the same final dataset is `1.0413196087 / 2.0136578083`. Fresh training is stochastic, so this is a regression target rather than a bit-for-bit guarantee.

## Evaluate a frozen final bundle

```bash
python scripts/evaluate_final_pipeline.py \
  --bundle checkpoints/canonical_fresh_wz/final_pipeline_bundle.pt \
  --data-roots "/path/to/processed_final_lap1,/path/to/processed_final_lap2" \
  --split test \
  --device cuda
```

The older `scripts/train.py` and `scripts/evaluate.py` remain available for Phase-B-only diagnostics; they are **not** the canonical full-paper reproduction path.

## Repository layout

```text
configs/                  canonical WZ / No-WZ experiment definitions
scripts/                  small public entry points
src/wztarf/data/          dataset loading and schema checks
src/wztarf/models/        model components
src/wztarf/losses/        supervised objectives
src/wztarf/training/      training loops and checkpoints
src/wztarf/evaluation/    exact metrics and report generation
src/wztarf/postprocess/   frozen A20 post-processing policy
artifacts/best_wz/        best-run proof files and hashes
docs/                     model, data, and reproducibility notes
tests/                    unit and forward/backward tests
```

The release intentionally excludes the thousands of experimental sweep/diagnostic scripts from the research workspace.

## Release checks

Before committing:

```bash
python scripts/verify_release.py
```

The check fails on machine-specific absolute Windows user-home paths or Python files above 2,000 lines, verifies the best-run artifact, compiles the source, and runs the test suite.

## Notes on checkpoints

The included A20 policy is small enough to keep with the repository. Larger legacy checkpoints can be copied from the research workspace with:

```bash
python scripts/import_best_artifacts.py --legacy-root "/path/to/old/WZ-TARF-workspace"
```

The script records SHA256 hashes automatically. Imported large checkpoints remain ignored by Git unless you intentionally publish them or attach them to a release.
