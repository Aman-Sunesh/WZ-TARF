# Repository layout

- `data/`: dataset loading, schema, derived features, map coverage.
- `geometry/`: reusable relative, lane, and WorkZone geometry.
- `models/`: only neural-network architecture components.
- `pretraining/`: masking, topology reconstruction, future contrastive learning.
- `losses/`: one supervised objective per file.
- `metrics/forecasting/`: one forecasting metric per file.
- `metrics/safety/`: one safety metric per file.
- `metrics/efficiency/`: one efficiency metric per file.
- `training/`: training/pretraining loops and checkpointing.
- `evaluation/`: metric orchestration and prediction export.
- `reporting/`: logs, reports, environment snapshots, master experiment table.
- `scripts/`: thin CLI entry points only.
