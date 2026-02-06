# JEPA World Model Analysis Tools

This folder contains analysis utilities for diagnosing planning behavior in `jepa_world_model_trainer` runs.

## Existing tools

### `scripts/analyze_planning_connectivity.py`
- Purpose: summarize whether planning graph centers are sparse/disconnected based on `planning_metrics.csv` and `planning_anchor_metrics.csv`.
- Inputs: run directory path (e.g. `out.jepa_world_model_trainer/2026-02-05_17-22-39`).
- Outputs:
  - text summary (default)
  - JSON summary with `--json`
- Example:

```bash
source /Users/dave/rl/nes-ai/.venv/bin/activate
uv run --active --no-sync python jepa_world_model_analysis/scripts/analyze_planning_connectivity.py \
  /Users/dave/rl/nes-ai/out.jepa_world_model_trainer/2026-02-05_17-22-39 --json
```
