---
name: pipeflush
description: Run a short pipeflush smoke test for non-trivial changes to jepa_world_model_trainer.py, including a 10-step run to the first visualization/planning dump. Use when editing training loop, planning diagnostics, plotting, or configuration in jepa_world_model_trainer.py.
---

# Pipeflush

## Overview
Run a fast, minimal training run to verify that non-trivial changes to `jepa_world_model_trainer.py` still reach the first visualization/planning dump without errors.

## Workflow
1. Decide whether the change is non-trivial (logic changes, diagnostics, plotting, config, training loop, or data flow). Skip the run for trivial comment/format-only edits.
2. Run a 10-step smoke test with a clear test title so the run is easy to identify.
   - Example:
     ```bash
     uv run python jepa_world_model_trainer.py --steps 10 --vis_schedule "10:10" --plan_schedule "10:10" --title "test: pipeflush <short desc>"
     ```
   - If you need a smaller dataset, prefer the smallest pipeflush dataset available (for example `--data_root data.gridworldkey_pipeflush_10`).
3. Confirm the run produces the first visualization/planning outputs under `out.jepa_world_model_trainer/` (e.g., `vis_*`, `vis_planning_*`).
4. Report the run outcome in the response and call out any failures or anomalies.
