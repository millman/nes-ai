---
name: jepa-planning-experiment
description: Run iterative JEPA world model trainer planning experiments focused on h-based planning quality. Use when asked to search for a working planning setup in `jepa_world_model_trainer.py`, run repeated training/evaluation cycles, checkpoint at key steps, disable visualization-heavy outputs, and publish experiment worklogs/blog posts with outcomes and next actions.
---

# JEPA Planning Experiment

## Overview
Follow a repeatable loop: isolate changes in a worktree, run trainer with fast diagnostics settings, evaluate h-planning at checkpoint milestones, document findings in a timestamped worklog, then iterate on code/config.

## Hypothesis Gate (Required Before Any Attempt)
Before changing code/config or running any experiment attempt, explicitly do all of the following:
1. State at least 3 plausible hypotheses for the failure mode.
2. Give a concrete recommendation for each hypothesis.
3. Analyze tradeoffs between approaches (expected upside, downside, risk of false positives, and runtime cost).
4. Pick one hypothesis for the next attempt and justify why it is the highest-value test.

Do not run an attempt without recording this hypothesis/tradeoff analysis first.

## Workflow
1. Create an isolated worktree for the exploration branch.
2. Configure trainer defaults for experiment speed and inspectability.
3. Before each attempt, run the required multi-hypothesis + tradeoff analysis gate above.
4. Run a short smoke test if trainer logic changed.
5. Run a long training job (up to 10k steps).
6. Evaluate h-based planning effectiveness at checkpoints.
7. Write a worklog blog post under `worklog/<timestamp>.<concept>/`.
8. If planning is not effective, propose and apply the next experiment changes, then repeat.

## Step 1: Isolate Work
- Create a new branch with prefix `codex/` and a new worktree.
- Keep source edits and outputs in that worktree for experiment hygiene.
- Create worktrees under `/Users/dave/rl/nes-ai.worktrees/` (next to `/Users/dave/rl/nes-ai`), not under `/tmp`.

Example:
```bash
ts=$(date +%Y-%m-%d_%H-%M-%S)
wt=/Users/dave/rl/nes-ai.worktrees/planning.$ts
br=codex/planning-explore-$ts
git worktree add -b "$br" "$wt" HEAD
```

## Step 2: Trainer Experiment Defaults
Set or pass settings that keep metrics while avoiding expensive visual output.

- Dataset: use `data.gwbasic_rand_corner_loops3`.
- Disable visualization schedules:
  - `vis_schedule=""`
  - `plan_schedule=""`
- Keep metrics/logging enabled.
- Emit checkpoints for inspection:
  - exact: `100`, `1000`
  - recurring: every `1000` afterward
  - keep `last.pt` as rolling resume checkpoint.

If relative dataset lookup fails from a worktree, use an absolute `data_root` path.

## Step 3: Runtime Environment (Learned Constraint)
Use standard `uv run` in project context. Do not pin Python 3.9.

Preferred pattern:
```bash
uv run python jepa_world_model_trainer.py ...
```

For repeated experiment loops where the environment is already prepared, optionally use:
```bash
uv run --no-sync python jepa_world_model_trainer.py ...
```

Reason: forcing fresh env resolution or interpreter pinning can trigger heavyweight optional dependency builds.

## Step 4: Required Smoke Run For Non-Trivial Trainer Edits
When changing `jepa_world_model_trainer.py` logic/config/training flow, run a short smoke run first.

- Use a clear test title, e.g. `test: planning setup smoke`.
- Use 10 steps.
- If you need to confirm vis/plan pipeline integrity, temporarily pass short schedules for the smoke run only.

## Step 5: Long Run
Run up to 10k steps for each experiment iteration.

Example:
```bash
uv run --no-sync python jepa_world_model_trainer.py \
  --steps 10000 \
  --data-root /Users/dave/rl/nes-ai/data.gwbasic_rand_corner_loops3 \
  --vis-schedule "" \
  --plan-schedule "" \
  --title "planning explore iter<N>"
```

## Step 6: Evaluate h-Planning At Checkpoints
For each available checkpoint (`100`, `1000`, `2000`, ...):
- Evaluate whether h and actions can move from start tile to goal tile reliably.
- Track at minimum:
  - h-graph reachability (median/p10/p90)
  - local h planning sanity success
  - success/failure on canonical planning tests (test1/test2)
  - number of h graph nodes
- Save metrics to CSV under run `metrics/`.

If there is no standalone evaluator script yet, add one that computes planning metrics without producing visualization files.

## Step 7: Worklog Blog Post
For each iteration, write a post in:
- `worklog/<timestamp>.<concept>/post.md`

Include:
- experiment objective
- exact config/changes
- dataset stats used
- checkpoint-by-checkpoint planning outcomes
- diagnosis (why success/failure happened)
- if failure: concrete next experiments to try
- if success: what made planning effective and why

Keep plots/tables/extra artifacts in the same `worklog/<timestamp>/` folder.

## Iteration Rules
Before each new iteration/attempt:
- Revisit at least 3 hypotheses (carry over prior ones or add new ones from latest evidence).
- Re-evaluate tradeoffs before deciding the next single hypothesis to test.
- Record why rejected hypotheses were deferred.

- Success condition: h-based planning reliably reaches goal in canonical tests and is supported by reachability metrics.
- On success:
  - mark the setup as successful
  - write the success post with evidence
- On failure:
  - write failure post with diagnosis
  - propose the next small set of changes
  - implement and rerun the loop.

## Guardrails
- Preserve metric outputs even when visual outputs are disabled.
- Prefer explicit assertions for invalid prerequisites.
- Avoid silent no-op behavior for required inputs/toggles.
- Keep experiment changes scoped and reversible.
