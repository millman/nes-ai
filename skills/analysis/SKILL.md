---
name: analysis
description: Analyze JEPA world model experiment behavior (especially planning graph sparsity/disconnected centers) with a reproducible workflow: create a dedicated worktree/branch, use jepa_world_model_analysis utilities, record hypotheses/results in worklog, and prepare reviewable commits.
---

# Analysis Skill

Use this workflow for diagnosis-heavy experiment analysis.

## 1) Create isolated analysis workspace
1. Create a dedicated worktree and branch.
2. Name the worktree `<concept>.<YYYY-MM-DD_HH-MM-SS>`.
3. Keep the branch prefixed with `codex/`.
4. Create the worktree under `/Users/dave/rl/nes-ai.worktrees/` (next to `/Users/dave/rl/nes-ai`), not under `/tmp`.
5. Create a matching worklog folder in the main repo at `worklog/<YYYY-MM-DD_HH-MM-SS>.<concept>/`.

## 2) Use analysis utility folder
1. Place helper scripts under `jepa_world_model_analysis/scripts/`.
2. Keep an inventory in `jepa_world_model_analysis/tools.md`.
3. Reuse existing scripts before adding new ones.

## 3) Plan the analysis before editing
1. Write hypotheses and tradeoffs to `worklog/<timestamp>.<concept>/hypotheses.md`.
2. Prefer tests that distinguish between:
   - latent quality issues,
   - graph-construction threshold issues,
   - sample-coverage issues.

## 4) Add/maintain utilities only when needed
1. Add scripts only when ad hoc one-liners are no longer enough.
2. Use fail-fast assertions for missing files/columns.
3. After creating a script, document usage in `jepa_world_model_analysis/tools.md`.

## 5) Write results and decision-ready conclusions
1. Create `worklog/<timestamp>.<concept>/<timestamp>.<concept>.md`.
2. Include:
   - key metrics,
   - hypothesis outcomes,
   - probable root cause(s),
   - next experiment changes.

## 6) Commit policy
1. If utility scripts were added/changed, commit those changes first.
2. If additional improvements are made, commit them separately.
3. Rebase the analysis branch onto `main` (no merge commits).
4. Follow local `.agent` prompt-logging and commit-message protocol.

## 7) Hand-off
1. Provide a concise summary of findings and changed files.
2. Ask the user to review and confirm whether to merge to `main`.
