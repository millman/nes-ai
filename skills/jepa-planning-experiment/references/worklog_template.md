# Worklog Post Template

## Objective
- What planning behavior is targeted in this iteration?

## Setup
- Branch/worktree:
- Dataset:
- Trainer command:
- Code/config deltas:

## Data Stats
- Number of trajectories:
- Frames used:
- Sequence length / batch size:

## Checkpoint Results
| checkpoint_step | num_nodes_h | reach_h_median | reach_h_p10 | reach_h_p90 | h_local_success | test1_success | test2_success |
|---|---:|---:|---:|---:|---:|---:|---:|

## Diagnosis
- What seems to be working?
- What is failing and why?

## Outcome
- `SUCCESS` or `FAILURE`
- Brief justification based on metrics and planning traces.

## Next Changes
1. Smallest high-impact change to try next.
2. Secondary change if first does not improve planning.
