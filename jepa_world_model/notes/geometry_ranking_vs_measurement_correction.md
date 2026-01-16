# Geometry/Ranking vs Measurement Correction

This note contrasts two tools that are often confused:

- **Geometry/ranking losses**: shape the pose space so it provides a global ordering for planning.
- **Measurement correction from z**: anchor pose to observations to reduce drift and enable loop closure.

They are complementary, not substitutes.

## Quick guide

Use **geometry/ranking** when you need:
- global ordering toward goals
- admissible heuristics for graph search
- topology-aware structure (bottlenecks, walls, one-way paths)

Use **measurement correction** when you need:
- drift reduction over long horizons
- loop-closure behavior (revisits map to the same pose)
- pose grounding to observations

## Examples

### Geometry/ranking helps; measurement correction does not

**Scenario**: a maze with two corridors that look visually similar. The agent must choose the correct corridor to reach the goal.

- A geometry/ranking loss can shape pose so that "closer to goal" is globally ordered, making search efficient.
- A measurement correction from z only anchors pose to what is seen; it does not create the correct ordering over corridors.

### Measurement correction helps; geometry/ranking does not

**Scenario**: a long hallway loop where odometry drift accumulates, and the agent revisits the same location.

- A measurement correction from z pulls pose back on revisits, fixing drift and enabling loop closure.
- A geometry/ranking loss does not anchor pose to observations, so drift remains.

## Summary

- **Geometry/ranking** = global planning structure.
- **Measurement correction** = observation anchoring and drift control.

Use both if you want long-horizon planning plus loop closure. If you must choose, pick the one aligned with your failure mode.
