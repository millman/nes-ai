# Pose Targets: Measurement-from-z vs Self-Consistent Pose

This note explains the design tradeoffs between two ways of defining the
"pose target" used to train odometry-style losses.

## Background

In the current framing, pose is an accumulated state produced by integrating
per-step increments:

```
pose_{t+1} = pose_t + Δpose_t
Δpose_t = g(pose_t, h_t, a_t)
```

Odometry losses need a "pose target" to compare against. There are two choices:

1) **Measurement pose from observations**
2) **Self-consistent pose (no measurement)**

Both are valid, but they encode different priorities.

## Option A: Measurement-from-z (observation-anchored)

Define a measurement head that maps observations into pose space:

```
pose_obs_t = m(z_t)
```

Train the odometry rollout to match these measurement poses.

**What you get**
- an external anchor tied to what is seen
- loop-closure behavior: revisits pull pose back to the same place
- reduced long-horizon drift
- better mapping/planning in environments with repeated structure

**Costs**
- you must design and train a stable measurement head
- aliasing in observations can inject noise into pose
- extra loss can fight motion consistency if over-weighted

**When to use**
- you care about mapping, loop closure, and long-horizon planning
- you have strong observation signals that can anchor pose

## Option B: Self-consistent pose (no measurement head)

Do not create a measurement head. Train only internal consistency:

```
pose_{t+1} = pose_t + Δpose_t
```

Use multi-step consistency and additivity losses without a target from z.

**What you get**
- clean algebraic structure (local additivity and composition)
- simpler optimization (fewer competing objectives)
- less sensitivity to observation noise/aliasing

**Costs**
- pose can drift arbitrarily over long horizons
- no guarantee that revisits map to the same pose
- weak loop-closure behavior unless you add other anchors (e.g., ranking)

**When to use**
- you only need local planning or short-horizon composition
- you want a purely geometric pose derived from dynamics

## Practical rule of thumb

- Use **measurement-from-z** when you want mapping and loop closure.
- Use **self-consistent pose** when you want clean local geometry and
  are willing to accept drift.

## Hybrid option (if needed later)

You can combine them by using self-consistent pose as the base and a small
correction term conditioned on z:

```
pose_{t+1} = pose_t + Δpose_t
pose_{t+1} += c(pose_{t+1}, z_{t+1})
```

This preserves local algebra while adding a weak observation anchor.
