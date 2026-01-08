## Guideline: Split `s` into pose + descriptor (for planning + graph/SLAM)

### Goal

You’re trying to use learned state for **two different jobs** that want different invariances:

1. **Planning / geometry:** needs a state that composes under actions and supports distances/costs.
2. **Graph building / loop closure:** needs a stable key that recognizes the *same place* across time/animation/nuisance.

So you split `s` into two heads:

* **pose state**: used by the planner and for map geometry
* **descriptor**: used for place recognition (node merge / loop closure)

---

## 1) What each part is for

### `pose` (formerly `s_pose`)

**Purpose**

* planning state for A*/Dijkstra/MPC
* map coordinates for graph nodes
* supports “odometry” and local algebra

**Must be true (locally, in free space)**

* additivity / composition
* approximate inverses when physics allows
* stable scale

**Algebra status**

* **pose is the algebraic object** (errors = planning errors)

---

### `feature` / `descriptor` (formerly `s_desc`)

**Purpose**

* node key / retrieval for revisits
* loop closure candidate generation
* node merging / de-duplication

**Must be true**

* invariant to animation frame, Mario sprite pose, HUD, small pixel jitter
* stable across time
* discriminative at “place” scale

**Algebra status**

* **no algebra required** (it should *not* be forced to compose)

---

## 2) Losses (what to train, and what NOT to mix)

### 2.1 Pose losses (algebra + odometry)

**Structural update (recommended)**

* Treat pose as odometry-integrated:

  ```
  pose_{t+1} = pose_t + Δpose_t
  Δpose_t = g(pose_t, a_t)  or  g(h_t, a_t)
  ```

  (If you keep pose as a pure projection `pose = head(h)`, it can “teleport” and you lose odometry semantics.)

**Core losses**

1. **1-step odometry consistency**

   ```
   L_pose_1 = || (pose_t + Δpose_t) - stopgrad(pose_{t+1}) ||^2
   ```

2. **k-step composition (2–5 steps is enough)**

   ```
   pose_hat_{t+k} = pose_t + Σ_{i=0..k-1} Δpose_{t+i}
   L_pose_k = || pose_hat_{t+k} - stopgrad(pose_{t+k}) ||^2
   ```

3. **Inverse / cancellation (gated)**
   Only enforce when motion is not blocked / contact-free:

   ```
   L_inv = w_t * || Δpose(a_t) + Δpose(a_inv) ||^2
   ```

   with `w_t` based on:

   * environment contact/blocked flag (best), or
   * motion magnitude (self-supervised): `w_t ~ 1[||Δpose|| > eps]`, or soft weight.

**Stabilizers**

* magnitude regularizer:

  ```
  L_mag = ||Δpose_t||^2
  ```
* optional scale control: whitening / running stats on pose dims

**What not to do**

* do **not** apply strong reconstruction/perceptual losses directly on `pose` unless you really want pose to carry appearance.

---

### 2.2 Descriptor/feature losses (place recognition)

**Invariance**

* same observation under augmentation should match:

  ```
  L_aug = || f(x) - f(aug(x)) ||
  ```

**Discriminative**

* contrastive / triplet:

  * positives: same place (time-near, or loop-closure-labeled, or “same node”)
  * negatives: far in time / different episodes / far in pose (once pose is decent)

**Optional: “pose-informed positives”**

* if `||pose_i - pose_j|| < r` treat as positive
* if `||pose_i - pose_j|| > R` treat as negative
  (works well once pose is somewhat stable)

**What not to do**

* do **not** enforce delta/additivity/inverse constraints on `f_t` (it will fight invariance).

---

## 3) Diagnostics (what to measure to know it’s working)

### 3.1 Pose diagnostics (algebra + planning readiness)

1. **Straight-line test**

* rollout repeated same action and check whether points lie on a line.
* failures:

  * curved → bad local algebra
  * expanding step size → drift/scale issues
  * zig-zag → sign/inverse inconsistency

2. **k-step error growth**

* plot:

  ```
  e(k) = || (pose_t + Σ Δpose) - pose_{t+k} ||
  ```
* expected: slow (near-linear) growth in free space; no sudden jumps

3. **Conditional inverse check**

* compute cosine:

  ```
  cos(Δpose(a), -Δpose(a_inv))
  ```

  only on transitions with motion (or not blocked)
* expected: high in free space, low/undefined near walls

4. **Graph/planner sanity**

* build a graph using observed transitions, run shortest path
* expect planned paths to resemble real trajectories; heuristics not crazy
* if this fails while losses look good → pose algebra isn’t usable

---

### 3.2 Descriptor diagnostics (graph stability)

1. **Revisit stability**

* same place revisited → features nearest-neighbor match reliably
  (“node explosion” is the failure mode)

2. **False merge rate**

* different places should not collapse to same feature neighborhood

3. **Loop closure PR curve**

* evaluate precision/recall of loop closure candidates (even with weak labels)

---

## 4) How this fits the “algebra” idea (the new framing)

* **pose**: algebra is the semantics. It must behave like a locally additive coordinate chart with conditional inverses.
* **feature**: no algebra. It’s “place identity,” not motion.

This resolves the old confusion about “should z/h/s all obey Level-3 algebra?”
Now it’s: only **pose** must.

---

## 5) Suggested renames (avoid collisions and ambiguity)

You already use `x_t` for pixels, so don’t reuse `x_t` for pose.

**Recommended**

* `x_t` = pixels / observation
* `z_t` = perceptual embedding (encoder output)
* `h_t` = belief / memory state
* `p_t` = **pose** (planning state)  ← rename `s_pose` → `p_t`
* `Δp_t` = pose increment / odometry  ← rename `Δs` → `Δp`
* `f_t` = **feature / descriptor**   ← rename `s_desc` → `f_t`

Why `f_t`:

* common in vision/SLAM
* avoids confusion with distance `d(·,·)`

**If you prefer to keep `s`**

* `s_pose_t`, `s_feat_t` are fine, but `p_t` / `f_t` reads cleaner in equations.

---

If you tell me whether your graph is built **online** (during exploration) or **offline** (from replay/dataset), I can add one extra section: *node creation/merge rules and what statistics to log so the graph doesn’t explode*.
