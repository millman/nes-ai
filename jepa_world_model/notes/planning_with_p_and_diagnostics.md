## Summary of what we discussed

* **Planning over `h`** (Dreamer-style hidden state) is mainly a **coverage / replay** test when you don’t use imagination:

  * A dataset-graph over `h` can only plan to states that already appear in your data.
  * With “perimeter loops only,” you should *not* expect `h`-planning to reach interior goals.
  * Still very useful as a **model health check** via a *local* test (plan from `t` to `t+k` inside the dataset).

* **Planning over `p`** is what can show **geometry / generalization**:

  * Your `p` is high-D; you can still plan in high-D using **action deltas** `mu[a] = median(p_{t+1}-p_t | a)`.
  * This “delta-lattice” planner can propose interior states even if the dataset never visited them.
  * Key diagnostics are whether action deltas are **consistent** (low spread) and whether inverse actions cancel.

* **Clustering** (even if embeddings vary smoothly) is mainly for:

  * building finite graphs for BFS,
  * detecting drift/history leakage (node count grows with laps),
  * measuring dataset connectivity/coverage (“reachable fraction”).
  * But for your generalization-to-interior tests, prefer **continuous/high-D delta planning** (no clustering needed).

* A **pose head** means a low-D readout `y = g(p)` (e.g., 2D/3D) trained with **action-equivariance** losses. It does *not* replace `p`.

  * Safest version for now: train it with `stopgrad(p)` so it’s **diagnostic-only** and won’t destabilize training.

---

## What to implement

### 0) One-time: define “true pose” grid indexing (for eval only)

Since your world is discrete:

* top-left x positions: `{0,8,16,24,32,40,48}` (7)
* same for y (7) → 49 cells.

Implement:

* `cell_from_frame_or_env()` → `(ix, iy)` in `[0..6]×[0..6]` (if you can query env state; otherwise infer from known renderer state).

This powers crisp success metrics + clean plots.

---

## 1) Data extraction for eval (runs every N training steps)

From a frozen snapshot of the model and a sample of recent replay:

* Gather transitions: `(x_t, a_t, x_{t+1})` (and optionally `done`)
* Compute:

  * `p_t = P(x_t)`, `p_{t+1} = P(x_{t+1})`
  * `h_t` (whatever you define as planning state; if it depends on recurrence, use a consistent teacher-forced extraction pass)

Store these arrays for the eval step.

---

## 2) Core high-D `p` delta stats (drives thresholds + planner)

For each action `a ∈ {L,R,U,D,Noop}`:

* `d_p = p_{t+1} - p_t` for transitions with `a_t=a`
* `mu[a] = coordinatewise median(d_p)`
* `L[a] = median(||d_p||)`
* `S[a] = median(||d_p - mu[a]||)`

Define:

* `L = median(L[L], L[R], L[U], L[D])`

Thresholds (adaptive, scale-free):

* `r_goal  = 0.35 * L`
* `r_merge = 0.25 * L`
* `r_cluster_p = 0.35 * L` (if you cluster p for dataset-graph diagnostics)

Quality diagnostics:

* `q[a] = S[a] / max(||mu[a]||, eps)`  (want <0.4, ideally <0.2)
* `inv_lr = ||mu[L] + mu[R]|| / L` (want <0.4)
* `inv_ud = ||mu[U] + mu[D]|| / L` (want <0.4)
* `noop_ratio = L[Noop] / L` (want <0.25, ideally <0.15)

---

## 3) Implement planners

### 3A) `h` dataset-graph planner (coverage/health)

* Build nodes by clustering `h` using cosine distance:

  * compute neighbor distance baseline `d_nn = median(1 - cos(h_t, h_{t+1}))` for non-noop
  * `tau_h_merge = clamp(3*d_nn, 0.02, 0.08)`
* Map each `h_t` to a node id.
* Add directed edges labeled by action from `(node(h_t), a_t) -> node(h_{t+1})`.

Run:

* **H-local sanity test** (should pass earlier than anything else):

  * pick random dataset index `t`
  * goal is `t+k` (k=3..10)
  * plan via BFS in h-graph and execute actions in env from `x_t`
* Your “interior” tests with h-graph:

  * expected to fail with loop-only data; log failure as coverage info.

### 3B) `p` dataset-graph planner (coverage only)

Same as h but cluster `p` (radius `r_cluster_p`), build dataset-only graph, BFS.

* Expect interior tests to fail (no interior nodes).
* Useful for reachable-fraction and stability (node count).

### 3C) `p` delta-lattice planner (the real generalization test)

State is high-D `p` (no clustering required), transitions:

* `p_next = p + mu[a]` for `a ∈ {L,R,U,D,Noop}`

Search:

* A* strongly recommended:

  * cost per step = 1
  * heuristic `h(p)=||p-p_goal||/L`
* Dedup explored states using `r_merge` (treat as same if within `r_merge`).

Stop when:

* `||p - p_goal|| <= r_goal`

This is what you use for:

* Test 1: loop start → interior goal
* Test 2: interior start → interior goal

Execution:

* Execute planned action list in env.
* Measure success in **true grid cell space** and also track p-distance to goal across execution.

---

## 4) Periodic evaluation tests (your exact setup)

### Test 1: start on loop near lower-left → goal at center (unseen)

* Reset env to `S_loop`.
* Goal cell: center `(3,3)` (or equivalent pixel top-left `(24,24)`).
* Run:

  * h dataset-graph BFS (expected fail)
  * p dataset-graph BFS (expected fail)
  * p delta-lattice A* (expected to start failing, eventually pass)

### Test 2: start in interior (unseen) → another interior goal (unseen)

* Reset env to `S_mid` and goal `G_mid2`.
* Run same planners.
* This is the “is p globally consistent?” test.

### Always include: H-local sanity

* random `t → t+k` test inside dataset
* should improve early, catches broken dynamics/stability.

---

## 5) Diagnostics & plots to generate each eval step

### Scalars to log (time series)

* `q[a]` for each action
* `inv_lr`, `inv_ud`, `noop_ratio`
* `L` (step scale)
* Test1/Test2: success, steps, final p-distance-to-goal / L
* `num_nodes_h`, `num_nodes_p` (from clustering)
* reachable fraction stats (median/10th/90th) for:

  * h dataset graph
  * p dataset graph

### Plots (save per eval snapshot)

1. **High-D “action vector field” (numeric)**

* For each action:

  * histogram of `||d_p||`
  * histogram of `cos(d_p, mu[a])`
* This replaces 2D arrows and is the most informative in high-D.

2. **PCA2 visualization (for intuition only)**

* Fit PCA on a sample of `p_t` (and/or deltas)
* Scatter dataset `p_t` (loop)
* Overlay:

  * planned path nodes (from delta planner)
  * start/goal markers

3. **Execution trace in true grid coordinates (7×7)**

* Plot visited cells over time for executed plan.
* Mark start/goal.

4. **Reachable fraction via BFS**

* Histogram of reachable fractions across start nodes
* And a line plot over training steps (median + bands)

5. **Node growth vs laps (especially for h)**

* Cluster using increasing amount of data (1 lap, 2 laps, …)
* Plot `num_nodes` vs #laps
* Linear growth = history/phase leakage; flat = map-like.

---

## 6) Optional: add a pose head (diagnostic-only at first)

Add `y = g(p)` with `y ∈ R^2` or `R^3`, trained with **stopgrad(p)** so it can’t destabilize p:

Losses on `y`:

* `L_delta = ||(y_{t+1}-y_t) - c[a_t]||^2` (learn c[a])
* `L_inv = ||c[L]+c[R]||^2 + ||c[U]+c[D]||^2`
* `L_noop = ||y_{t+1}-y_t||^2` for noop
* light scale regularizer

Use it to:

* produce clean 2D vector-field plots
* compare pose-head planning vs high-D p planning
* later (optionally) allow weak gradients into p if it’s stable and useful

---

## What “good” looks like over training

* Early: H-local starts passing; p delta stats noisy (`q` high), Test 1/2 fail.
* Mid: `q` drops, inverse checks improve, Test 1 starts passing.
* Later: Test 2 starts passing; node counts stabilize; reachable fractions stabilize.

If you want, I can turn this into a concrete “eval_planning.py” implementation plan (function list + pseudo-code + file outputs) that matches your training loop and replay buffer layout.
