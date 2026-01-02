From ChatGPT discussion:
https://chatgpt.com/share/6956e5c5-2f4c-8005-b0ee-0fb64a6502a8

Below is a **Codex-ready, implementation-oriented review** of the roles, training signals, constraints, diagnostics, and proposed refactors for **`z`**, **`h`**, and **`s`** (a.k.a. `g(h)` geometry / planning head).
This version incorporates the latest decisions: **single-stage training**, **no A*/Dijkstra evaluation**, **pixel-delta auxiliary loss**, and **delta/delta-recon rows in batch visualization**.

---

# Representation Roles and Training Contracts

This system intentionally separates **appearance**, **dynamics**, and **planning geometry**.
Each representation has a narrow job. Mixing jobs causes failure modes.

---

## 1. `z` — Observation / Appearance Representation

### What `z` is

* An embedding of the current observation (frame or frame stack)
* Represents **what the screen looks like**
* No memory, no planning semantics

### What `z` should encode

* Visual similarity
* Spatial layout as seen in pixels
* Objects, textures, sprites
* Invariance to nuisance transformations

### What `z` must NOT encode

* Distance-to-goal
* Reachability
* Action effects
* Planning geometry
* Rewards or value

### Training signals for `z`

* JEPA-style observation consistency
* Augmentation invariance (crop, jitter, noise)
* Optional: very lightweight reconstruction (edges or deltas only)
* **Pixel-delta auxiliary loss** (short-horizon only)

### Pixel-delta auxiliary loss (new)

* **Goal:** encourage z to preserve short-term visual change without leaking dynamics/geometry.
* **Targets:**

  * `delta_target = x_{t+1} - x_t`
  * `delta_pred = recon(z_{t+1}) - recon(z_t)` (or a tiny delta decoder)
* **Loss:** L1 or hardness-weighted L1 on delta only.
* **Constraints:** no action-conditioning, no long-horizon rollout.

### Explicit non-goals

* Do NOT train `z` with:

  * ranking
  * Bellman / value
  * goal-conditioned losses
  * A* or planning signals
  * action-conditioned prediction

### Diagnostics for `z` (web viewer)

* Nearest-neighbor retrieval / embedding projection (reuse `save_embedding_projection`).
* Reconstruction grids + hard samples (reuse `save_input_batch_visualization`, `save_hard_example_grid`).
* Temporal smoothness + random-pair distance histogram (extend `vis_vis_ctrl`).
* **Delta recon grid** (new; see visualization section).

### Mental model

* `z` ≈ appearance manifold
* Used for retrieval, map compression, loop closure
* **Never used directly for planning**

---

## 2. `h` — Belief / Dynamics State (Predictive State)

### What `h` is

* The internal state used to predict the future
* Encodes **everything needed for dynamics**
* Can be discontinuous and hybrid

### What `h` should encode

* Velocity, acceleration
* Contact / collision state
* Coins, power-ups, switches
* Timers, hidden variables
* Any latent needed for accurate prediction

### What `h` must NOT be forced to be

* Globally smooth
* Metric
* Geometry-like
* Distance-preserving

### State update (recommended form)

```
delta_h = DeltaNet(h_t, action_t, z_{t+1})
h_{t+1} = h_t + delta_h
```

Residual form is preferred to avoid teleporting / re-anchoring.

### Training signals for `h`

* 1-step prediction loss (JEPA or latent prediction)
* Short-horizon rollout consistency (optional)
* Optional stochastic latent losses (RSSM-style)
* Optional delta prediction auxiliary losses

### Required property: **local composability**

Within the same semantic mode:

```
f(f(h, a), b) ≈ h + Δ(h, a) + Δ(h + Δ(h, a), b)
```

This does NOT require smoothness across:

* collisions
* teleports
* power-up changes

It DOES require:

* reuse of action semantics
* no hidden phase / timestep hacks

### What `h` must NOT be trained on

* Geometry losses
* Ranking losses
* Goal distance
* Planning heuristics

Those cause gradient conflict with prediction.

### Diagnostics for `h` (web viewer)

* Two-step composition error (reuse `vis_vis_ctrl` applied to h).
* Stability plots and knn distance curves (reuse `vis_vis_ctrl`).
* Cycle error plots (reuse `compute_cycle_errors`, `save_cycle_error_plot`).
* Action-conditioned delta clustering (new: per-action delta means, entropy).

### Mental model

* `h` ≈ hybrid automaton state
* Piecewise predictable
* Allowed to jump at real events
* **Predictable, not geometric**

---

## 3. `s` / `g(h)` — Geometry / Planning Representation

(Here called `g(h)` for clarity.)

### What `g` is

* A projection of `h` into a **planning space**
* Used for ordering, heuristics, graph search
* Lower dimensional than `h`

### What `g` should encode

* Reachability ordering
* Topology (walls, doors, bottlenecks)
* Monotonic progress toward goals

### What `g` should ignore

* Coin history
* Animation phase
* Velocity (unless it changes reachability)
* Timers
* History quirks

### Training rule (critical)

```
e = g(stopgrad(h))
```

Never backprop geometry loss into `h`.

### Primary training signal: **goal-conditioned ranking / ordering**

* Triplet / InfoNCE / OT ranking within trajectory windows
* Local comparisons dominate

Example constraint:

```
D(g(h_near), g(h_goal)) < D(g(h_far), g(h_goal))
```

### What `g` must NOT be trained on

* Reconstruction
* Next-state prediction
* Action-conditioned prediction
* Raw trajectory length regression

### Diagnostics for `g` (web viewer)

* Ranking accuracy (pairwise ordering, NDCG-style).
* Smoothness / stability plots (reuse `vis_vis_ctrl` for s).
* Nuisance-collapse check: same position, different coin history -> small s distance (if labels available).

### Mental model

* `g(h)` ≈ navigation chart
* Topological, not exact metric
* Geometry emerges from ordering constraints

---

# Single-Stage Training (No Phases)

Training runs in a **single stage** with careful loss gating to enforce role separation.

### Keep

* JEPA consistency for z
* Optional light recon + sigreg
* h dynamics losses (h2z, delta_z)
* geometry ranking loss on s with stopgrad(h)

### Remove / disable by default

* Action prediction from z deltas
* z rollout losses
* s rollout losses
* action prediction from s deltas
* delta_s losses
* adjacency losses as geometry

### Replace

* Adjacency losses -> goal-conditioned ranking loss on `s` (new module, e.g. `loss_geometry.py`).

---

# Diagnostics Additions (Web Viewer)

## z tab

* Embedding projection (existing)
* Input recon grid (existing)
* Hard samples (existing)
* **Delta target / delta recon rows** (new)
* Temporal smoothness histogram (new)

## h tab

* Two-step composition error (existing vis_vis_ctrl machinery)
* Stability plot and knn distance curves (existing)
* Cycle error plots (existing)
* Action-delta clustering summary (new)

## s tab

* Ranking accuracy curves (new)
* Smoothness / stability (existing vis_vis_ctrl for s)
* Nuisance-collapse checks (new, if labels)

---

# Batch Visualization Additions (New Requirement)

Include **two additional rows** in the batch visualization output:

1. **Delta target** (pixel delta between consecutive frames)
2. **Delta recon** (pixel delta between consecutive reconstructions)

Notes:

* Use existing input batch visualization path; add rows instead of creating a new figure.
* Keep deltas short-horizon only (t to t+1).
* Use a fixed normalization/contrast for deltas so outputs are comparable across steps.

---

# One-Line Summary

```
z = appearance (+ pixel-delta aux)
h = dynamics (locally composable)
g(h) = geometry (ordering only, stopgrad)
```

This separation is what makes planning, ranking, and composability work together without fighting.
