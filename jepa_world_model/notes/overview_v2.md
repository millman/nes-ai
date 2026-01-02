From ChatGPT discussion:
https://chatgpt.com/share/6956e5c5-2f4c-8005-b0ee-0fb64a6502a8

Below is a **Codex-ready, implementation-oriented review** of the roles, training signals, constraints, and diagnostics for **`z`**, **`h`**, and **`s`** (a.k.a. `g(h)` geometry / planning head).
This is written to be pasted directly into a design doc or used as guidance for implementation.

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

### Explicit non-goals

* Do NOT train `z` with:

  * ranking
  * Bellman / value
  * goal-conditioned losses
  * A* or planning signals

### Diagnostics for `z`

* Nearest-neighbor retrieval: similar frames → similar `z`
* Temporal smoothness vs random frames
* Probe-only x/y prediction (Gridworld only; diagnostic)

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
* Short-horizon rollout consistency
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

### Diagnostics for `h`

**Delta consistency**

```
Δ1 = f(h, a) - h
Δ2 = f(h + Δ1, a) - (h + Δ1)
Check ||Δ1 - Δ2||
```

**Action delta clustering**

* For fixed action, cluster Δ(h, a)
* Expect small number of clusters (real modes)
* Many unstructured clusters → hidden phase variable

**Mode predictability test**

* Cluster Δ(h, a)
* Train probe: h → cluster_id
* If probe fails → fake/hidden mode

**Counterfactual consistency**

* Same h from different histories → same predictions

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

### Primary training signal: **ranking / ordering**

* Goal-conditioned ranking
* Multi-goal per trajectory window
* Local comparisons dominate

Example constraint:

```
D(g(h_near), g(h_goal)) < D(g(h_far), g(h_goal))
```

### Dense option: OT / Sinkhorn ranking

Within a window:

* Compute distances to goal
* Match distance distribution to time-to-goal ranks
* Enforces ordering + budget (anti-collapse)

### What `g` must NOT be trained on

* Reconstruction
* Next-state prediction
* Action-conditioned prediction
* Raw trajectory length regression

### Diagnostics for `g`

* Same position, different coin history:

  * ||h1 - h2|| large
  * ||g(h1) - g(h2)|| small
* Visually similar but unreachable states:

  * far in `g`
* A* heuristic sanity:

  * helps search
  * does not hallucinate shortcuts

### Mental model

* `g(h)` ≈ navigation chart
* Topological, not exact metric
* Geometry emerges from ordering constraints

---

## 4. Training Order (Strongly Recommended)

### Phase 1 — Perception

* Train encoder + `z`
* Validate visual similarity

### Phase 2 — Dynamics

* Train `h` with 1-step + short rollout losses
* Add delta-h prediction
* Diagnose composability issues

### Phase 3 — Geometry

* Freeze or stopgrad `h`
* Train `g` with ranking (triplets or OT)
* Verify nuisance collapse (coins, timers)

### Phase 4 — Planning Evaluation

* Nodes = `h`
* Successors = learned dynamics
* Heuristic / pruning = `g(h)`
* Measure success + efficiency

---

## 5. Key Invariants (Do Not Break These)

* `z` answers: “What does this look like?”
* `h` answers: “What will happen next?”
* `g` answers: “What is closer / reachable?”

If any representation answers more than its question, expect failure.

---

## 6. One-Line Summary

```
z = appearance
h = belief + dynamics (locally composable)
g(h) = planning geometry (ordering only)
```

This separation is what makes planning, ranking, and composability work together without fighting.

---


Below is a **drop-in addendum** to the earlier **Codex-ready notes**, written in plain markdown and structured so it can be pasted directly under the `z / h / g` design section.

This covers **inverse dynamics** and **edge / delta-frame prediction** precisely, without contradicting any earlier constraints.

---

# Addendum: Auxiliary Losses for Encoder Support

(Delta-Edge Prediction & Inverse Dynamics)

These auxiliary losses exist **only** to ensure the encoder (`z`) retains information needed for dynamics (`h`).
They must **never** introduce planning or geometry structure.

---

## A1. Delta-Edge (Delta-Frame) Prediction Loss

### Purpose

* Prevent the encoder from discarding small but critical visual changes
* Preserve 1–2px motion, contacts, collisions, sprite movement
* Improve dynamics learning without adding planning semantics

This is an **observation-space auxiliary**, not a latent-space constraint.

---

### Target construction (fixed, non-learned)

Compute targets directly from observations:

```
E_t     = Edge(x_t)
E_{t+1} = Edge(x_{t+1})

target_delta = E_{t+1} - E_t
```

Where `Edge(·)` is a fixed operator (e.g. Sobel, LoG, Laplacian).

Optional multi-scale variant:

```
target_delta_s = Edge(downsample_s(x_{t+1})) - Edge(downsample_s(x_t))
```

---

### Prediction head

```
delta_hat = DeltaDecoder(z_t)
```

Optional (slightly stronger controllability bias):

```
delta_hat = DeltaDecoder(z_t, a_t)
```

Constraints:

* Shallow head (few convs or MLP)
* No recurrence
* No access to `h`
* No access to goals

---

### Loss definition

Base loss:

```
L_delta = mean( |delta_hat - target_delta| )
```

Masked / weighted variant (recommended):

```
weights = |target_delta|^α     # α ≈ 0.5–1.0
L_delta = mean( weights * |delta_hat - target_delta| )
```

Purpose of weighting:

* Avoid trivial zero-prediction
* Focus learning on motion/edge regions

---

### Gradient routing

* Gradients flow into:

  * Encoder producing `z`
  * DeltaDecoder
* Gradients must NOT flow into:

  * `h`
  * `g(h)` or geometry head

---

### What this loss enforces

* `z` retains visually grounded change information
* No metric, ordering, or planning structure imposed
* No temporal shortcutting

---

### What this loss must NOT do

* Predict full next frame
* Predict latent deltas (`z_{t+1} - z_t`)
* Encode goal distance or reachability

---

### Typical weighting

```
L_total += 0.05 – 0.2 * L_delta
```

Relative to main dynamics / JEPA loss weight of `1.0`.

---

## A2. Inverse Dynamics Auxiliary Loss

### Purpose

* Encourage `z` to retain *controllable* factors
* Ensure action-relevant changes are represented
* Support dynamics learning under partial observability

This is a **dynamics-support loss**, not planning.

---

### Target construction

Ground-truth action from environment:

```
a_t
```

Inputs:

```
(z_t, z_{t+1})
```

---

### Prediction head

```
a_hat = InverseDyn(z_t, z_{t+1})
```

Constraints:

* Small MLP
* No recurrence
* No access to `h`
* No access to goals

---

### Loss definition

Discrete actions:

```
L_inv = CrossEntropy(a_hat, a_t)
```

Continuous actions:

```
L_inv = ||a_hat - a_t||^2
```

---

### Gradient routing

* Gradients flow into:

  * Encoder producing `z`
  * Inverse dynamics head
* Gradients must NOT flow into:

  * `h`
  * `g(h)` or geometry head

---

### What this loss enforces

* `z` retains information sufficient to infer action effects
* Biases representation toward controllable aspects
* Complements delta-edge loss (structure + controllability)

---

### Risks and mitigations

**Risk:** `z` becomes overly action-centric
**Mitigation:**

* Keep loss weight small
* Combine with strong augmentation invariance

---

### Typical weighting

```
L_total += 0.05 * L_inv
```

Often lower than delta-edge loss.

---

## A3. Interaction with Core Representations

### Interaction with `z`

* Both losses shape `z`
* Neither imposes geometry or planning semantics
* Both preserve information needed for `h`

### Interaction with `h`

* Indirect only (via improved `z`)
* No direct gradient
* No interference with composability constraints

### Interaction with `g(h)`

* None
* Geometry head uses `stopgrad(h)`

---

## A4. Explicit Non-Goals (Do Not Do These)

* Do NOT compute delta loss in latent space (`z_{t+1} - z_t`)
* Do NOT apply inverse dynamics to `h`
* Do NOT backprop these losses into geometry/ranking heads
* Do NOT condition these losses on goals

---

## A5. Summary Table

```
Loss                Shapes        Target space     Affects
---------------------------------------------------------------
Delta-edge loss     z             pixel / edge     encoder only
Inverse dynamics    z             action label     encoder only
Ranking / OT        g(h)           ordering         geometry head
JEPA / dynamics     h              latent future    encoder + h
```

---

## One-line summary (Codex-friendly)

```
Delta-edge and inverse-dynamics losses are small, observation-anchored auxiliaries that shape z to retain motion and controllability, without imposing geometry, planning, or metric structure.
```

---

If you want next, I can:

* merge this addendum directly into the earlier full z/h/g spec
* provide a minimal loss-combination pseudocode block
* suggest when to anneal or turn off these auxiliaries later in training
* give diagnostic plots to confirm they’re helping rather than harming
