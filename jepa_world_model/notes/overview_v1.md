From ChatGPT discussion:
https://chatgpt.com/share/6956e5c5-2f4c-8005-b0ee-0fb64a6502a8

Below is a **clean, role-separated contract** for what **`z`**, **`h`**, and **`s`** (I’ll treat `s` as your action-conditioned predictive state / successor, possibly the same as `h` depending on your code) should be trained to do — **and just as important, what they should *not* be trained to do**.

I’ll phrase each as: **job → training signals → diagnostics → anti-goals**.

---

# 1) `z` — perceptual / similarity representation

### What `z` *is*

> “What does the screen look like?”

`z` is an **observation embedding**, not a planning state.

### What `z` should be good at

* Visual similarity
* Invariance to nuisance factors
* Representing *what is visible now*

### Training signals for `z`

Use **purely observational signals**:

* **JEPA / predictive consistency**

  * same underlying observation → same `z`
  * different but temporally adjacent frames → nearby `z`
* **Augmentation invariance**

  * crop, jitter, noise, color shifts
* (Optional) **light reconstruction**

  * edges, deltas, or low-capacity decoder
  * *never* long-horizon rollout loss

### What `z` must NOT be trained on

❌ Distance-to-goal
❌ Planning loss
❌ Bellman / ranking
❌ Reward
❌ Action-conditioned prediction

Those signals will **poison perceptual similarity**.

### Diagnostics for `z`

* Nearest-neighbor retrieval: “do neighbors *look* similar?”
* Temporal smoothness vs random pairs
* Probe-only x/y (Gridworld): *diagnostic only*

### Mental model

> `z` ≈ appearance manifold
> Useful for map compression, retrieval, loop closure
> **Not a planner**

---

# 2) `h` — belief / dynamics state (hybrid, discontinuous)

### What `h` *is*

> “Everything I need to predict what happens next.”

This is your **hybrid automaton state**.

### What `h` should be good at

* Predicting future observations
* Encoding hidden state:

  * velocity
  * timers
  * coin / power-up flags
* Supporting *local* rollouts

### Training signals for `h`

These are **dynamics-aligned**, not geometric:

* **Action-conditioned JEPA prediction**

  * `h_{t+1} = f(h_t, a_t, z_{t+1})`
* **Short-horizon rollout consistency**

  * 1–2 steps (not long rollouts)
* **Auxiliary dynamics losses**

  * delta-prediction
  * residual prediction
* (Optional) stochasticity / RSSM-style losses

### What `h` must NOT be trained on

❌ Geometry / distance losses
❌ Ranking
❌ Goal conditioning
❌ A* signals

Otherwise you get gradient conflict:

* prediction wants discreteness
* geometry wants smoothness

### Diagnostics for `h`

* **Delta consistency (mode-conditioned)**

  * same action, same mode → similar deltas
* **2-step composition sanity**
* **Rollout stability**

  * does error explode in 3–5 steps?

### Mental model

> `h` ≈ hybrid belief state
> Piecewise smooth, allowed to jump
> **Predictable, not metric**

---

# 3) `s` / `g(h)` — geometry / planning representation (ranking-trained)

(I’ll call it `g` here for clarity.)

### What `g` *is*

> “Where am I in navigation space?”

This is your **planning chart**.

### What `g` should be good at

* Reachability ordering
* Goal-directed monotonicity
* Ignoring irrelevant discrete state (coins, animation, timers)

### Training signals for `g`

**Ranking / topology only**, with `stopgrad(h)`:

#### Core signal (recommended)

From trajectories:

```
(h_t, goal) < (h_{t+k}, goal)   for k > 0
```

Loss:

* ranking / margin loss
* InfoNCE-style ordering
* multi-goal ranking (stronger)

#### Optional helpers

* Bottleneck (low dimensionality)
* Mild L2 regularization
* Temporal contrastive (reachable vs unreachable)

### What `g` must NOT be trained on

❌ Reconstruction
❌ Next-state prediction
❌ Raw trajectory step regression
❌ Reward prediction (unless explicitly value-based)

### Diagnostics for `g`

* Same position, different coin history:

  * `||h1 - h2||` large
  * `||g(h1) - g(h2)||` small
* Wall / door separation:

  * visually similar but unreachable → far in `g`
* Planning test:

  * A* with `g` heuristic improves efficiency
  * does not hallucinate shortcuts

### Mental model

> `g` ≈ navigation topology
> Not exact distance, but correct ordering
> **Geometry emerges from constraints**

---

# 4) How they work together (end-to-end)

| Component | Role              | Used by                       |
| --------- | ----------------- | ----------------------------- |
| `z`       | appearance        | retrieval, map, novelty       |
| `h`       | belief + dynamics | rollout, successor generation |
| `g(h)`    | planning geometry | heuristic, graph search       |

### Planning loop

* Nodes: `h`
* Successors: `h' = f(h,a)`
* Heuristic / pruning: `g(h)` relative to goal
* Optional: `z` for deduping / visualization

---

# 5) Training order (important)

### Phase 1 — Perception

Train `z` + encoder invariances

### Phase 2 — Dynamics

Train `h` with JEPA + action prediction
Verify delta consistency

### Phase 3 — Geometry

Freeze or stopgrad `h`
Train `g` with ranking

### Phase 4 — Planning eval

Run A* / Dijkstra over learned model
Measure success + efficiency

---

# 6) One-sentence summary (the contract)

> **`z` learns what looks the same, `h` learns what happens next, and `g` learns what is closer — and none of them are allowed to steal each other’s job.**

If you want, next we can:

* turn this into a **loss table + wiring diagram**
* define **exact ranking losses** (single-goal vs multi-goal)
* discuss **how many goals per batch** is ideal
* sketch **A* pseudocode** using `(h, g)` cleanly
