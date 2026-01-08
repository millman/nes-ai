Below is a **new, self-contained algebra document** written in the *current* framing:

> **S is the state you plan on, map with, and treat like pose/odometry.**
> Algebra is not a regularizer; it is the *meaning* of S.

I’ve explicitly included **design diagnostics**—not just losses—so this doc can be used both as a **spec** and a **debugging guide**.

---

# Algebra for Planning on S

### (Design, Losses, and Diagnostics)

## 0. Purpose (what this doc is for)

This document defines what it means for a learned state **S** to be:

* usable for **planning**
* usable for **map building**
* compatible with **SLAM-like reasoning**

It answers three questions:

1. **What algebraic structure must S satisfy?**
2. **What losses enforce that structure?**
3. **How do we diagnose whether it is actually satisfied?**

---

## 1. Roles of the representations (non-negotiable)

We assume three distinct roles:

```
z : perceptual snapshot (what I see)
h : belief / memory / dynamics (what is happening)
s : geometric planning state (where I am, algebraically)
```

Only **s** is required to obey strong algebraic structure.

z and h exist to *support* s.

---

## 2. What S is (algebraically)

We treat S as a **local coordinate chart** over the agent’s reachable configuration space.

Formally (locally):

```
s ∈ R^d
s_{t+1} = s_t + Δs_t
Δs_t = g(s_t, a_t)        (or g(h_t, a_t))
```

Interpretation:

* `s_t` = pose / coordinate
* `Δs_t` = motion primitive
* addition = composition of actions
* distance in s-space = planning cost (locally)

This is the *same abstraction* used in odometry and SLAM.

---

## 3. Algebraic properties S must satisfy

### A. Local additivity (mandatory)

For short horizons in free space:

```
s_{t+k} ≈ s_t + Σ_{i=0..k-1} Δs_{t+i}
```

This is the foundation of:

* rollout stability
* straight-line trajectories
* additive path cost

---

### B. Local inverses (very important)

If an action has an inverse:

```
Δs(a⁻¹) ≈ −Δs(a)
```

This ensures:

* reversibility
* symmetric distances
* backtracking in planning

---

### C. Path independence / cycle consistency (critical for mapping)

If two paths reach the same place:

```
Σ Δs(path A) ≈ Σ Δs(path B)
```

This is exactly the **pose-graph constraint** in SLAM.

---

### D. Controlled algebra breaking (intentional)

Algebra is **not global**.

It is allowed to break under:

* contact / collision
* blocked motion
* teleport / warp
* discrete mode transitions

So the real rule is:

> S obeys a **local Abelian group structure in free space**,
> with **gated violations** elsewhere.

---

## 4. Losses that enforce this algebra

### 4.1 Residual update (structural, not a loss)

```
s_{t+1} = s_t + Δs_t
```

If you do not do this, S is not an odometry state.

---

### 4.2 1-step odometry consistency

```
L_s_1 = || (s_t + Δs_t) − stopgrad(s_{t+1}) ||^2
```

Purpose:

* makes Δs meaningful
* prevents teleporting via projection

---

### 4.3 k-step composition loss (core algebra loss)

```
ŝ_{t+k} = s_t + Σ_{i=0..k-1} Δs_{t+i}
L_s_k = || ŝ_{t+k} − stopgrad(s_{t+k}) ||^2
```

k = 2–5 is usually sufficient.

This enforces:

* additivity
* associativity (path independence over short loops)

---

### 4.4 Inverse / cycle loss (gated)

If inverse actions exist:

```
L_inv = || Δs(a) + Δs(a⁻¹) ||^2
```

Or via rollout:

```
L_cycle = || rollout(s_t, [a, a⁻¹]) − s_t ||^2
```

Gate this loss when:

* contact occurs
* motion is blocked

---

### 4.5 Anchoring to perception (anti-drift)

Pure odometry drifts. You need anchors.

Two common forms:

**(A) Alignment head**

```
L_align = || q(s_t) − stopgrad(z_t) ||^2
```

**(B) Loop closure**
If two observations are perceptually similar:

```
L_loop = w_ij * || s_i − s_j ||^2
```

This turns algebra into a **pose graph with loop closures**.

---

### 4.6 Δs regularization (stability)

```
L_mag = ||Δs_t||^2
```

Keeps integration stable and prevents runaway drift.

---

## 5. Design diagnostics (this is new and important)

Losses alone are not enough.
You must *measure* whether S actually behaves algebraically.

Below are **must-have diagnostics**.

---

### Diagnostic 1: Straight-line test (local linearity)

**Test**

* Pick a state `s_t`
* Roll out repeated same action: `a, a, a, ...`
* Plot `s_t, s_{t+1}, s_{t+2}, ...`

**Expected**

* Points lie approximately on a straight line
* Step length roughly constant (until contact)

**Failure modes**

* Curving → Δs depends incorrectly on history
* Expanding → drift / scale instability
* Zig-zag → sign or inverse inconsistency

---

### Diagnostic 2: k-step error vs k (composition quality)

**Measure**

```
e(k) = || (s_t + Σ_{i=0..k-1} Δs_{t+i}) − s_{t+k} ||
```

Plot e(k) vs k.

**Expected**

* roughly linear or sublinear growth
* no sudden jumps in free space

**Failure**

* superlinear growth → algebra not composing
* oscillations → inverse inconsistency

---

### Diagnostic 3: Inverse symmetry

For inverse action pairs:

```
cos(Δs(a), −Δs(a⁻¹))  ≈ 1
```

**Expected**

* high cosine similarity in free space

**Failure**

* biased drift
* planner prefers one direction

---

### Diagnostic 4: Local commutativity probe (optional)

Check:

```
Δs(a) + Δs(b)  vs  Δs(b) + Δs(a)
```

**Expected**

* similar only for small, free-space moves

**Important**

* This is a *probe*, not a hard requirement
* Strong violation everywhere → twisted geometry
* Mild violation → physics / constraints (fine)

---

### Diagnostic 5: Loop closure residuals (mapping sanity)

If two states look the same perceptually:

```
|| s_i − s_j ||
```

**Expected**

* small (after alignment)
* decreases over training

**Failure**

* map tearing
* inconsistent planning distances

---

### Diagnostic 6: Planner sanity check (end-to-end)

Build a small graph in s-space:

* nodes = visited states
* edges = Δs with cost = ||Δs||

Run A* or Dijkstra.

**Expected**

* planned paths resemble real trajectories
* heuristic distances are admissible

If this fails, **algebra is broken**, even if losses look good.

---

## 6. How this differs from earlier algebra discussions (explicit)

Earlier algebra work focused on:

* z or h
* learning stability
* diagnostic curiosity

This document asserts:

* **S is the algebraic object**
* algebra violations = planning failures
* losses correspond to **pose-graph constraints**
* diagnostics correspond to **SLAM sanity checks**

This is a categorical shift, not an incremental one.

---

## 7. One-paragraph summary (keep this)

> S is a learned coordinate system over the agent’s configuration space.
> Actions induce local, approximately additive transformations in S.
> Planning, mapping, and SLAM rely on this algebra being correct in free space and allowed to break under contact or discrete transitions.
> Losses enforce local additivity and consistency; diagnostics verify that S actually behaves like geometry.

---

If you want next, I can:

* turn this into a **checklist you can paste into your training code**
* map your *current* losses to these algebra requirements
* or design a **single composite “algebra score”** from the diagnostics
