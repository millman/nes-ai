# Base + Residual Pose Increments (NOOP Baseline, Soft Cancellation)

## Goal
Prevent "p as a timer" and make p a usable odometry/geometry in environments with background motion (scrolling, moving platforms) **without** hard-coded inverse action pairs or brittle gates.

This note captures the discussion about:
- splitting pose increments into baseline + control residual,
- using NOOP to define the baseline (no-control input),
- learning inverse/cancel behavior from data via soft multi-step residual cancellation,
- avoiding hard thresholds by using learned/statistical weighting.

---

## 1) Core idea: baseline drift + control residual

**Total motion is not invertible** in scrolling worlds, but **control effects often are**. So we separate them:

- **Baseline** = motion that happens regardless of control input
- **Residual** = motion caused by control input

### Option A: Explicit split (two heads)
```
Δp_base = g(h_t)               # action-independent drift
Δp_res  = r(h_t, a_t)          # control residual
Δp_total = Δp_base + Δp_res
p_{t+1} = p_t + Δp_total
```

### Option B: Implicit split (single head + NOOP)
If NOOP is a real action (all zeros), avoid a second head:
```
Δp_total(h,a) = r(h, a)
Δp_base(h)    = r(h, NOOP)
Δp_res(h,a)   = r(h, a) - r(h, NOOP)
```

This makes the **baseline** and **residual** computable from a single action-conditioned projector.

---

## 2) NOOP semantics

NOOP means **no control input**, not “no motion”. In scrolling or physics-driven scenes:
- baseline motion can be nonzero under NOOP
- residual should be near zero under NOOP

**NOOP residual-zero constraint**:
```
L_noop = E[ ||Δp_res|| | a = NOOP ]
```

---

## 3) What to enforce on total vs residual

**Total (baseline + residual):**
- odometry and rollout consistency
  - `action_delta_dp`
  - `rollout_kstep_p`

**Residual only:**
- additivity / composability
- inverse/cancel structure
- scale anchoring (optional)

The reason: background motion breaks inverse/cancel in total p, but not in control residual.

---

## 4) Soft multi-step cancellation (no hard gate)

We want cancellation to be **learned from data**, not from explicit inverse labels. The weak assumption is:

> If the system ends up in (approximately) the same place, then **control residuals should sum to ~0**.

### Cancellation loss (short window, soft weighted)
For a short window of length k (e.g., 4–6):
```
R = Σ_{i=0..k-1} Δp_res(t+i)
```

We weight this by **relative z similarity** (no fixed threshold):

**Rank-based weight (preferred)**
```
d = ||z_{t+k} - z_t||
w = exp(-rank(d) / τ_rank)
L_cancel = w * ||R|| / (mean||Δp_res|| + eps)
```

**Z-score weight (alternative)**
```
w = exp(-(d - mean_d) / std_d)
```

This avoids brittle gates and adapts to learned z statistics.

**Key properties**
- No hard ε threshold
- Most weight goes to “most similar” windows
- Works even when many windows do **not** cancel

---

## 5) Why this kills timer-like p

A timer-like p implies residuals that never cancel. The cancellation loss makes that impossible whenever the model predicts locally similar states. The model is forced to learn **directional, non-monotonic residuals**.

---

## 6) How this compares to inverse_cycle_dp

**Existing inverse_cycle_dp** (current code):
- learns an explicit inverse action mapping via `inverse_action_head`
- enforces a **2-step** cycle in total p

**Residual cancellation (proposed):**
- no explicit inverse labels
- multi-step cancellation
- applies to **residual only**
- compatible with scrolling or forced motion

**Gridworld note:**
In simple deterministic gridworlds (no forced motion), `inverse_cycle_dp` + `additivity_dp` on total p is often sufficient and easier to train. The residual/cancel system is mainly for environments where total motion is not invertible.

---

## 7) Integration checklist (implicit split)

1) Compute baseline/residual from the same action head:
```
Δp_base = r(h, NOOP)
Δp_res  = r(h, a) - r(h, NOOP)
Δp_total = r(h, a)
```

2) Apply existing odometry losses to **Δp_total**:
- `action_delta_dp`
- `rollout_kstep_p`

3) Add NOOP residual zero loss:
- `L_noop`

4) Move additivity/inverse constraints to **Δp_res**:
- `additivity_dp` on residual
- `inverse_cycle_dp` (if kept) on residual, not total

5) Add soft multi-step cancellation loss:
- `L_cancel` with rank- or stats-based weights

6) (Optional) `scale_dp` applies to **Δp_res**, not total.

---

## 8) Suggested starting weights (ballpark)

- `action_delta_dp`: 0.05–0.2
- `rollout_kstep_p`: 0.05–0.2 (k=3–8)
- `additivity_dp` (residual): 0.01–0.05
- `inverse_cycle_dp` (residual, optional): 0.01–0.05
- `noop_residual_zero_p`: 0.05–0.2
- `cancel_residual_p`: 0.01–0.05

Ramp cancellation after odometry stabilizes.

---

## 9) Diagnostics to confirm it’s working

- **NOOP residual norm:** E[||Δp_res|| | NOOP] → small
- **Residual inverse check:** Δp_res(LEFT) + Δp_res(RIGHT) ≈ 0 (in free regimes)
- **Timer leak probe:** linear probe on Δp_res to predict timestep should fail
- **Short-window cancel:** weighted ||ΣΔp_res|| decreases over training

