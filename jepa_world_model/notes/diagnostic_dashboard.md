Here’s a **single diagnostic dashboard** that’s “one page, always-on” and will tell you (a) whether z/h/s are doing their jobs, and (b) *why* rollouts flicker when they don’t. I’ll describe it as a layout + exactly what each panel computes, so you can hand it to Codex.

---

# JEPA z/h/s Diagnostic Dashboard (one screen)

## Row 0 — “Status strip” (numbers only)

Show these as live scalars (current step + EMA):

1. **Recon**: `L_recon`
2. **1-step latent dyn**: `L_s_1` and `L_z_1` (if you have both)
3. **2-step latent dyn**: `L_s_2` (or `L_cycle`)
4. **Norms**: `||z||`, `||h||`, `||s||` (mean and p95)
5. **Drift**: `E[||h_{t+1}-h_t||]` (and same for s)
6. **Action accuracy (optional)**: inverse dynamics acc on s: `Acc(ID_s)`

**Why:** these instantly catch “z tiny / h exploding”, “h drifting”, and “2-step blows up”.

---

## Row 1 — Rollout sanity (what you look at first)

### Panel A: Ground-truth vs rollout filmstrip (images)

A 2×N strip (N=8 or 12):

* top: `x_t, x_{t+1}, …, x_{t+N}` (ground truth)
* bottom: `x̂_t (optional), x̂_{t+1|roll}, …, x̂_{t+N|roll}` (decoded from latent rollout)

Annotate each frame with:

* action taken
* `||z||, ||h||, ||s||` (tiny text)

**Why:** shows flicker/hallucination immediately, and whether it correlates with latent norm drift.

### Panel B: “Rollout divergence curves” (line plot)

For k = 1..N:

* **pixel error**: `E[ ||x_roll(k) - x_gt(k)|| ]`
* **latent error** (preferred): `E[ ||s_roll(k) - s_gt(k)|| ]`

**Why:** tells you if errors compound (dynamics drift) vs stay flat (decoder issue).

---

## Row 2 — s: action algebra health (geometry)

### Panel C: Straight-line plot (2D)

Pick 3 random start states and 2 actions (e.g. RIGHT and LEFT). For each:

* roll out in latent space: `s_0 … s_K`
* PCA-project to 2D on the fly (fit PCA on a batch of s’s)
* plot trajectory with arrows

**Pass:** repeated RIGHT looks like a ray; LEFT is opposite ray; near walls it stops.

### Panel D: Delta alignment heatmap (per-action)

For each action `a`:

* collect `Δ_i(a) = s_{t+1} - s_t` over a batch where action=a
* compute mean delta `μ_a`
* report:

  * `mean cos(Δ_i, μ_a)`
  * `std cos(Δ_i, μ_a)`
  * `mean ||Δ_i||`

Render as a small table or bar plot.

**Pass (toy gridworld):**

* cos mean > 0.95, std small
* step norms consistent (except at borders)

### Panel E: Cycle error (numbers + histogram)

Compute for opposite actions (RIGHT then LEFT, UP then DOWN):

* `cycle = || S(S(s, RIGHT), LEFT) - s ||`

Show mean + p95 + histogram.

**Why:** catches curvature and contact/boundary issues.

---

## Row 3 — z: perceptual latent health

### Panel F: Same-frame consistency (scatter / histogram)

Take one frame `x` and encode it M times (M=8) under whatever stochasticity you use (dropout/augment/noise).
Plot histogram of:

* `||z_i - mean(z)||`
  Also show `cos(z_i, mean(z))`.

**Pass:** very tight distribution.

If this fails, rollouts *will* flicker even if everything else is perfect.

### Panel G: Visual similarity monotonicity (line plot)

In your toy env you can generate controlled perturbations:

* shift object by d pixels (d=0..D)
  Compute:
* `dist_z(d) = E[ || z(x) - z(shift(x,d)) || ]`
  Plot dist vs d.

**Pass:** monotonic increasing, smooth.

### Panel H: Path-independence (two bars)

Construct two different action sequences that end in the same true state (in gridworld this is easy):

* path A: RIGHT×k then LEFT×k
* path B: NOOP×(2k) (or alternate)
  Compare:
* `||z_end(A) - z_end(B)||`
* `||s_end(A) - s_end(B)||`
  Optionally also decoded image difference.

**Why:** detects “z or s leaks history” and “h is being used when it shouldn’t”.

---

## Row 4 — h: memory health

### Panel I: Zero-out ablation (two rollout divergence curves)

Run two latent rollouts from same start:

* normal
* with `h_t := 0` (or running mean) at every step

Plot `s-error(k)` and/or `pixel-error(k)` for both.

**Pass in fully observable toy env:** curves nearly identical.

### Panel J: h drift vs action (small table)

For each action `a`:

* `E[ ||h_{t+1} - h_t|| | a ]`

**Pass:** small and roughly constant (unless you intentionally added hidden velocity/contact).

### Panel K: h scale stability (time series)

Plot `mean ||h||` and `p95 ||h||` over training steps.

**Why:** catches the “h explodes slowly” failure mode before images break.

---

# Implementation notes (so it’s practical)

* Sample a **fixed evaluation batch** every N steps (e.g. every 500 training steps). Keep it deterministic so trends are real.
* Prefer **latent-space comparisons** (`s_roll` vs `s_gt`) over pixel-space for diagnosing drift vs decoder sensitivity.
* Use **PCA-on-batch** for s trajectory plots; don’t try to maintain a global embedding early on.

---

# Reading the dashboard quickly (common patterns)

* **Filmstrip flickers + s-error grows with k** → dynamics drift (add/weight 2-step, normalize, reduce h influence)
* **Filmstrip flickers + s-error flat but pixel-error grows** → decoder off-manifold sensitivity (stronger recon manifold, reduce stochasticity, stabilize z)
* **Zero-out h changes everything in toy env** → h is doing the wrong job (add h-stability, reduce conditioning of S_dyn on h, or gate h)
* **Straight-line fails but ID accuracy is high** → inverse dynamics is not enforcing compositionality (need 2-step / cycle)

---

If you tell me what you’re using to decode (Dec(z) vs Dec(z,s) vs Dec(h,s)), I can suggest the exact minimal set of panels you can start with (often 8–10 panels is enough) and the most informative “red/green” thresholds for your setup.
