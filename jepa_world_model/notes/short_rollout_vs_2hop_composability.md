From ChatGPT discussion:
https://chatgpt.com/s/t_695706dce31881918cd3d49aeeda377c


Below is a **Codex-oriented, detailed comparison checklist** of **(A) short rollout consistency** vs **(B) 2-hop composability**, including: definitions, what tensors are involved, what to log, what common “wrong implementations” look like, and how to auto-detect which one your code currently implements.

---

# A. Short Rollout Consistency vs B. 2-Hop Composability

(Codex Checklist + “how to tell what you implemented”)

## 0) Shared notation / assumptions

* `Enc(x) -> z`
* `f(h, z, a) -> h_next`  (or `f(h, a) -> h_next` if z not used)
* `Dec(h or z) -> x_hat`  (optional)
* You have sequences: `(x_t, a_t, x_{t+1}, a_{t+1}, x_{t+2}, ...)`
* You may store “target” latents `h_{t+1}^*`, `h_{t+2}^*` computed from encodings (stopgrad targets) or from a separate target network (JEPA-style).

Important: Many codebases use `h_target[t] = stopgrad(EncoderTarget(x_t))` rather than “true h”. That’s fine; just be consistent.

---

# 1) SHORT ROLLOUT CONSISTENCY (multi-step prediction)

## 1.1 Definition (what it is)

Unroll the model for `K` steps and match predicted state (or predicted decoded frame) at step `K` to a target at step `K`.

### Core computation

```
h_hat = h_t
for i in 0..K-1:
    h_hat = f(h_hat, z_{t+i}, a_{t+i})   # or f(h_hat, a_{t+i})
```

### Typical losses

**State-space rollout loss**

```
L_rollout_state = || h_hat_{t+K} - h_target_{t+K} ||
```

**Frame-space rollout loss**

```
x_hat_{t+K} = Dec(h_hat_{t+K}) or Dec(z_hat_{t+K})
L_rollout_img = || x_hat_{t+K} - x_{t+K} ||  (or edge loss, etc.)
```

## 1.2 What it enforces

* Endpoint accuracy after K steps
* Reduces drift/explosion
* Improves long-ish rollouts

## 1.3 What it does NOT enforce

* Reuse of action semantics
* Algebraic consistency of deltas
* No guarantee that “RIGHT then RIGHT” behaves like two reusable effects

## 1.4 How to detect it in code

Look for:

* explicit unrolling loop over time steps
* a loss comparing `t+K` prediction to `t+K` target
* no explicit comparison between “two different ways to compute the same 2-step state”

**Key signature**

* The loss depends on a single path: `f(f(h_t,a_t),a_{t+1})` (and longer)
* There is no second expression for the same time index.

---

# 2) 2-HOP COMPOSABILITY (algebraic self-consistency)

## 2.1 Definition (what it is)

Compare two *different internal computations* that should agree for the same 2-step future.

There are multiple equivalent formulations. The defining feature is:

> The loss compares **Path A** vs **Path B** for the same future time index.

### Path A (sequential)

```
h1 = f(h_t, z_t, a_t)
h2 = f(h1,  z_{t+1}, a_{t+1})
```

### Path B (composed effects)

This requires an explicit notion of “delta” or “effect” that can be composed.

#### If your model predicts deltas explicitly:

```
dh0 = Δ(h_t,  z_t, a_t)
h1b = h_t + dh0

dh1 = Δ(h1b, z_{t+1}, a_{t+1})
h2b = h1b + dh1
```

Then the 2-hop loss can be framed as:

```
L_2hop = || h2 - h2b ||
```

#### Equivalent effect-consistency variant (common and simpler):

Compare *effects* rather than states:

Define:

```
dh0_A = h1 - h_t
dh1_A = h2 - h1
```

Then add a constraint that the “second step effect” is stable under reuse. Two common choices:

**(B1) Same-action reuse test (if a_{t+1}=a_t or you sample same action twice)**

```
h1 = f(h, a)
h2 = f(h1, a)
dh0 = h1 - h
dh1 = h2 - h1
L = || dh1 - dh0 ||
```

This is the clearest “first RIGHT vs second RIGHT” composability check.

**(B2) General 2-hop consistency (always available)**
Compare:

* direct 2-step predicted state `h2`
  vs
* state reconstructed by “sum of predicted deltas” using the same network outputs

This requires delta outputs to exist (recommended).

## 2.2 What it enforces

* Reuse / stable meaning of actions
* No hidden phase variables (odd/even step hacks)
* Discourages time-skipping / shortcutting representations
* “Local algebra” correctness (self-consistency)

## 2.3 What it does NOT enforce

* Long-horizon accuracy by itself
* Avoiding drift over 20–100 steps (you can still drift slowly)

## 2.4 How to detect it in code

Look for:

* two expressions for a 2-step quantity
* a loss term that compares them
* explicit use of deltas or effect comparisons
* special-case “apply same action twice” check

**Key signature**

* Something like:

  * `h2 = f(f(h_t, a_t), a_{t+1})`
  * `h2_alt = ...` (constructed differently)
  * `loss += norm(h2 - h2_alt)` (or equivalent delta comparisons)

If you only see “unroll 2 steps and match h_target[t+2]”, that is rollout, not 2-hop composability.

---

# 3) Common confusions / incorrect implementations

## 3.1 “K=2 rollout loss” is NOT 2-hop composability

This is the #1 confusion.

If your code does:

```
h2 = f(f(h_t, a_t), a_{t+1})
loss = || h2 - h_target_{t+2} ||
```

That is **rollout consistency**, even though K=2.

2-hop composability requires comparing `h2` to an alternative internally-composed `h2_alt`, not to a target.

## 3.2 Using only decoded frames for “2-hop”

If you do:

```
x_hat2 = Dec(h2)
loss = || x_hat2 - x_{t+2} ||
```

Still rollout consistency. It says nothing about internal algebra.

---

# 4) “Which is preferred?” (implementation guidance)

If your goal is “h is composable for planning”:

* **2-hop composability is critical**
* short rollout is optional and small

Recommended mix:

```
L_total =
  L_1step_pred
+ λ_2hop * L_2hop_composability
+ λ_roll * L_rollout_short    # optional
```

Where:

* `λ_2hop` is non-zero early (e.g., 0.1–1.0 depending on scale)
* `λ_roll` is small (e.g., 0.05–0.2) or off unless needed

---

# 5) Concrete “Codex checks” to automatically identify what exists

Ask Codex to search for patterns:

## 5.1 Search patterns for rollout loss

* loops: `for k in range(K): h = ...`
* names: `rollout`, `unroll`, `k_step`, `multi_step`, `h_pred_k`
* losses against `t+k` targets: `loss += mse(h_pred, h_target[t+k])`

## 5.2 Search patterns for 2-hop composability

* two-step computed twice:

  * `h2 = f(f(h, ...), ...)`
  * `h2_alt = ...`
* delta variables:

  * `dh = ...`
  * `delta_h`
  * `residual`
  * `h_next = h + dh`
* explicit comparisons between deltas:

  * `loss += norm((h2-h1) - (h1-h0))` (same-action case)
  * `loss += norm(h2 - (h0 + dh0 + dh1))`

If none of these exist, you probably only have rollout consistency.

---

# 6) Minimal log outputs to confirm behavior (runtime verification)

Implement these debug metrics:

## 6.1 Rollout error curve (K=1..N)

* `err_k = || h_hat_{t+k} - h_target_{t+k} ||`
  Expect increasing curve if drift exists.

## 6.2 Composability error (2-hop)

* `comp_err = || h2 - h2_alt ||`
  or for same-action reuse:
* `reuse_err = || (h2-h1) - (h1-h0) ||`

If rollout error is low but composability error is high → hidden phase / shortcutting.

---

# 7) Final “tell me which is implemented” decision rule

* If the loss compares **predicted** `h_{t+k}` to a **target** at `t+k`: **rollout consistency**
* If the loss compares two **different internal computations** of the same `t+2` predicted state/effect: **2-hop composability**

---

If you paste a snippet of your current loss code (or file names), I can point to the exact lines/patterns and say “this is rollout” vs “this is composability,” but the above should be enough for Codex to reliably detect it.
