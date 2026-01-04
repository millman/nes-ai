https://chatgpt.com/share/6959f25d-56ac-8005-85b8-1a31804c09df

# What “action algebra” means (one clear intuition)

An **action algebra** is just the set of rules that describe:

• what an action *does* to state
• how actions *combine over time*
• when actions *cancel or depend on order*

Losses don’t just make predictions correct — they **force actions to behave in consistent, reusable ways** in latent space.

---

# Notation (ASCII only)

x_t        = observation (image)
z_t        = E(x_t)                 perceptual embedding
h_t        = recurrent / memory state
h_{t+1}    = U(z_t, h_t, a_t)        state update

dz_t       = z_{t+1} - z_t
dh_t       = h_{t+1} - h_t

Pz(h,a)    = JEPA-style predictor for z
e(a)       = learned prototype vector per action
E(h,a)     = learned state-conditioned delta model

---

# Level 0 — No algebra (black-box dynamics)

### Intuition

Gridworld or Mario:

> “Given state + action, predict what happens next.
> I don’t care how the state space is shaped.”

Actions are just inputs. Geometry can be arbitrary.

### Losses

Reconstruction (anchor perception):

```
L_recon = || D(z_t) - x_t ||
```

Dynamics / JEPA:

```
L_jepa = || Pz(h_t, a_t) - stopgrad(z_{t+1}) ||^2
```

### What you get

• Correct rollouts possible
• Actions used causally
• NO compositional meaning
• Planning is fragile

This is fine for video prediction, not for planning.

---

# Level 1 — Fixed translation algebra (Gridworld)

### Intuition

Gridworld:

> “Each action always means the same move.”

LEFT always shifts left by one cell.
RIGHT always shifts right.

### Enforced idea

Actions are **fixed vectors** in latent space.

### Loss

```
L_delta_proto = || (h_{t+1} - h_t) - e(a_t) ||^2
```

(You can do this on z instead of h in very simple worlds.)

### What you get

• LEFT + LEFT ≈ twice LEFT
• LEFT + RIGHT ≈ cancel
• Distances mean something
• A* works trivially

### Where it breaks

Mario:
• velocity
• collisions
• slopes
• momentum

A single vector per action is too rigid.

---

# Level 2 — State-conditioned translation algebra (⭐ Mario sweet spot)

### Intuition

Mario:

> “Actions are moves, but how big or effective they are depends on state.”

Examples:
• LEFT while running → big move
• LEFT into wall → no move
• JUMP while airborne ≠ JUMP on ground

### Enforced idea

Actions define **local directions** that depend on state.

### Core loss (put this on h)

```
L_delta_h = || (h_{t+1} - h_t) - E(h_t, a_t) ||^2
```

### Optional perceptual probe (weaker)

```
L_delta_z = || (z_{t+1} - z_t) - Ez(h_t, a_t) ||^2
```

### What you get

• Actions behave similarly in nearby states
• Velocity & momentum live in h
• Local geometry is predictable
• Short-horizon planning works

This is the **minimum algebra Mario needs**.

---

# Level 3 — Compositional algebra (multi-step consistency)

### Intuition

Gridworld:

> “Two small moves equal one big move.”

Mario:

> “Short sequences compose predictably (when physics allows).”

### 3a. k-step rollout consistency

Roll out the model and compare to reality:

```
L_kstep = sum_{i=1..k} || Pz(h_hat_{t+i-1}, a_{t+i-1})
                          - stopgrad(z_{t+i}) ||^2
```

### 3b. Explicit additivity (stronger)

Let dh_hat_t = E(h_t, a_t):

```
L_add = || (h_{t+2} - h_t) - (dh_hat_t + dh_hat_{t+1}) ||^2
```

### What you get

• LEFT twice ≈ longer LEFT
• Rollouts stay straight
• Planning horizon increases

This is where “algebra” becomes *usable*.

---

# Level 4 — Cancellation / inverse actions

### Intuition

Gridworld:

> “LEFT then RIGHT returns you.”

Mario:

> “Only cancels when not blocked.”

### Loss

```
L_cycle = || rollout(h_t, [a, a_inv]) - h_t ||^2
```

### Notes

• Must be gated (no walls, no collisions)
• Low weight

### What you get

• Loop closure
• Cleaner graph topology

Optional, but powerful for planning.

---

# Level 5 — Physics-like behavior (emergent)

### Intuition

Mario:

> “Action order matters.”

JUMP then LEFT ≠ LEFT then JUMP.

### Key point

No new loss.

This **emerges** if:
• h has memory (velocity)
• Level 2 deltas exist
• Level 3 composition exists

You *allow* this, you don’t force it.

---

# Where z vs h fit (important)

### z (perception)

Think:

> “What do I see right now?”

• Anchored to pixels
• Weak algebra only
• Losses: L_recon, maybe light L_delta_z

### h (state / memory)

Think:

> “What is happening and where am I going?”

• Carries velocity & hidden state
• Algebra lives here
• Losses: L_delta_h, L_kstep, L_add, L_cycle

---

# One-paragraph takeaway

Gridworld wants **fixed action vectors**.
Mario wants **state-conditioned local moves**.

Dynamics losses make predictions correct.
Algebra losses make actions behave consistently when combined.

For Mario, the sweet spot is:
• **Level 2 (local translation)**
• plus **light Level 3 (composition)**

Everything else is optional and should be added carefully.

If you want next, I can:
• map your current losses to these levels
• suggest a training schedule
• or show diagnostics to tell which level you actually have
