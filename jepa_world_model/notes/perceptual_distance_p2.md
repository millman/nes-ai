https://chatgpt.com/c/697d85ea-030c-8327-a42c-9c0d03826a7a

---

# Perceptual Similarity, Distance, and Robust Smoothness in z / h / p World Models

This note summarizes design choices, motivations, and techniques for learning **perceptual similarity** and **distance** in a world-model setup with latent variables:

- **z**: perceptual descriptor
- **h**: belief / dynamics state
- **p**: geometry / action-distance (pose, odometry)

The motivating environment is a simple one: a black background with a small white box that moves around. Despite its simplicity, this environment exposes fundamental questions about *what “distance” means* in learned representations.

---

## 1. The Core Question

**How should we define and learn “nearness” between observations?**

Key tension:
- Pixel similarity suggests one notion of nearness
- Action reachability suggests another
- Time adjacency is often misleading

We want:
- Nearby-looking images → nearby in **z**
- Reachability and planning distance → encoded elsewhere (**p**, **h**)

---

## 2. Why Pixel Distance Alone Is Not Enough

### Human intuition
Even without knowing actions, humans can:
- See two frames
- Estimate how far the object moved
- Infer similarity from visual displacement

This suggests **perceptual distance is meaningful before actions are known**.

### But…
Pixel similarity can be misleading:
- Teleportation environments
- Pipes / doors (Mario)
- End-of-level transitions
- Variable-speed motion

Thus:
- Pixel distance ≠ causal distance
- Perceptual similarity ≠ reachability

---

## 3. Time-Based vs Reachability-Based Distance

### Time-based (temporal smoothness)
Assumes:
> “Consecutive frames should be similar.”

Fails when:
- Large jumps occur
- Teleports or resets happen
- One step ≠ one unit of distance

### Reachability-based
Defines distance as:
> “Minimum action cost to get from A to B.”

Correct for planning, but:
- Requires known actions
- Sparse early
- Not purely perceptual

---

## 4. Key Insight: Separate Similarity from Distance

**Distance is a property of control.**
**Similarity is a property of observation.**

Trying to encode both in a single latent leads to collapse or distortion.

### Proper division of labor

| Latent | Role |
|------|------|
| z | Perceptual similarity / visual descriptor |
| h | Belief, dynamics, mode switches |
| p | Action distance / geometry / odometry |

---

## 5. Training z for Perceptual Similarity (from trajectories only)

The goal for **z**:
> Nearby-looking images should be nearby in latent space, independent of reachability.

### Common approaches

#### 5.1 Reconstruction-based
- Autoencoders / VAEs
- Masked autoencoding (MAE)

**Pros**
- Stable
- Learns “what’s in the image”

**Cons**
- Can over-focus on pixels
- Needs bottlenecking to avoid trivial identity mapping

---

#### 5.2 Augmentation-invariance (contrastive / non-contrastive)

Examples:
- SimCLR / InfoNCE
- BYOL / SimSiam
- VICReg / Barlow Twins

**Key rule**
> Augmentations must preserve spatial meaning.

Avoid:
- Random crops
- Large translations

Prefer:
- Noise
- Blur
- Brightness / contrast jitter

VICReg-style losses are especially effective in simple environments because they:
- Prevent collapse
- Enforce variance
- Encourage smooth geometry

---

#### 5.3 Temporal coherence (carefully!)

Naïve loss:
```text
||z_t - z_{t+1}||
````

Problem:

* Forces smoothness even during teleports

Solution:

* Use **robust or gated** smoothness (see Section 6)

---

#### 5.4 Pixel-derived self-supervision (still “environment-only”)

In toy worlds:

* Estimate displacement directly from pixels (e.g., center-of-mass of white pixels)
* Use ranking or triplet losses:

  * small displacement → closer in z
  * large displacement → farther in z

This strongly aligns z-distance with visible movement without privileged state.

---

## 6. Robust Smoothness Loss on z

### Motivation

Temporal smoothness is useful as a **prior**, but dangerous as a rule.

We want:

> “Keep z smooth during normal motion, but allow large jumps.”

---

### Canonical formulation

Let:

```text
d_t = || z_{t+1} - z_t ||
```

Apply a **robust penalty** ρ:

```text
L_smooth = ρ(d_t)
```

---

### Common robust penalties

#### Huber loss

```text
ρ(d) = 0.5 d^2           if d ≤ δ
       δ (d − 0.5 δ)     if d > δ
```

* Quadratic for small motion
* Linear for big jumps

#### Charbonnier (smooth L1)

```text
ρ(d) = sqrt(d^2 + ε^2)
```

* Always differentiable
* Soft saturation

#### Truncated L2

```text
ρ(d) = min(d^2, c)
```

* Explicitly ignores large jumps

---

### Interpretation

* Walking / running → smooth z trajectory
* Pipe / teleport → sharp jump, no penalty
* Smoothness becomes a **soft prior**

---

### Weighting

Robust smoothness should:

* Have **low weight**
* Never be the dominant loss

Example:

```text
L_total = L_perceptual + λ_smooth * L_smooth
λ_smooth ≪ 1
```

---

## 7. Handling Big Jumps (Mario pipes, resets)

Big jumps are **events**, not failures.

Correct behavior:

* z is allowed to jump
* h absorbs surprise / mode change
* p encodes the action cost or special transition

Strategies:

* Robust penalties (above)
* Gating smoothness by prediction error
* Disabling smoothness at known episode boundaries

---

## 8. What Robust Smoothness Does NOT Do

Robust smoothness on z:

* ❌ does NOT define distance
* ❌ does NOT encode reachability
* ❌ does NOT prevent teleports
* ❌ does NOT support planning

Those belong in **p** and **h**.

---

## 9. Recommended Training Recipe (Toy Gridworld / White Box)

1. Train **z** with:

   * VICReg or similar (anti-collapse)
   * Mild augmentations
   * Robust temporal smoothness (low weight)

2. Train **p** separately with:

   * Action deltas
   * Additivity
   * Multi-step rollouts

3. Let **h** absorb:

   * Surprise
   * Discontinuities
   * Partial observability

Allow disagreement:

* z says “far”
* p says “close”
  → teleport or shortcut

That disagreement is **information**, not error.

---

## 10. One-Sentence Takeaway

> **Perceptual similarity and action distance are different geometries — treat them as such, and your model will stay stable, interpretable, and useful for planning.**

```

---

If you want next:
- a matching **loss diagram** for z / h / p
- a minimal PyTorch loss implementation section
- or a short “design rules” checklist to put at the top of the file
```
