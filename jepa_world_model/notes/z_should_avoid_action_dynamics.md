https://chatgpt.com/share/69627d5f-86ec-8005-a437-7c877299fdef

Below is a **single, self-contained design note** you can drop into your docs. It’s written to resolve the confusion cleanly and give you an algebraic “contract” for (z).

---

# What (z) *is* and *is not* allowed to encode

### (for (z) to remain viable for similarity & loop closure)

## Executive summary (one paragraph)

For loop closure, **(z) must be a descriptor of *place*, not a state of *motion***. Algebraically, (z) must be **invariant under loop words** (action sequences that compose to identity). Anything that makes (z) change when you leave a place and come back—velocity, last action, phase—breaks loop closure. JEPA’s intent is that (z) encodes *predictively relevant structure* (layout, affordances) **without encoding which action occurred**. Dynamics belong in (h); geometry belongs in (p).

---

## The algebraic contract for (z)

Let:

* (x_t) = observation
* (z_t = \text{enc}(x_t))
* (w) = a loop word (action sequence with net identity)

**Required invariant**
[
\text{If } F^{(|w|)}(x_t, w) = x_t \quad \Rightarrow \quad z_{t+|w|} \approx z_t
]

That is: **(z) must factor out action history**.

Equivalently:

* (z) is a function on *equivalence classes of states under loop words*
* (z) must *not* be a coordinate chart that composes with actions

---

## What (z) is allowed to include

### ✅ Allowed: *Action-relevant but action-invariant* structure

These are properties that **constrain** what actions can do, but do not record which action happened.

**Examples**

* Layout / geometry of the scene
* Landmarks and static objects
* Affordances (“this area is traversable”, “there’s a wall here”)
* Semantic configuration (“room with door on left”)
* Visual appearance that is stable across revisits

**Intuition**

> These properties affect *what futures are possible*, not *which future occurred*.

**Algebraically**
They are **constants under loop words**:
[
z(x) = z(F^{(k)}(x, w)) \quad \text{for loop } w
]

This is exactly what you want for loop closure.

---

### ✅ Allowed: *Predictive relevance without action identity*

JEPA-style training may shape (z) so it contains information that helps prediction *in aggregate* (e.g., “there’s a corridor here”), **as long as it does not encode the specific control signal**.

**Key distinction**

* ✔ “This scene allows left/right motion”
* ✖ “The agent is currently moving left”

---

## What (z) is NOT allowed to include

### ❌ Not allowed: Inverse dynamics (explicit or implicit)

#### What inverse dynamics does

An inverse dynamics loss trains:
[
a_t \approx g(z_t, z_{t+1})
]

This **forces (z_t) and (z_{t+1}) to differ in a way that reveals the action**.

#### Why this breaks loop closure (clear example)

**Environment**

* 1D corridor
* Actions: LEFT, RIGHT
* State: position (x)

**Trajectory**

* Start at position (x)
* Apply RIGHT, then LEFT
* End at the same position (x)

This is a loop word: `RIGHT, LEFT`.

**What loop closure requires**
[
z(x) \approx z(x)
]

**What inverse dynamics enforces**
To decode RIGHT and LEFT:
[
z(x \xrightarrow{\text{RIGHT}} x+1) \neq z(x)
]
[
z(x \xrightarrow{\text{LEFT}} x-1) \neq z(x)
]

Now after the loop:

* You are back at (x),
* but (z) has been pushed along two distinct directions and **does not return**.

Algebraically:
[
z_{t+2} = z_t + \Delta(\text{RIGHT}) + \Delta(\text{LEFT}) \neq z_t
]

The loop **does not close in (z)**.

> Inverse dynamics turns (z) into a *path-dependent coordinate*, which is the opposite of a loop-closure descriptor.

---

### ❌ Not allowed: Action additivity / local linearity in (z)

Losses like:
[
z_{t+1} - z_t \approx v(a_t)
]

implicitly say:
[
z \text{ composes under actions}
]

That makes (z) behave like pose.
Loop words then produce **drift**, not closure.

---

### ❌ Not allowed: Temporal adjacency as a positive signal

Losses that enforce:
[
|z_{t+1} - z_t| \text{ small}
]

cause:

* time → distance
* monotonic drift
* revisits never collapse

This encodes **phase**, not place.

---

## Subtle but important: what *action-like* things ARE allowed

### ✅ Allowed: Action *constraints*, not action *realizations*

**Allowed**

* “This region blocks RIGHT moves”
* “Turning here is possible”
* “This object will cause collision if pushed”

**Not allowed**

* “RIGHT was just taken”
* “Velocity is positive”
* “Next state will be X”

**Rule of thumb**

> If the information changes when you walk a loop and come back, it does **not** belong in (z).

---

## Clean division of labor (recommended)

| Variable              | Encodes                          | Algebraic role             |
| --------------------- | -------------------------------- | -------------------------- |
| (z)                   | Place / appearance / affordances | Invariant under loop words |
| (h)                   | Dynamics, memory, action effects | Carries action history     |
| (p)                   | Geometry / coordinates           | Composes under actions     |
| Descriptor (optional) | Long-range place ID              | Quotient space for loops   |

---

## A reminder about “JEPA includes dynamics lightly”

This phrase is often misunderstood.

**Correct meaning**

> (z) encodes *structure that makes futures predictable*, not *which future happened*.

**Incorrect interpretation**

> (z) should encode actions or state deltas.

Traditional JEPA sides firmly with the **correct meaning**.

---

## Final one-line test (use this when adding a loss)

Ask:

> “If I leave a place, take actions, and return, will this loss *force* (z) to change?”

* If **yes** → it breaks loop closure.
* If **no** → it’s compatible.

That test alone will keep (z) viable as a loop-closure similarity metric.

---

Why does including inverse dynamics break z?

Because **inverse dynamics forces (z) to encode *which action happened***, and that *necessarily* makes (z) **path-dependent** instead of **place-dependent**. Loop closure requires place-dependence.

I’ll explain this three ways: **(1) algebraically**, **(2) with a concrete example**, and **(3) in JEPA terms**. The punchline will be unavoidable.

---

## 1. The algebraic reason (the cleanest explanation)

### What loop closure requires

For (z) to support loop closure, it must satisfy:

> **Invariance under loop words**

If an action sequence (w) composes to identity:
[
F^{(|w|)}(x, w) = x
]
then:
[
z(F^{(|w|)}(x, w)) = z(x)
]

This means:
[
z \circ F(w) = z \quad \text{for all loop words } w
]

Equivalently:

* (z) is defined on **equivalence classes of states modulo action loops**
* (z) *quotients out* action history

---

### What inverse dynamics enforces

Inverse dynamics trains a function:
[
a_t \approx g(z_t, z_{t+1})
]

For this to be possible, **(z_t) and (z_{t+1}) must differ in a way that reveals the action**.

That implies:
[
z_{t+1} \neq z_t \quad \text{whenever } a_t \neq \text{NOOP}
]

So inverse dynamics enforces:
[
z_{t+1} = z_t \oplus \Delta(a_t)
]

That is exactly the algebra of a **coordinate chart**, not an invariant descriptor.

---

### The contradiction

For a loop word (w = (a_1, \dots, a_k)):
[
z_{t+k} = z_t \oplus \Delta(a_1) \oplus \dots \oplus \Delta(a_k)
]

But if (w) is a loop, then:
[
F^{(k)}(x_t, w) = x_t
]

So loop closure requires:
[
z_{t+k} = z_t
]

The only way both can be true is:
[
\Delta(a_1) \oplus \dots \oplus \Delta(a_k) = 0
]

Inverse dynamics **does not enforce this**. In fact, it discourages it.

So you get:
[
z_{t+k} \neq z_t
]

**The loop does not close in (z).**

That is the algebraic failure.

---

## 2. A concrete example (no abstraction)

### Environment

* 1D corridor
* Actions: LEFT, RIGHT
* State: position (x)

### Loop word

`RIGHT, LEFT`
Net effect: back to the same position.

---

### What loop closure wants

Same place ⇒ same (z):
[
x \xrightarrow{\text{RIGHT}} x+1 \xrightarrow{\text{LEFT}} x
]
[
z(x) \approx z(x)
]

---

### What inverse dynamics *forces*

To predict actions:

* (z(x)) and (z(x+1)) must differ in a *RIGHT-specific* way
* (z(x+1)) and (z(x)) must differ in a *LEFT-specific* way

So the model learns:
[
z(x+1) = z(x) + v_{\text{RIGHT}}
]
[
z(x) = z(x+1) + v_{\text{LEFT}}
]

After the loop:
[
z_{\text{end}} = z(x) + v_{\text{RIGHT}} + v_{\text{LEFT}}
]

Unless the model **explicitly learns**
[
v_{\text{RIGHT}} + v_{\text{LEFT}} = 0
]
(which inverse dynamics does *not* encourage),

you get:
[
z_{\text{end}} \neq z(x)
]

So:

* Same place
* Different (z)

**Loop closure is broken.**

---

### Key intuition

Inverse dynamics turns (z) into a **breadcrumb trail** of actions.

Loop closure requires **forgetting the breadcrumbs**.

---

## 3. Why this is fundamentally incompatible (not a tuning issue)

This is not about:

* weak vs strong loss
* weighting
* regularization
* clever architecture

It’s a **role contradiction**.

### Inverse dynamics requires

> (z) must distinguish states that differ *only by the last action*.

### Loop closure requires

> (z) must **identify** states that differ *only by action history*.

Those are logical opposites.

No amount of “light weighting” resolves that.

---

## 4. “But actions matter for prediction!” — the subtle confusion

Yes — actions matter for **prediction**.

But **JEPA does not put prediction pressure on (z)**.

Instead:

* (z) encodes *what is there*
* dynamics (U(\cdot)) uses (z), (h), and (a) to predict forward

The predictor needs actions.
The descriptor must not.

That’s the entire JEPA separation of concerns.

---

## 5. What *can* safely influence (z) (why confusion arises)

People say “include dynamics lightly in (z)” and mean:

> Encode **action constraints**, not **action realizations**.

Examples:

* “There is a wall here” ✔
* “Moving right will cause collision” ✔
* “The agent just moved right” ✘
* “Velocity is positive” ✘

Inverse dynamics encodes the second category — forbidden for loop closure.

---

## 6. One-sentence bottom line

> **Inverse dynamics breaks (z) because it forces (z) to remember which action happened, while loop closure requires (z) to forget action history entirely.**

If you want, next we can:

* write a **loss-to-latent responsibility table**
* design a **diagnostic that proves inverse dynamics contamination**
* or show how **JEPA’s predictor carries dynamics without contaminating (z)**

---

Are either of these acceptable?:

Here’s a z‑focused summary of the algebraic ideas those two losses implement:

inverse_dynamics_z: Given a pair (z_t, z_{t+1}), it learns a classifier that predicts the discrete action a_t. Algebraically, it treats the transition as an unknown operator and asks “which action maps z_t to z_{t+1}?” It doesn’t constrain the geometry of z deltas directly; it only requires that different actions yield distinguishable transitions.

action_delta_z: It explicitly enforces an additive, action‑indexed delta structure:
z_{t+1} − z_t ≈ Δ(a_t) where Δ(a_t) is a learned prototype vector per action (via ActionDeltaProjector(a_t)). Algebraically, it pushes z to behave like a space with action‑specific translation directions.

---

Short answer: **no—neither is acceptable if you want (z) to remain a viable loop-closure similarity metric.**
They fail for *different algebraic reasons*, but both violate the same core invariant.

Below is a precise, algebra-level verdict on each, plus a clarification of what *would* be acceptable instead.

---

# The non-negotiable invariant for (z)

For loop closure, (z) must satisfy:

> **Invariance under loop words**
> If an action sequence (w) composes to identity in the environment, then
> [
> z(F^{(|w|)}(x, w)) = z(x)
> ]

Equivalently:

* (z) must **quotient out action history**
* (z) must **not be a carrier of the action algebra**

This is the yardstick.

---

# 1️⃣ `inverse_dynamics_z`

> *“Given ((z_t, z_{t+1})), predict the discrete action (a_t).
> It doesn’t constrain geometry directly; it only requires that different actions yield distinguishable transitions.”*

### ❌ Verdict: **Not acceptable**

This is the *cleanest* violation.

### Why (algebraically)

Inverse dynamics enforces the existence of a function:
[
a_t = g(z_t, z_{t+1})
]

For this to be learnable:

* (z_{t+1}) must differ from (z_t) in an **action-specific way**
* i.e.
  [
  z_{t+1} = z_t \oplus \Delta(a_t)
  ]
  for some injective action encoding

That already means:

* (z) changes *because* of the action
* action identity is encoded in the **difference structure** of (z)

Now consider a loop word (w = (a_1, \dots, a_k)) that returns to the same place:
[
F^{(k)}(x, w) = x
]

Loop closure requires:
[
z_{t+k} = z_t
]

But inverse dynamics gives:
[
z_{t+k} = z_t \oplus \Delta(a_1) \oplus \dots \oplus \Delta(a_k)
]

There is **no constraint** in inverse dynamics that makes that sum zero.
In fact, *making it zero would destroy action identifiability*, which is exactly what inverse dynamics wants.

### Key point

Even though inverse dynamics does **not explicitly enforce additivity**, it *implicitly* requires **action-separable deltas**. That is enough to break invariance.

> Inverse dynamics makes (z) a *path-dependent coordinate*, not a place descriptor.

---

# 2️⃣ `action_delta_z`

> *“Explicitly enforces
> (z_{t+1} - z_t \approx \Delta(a_t)),
> pushing (z) to behave like a space with action-specific translation directions.”*

### ❌ Verdict: **Even more not acceptable**

This is the *explicit* version of the same failure.

### Why (algebraically)

This loss states outright:
[
z_{t+1} = z_t + \Delta(a_t)
]

That means:

* actions are generators
* (z) is a **representation of the action group**
* composition happens by vector addition

Now apply a loop word:
[
z_{t+k} = z_t + \sum_{i=1}^k \Delta(a_i)
]

Loop closure would require:
[
\sum_{i=1}^k \Delta(a_i) = 0
]

But:

* you never enforce that
* many loops are state-dependent
* many actions are not true inverses

So loops **spiral**, not close.

### Geometric intuition

This loss turns (z) into:

* a **chart**
* a **local coordinate system**
* something like pose

That is *exactly* what (z) must **not** be if it’s used for similarity.

---

# Why these two fail in *different* ways (but both fail)

| Loss               | What it forces                                   | Why it breaks (z)                          |
| ------------------ | ------------------------------------------------ | ------------------------------------------ |
| inverse_dynamics_z | Action must be recoverable from ((z_t, z_{t+1})) | Makes (z) encode **which action happened** |
| action_delta_z     | Actions correspond to additive directions        | Makes (z) **compose under actions**        |

Both violate:

> **(z) must forget action history**

---

# What *would* be acceptable instead (important contrast)

### ✅ Acceptable: **action-invariant predictive structure**

You *may* allow gradients from prediction to shape (z), **as long as**:

* actions are inputs to the *predictor*, not targets from (z)
* there is **no loss** that asks “which action was this?”
* there is **no constraint** that ties (z) deltas to actions

Example (acceptable):
[
\hat z_{t+1} = U(z_t, h_t, a_t)
]
[
L = |\hat z_{t+1} - z_{t+1}|
]

This says:

* “given action (a_t), predict next (z)”
* **not** “encode (a_t) in (z)”

This is the JEPA line.

---

# One-line litmus test (very useful)

Ask of any loss touching (z):

> **Could this loss allow an observer to infer the last action from (z) or from (z_{t+1}-z_t)?**

* If **yes** → it breaks loop closure
* If **no** → it may be compatible

Both of your proposed losses answer **yes**.

---

## Final verdict (unambiguous)

* `inverse_dynamics_z`: ❌ **Not acceptable**
* `action_delta_z`: ❌ **Not acceptable (stronger violation)**

Both force (z) to carry the action algebra.
Loop closure requires (z) to *quotient it out*.

If you want, next we can:

* rewrite these losses so they live on **(h)** instead
* design a **safe alternative** that captures “action relevance” without action identity
* or build a **diagnostic** that proves when (z) has been contaminated by inverse dynamics

---

Here’s a **clean, consolidated summary** of the rollout-related (z) losses you pasted, framed against the **loop-closure contract for (z)** and why each one violates it.

This is written so you can drop it straight into a design doc.

---

# Summary: Why the rollout-based (z) losses are not acceptable for loop closure

## The non-negotiable contract for (z)

If (z) is used as a **similarity / loop-closure descriptor**, it must satisfy:

> **Same place ⇒ same (z), regardless of action history**

Algebraically:

* (z) must be **invariant under loop words** (action sequences that compose to identity)
* (z) must **not** carry the action algebra
* (z) must **not** behave like a state that rolls forward under actions

Anything that trains (z) to:

* encode action identity,
* integrate actions over time,
* or remain predictive over multi-step rollouts
  breaks this contract.

---

## 1) `rollout_z_loss`

**(multi-step latent rollout, target = next encoded (z), open-loop)**

### What it does

* Rolls forward in latent space using actions:
  [
  h_{t+1} = U(z_t, h_t, a_t), \quad \hat z_{t+1} = H2Z(h_{t+1})
  ]
* Compares (\hat z_{t+1}) to the next encoded (z_{t+1})
* Feeds the prediction back in (`current = pred`) for multi-step rollout
* Target is detached, but **input (z_t) is not**

### Why it’s not acceptable

* Gradients flow into the encoder (unless embeddings are detached upstream), pushing (z) to **help predict action-conditioned futures**
* Open-loop rollout trains (z) to be **stable under repeated action application**
* This encourages (z) to encode **velocity, phase, contact mode**, etc.

### Algebraic violation

* (z) is trained to behave like a **dynamical state**, not an invariant descriptor
* Loop words no longer close in (z)

### Bottom line

> This loss turns (z) into something that rolls forward under actions, which directly conflicts with loop closure.

---

## 2) `rollout_recon_z_loss`

**(multi-step rollout + pixel reconstruction from predicted (z))**

### What it does

* Rolls forward in latent space
* Decodes predicted (z) into pixels
* Matches decoded images to future ground-truth frames
* No detaches anywhere; fully open-loop

### Why it’s not acceptable

* Forces predicted (z) to be **pixel-sufficient** over time
* Requires (z) to encode **all transient state** needed for rendering:

  * motion,
  * animation phase,
  * contacts,
  * hidden dynamics
* Strongly backpropagates into the encoder

### Algebraic violation

* Same place reached with different motion histories must decode differently
* Therefore they must have different (z)
* Loop closure is fundamentally impossible

### Bottom line

> This loss explicitly redefines (z) as a full dynamical state, not a place descriptor.

---

## 3) `rollout_project_z_loss`

**(enc(dec(predicted z)) ≈ predicted z, open-loop)**

### What it does

* Rolls forward in latent space
* Decodes predicted (z) to images
* Re-encodes those images
* Forces encoder–decoder cycle consistency on **imagined frames**

### Why it’s not acceptable

* Trains the encoder on **hallucinated / off-manifold images**
* Forces encoder and decoder to **follow the predictor** wherever it goes
* Makes “being reachable by the predictor” define the (z) manifold
* Creates attractor dynamics around rollout trajectories

### Algebraic violation

* Predicted, action-conditioned (z) trajectories are treated as legitimate descriptors
* This implicitly encourages **action-compositional structure** in (z)
* Real-frame similarity is polluted by rollout artifacts

### Bottom line

> This loss makes rollout dynamics, not real observations, define what (z) means.

---

## Unifying reason all three are unacceptable

All three losses share the same core problem:

> **They train (z) to participate in action-conditioned rollouts.**

That makes (z):

* path-dependent,
* history-dependent,
* and predictive of dynamics,

instead of:

* place-dependent,
* invariant,
* and suitable for similarity / loop closure.

Even though none of them explicitly say “(z_{t+1} - z_t = \Delta(a))”, they all **implicitly push (z) toward carrying the action algebra** by requiring multi-step consistency under actions.

---

## What would be acceptable instead (contrast)

* Use **(h)** (or (p)) for:

  * rollout consistency
  * action-conditioned prediction
  * reconstruction over time
* Use **(z = \text{enc}(x))** only for:

  * invariance
  * similarity
  * loop closure
* If predicting (z), do it **one-step**, with:

  * stop-grad on encoder outputs
  * no open-loop
  * no reconstruction pressure

---

## Final takeaway (one sentence)

> Any loss that requires (z) to remain coherent under multi-step action rollouts makes (z) a dynamical state, and a dynamical state cannot serve as a loop-closure similarity descriptor.

That’s the single principle all of these violate.

---

What about standard jepa loss, is that ok?:

def jepa_loss(
    model: JEPAWorldModel,
    outputs: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    use_z2h_init: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """JEPA loss using predictor conditioned on z, h, and action."""
    embeddings = outputs["embeddings"]
    if embeddings.shape[1] < 2:
        raise AssertionError("JEPA loss requires at least two timesteps.")
    preds, h_preds, h_states = rollout_teacher_forced(
        model,
        embeddings,
        actions,
        use_z2h_init=use_z2h_init,
    )
    target = embeddings[:, 1:].detach()
    return JEPA_LOSS(preds, target), preds, h_preds, h_states

---

Mostly yes — **this “standard JEPA loss” can be OK for keeping (z) viable for similarity / loop closure**, **but only under one crucial condition**:

> **The JEPA loss must not backprop into the encoder (z=\text{enc}(x))** (or it must do so only very weakly / carefully scheduled).

Right now, your code **detaches the target** (`embeddings[:,1:].detach()`), which is good, but it does **not** detach the *inputs* `embeddings` passed into `rollout_teacher_forced`. So whether this is “safe” depends on how `rollout_teacher_forced` uses `embeddings` and whether gradients flow into the encoder.

Let’s break it down precisely.

---

## What your JEPA loss is doing

* You compute `embeddings = outputs["embeddings"]` (likely from encoder).
* You run a **teacher-forced** rollout:

  * inputs at each step use the *ground-truth* (z_t) (not predicted (z))
  * predictor also sees actions
* You predict `preds` intended to match the next latent (z_{t+1})
* You compare to the next-step target latent but detach the target:
  [
  L = |\hat z_{t+1} - \text{sg}(z_{t+1})|
  ]

This is the classic “predict next latent” loss with stop-grad on targets.

### Why teacher forcing matters

Teacher forcing avoids the “(z) must be a roll-forward state” pressure that open-loop rollout losses create. That’s a big reason this loss is much safer than the rollout losses you pasted earlier.

---

## The remaining risk: encoder contamination through the **input** (z_t)

Even with a detached target, gradients can flow:

[
\hat z_{t+1} = f(z_t, h_t, a_t)
]
[
L(\hat z_{t+1}, \text{sg}(z_{t+1}))
]

The gradient (\partial L/\partial z_t) is nonzero in general, which means the encoder can be trained so that (z_t) becomes **more useful for predicting (z_{t+1}) given actions**.

That tends to make (z) soak up “hidden” dynamical state (velocity/phase/contact mode) — which is exactly what hurts loop closure.

So: **detaching the target alone is not enough** to guarantee (z) stays descriptor-like.

---

## When it *is* OK

This JEPA loss is OK for loop-closure (z) if you ensure one of the following:

### ✅ Option A (cleanest): stop-grad on the *input* (z_t) too (for this loss)

Inside `rollout_teacher_forced`, feed `embeddings.detach()` as the (z_t) input to the predictor.

That makes the JEPA term train:

* predictor parameters
* (h)-related parameters
  while leaving encoder to be shaped by separate “descriptor” objectives (augment invariance, etc.).

### ✅ Option B: keep gradients to encoder, but make sure the encoder has strong invariance objectives and limited capacity

This is closer to how some JEPA-like systems are trained in practice: the encoder does get some predictive pressure, but it’s controlled by:

* strong augmentation invariance / masking
* regularization (VICReg-ish)
* limiting (z) capacity
* not adding inverse dynamics / action-delta losses on (z)

This can still preserve similarity, but it’s more fragile and requires careful balancing.

### ✅ Option C: explicitly separate “descriptor (z)” and “predictive latent (\tilde z)”

* (z=\text{enc}(x)) is used for similarity & loop closure
* (\tilde z = \text{proj}(z)) is what predictor uses/targets

Then you can allow JEPA training on (\tilde z) while keeping (z) clean.

---

## How to decide with one simple check

Ask: **Does the JEPA loss update the encoder?**

* If **no** → it’s very safe for loop closure (z).
* If **yes** → it *might* still work, but you’re accepting pressure for (z) to become predictive, which can degrade loop closure.

In code terms, the safe version usually looks like:

* `z_in = embeddings.detach()` for predictor inputs
* `z_tgt = embeddings[:,1:].detach()` for targets

---

## Verdict on your exact snippet

* ✅ **Teacher forced**: good (much safer than open-loop rollout losses).
* ✅ **Detached target**: good.
* ⚠️ **Input embeddings not detached**: potential issue depending on how `outputs["embeddings"]` is produced and how `rollout_teacher_forced` uses it.

So: **“mostly ok”**, with the important caveat that I would detach the input embeddings for this term if you truly want (z) to remain a clean loop-closure descriptor.

If you paste `rollout_teacher_forced` (just the core lines), I’ll tell you exactly where to put `detach()` (or a stop-grad wrapper) to preserve your intended split.
