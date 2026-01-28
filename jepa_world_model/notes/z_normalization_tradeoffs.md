# z normalization tradeoffs

## Summary
Normalizing z can stabilize training and make JEPA-style losses scale-invariant, but it removes magnitude as a signal and can create distribution mismatches if other heads are trained on raw z. This note summarizes when normalization helps, when it hurts, and why JEPA-only normalization did not stop raw z norm drift.

## Why normalize z?
- Stabilizes training by removing scale drift from JEPA-style prediction losses.
- Keeps decoder inputs at a consistent scale (easier optimization).
- Makes distances and deltas less sensitive to encoder scale changes.
- Avoids saturation or exploding activations in downstream heads.

## Why not normalize z?
- Removes magnitude as a signal ("how far apart" or confidence).
- Can degrade tasks that rely on Euclidean distances in z.
- Can reduce representational capacity if magnitude carries state information.

## What we observed
- z norm kept increasing even with SIGReg (0.01 and 0.1).
- Normalizing only inside JEPA losses did not stop raw z norm drift.
- After encoder normalization, raw z norms became stable, but predicted rollouts initially looked wrong because predicted z was not normalized (decoder trained on normalized z).

## Q&A from the discussion

### Q: Doesn't normalizing z make it so z always has the same target? Why didn't that work?
A: JEPA-only normalization makes the *loss* scale-invariant, but it does not constrain the *raw encoder output*. Diagnostics plot raw z, so it can still drift. The encoder is free to scale z unless there is an explicit norm anchor or global normalization.

### Q: Is there a direct benefit to keeping raw z norm bounded?
A: Yes. It improves stability, keeps downstream heads well-conditioned, makes Euclidean distances more consistent, and avoids saturation.

### Q: Should SIGReg be enough to constrain z?
A: Usually not. SIGReg is a soft distribution-matching regularizer on random projections. It often does not pin the norm unless heavily weighted.

### Q: Is it a problem if I don’t normalize z? What does that really do?
A: Not necessarily. It allows the model to use magnitude as a signal, but also allows uncontrolled scale drift. If raw z is used by other heads, drift can destabilize training.

### Q: Why would predicted rollouts look wrong after normalization?
A: If the encoder output is normalized but predicted z (from h_to_z) is not, the decoder sees out-of-distribution inputs. Normalizing h_to_z outputs fixes this mismatch.

## Practical guidance
- If you want stability and don’t need magnitude information, normalize z globally (encoder output + h_to_z).
- If you need magnitude as signal, keep raw z but add a norm anchor loss (e.g., target mean norm) or increase SIGReg heavily.
- If you normalize only inside JEPA losses, be aware this removes scale pressure and won’t stop raw z drift.

