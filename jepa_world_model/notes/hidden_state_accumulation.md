# Hidden State Accumulation Problem

## Context

Around commit b0843a68, we encountered issues with the model architecture that motivated a reversion to the b611179c architecture. This note documents the discussion around possible causes and solutions.

## The Problem: Unbounded Accumulation in Residual Models

When using residual connections for hidden state updates:

```python
h_next = h_t + delta
```

The hidden state norm grows linearly with the number of steps:

```
||h_T|| ≈ ||h_0|| + T * E[||delta||]
```

This leads to several issues:
1. **Unbounded growth**: Over long rollouts, hidden states grow without bound
2. **Gradient instability**: Large activations can cause gradient explosion
3. **Numerical issues**: Very large values can lead to overflow or precision loss

## Why LayerNorm Alone Isn't Sufficient

LayerNorm normalizes *inputs* to a layer:

```python
def forward(self, h: torch.Tensor) -> torch.Tensor:
    if self.norm is not None:
        h = self.norm(h)  # normalize input
    return self.net(h)    # but output is unbounded
```

Key limitations:
- LayerNorm stabilizes gradients during training
- But it doesn't bound the *output* of the network
- A linear layer after LayerNorm can still produce arbitrarily large outputs
- The accumulation problem persists: `h_next = h_t + unbounded_delta`

## Solution 1: Direct Prediction (Current Approach)

Instead of predicting deltas, directly predict the next state:

```python
h_next = predictor(z_t, h_t, action_t)  # direct prediction
```

Advantages:
- No accumulation problem
- Network learns absolute states rather than increments
- More stable for long rollouts

Disadvantages:
- May be harder to learn (predicting full state vs small changes)
- Loses explicit notion of state transitions

## Solution 2: Gated Updates (GRU/LSTM-style)

Use learned gates to control update magnitude:

```python
# Predict candidate update and gate
h_candidate = predictor(z_t, h_t, action_t)
alpha = sigmoid(gate_network(z_t, h_t, action_t))

# Bounded update
h_next = (1 - alpha) * h_t + alpha * h_candidate
```

How this helps:
- Gate α ∈ [0, 1] bounds the update magnitude
- Even if `h_candidate` is large, the update is controlled
- Convex combination keeps `h_next` in a reasonable range
- Network learns when to make small vs large updates

Mathematical analysis:
```
||h_next|| ≤ (1-α)||h_t|| + α||h_candidate||
```

If we additionally normalize `h_candidate`:
```
||h_next|| ≤ ||h_t||  # stays bounded!
```

## Solution 3: Unit Normalization

Project hidden states onto unit sphere:

```python
h_next_raw = h_t + delta
h_next = F.normalize(h_next_raw, dim=-1)  # project to unit sphere
```

Advantages:
- Strictly bounds ||h_next|| = 1
- Simple to implement
- Works with residual updates

Disadvantages:
- Loses magnitude information (only direction is preserved)
- May limit model expressiveness
- Can cause gradient issues at poles

## Why b611179c Architecture Worked

The b611179c architecture used `PredictorNetwork` that directly outputs:
```python
pred, delta, h_next = model.predictor(z_t, h_t, act_t)
```

Where `h_next` is a direct prediction, not a residual update. This avoids the accumulation problem entirely.

## Action Items for Future Architectures

When designing hidden state updates:

1. **Be explicit about accumulation**: If using residuals, document why accumulation is safe
2. **Consider rollout length**: Long rollouts require more careful design
3. **Add monitoring**: Track hidden state norms during training
4. **Test long rollouts**: Ensure model is stable for longer horizons than trained on
5. **Consider gates**: For residual updates, use gating mechanisms to bound updates

## References

- GRU paper: Learning Phrase Representations using RNN Encoder-Decoder (Cho et al., 2014)
- LSTM paper: Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)
- LayerNorm paper: Layer Normalization (Ba et al., 2016)

## Discussion Date

January 2, 2026
