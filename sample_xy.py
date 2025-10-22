# %%

import numpy as np

def combined_weighted_sample(tuples, T=1.0, random_state=None):
    arr = np.array(tuples)
    xs = arr[:, 0]
    counts = arr[:, 2]

    # Unique sorted xs, multiplicity, and index mapping
    unique_xs, inverse_indices, counts_per_x = np.unique(xs, return_inverse=True, return_counts=True)

    print(f"{unique_xs}")
    print(f"{inverse_indices=}")
    print(f"{counts_per_x=}")

    x_ranks = np.argsort(np.argsort(unique_xs))  # ranks (0 for smallest x)

    print(f"{x_ranks=}")

    x_weights_raw = 1.0 / (x_ranks + 1)

    print(f"{x_weights_raw=}")

    x_weights = x_weights_raw / counts_per_x

    print(f"{x_weights=}")

    tuple_x_weights = x_weights[inverse_indices]

    print(f"{tuple_x_weights=}")

    # Boltzmann y weights (using count)
    y_weights = np.exp(-counts / T)

    print(f"{y_weights=}")

    # Combined weights
    combined_weights = tuple_x_weights * y_weights

    print(f"{combined_weights=}")

    probs = combined_weights / combined_weights.sum()

    print(f"{probs=}")

    # Sample
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(tuples), p=probs)
    return tuples[idx]

# --- Example usage ---
tuples = [
    (10, 10, 3),
    (20, 15, 0),
    (20, 16, 2),
    (30, 10, 1),
    (30, 12, 4),
    (30, 12, 5),
]

sampled = combined_weighted_sample(tuples, T=1.0)
print("Sampled tuple:", sampled)


# %%

rng = np.random.default_rng()

w = np.asarray([5, 6, 7])
probs = w / w.sum()

rng.choice([(10, 'a'), (20, 'b'), (30, 'c')], p=probs)


# %%

import numpy as np
import pandas as pd

def build_weight_grid(tuples, T=1.0):
    arr = np.array(tuples)
    xs = arr[:, 0]
    ys = arr[:, 1]
    counts = arr[:, 2]

    unique_xs = np.unique(xs)
    unique_ys = np.unique(ys)

    # Stage 1: x weights (hyperbolic, normalized)
    x_ranks = np.argsort(np.argsort(unique_xs))
    x_weights = 1.0 / (x_ranks + 1)
    x_probs = x_weights / x_weights.sum()

    # Build empty grid (DataFrame for nice labeling)
    grid = pd.DataFrame(
        np.zeros((len(unique_xs), len(unique_ys))),
        index=unique_xs,
        columns=unique_ys
    )

    # For each x, fill probabilities for y in that x group
    for i, x in enumerate(unique_xs):
        idxs = np.where(xs == x)[0]
        # Stage 2: Boltzmann over counts within x group
        group_ys = ys[idxs]
        group_counts = counts[idxs]
        y_weights = np.exp(-group_counts / T)
        y_probs = y_weights / y_weights.sum()

        # Fill in the joint probability for each y in this group
        for j, (yy, prob) in enumerate(zip(group_ys, y_probs)):
            grid.at[x, yy] = x_probs[i] * prob  # joint probability

    return grid

# --- Example usage ---
tuples = [
    (1, 10, 3),
    (2, 15, 0),
    (2, 16, 2),
    (3, 10, 1),
    (3, 12, 4),
    (3, 12, 5),
]

weight_grid = build_weight_grid(tuples, T=1.0)
print(weight_grid)
print(f"Sum of all weights: {weight_grid.values.sum()}")  # Should be 1.0


# %%

grid = np.ones((7, 3))

x = np.random.random(3)
y = np.random.random(7)

print(x)
print(y)

r = grid * x[None, :]
r *= y[:, None]

print(r.shape)

r

# %%

0.73987042 * 0.53281468