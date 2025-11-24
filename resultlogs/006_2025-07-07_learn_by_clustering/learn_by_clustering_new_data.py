# %%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(44)

def make_semantic_points(center, n, scale, label):
    return (np.random.randn(n, 2) * scale + center, np.full(n, label))

# Color map for semantic and cluster
sem_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
cluster_colors = ['#FFD700', '#6495ED', '#32CD32', '#FF69B4', '#FFA500']

# --- Iteration 0: Start with 2 semantic groups ---
X0a, S0a = make_semantic_points([0,0], 8, 0.7, 0)
X0b, S0b = make_semantic_points([5,5], 8, 0.7, 1)
X = np.concatenate([X0a, X0b])
S = np.concatenate([S0a, S0b])

embeddings = [X.copy()]
semantics = [S.copy()]
n_clusters = 2
cluster_assignments = []

plt.figure(figsize=(18,6))

for it in range(5):
    # Cluster on current embedding
    km = KMeans(n_clusters=n_clusters, random_state=it)
    C = km.fit_predict(embeddings[-1])
    cluster_assignments.append(C)

    # Plot
    ax = plt.subplot(1, 5, it+1)
    for c in range(n_clusters):
        idxs = np.where(C == c)[0]
        ax.scatter(
            embeddings[-1][idxs,0], embeddings[-1][idxs,1],
            c=cluster_colors[c], edgecolors=[sem_colors[S[i]] for i in idxs],
            s=90, linewidths=2, alpha=0.6, marker='o', label=f'Cluster {c}')
    # Draw semantics (edge color) for visibility
    for i in range(len(embeddings[-1])):
        ax.text(
            embeddings[-1][i,0], embeddings[-1][i,1], f"{i}",
            fontsize=7, ha='center', va='center',
            color=sem_colors[S[i]], weight='bold')
    ax.set_title(f"Iteration {it}\nKMeans (K={n_clusters})")
    ax.set_xlim(-3, 10); ax.set_ylim(-3, 10)
    ax.grid(True)
    if it == 0:
        ax.legend(fontsize=8)

    # --- Add new data each round ---
    if it == 0:
        # Add new semantic group C in between A and B (requires new cluster!)
        Xnew, Snew = make_semantic_points([2.5,2.5], 8, 0.7, 2)
        X = np.concatenate([X, Xnew])
        S = np.concatenate([S, Snew])
        embeddings.append(X.copy())
        semantics.append(S.copy())
        n_clusters = 3  # Must increase cluster count to group C
    elif it == 1:
        # Add new points for group A and C, spatially overlapping group B
        Xnew, Snew = make_semantic_points([5,5], 6, 0.7, 0)
        Xnew2, Snew2 = make_semantic_points([2.5,2.5], 6, 0.7, 2)
        X = np.concatenate([X, Xnew, Xnew2])
        S = np.concatenate([S, Snew, Snew2])
        embeddings.append(X.copy())
        semantics.append(S.copy())
        # Keep n_clusters the same, but clusters must shift
    elif it == 2:
        # Add new points for group B, overlapping group A
        Xnew, Snew = make_semantic_points([0,0], 6, 0.7, 1)
        X = np.concatenate([X, Xnew])
        S = np.concatenate([S, Snew])
        embeddings.append(X.copy())
        semantics.append(S.copy())
    elif it == 3:
        # Add new group D far away
        Xnew, Snew = make_semantic_points([8,8], 8, 0.7, 3)
        X = np.concatenate([X, Xnew])
        S = np.concatenate([S, Snew])
        embeddings.append(X.copy())
        semantics.append(S.copy())
        n_clusters = 4  # Now need 4 clusters
    else:
        # Final round, no new data
        embeddings.append(X.copy())
        semantics.append(S.copy())

plt.tight_layout()
plt.show()

# Print table
print("Round | idx | (x, y)      | semantic | cluster")
for it in range(5):
    print(f"\n=== Iteration {it} ===")
    for i in range(len(embeddings[it])):
        print(f"{it:2d}   | {i:2d} | ({embeddings[it][i,0]:5.2f},{embeddings[it][i,1]:5.2f}) |   {semantics[it][i]}     |   {cluster_assignments[it][i]}")

# %%
