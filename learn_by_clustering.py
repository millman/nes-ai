# %%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(42)

# Parameters
n_semantics = 3
points_per_sem = 10
wiggle_scale = 4.5   # **Even more wiggle!**

# Semantic centers for each class
semantic_centers = np.array([
    [0, 0],
    [10, 0],
    [5, 8]
])

# Interleave order: repeat pattern, then shuffle
semantics = np.tile(np.arange(n_semantics), points_per_sem)
chunks = []
for i in range(points_per_sem):
    chunk = np.arange(n_semantics)
    np.random.shuffle(chunk)
    chunks.append(chunk)
semantics = np.concatenate(chunks)
t_steps = len(semantics)

# Random "wiggle" for each instance
x = semantic_centers[semantics, 0] + np.random.randn(t_steps) * wiggle_scale
y = semantic_centers[semantics, 1] + np.random.randn(t_steps) * wiggle_scale
points = np.stack([x, y], axis=1)
true_semantic = semantics.copy()

# Colors
true_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown']
cluster_colors = ['#FFD700', '#6495ED', '#32CD32', '#FF69B4', '#FFA500', '#6A5ACD']

# Time-based initial clustering (force K clusters)
K = n_semantics
window = t_steps // K
time_clusters = np.zeros(t_steps, dtype=int)
for k in range(K):
    time_clusters[k*window:(k+1)*window] = k
if (K * window < t_steps):
    time_clusters[K*window:] = K-1

# Clustering/refinement iterations
n_iters = 6
embeddings = [points.copy()]
clusters = [time_clusters.copy()]
all_kmeans = []

for it in range(n_iters):
    km = KMeans(n_clusters=K, n_init=20, random_state=it)
    c = km.fit_predict(embeddings[-1])
    all_kmeans.append(km)
    clusters.append(c)
    # Move toward cluster center (blend)
    blend = 0.5
    centers = km.cluster_centers_
    new_embed = (1-blend)*embeddings[-1] + blend*centers[c]
    embeddings.append(new_embed)

# Visualization: plot each iteration
fig, axes = plt.subplots(2, 3, figsize=(19, 10))
for it in range(n_iters):
    ax = axes[it // 3, it % 3]
    ax.set_title(f"Iteration {it}: KMeans Clusters (K={K})", fontsize=14)
    # Plot points: facecolor=cluster, edgecolor=true_semantic
    for k in range(K):
        idxs = np.where(clusters[it+1] == k)[0]
        ax.scatter(
            embeddings[it+1][idxs, 0], embeddings[it+1][idxs, 1],
            facecolors=cluster_colors[k],
            edgecolors=[true_colors[true_semantic[j]] for j in idxs],
            linewidths=1.5, s=46, marker='o',
            alpha=0.34, label=f"Cluster {k}")
    # Draw arrows showing movement from previous embedding
    for i in range(t_steps):
        changed = clusters[it+1][i] != clusters[it][i]
        color = cluster_colors[clusters[it+1][i]] if changed else "gray"
        zorder = 4 if changed else 3
        ax.arrow(
            embeddings[it][i,0], embeddings[it][i,1],
            embeddings[it+1][i,0] - embeddings[it][i,0],
            embeddings[it+1][i,1] - embeddings[it][i,1],
            head_width=0.14, length_includes_head=True,
            color=color, alpha=0.68 if changed else 0.18, linewidth=1.5 if changed else 0.7, zorder=zorder
        )
    # Add tiny labels for time index (white text on true color circle)
    for t in range(t_steps):
        ax.text(
            embeddings[it+1][t,0], embeddings[it+1][t,1],
            f"{t}", fontsize=7, ha='center', va='center',
            color='white', weight='bold',
            bbox=dict(facecolor=true_colors[true_semantic[t]], alpha=0.35, edgecolor='none', boxstyle='circle,pad=0.13')
        )
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_xlim(np.min(points[:,0])-3, np.max(points[:,0])+3)
    ax.set_ylim(np.min(points[:,1])-3, np.max(points[:,1])+3)
    ax.grid(True)
    if it == 0:
        ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.show()

# Print cluster assignments at each step
header = "t\t(x,y)\t\ttrue_sem\t" + "\t".join([f"Iter{i}" for i in range(n_iters+1)])
print(header)
for t in range(t_steps):
    embed_str = f"({x[t]:6.2f},{y[t]:6.2f})"
    clusts = "\t".join(str(clusters[it][t]) for it in range(n_iters+1))
    print(f"{t:2d}\t{embed_str}\t   {true_semantic[t]}   \t{clusts}")
