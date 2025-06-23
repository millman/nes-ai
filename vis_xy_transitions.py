# %%

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _get_most_recent_file(directory: str):
    directory = Path(directory)

    # Filter to only files (not directories)
    files = [f for f in directory.iterdir() if f.is_file()]

    if not files:
        return None  # or raise an exception if preferred

    # Get the file with the latest modification time
    most_recent = max(files, key=lambda f: f.stat().st_mtime)
    return most_recent


runs_dir = Path("~/rl/nes-ai/runs").expanduser()

#load_path = runs_dir / "smb-search-v0__search_mario__1__2025-06-22_13-15-00/xy_transitions/level_1-1_30156.pkl"
#load_path = runs_dir / "smb-search-v0__search_mario__1__2025-06-23_00-12-20/xy_transitions/level_1-2_129941_end.pkl"
#load_path = runs_dir / "smb-search-v0__search_mario__1__2025-06-22_13-15-00/xy_transitions/"
load_path = runs_dir / "smb-search-v0__search_mario__1__2025-06-23_09-24-37/xy_transitions/level_8-4_616299.pkl"

if load_path.is_dir():
    # Find most recent path in transition directory.
    filepath = _get_most_recent_file(load_path)
else:
    filepath = load_path



print(f"xy transitions file: {filepath}")

with open(filepath, "rb") as infile:
    xy_transitions = pickle.load(infile)

print(f"transition count: {xy_transitions.total()}")

# Extract all x, y coordinates.
x0s = [x0 for (x0, y0), (x1, y1) in xy_transitions.keys()]
y0s = [y0 for (x0, y0), (x1, y1) in xy_transitions.keys()]
x1s = [x1 for (x0, y0), (x1, y1) in xy_transitions.keys()]
y1s = [y1 for (x0, y0), (x1, y1) in xy_transitions.keys()]

# Determine coordinate bounds (add padding if desired).
x_min = int(min(min(x0s), min(x1s)))
x_max = int(max(max(x0s), max(x1s)))

y_min = int(min(min(y0s), min(y1s)))
y_max = int(max(max(y0s), max(y1s)))

# Create a grid for the heatmap
x_range = x_max - x_min + 1
y_range = y_max - y_min + 1
heatmap = np.zeros((y_range, x_range), dtype=int)

print(f"x range: {x_range}, ({x_min} -> {x_max})")
print(f"y range: {y_range}, ({y_min} -> {y_max})")

# Fill heatmap with transition counts
for (x0, y0), (x1, y1) in xy_transitions:
    xi = x1 - x_min
    yi = y1 - y_min
    heatmap[yi, xi] += 1

# Mask zeros to show them in black.
masked_heatmap = np.ma.masked_where(heatmap == 0, heatmap)

# Define colormap with black for masked values
cmap = plt.cm.viridis
# cmap.set_bad(color='white')

# Figure size: 1 inch per N units (adjust scaling_factor as needed)
# Normally use 8,6 for 800x600.  We expect 3000x224
scaling_factor = 0.02  # inch per unit
fig_width = x_range * scaling_factor
fig_height = y_range * scaling_factor

# Plot heatmap with axis labels reflecting real x,y positions
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
im = ax.imshow(
    masked_heatmap,
    origin='upper',
    cmap=cmap,
    extent=(x_min, x_max + 1, y_min, y_max + 1),
    interpolation='none',
    vmin=1, vmax=3,
    # masked_heatmap.max(),
)

ax.set_title("Transition Endpoint Heatmap with World Coordinates")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
plt.colorbar(im, ax=ax, label="Transition Count")
plt.tight_layout()
plt.show()


# %%

if True:
    print(f"vmax: {masked_heatmap.max()}")
    b_min = 0
    b_max = masked_heatmap.max()

    # Quick and dirty histogram.
    n_bins = 30
    bin_width = (b_max - b_min) / n_bins
    bins = np.zeros(n_bins + 1, dtype=np.int64)

    for j in range(y_range):
        for i in range(x_range):
            count = heatmap[j, i]
            if count == 0:
                continue
            bin = int((count  - b_min) / bin_width)
            bins[bin] += 1

    print("heatmap histogram")
    print()
    for i, count in enumerate(bins):
        b0 = b_min + i*bin_width
        b1 = b_min + (i+1)*bin_width
        print(f"  [{b0:.2f} -> {b1:.2f}]: {count}")

