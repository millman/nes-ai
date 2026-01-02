"""Variance report helper for diagnostics."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_variance_report(
    variance_ratio: np.ndarray,
    out_dir: Path,
    global_step: int,
    embedding_label: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"delta_{embedding_label}_pca_report_{global_step:07d}.txt"
    with report_path.open("w") as handle:
        if variance_ratio.size == 0:
            handle.write("No variance ratios available.\n")
            return
        cumulative = np.cumsum(variance_ratio)
        targets = [1, 2, 4, 8, 16, 32]
        handle.write("Explained variance coverage by component count:\n")
        for t in targets:
            if t <= len(cumulative):
                handle.write(f"top_{t:02d}: {cumulative[t-1]:.4f}\n")
        handle.write("\nTop variance ratios:\n")
        top_k = min(10, len(variance_ratio))
        for i in range(top_k):
            handle.write(f"comp_{i:02d}: {variance_ratio[i]:.6f}\n")
        handle.write(f"\nTotal components: {len(variance_ratio)}\n")
