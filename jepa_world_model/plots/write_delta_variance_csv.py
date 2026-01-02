"""Delta variance CSV writer."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def write_delta_variance_csv(
    delta_dir: Path,
    global_step: int,
    variance_ratio: np.ndarray,
    embedding_label: str,
) -> None:
    delta_dir.mkdir(parents=True, exist_ok=True)
    delta_var_csv = delta_dir / f"delta_{embedding_label}_pca_variance_{global_step:07d}.csv"
    with delta_var_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component", "variance_ratio"])
        for idx, val in enumerate(variance_ratio[:64]):  # cap rows
            writer.writerow([idx, float(val)])
