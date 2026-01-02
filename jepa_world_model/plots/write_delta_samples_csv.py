"""Delta sample CSV writer."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict


def write_delta_samples_csv(
    delta_dir: Path,
    global_step: int,
    motion: Dict[str, Any],
    embedding_label: str,
) -> None:
    delta_dir.mkdir(parents=True, exist_ok=True)
    delta_samples_csv = delta_dir / f"delta_{embedding_label}_pca_samples_{global_step:07d}.csv"
    with delta_samples_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "frame_index", "frame_path"])
        paths = motion.get("paths") or []
        if paths:
            for sample_idx, frame_list in enumerate(paths):
                if not frame_list:
                    continue
                writer.writerow([sample_idx, 0, frame_list[0]])
