"""Action alignment overview text writer."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np


def write_action_alignment_overview_txt(
    alignment_dir: Path,
    global_step: int,
    cosine_high_threshold: float,
    alignment_debug: Dict[str, Any],
) -> None:
    alignment_dir.mkdir(parents=True, exist_ok=True)
    overview_txt = alignment_dir / f"action_alignment_overview_{global_step:07d}.txt"
    overall_cos_raw = alignment_debug.get("overall_cos")
    overall_norms_raw = alignment_debug.get("overall_norms")
    overall_cos = np.asarray([] if overall_cos_raw is None else overall_cos_raw, dtype=np.float32)
    overall_norms = np.asarray([] if overall_norms_raw is None else overall_norms_raw, dtype=np.float32)
    with overview_txt.open("w") as handle:
        handle.write("Global alignment summary (cosine vs per-action mean)\n")
        if overall_cos.size == 0:
            handle.write("No valid cosine samples.\n")
        else:
            handle.write(f"samples: {overall_cos.size}\n")
            handle.write(f"mean: {float(overall_cos.mean()):.4f}\n")
            handle.write(f"median: {float(np.median(overall_cos)):.4f}\n")
            handle.write(f"std: {float(overall_cos.std()):.4f}\n")
            handle.write(
                f"pct_gt_thr({cosine_high_threshold}): {float((overall_cos > cosine_high_threshold).mean()):.4f}\n"
            )
            handle.write(f"frac_negative: {float((overall_cos < 0).mean()):.4f}\n")
            if overall_norms.size:
                handle.write("\nDelta norm stats (all actions):\n")
                handle.write(f"median: {float(np.median(overall_norms)):.6f}\n")
                handle.write(f"p10: {float(np.percentile(overall_norms, 10)):.6f}\n")
                handle.write(f"p90: {float(np.percentile(overall_norms, 90)):.6f}\n")
