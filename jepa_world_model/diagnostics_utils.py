from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import csv
import torch


def append_csv_row(path: Path, header: Sequence[str], row: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(header)
        writer.writerow(row)


def write_step_csv(out_dir: Path, name: str, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def compute_norm_stats(tensor: torch.Tensor, quantile: float = 0.95) -> Tuple[float, float]:
    if tensor.numel() == 0:
        return 0.0, 0.0
    norms = tensor.detach().flatten(0, -2).norm(dim=-1)
    if norms.numel() == 0:
        return 0.0, 0.0
    mean_val = float(norms.mean().item())
    quantile_val = float(torch.quantile(norms, quantile).item())
    return mean_val, quantile_val
