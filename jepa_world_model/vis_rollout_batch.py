from __future__ import annotations

from pathlib import Path
from typing import Sequence

from jepa_world_model.vis_rollout import VisualizationSequence
from jepa_world_model.vis_rollout_visualization import save_rollout_visualization


def save_rollout_sequence_batch(
    template_dir: Path,
    sequences: Sequence[VisualizationSequence],
    grad_label: str,
    global_step: int,
    include_pixel_delta: bool,
) -> None:
    if not sequences:
        return
    base_parent = template_dir.parent
    base_name = template_dir.name
    for idx, sequence in enumerate(sequences):
        sample_dir = base_parent / f"{base_name}_{idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        out_path = sample_dir / f"rollout_{global_step:07d}.png"
        save_rollout_visualization(out_path, sequence, grad_label, include_pixel_delta)


__all__ = ["save_rollout_sequence_batch"]
