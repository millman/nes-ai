from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch

from jepa_world_model.vis import (
    _annotate_with_text,
    describe_action_tensor,
    tensor_to_uint8_image,
)


def save_temporal_pair_visualization(
    out_path: Path,
    frames: torch.Tensor,
    actions: torch.Tensor,
    rows: int,
    generator: torch.Generator | None = None,
) -> None:
    if frames.shape[1] < 2:
        return
    frames = frames.detach().cpu()
    actions = actions.detach().cpu()
    batch_size = frames.shape[0]
    num_rows = min(rows, batch_size)
    order = torch.randperm(batch_size, generator=generator)[:num_rows]
    pairs: list[np.ndarray] = []
    for idx in order:
        time_idx = torch.randint(1, frames.shape[1], (), generator=generator).item()
        prev_frame = tensor_to_uint8_image(frames[idx, time_idx - 1])
        next_frame = tensor_to_uint8_image(frames[idx, time_idx])
        prev_frame = _annotate_with_text(prev_frame, describe_action_tensor(actions[idx, time_idx - 1]))
        next_frame = _annotate_with_text(next_frame, describe_action_tensor(actions[idx, time_idx]))
        pairs.append(np.concatenate([prev_frame, next_frame], axis=1))
    grid = np.concatenate(pairs, axis=0) if pairs else np.zeros((1, 1, 3), dtype=np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)


__all__ = ["save_temporal_pair_visualization"]
