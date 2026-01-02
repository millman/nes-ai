from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch

from jepa_world_model.vis import (
    _annotate_with_text,
    delta_to_uint8_image,
    describe_action_tensor,
    tensor_to_uint8_image,
)


def save_input_batch_visualization(
    out_path: Path,
    frames: torch.Tensor,
    actions: torch.Tensor,
    rows: int,
    recon: Optional[torch.Tensor] = None,
    include_deltas: bool = False,
) -> None:
    frames = frames.detach().cpu()
    actions = actions.detach().cpu()
    if recon is not None:
        recon = recon.detach().cpu()
    batch_size, seq_len = frames.shape[0], frames.shape[1]
    num_rows = min(rows, batch_size)
    if num_rows <= 0:
        return
    grid_rows: list[np.ndarray] = []
    for row_idx in range(num_rows):
        columns: list[np.ndarray] = []
        for step in range(seq_len):
            frame_img = tensor_to_uint8_image(frames[row_idx, step])
            desc = describe_action_tensor(actions[row_idx, step])
            frame_img = _annotate_with_text(frame_img, desc)
            columns.append(frame_img)
        row_image = np.concatenate(columns, axis=1)
        grid_rows.append(row_image)
        if include_deltas and recon is not None and seq_len > 1:
            delta_target = frames[row_idx, 1:] - frames[row_idx, :-1]
            delta_recon = recon[row_idx, 1:] - recon[row_idx, :-1]
            delta_rows = []
            delta_recon_rows = []
            zero = np.zeros_like(tensor_to_uint8_image(frames[row_idx, 0]))
            delta_rows.append(_annotate_with_text(zero, "delta_tgt"))
            delta_recon_rows.append(_annotate_with_text(zero, "delta_rec"))
            for step in range(seq_len - 1):
                delta_img = delta_to_uint8_image(delta_target[step])
                delta_img = _annotate_with_text(delta_img, f"t{step+1}")
                delta_rows.append(delta_img)
                delta_rec_img = delta_to_uint8_image(delta_recon[step])
                delta_rec_img = _annotate_with_text(delta_rec_img, f"t{step+1}")
                delta_recon_rows.append(delta_rec_img)
            grid_rows.append(np.concatenate(delta_rows, axis=1))
            grid_rows.append(np.concatenate(delta_recon_rows, axis=1))
    grid = np.concatenate(grid_rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)


__all__ = ["save_input_batch_visualization"]
