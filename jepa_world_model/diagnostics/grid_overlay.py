"""Grid overlay embedding helpers for diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch

from gridworldkey_env import GridworldKeyEnv, create_env_with_theme
from jepa_world_model.rollout import rollout_teacher_forced
from jepa_world_model.pose_rollout import rollout_pose


@dataclass(frozen=True)
class GridOverlayEmbeddings:
    positions: np.ndarray
    grid_rows: int
    grid_cols: int
    z: np.ndarray
    h: np.ndarray
    p: Optional[np.ndarray]


@dataclass(frozen=True)
class GridOverlayFrames:
    frames: Sequence[np.ndarray]
    positions: np.ndarray
    grid_rows: int
    grid_cols: int


def _frame_to_tensor(frame: np.ndarray, image_size: int) -> torch.Tensor:
    pil = Image.fromarray(frame)
    pil = pil.resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _collect_grid_frames(env: GridworldKeyEnv) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    if env.grid_rows <= 0 or env.grid_cols <= 0:
        raise AssertionError("Grid overlay requires positive grid size.")
    frames = []
    positions = []
    for row in range(env.grid_rows):
        for col in range(env.grid_cols):
            obs, _ = env.reset(options={"start_tile": (row, col)})
            frames.append(obs)
            positions.append((row, col))
    return frames, np.asarray(positions, dtype=np.int64)


def build_grid_overlay_frames(
    *,
    theme: Optional[str],
    render_mode: str = "rgb_array",
) -> GridOverlayFrames:
    overlay_env = create_env_with_theme(
        theme=theme,
        render_mode=render_mode,
        keyboard_override=False,
        start_manual_control=False,
    )
    frames, positions = _collect_grid_frames(overlay_env)
    if not frames:
        raise AssertionError("No frames collected for grid overlay.")
    grid_rows = overlay_env.grid_rows
    grid_cols = overlay_env.grid_cols
    overlay_env.close()
    return GridOverlayFrames(
        frames=frames,
        positions=positions,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )


def save_grid_overlay_frame_grid(
    *,
    image_size: int,
    out_path: Path,
    frames_data: GridOverlayFrames,
) -> None:
    if image_size <= 0:
        raise AssertionError("image_size must be positive.")
    frames = frames_data.frames
    grid_rows = frames_data.grid_rows
    grid_cols = frames_data.grid_cols
    canvas = Image.new("RGB", (grid_cols * image_size, grid_rows * image_size))
    for idx, frame in enumerate(frames):
        row = idx // grid_cols
        col = idx % grid_cols
        img = Image.fromarray(frame).resize((image_size, image_size), Image.BILINEAR)
        canvas.paste(img, (col * image_size, row * image_size))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def build_grid_overlay_embeddings(
    *,
    model,
    model_cfg,
    device: torch.device,
    action_dim: int,
    use_z2h_init: bool,
    force_h_zero: bool = False,
    frames_data: Optional[GridOverlayFrames] = None,
    theme: Optional[str] = None,
) -> GridOverlayEmbeddings:
    if action_dim <= 0:
        raise AssertionError("Grid overlay requires positive action_dim.")
    if frames_data is None:
        if theme is None:
            raise AssertionError("Grid overlay requires frames_data or a theme.")
        frames_data = build_grid_overlay_frames(theme=theme)
    frames = frames_data.frames
    positions = frames_data.positions
    frames_tensor = torch.stack(
        [_frame_to_tensor(f, model_cfg.image_size) for f in frames],
        dim=0,
    ).to(device)

    with torch.no_grad():
        z_seq = model.encode_sequence(frames_tensor.unsqueeze(1))["embeddings"]
        z_embed = z_seq[:, 0]

        frames_seq2 = frames_tensor.unsqueeze(1).repeat(1, 2, 1, 1, 1)
        actions_zero = torch.zeros((frames_seq2.shape[0], 1, action_dim), device=device)
        z_seq2 = model.encode_sequence(frames_seq2)["embeddings"]
        _, _, h_states = rollout_teacher_forced(
            model,
            z_seq2,
            actions_zero,
            use_z2h_init=use_z2h_init,
            force_h_zero=force_h_zero,
        )
        h_embed = h_states[:, 0]

        p_embed = None
        if model.p_action_delta_projector is not None:
            _, pose_pred, _ = rollout_pose(
                model,
                h_states,
                actions_zero,
                z_embeddings=z_seq2,
            )
            p_embed = pose_pred[:, 0]

    return GridOverlayEmbeddings(
        positions=positions,
        grid_rows=frames_data.grid_rows,
        grid_cols=frames_data.grid_cols,
        z=z_embed.detach().cpu().numpy(),
        h=h_embed.detach().cpu().numpy(),
        p=None if p_embed is None else p_embed.detach().cpu().numpy(),
    )


__all__ = [
    "GridOverlayEmbeddings",
    "GridOverlayFrames",
    "build_grid_overlay_embeddings",
    "build_grid_overlay_frames",
    "save_grid_overlay_frame_grid",
]
