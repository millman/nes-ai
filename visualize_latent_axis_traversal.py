#!/usr/bin/env python3
"""Axis-aligned latent traversal visualizer using predict_mario_multi decoder."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import torch
import tyro
from sklearn.decomposition import FastICA
import torchvision.transforms as T

from output_dir_utils import resolve_output_dir, TIMESTAMP_PLACEHOLDER
from predict_mario_multi import MultiHeadPredictor
from predict_mario_ms_ssim import (
    Mario4to1Dataset,
    pick_device,
    unnormalize,
)

OVERLAY_SCRIPT = """
(function() {
  const plotDiv = document.getElementById('latent-axis-plot');
  if (!plotDiv) { return; }

  let hoverDiv = document.getElementById('latent-axis-hover');
  if (!hoverDiv) {
    hoverDiv = document.createElement('div');
    hoverDiv.id = 'latent-axis-hover';
    hoverDiv.style.cssText = 'position:fixed; display:none; pointer-events:none; border:1px solid #999; background:rgba(255,255,255,0.98); padding:6px; z-index:1000; max-width:280px;';
    hoverDiv.innerHTML = `
      <img id="latent-axis-hover-img" src="" style="width:100%; height:auto; display:block; border-bottom:1px solid #ccc; margin-bottom:4px;">
      <div id="latent-axis-hover-info" style="font-size:12px; line-height:1.35;"></div>
    `;
    document.body.appendChild(hoverDiv);
  }

  const hoverImg = document.getElementById('latent-axis-hover-img');
  const hoverInfo = document.getElementById('latent-axis-hover-info');

  function positionHover(event) {
    if (!event) { return; }
    let x = (event.clientX || 0) + 16;
    let y = (event.clientY || 0) + 16;
    const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 0;
    const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;
    const rect = hoverDiv.getBoundingClientRect();
    const overlayWidth = rect.width || hoverDiv.offsetWidth || 0;
    const overlayHeight = rect.height || hoverDiv.offsetHeight || 0;
    if (overlayWidth && x + overlayWidth > viewportWidth - 4) {
      x = (event.clientX || 0) - overlayWidth - 16;
    }
    if (overlayHeight && y + overlayHeight > viewportHeight - 4) {
      y = (event.clientY || 0) - overlayHeight - 16;
    }
    hoverDiv.style.left = Math.max(x, 4) + 'px';
    hoverDiv.style.top = Math.max(y, 4) + 'px';
  }

  plotDiv.on('plotly_hover', function(data) {
    if (!data || !data.points || !data.points.length) { return; }
    const point = data.points[0];
    const custom = point.customdata || [];
    const imgPath = custom[0] || '';
    const gifPath = custom[1] || '';
    const axisVal = custom[2];
    hoverImg.src = imgPath;
    hoverInfo.innerHTML = [
      'Axis value: ' + (axisVal != null ? Number(axisVal).toFixed(4) : ''),
      gifPath ? ('Traversal GIF: ' + gifPath) : ''
    ].filter(Boolean).join('<br>');
    hoverDiv.style.display = 'block';
    positionHover(data.event);
  });

  plotDiv.on('plotly_unhover', function() {
    hoverDiv.style.display = 'none';
  });

  plotDiv.on('plotly_relayout', function() {
    hoverDiv.style.display = 'none';
  });

  plotDiv.addEventListener('mousemove', function(evt) {
    if (hoverDiv.style.display === 'block') {
      positionHover(evt);
    }
  });
})();
"""

DEFAULT_OUT_DIR_TEMPLATE = f"out.latent_axis_traversal_{TIMESTAMP_PLACEHOLDER}"


@dataclass
class Args:
    traj_dir: str = "data.image_distance.train_levels_1_2"
    model_checkpoint: str = "out.predict_mario_multi/run__2025-10-19_17-00-49/checkpoint.pt"
    out_dir: str = DEFAULT_OUT_DIR_TEMPLATE
    rollout_steps: int = 4
    max_trajs: Optional[int] = None
    max_samples: Optional[int] = 2048
    axis_count: int = 5
    anchors_per_axis: int = 5
    samples_per_anchor: int = 10
    device: Optional[str] = None
    random_state: int = 0
    allow_incompatible_checkpoints: bool = False


def load_latents(args: Args, device: torch.device, out_dir: Path) -> tuple[np.ndarray, List[dict], MultiHeadPredictor]:
    dataset = Mario4to1Dataset(args.traj_dir, max_trajs=args.max_trajs, rollout=args.rollout_steps)
    model = MultiHeadPredictor(rollout=args.rollout_steps).to(device)
    state = torch.load(args.model_checkpoint, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=not args.allow_incompatible_checkpoints)
    if args.allow_incompatible_checkpoints:
        if missing:
            print(f"[warn] Missing keys when loading model: {missing}")
        if unexpected:
            print(f"[warn] Unexpected keys ignored: {unexpected}")
    model.eval()

    latents: List[np.ndarray] = []
    samples: List[dict] = []

    total = len(dataset)
    limit = min(total, args.max_samples or total)
    with torch.no_grad():
        for idx in range(limit):
            x, _ = dataset[idx]
            files, offset = dataset.index[idx]
            context_paths = files[offset:offset + 4]
            target_path = files[offset + 4]
            latent = model(x.unsqueeze(0).to(device))[2]
            latents.append(latent.squeeze(0).cpu().numpy())
            samples.append({
                "context_paths": [str(p) for p in context_paths],
                "target_path": str(target_path),
                "latent": latent.squeeze(0).cpu(),
            })
    return np.stack(latents, axis=0), samples, model


def select_anchor_indices(axis_values: np.ndarray, count: int) -> List[int]:
    targets = np.linspace(axis_values.min(), axis_values.max(), count)
    used: set[int] = set()
    indices: List[int] = []
    for target in targets:
        order = np.argsort(np.abs(axis_values - target))
        choice = next((idx for idx in order if idx not in used), order[0])
        used.add(choice)
        indices.append(int(choice))
    return indices


def compute_bounds(sorted_values: np.ndarray) -> np.ndarray:
    bounds = np.zeros((len(sorted_values), 2), dtype=float)
    for i, val in enumerate(sorted_values):
        if len(sorted_values) == 1:
            bounds[i] = (val - 0.5, val + 0.5)
        elif i == 0:
            next_val = sorted_values[i + 1]
            half = (next_val - val) / 2.0
            bounds[i] = (val - half, val + half)
        elif i == len(sorted_values) - 1:
            prev_val = sorted_values[i - 1]
            half = (val - prev_val) / 2.0
            bounds[i] = (val - half, val + half)
        else:
            prev_val = sorted_values[i - 1]
            next_val = sorted_values[i + 1]
            bounds[i] = ((prev_val + val) / 2.0, (val + next_val) / 2.0)
    return bounds


def decode_frames(model: MultiHeadPredictor, latents: np.ndarray, device: torch.device) -> List[Image.Image]:
    frames: List[Image.Image] = []
    to_pil = T.ToPILImage()
    with torch.no_grad():
        latent_tensor = torch.from_numpy(latents.astype(np.float32)).to(device)
        recon = model.decode_from_latent(latent_tensor)
        recon = unnormalize(recon).cpu().clamp(0, 1)
        for frame in recon:
            frames.append(to_pil(frame))
    return frames


def save_gif(frames: List[Image.Image], path: Path, duration: int = 120) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    base = frames[0].convert("RGB")
    rest = [img.convert("RGB") for img in frames[1:]]
    base.save(path, format="GIF", append_images=rest, save_all=True, duration=duration, loop=0)


def main(args: Args) -> None:
    device = pick_device(args.device)
    out_dir = resolve_output_dir(args.out_dir, default_template=DEFAULT_OUT_DIR_TEMPLATE)
    latents, samples, model = load_latents(args, device, out_dir)
    print(f"Collected {latents.shape[0]} latent vectors")

    axis_count = min(args.axis_count, latents.shape[1])
    ica = FastICA(n_components=axis_count, random_state=args.random_state)
    components = ica.fit_transform(latents)

    traversal_dir = out_dir / "traversals"
    traversal_dir.mkdir(parents=True, exist_ok=True)

    traj_names = [Path(info["target_path"]).parents[1].name for info in samples]
    unique_trajs = sorted(set(traj_names))
    palette = go.Figure().layout.colorway or ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    color_map = {traj: palette[i % len(palette)] for i, traj in enumerate(unique_trajs)}

    fig = make_subplots(rows=axis_count, cols=1,
                        subplot_titles=[f"ICA axis {i + 1}" for i in range(axis_count)])

    for axis_idx in range(axis_count):
        axis_values = components[:, axis_idx]

        anchor_indices = select_anchor_indices(axis_values, args.anchors_per_axis)
        anchor_values = axis_values[anchor_indices]
        sorted_idx = np.argsort(anchor_values)
        bounds_sorted = compute_bounds(anchor_values[sorted_idx])
        bounds = np.empty_like(bounds_sorted)
        bounds[sorted_idx] = bounds_sorted

        anchor_custom = []
        anchor_colors = []
        for local_idx, sample_idx in enumerate(anchor_indices):
            base_components = components[sample_idx].copy()
            lower, upper = bounds[local_idx]
            samples_vals = np.linspace(lower, upper, args.samples_per_anchor)

            traversal_samples = []
            for val in samples_vals:
                s_new = base_components.copy()
                s_new[axis_idx] = val
                latent_new = ica.inverse_transform(s_new.reshape(1, -1))[0]
                traversal_samples.append(latent_new)
            traversal_arr = np.stack(traversal_samples, axis=0)
            frames = decode_frames(model, traversal_arr, device)

            axis_dir = traversal_dir / f"axis_{axis_idx + 1}"
            sample_dir = axis_dir / f"anchor_{local_idx + 1}_frames"
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_images = []
            for idx_sample, frame_img in enumerate(frames):
                frame_path = sample_dir / f"sample_{idx_sample:03d}.png"
                frame_img.save(frame_path)
                sample_images.append(frame_path.relative_to(out_dir).as_posix())

            gif_path = axis_dir / f"anchor_{local_idx + 1}.gif"
            save_gif(frames, gif_path)

            anchor_src = Path(samples[sample_idx]["target_path"])
            anchor_copy = axis_dir / f"anchor_{local_idx + 1}_frame.png"
            anchor_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(anchor_src, anchor_copy)
            traj_name = anchor_src.parents[1].name
            anchor_colors.append(color_map[traj_name])

            anchor_custom.append([
                anchor_copy.relative_to(out_dir).as_posix(),
                gif_path.relative_to(out_dir).as_posix(),
                anchor_values[local_idx],
            ])

            sample_custom = np.zeros((len(frames), 3), dtype=object)
            sample_custom[:, 0] = sample_images
            sample_custom[:, 1] = gif_path.relative_to(out_dir).as_posix()
            sample_custom[:, 2] = samples_vals

            fig.add_trace(
                go.Scatter(
                    x=samples_vals,
                    y=np.zeros_like(samples_vals),
                    mode='markers',
                    marker=dict(size=6, color=color_map[traj_name], symbol='circle-open'),
                    showlegend=False,
                    customdata=sample_custom,
                    hovertemplate="Traversal frame %{customdata[0]}<br>Axis value %{customdata[2]:.4f}<extra></extra>",
                ),
                row=axis_idx + 1,
                col=1,
            )

        anchor_custom = np.array(anchor_custom, dtype=object)
        fig.add_trace(
            go.Scatter(
                x=anchor_values,
                y=np.zeros_like(anchor_values),
                mode='markers',
                marker=dict(size=10, color=anchor_colors, symbol='diamond'),
                name=f"Axis {axis_idx + 1} anchors",
                customdata=anchor_custom,
                hovertemplate="Anchor frame %{customdata[0]}<br>Axis value %{x:.4f}<extra></extra>",
            ),
            row=axis_idx + 1,
            col=1,
        )
        fig.update_yaxes(range=[-0.5, 0.5], showticklabels=False, row=axis_idx + 1, col=1)
        fig.update_xaxes(title_text=f"ICA axis {axis_idx + 1} value", row=axis_idx + 1, col=1)

    fig.update_layout(title="Latent axis traversal", height=320 * axis_count, hovermode='closest')
    html_path = out_dir / "latent_axis_traversal.html"
    fig.write_html(html_path, include_plotlyjs='cdn', full_html=True,
                   default_height='100vh', default_width='100%', div_id='latent-axis-plot',
                   post_script=OVERLAY_SCRIPT)
    print(f"Saved traversal plot to {html_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
