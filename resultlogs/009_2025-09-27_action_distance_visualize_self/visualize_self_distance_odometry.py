#!/usr/bin/env python3
"""Per-trajectory latent odometry visualization driven by PCA axes."""
from __future__ import annotations

import csv
import ctypes
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
import torch
import tyro

from output_dir_utils import TIMESTAMP_PLACEHOLDER, resolve_output_dir
from predict_mario_ms_ssim import pick_device
from self_distance_utils import compute_self_distance_results, copy_frames_for_visualization

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _set_max_active_levels(levels: int = 1) -> None:
    """Attempt to call omp_set_max_active_levels to avoid nested warning."""
    lib_names = [
        "libomp.dylib",
        "libiomp5.dylib",
        "libomp.so",
        "libiomp5.so",
    ]
    func = None
    for name in lib_names:
        try:
            lib = ctypes.CDLL(name)
            func = lib.omp_set_max_active_levels
            break
        except OSError:
            continue
        except AttributeError:
            func = None
            continue
    if func is None:
        return
    try:
        func.argtypes = [ctypes.c_int]
        func.restype = None
        func(levels)
    except Exception:
        pass


DEFAULT_OUT_DIR_TEMPLATE = f"out.self_distance_odometry_{TIMESTAMP_PLACEHOLDER}"
PLOT_DIV_ID = "self-distance-odometry-plot"

OVERLAY_POST_SCRIPT = f"""
(function() {{
  const plotDiv = document.getElementById('{PLOT_DIV_ID}');
  if (!plotDiv) {{
    return;
  }}

  const hoverDiv = document.createElement('div');
  hoverDiv.id = 'hover-image';
  hoverDiv.style.cssText = 'position:fixed; display:none; pointer-events:none; border:1px solid #999; background:rgba(255,255,255,0.98); padding:4px; z-index:1000; max-width:260px;';
  hoverDiv.innerHTML = `
    <img id="hover-image-img" src="" style="width:100%; height:auto; display:block; border-bottom:1px solid #ccc; margin-bottom:4px;">
    <div id="hover-image-details" style="font-size:12px; line-height:1.4;"></div>
  `;
  document.body.appendChild(hoverDiv);

  const hoverImg = document.getElementById('hover-image-img');
  const hoverDetails = document.getElementById('hover-image-details');

  function positionHover(event) {{
    if (!event) {{
      return;
    }}
    let x = (event.clientX || 0) + 16;
    let y = (event.clientY || 0) + 16;

    const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 0;
    const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;
    const rect = hoverDiv.getBoundingClientRect();
    const overlayWidth = rect.width || hoverDiv.offsetWidth || 0;
    const overlayHeight = rect.height || hoverDiv.offsetHeight || 0;

    if (overlayWidth && x + overlayWidth > viewportWidth - 4) {{
      x = (event.clientX || 0) - overlayWidth - 16;
    }}
    if (overlayHeight && y + overlayHeight > viewportHeight - 4) {{
      y = (event.clientY || 0) - overlayHeight - 16;
    }}

    if (x < 4) {{
      x = 4;
    }}
    if (y < 4) {{
      y = 4;
    }}

    hoverDiv.style.left = x + 'px';
    hoverDiv.style.top = y + 'px';
  }}

  plotDiv.on('plotly_hover', function(data) {{
    if (!data || !data.points || !data.points.length) {{
      return;
    }}
    const point = data.points[0];
    const custom = point.customdata || [];
    const imgPath = custom[0];
    const trajName = custom[1] || '';
    const frameIdx = custom[2] != null ? custom[2] : '';
    const l2 = custom[3];
    const cos = custom[4];

    hoverImg.src = imgPath || '';
    const l2Text = (l2 != null && !isNaN(l2)) ? Number(l2).toFixed(4) : '';
    const cosText = (cos != null && !isNaN(cos)) ? Number(cos).toFixed(4) : '';
    hoverDetails.innerHTML = [
      '<strong>' + trajName + '</strong>',
      'Frame offset: ' + frameIdx,
      'L2 distance: ' + l2Text,
      'Cosine distance: ' + cosText,
    ].join('<br>');

    hoverDiv.style.display = 'block';
    positionHover(data.event);
  }});

  plotDiv.on('plotly_unhover', function() {{
    hoverDiv.style.display = 'none';
  }});

  plotDiv.on('plotly_relayout', function() {{
    hoverDiv.style.display = 'none';
  }});

  plotDiv.addEventListener('mousemove', function(evt) {{
    if (hoverDiv.style.display === 'block') {{
      positionHover(evt);
    }}
  }});
}})();
"""


@dataclass
class Args:
    traj_dir: str
    out_dir: str = DEFAULT_OUT_DIR_TEMPLATE
    max_trajs: Optional[int] = None
    device: Optional[str] = None
    grid_cols: int = 3
    progress: bool = True


@dataclass
class TrajectoryPlotData:
    name: str
    safe_name: str
    positions: np.ndarray
    frame_offsets: np.ndarray
    l2_values: np.ndarray
    cos_values: np.ndarray
    image_paths: np.ndarray


def write_metadata_csv(records: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "trajectory",
            "frame_index",
            "l2_distance",
            "cosine_distance",
            "image_path",
            "safe_trajectory_name",
        ])
        for rec in records:
            writer.writerow([
                rec["traj"],
                rec["frame_index"],
                rec["l2"],
                rec["cos"],
                rec["image_path"],
                rec["safe_traj"],
            ])


def compute_principal_components(delta_vectors: torch.Tensor) -> torch.Tensor:
    """Return the top two principal directions for the provided deltas."""
    if delta_vectors.ndim != 2 or delta_vectors.numel() == 0:
        raise ValueError("delta_vectors must be a non-empty 2D tensor")
    mean = delta_vectors.mean(dim=0, keepdim=True)
    centered = delta_vectors - mean
    cov = centered.T @ centered
    # Numerical guard: if all deltas are identical, covariance is zero and eigh returns arbitrary basis.
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    top = evecs[:, order[:2]]
    if top.shape[1] < 2:
        pad = torch.zeros((top.shape[0], 2 - top.shape[1]))
        top = torch.cat([top, pad], dim=1)
    return top


def accumulate_odometry_positions(embeddings: torch.Tensor, components: torch.Tensor) -> np.ndarray:
    """Convert embeddings into planar odometry coordinates using PCA components."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D tensor")
    if embeddings.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    positions = torch.zeros((embeddings.shape[0], 2), dtype=torch.float32)
    for idx in range(1, embeddings.shape[0]):
        delta = embeddings[idx] - embeddings[idx - 1]
        delta_xy = delta @ components
        positions[idx] = positions[idx - 1] + delta_xy[:2]
    return positions.numpy()


def build_plot_inputs(
    results: Sequence,
    copied_paths: Sequence[Sequence[Path]],
    out_dir: Path,
    components: torch.Tensor,
) -> Tuple[List[dict], List[TrajectoryPlotData]]:
    records: List[dict] = []
    plot_data: List[TrajectoryPlotData] = []

    for res, paths in zip(results, copied_paths):
        display_name = res.relative_dir.as_posix()
        positions = accumulate_odometry_positions(res.embeddings, components)
        frame_offsets = np.arange(len(paths))
        l2_values = res.l2_distances.numpy()
        cos_values = res.cosine_distances.numpy()
        image_paths = np.array([
            (dst.relative_to(out_dir).as_posix()) for dst in paths
        ])
        for idx, rel_path in enumerate(image_paths):
            records.append(
                {
                    "traj": display_name,
                    "safe_traj": res.traj_name,
                    "frame_index": idx,
                    "l2": float(l2_values[idx]),
                    "cos": float(cos_values[idx]),
                    "image_path": rel_path,
                }
            )
        plot_data.append(
            TrajectoryPlotData(
                name=display_name,
                safe_name=res.traj_name,
                positions=positions,
                frame_offsets=frame_offsets,
                l2_values=l2_values,
                cos_values=cos_values,
                image_paths=image_paths,
            )
        )

    return records, plot_data


def build_fig(traj_data: List[TrajectoryPlotData], grid_cols: int) -> go.Figure:
    if not traj_data:
        raise ValueError("No trajectory data available for plotting")

    total = len(traj_data)
    cols = max(1, grid_cols)
    cols = min(cols, total)
    rows = math.ceil(total / cols)
    subplot_titles = [td.name for td in traj_data]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    hover_template = "<extra></extra>"
    for idx, data in enumerate(traj_data):
        row = idx // cols + 1
        col = idx % cols + 1
        offsets = data.frame_offsets.astype(float)
        if offsets.size:
            max_offset = float(offsets.max())
            if max_offset > 0:
                norm_offsets = offsets / max_offset
            else:
                norm_offsets = np.zeros_like(offsets)
        else:
            norm_offsets = np.zeros_like(offsets)
        color_strings = np.array(
            sample_colorscale('Viridis', norm_offsets.tolist()),
            dtype=object,
        )
        custom = np.array(
            list(
                zip(
                    data.image_paths,
                    np.repeat(data.name, len(data.image_paths)),
                    data.frame_offsets.astype(float),
                    data.l2_values,
                    data.cos_values,
                )
            ),
            dtype=object,
        )
        fig.add_trace(
            go.Scatter(
                x=data.positions[:, 0],
                y=data.positions[:, 1],
                mode='markers',
                name=data.name,
                showlegend=False,
                marker=dict(
                    size=7,
                    color=norm_offsets,
                    colorscale='Viridis',
                    cmin=0.0,
                    cmax=1.0,
                    opacity=0.9,
                    showscale=False,
                ),
                customdata=custom,
                hovertemplate=hover_template,
            ),
            row=row,
            col=col,
        )
        axis_index = (row - 1) * cols + col
        xref = 'x1' if axis_index == 1 else f'x{axis_index}'
        yref = 'y1' if axis_index == 1 else f'y{axis_index}'
        for point_idx in range(len(data.positions) - 1):
            start = data.positions[point_idx]
            end = data.positions[point_idx + 1]
            arrow_color = str(color_strings[point_idx])
            fig.add_annotation(
                x=float(end[0]),
                y=float(end[1]),
                ax=float(start[0]),
                ay=float(start[1]),
                xref=xref,
                yref=yref,
                axref=xref,
                ayref=yref,
                showarrow=True,
                arrowhead=3,
                arrowsize=1.0,
                arrowwidth=1.2,
                arrowcolor=arrow_color,
                opacity=0.9,
            )
        fig.update_xaxes(title_text='Δ PC1', row=row, col=col)
        fig.update_yaxes(title_text='Δ PC2', row=row, col=col)

    fig.update_layout(
        title='Per-trajectory latent odometry (PCA-aligned)',
        hovermode='closest',
        template='plotly_white',
        height=max(600, rows * 400),
    )
    return fig


def main(args: Args) -> None:
    _set_max_active_levels(1)
    device = pick_device(args.device)
    traj_dir = Path(args.traj_dir)
    out_dir = resolve_output_dir(
        args.out_dir,
        default_template=DEFAULT_OUT_DIR_TEMPLATE,
    )
    frames_dir = out_dir / "frames"

    results = compute_self_distance_results(
        traj_dir,
        device=device,
        max_trajs=args.max_trajs,
        progress=args.progress,
    )
    if not results:
        print(f"No trajectories processed in {traj_dir}")
        return

    copied_paths = copy_frames_for_visualization(results, frames_dir)

    delta_vectors: List[torch.Tensor] = []
    for res in results:
        if res.embeddings.shape[0] > 1:
            delta_vectors.append(res.embeddings[1:] - res.embeddings[:-1])
    if not delta_vectors:
        print("Insufficient frames to compute odometry deltas")
        return
    all_deltas = torch.cat(delta_vectors, dim=0)
    components = compute_principal_components(all_deltas)

    records, plot_data = build_plot_inputs(results, copied_paths, out_dir, components)
    write_metadata_csv(records, out_dir / "self_distance_odometry_records.csv")

    fig = build_fig(plot_data, args.grid_cols)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "self_distance_odometry.html"
    fig.write_html(
        html_path,
        include_plotlyjs='cdn',
        full_html=True,
        default_height='100vh',
        default_width='100%',
        div_id=PLOT_DIV_ID,
        post_script=OVERLAY_POST_SCRIPT,
    )
    print(f"Saved odometry visualization to {html_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
