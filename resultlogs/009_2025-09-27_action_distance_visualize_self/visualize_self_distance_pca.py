#!/usr/bin/env python3
"""Interactive PCA-based self-distance visualizer with Plotly hover images."""
from __future__ import annotations

import csv
import os
import ctypes
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import cm
from PIL import Image
import torch
import tyro
from sklearn.decomposition import FastICA

from output_dir_utils import TIMESTAMP_PLACEHOLDER, resolve_output_dir
from predict_mario_ms_ssim import pick_device
from self_distance_utils import compute_self_distance_results, copy_frames_for_visualization

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _set_max_active_levels(levels: int = 1) -> None:
    """Attempt to call omp_set_max_active_levels to avoid nested warning."""
    lib_names = [
        "libomp.dylib",  # macOS (Homebrew/Apple)
        "libiomp5.dylib",
        "libomp.so",      # Linux generic
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


DEFAULT_OUT_DIR_TEMPLATE = f"out.self_distance_pca_{TIMESTAMP_PLACEHOLDER}"
SCATTER_DIV_ID = "self-distance-pca-plot"
AXES_DIV_ID = "self-distance-pca-axes-plot"


def _build_scatter_overlay_script(div_id: str) -> str:
    return f"""
(function() {{
  const plotDiv = document.getElementById('{div_id}');
  if (!plotDiv) {{
    return;
  }}

  let hoverDiv = document.getElementById('{div_id}-hover');
  if (!hoverDiv) {{
    hoverDiv = document.createElement('div');
    hoverDiv.id = '{div_id}-hover';
    hoverDiv.style.cssText = 'position:fixed; display:none; pointer-events:none; border:1px solid #999; background:rgba(255,255,255,0.98); padding:4px; z-index:1000; max-width:260px;';
    hoverDiv.innerHTML = `
      <img id="{div_id}-hover-img" src="" style="width:100%; height:auto; display:block; border-bottom:1px solid #ccc; margin-bottom:4px;">
      <div id="{div_id}-hover-details" style="font-size:12px; line-height:1.4;"></div>
    `;
    document.body.appendChild(hoverDiv);
  }}

  const hoverImg = document.getElementById('{div_id}-hover-img');
  const hoverDetails = document.getElementById('{div_id}-hover-details');

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

    if (hoverImg) {{
      hoverImg.src = imgPath || '';
    }}
    if (hoverDetails) {{
      const l2Text = (l2 != null && !isNaN(l2)) ? Number(l2).toFixed(4) : '';
      const cosText = (cos != null && !isNaN(cos)) ? Number(cos).toFixed(4) : '';
      hoverDetails.innerHTML = [
        '<strong>' + trajName + '</strong>',
        'Frame offset: ' + frameIdx,
        'L2 distance: ' + l2Text,
        'Cosine distance: ' + cosText,
      ].join('<br>');
    }}

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


def _build_axes_overlay_script(div_id: str, value_label: str = "Axis value") -> str:
    label_js = value_label.replace("'", "\\'")
    return f"""
(function() {{
  const plotDiv = document.getElementById('{div_id}');
  if (!plotDiv) {{
    return;
  }}

  const hoverId = '{div_id}-hover';
  let hoverDiv = document.getElementById(hoverId);
  if (!hoverDiv) {{
    hoverDiv = document.createElement('div');
    hoverDiv.id = hoverId;
    hoverDiv.style.cssText = 'position:fixed; display:none; pointer-events:none; border:1px solid #999; background:rgba(255,255,255,0.98); padding:6px; z-index:1000; max-width:220px;';
    hoverDiv.innerHTML = `
      <img id="${{hoverId}}-img" src="" style="width:100%; height:auto; border-bottom:1px solid #ccc; margin-bottom:4px;">
      <div id="${{hoverId}}-meta" style="font-size:12px; line-height:1.35;"></div>
    `;
    document.body.appendChild(hoverDiv);
  }}

  const hoverImg = document.getElementById(`${{hoverId}}-img`);
  const hoverMeta = document.getElementById(`${{hoverId}}-meta`);

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
    const offset = custom[0];
    const localIdx = custom[1];
    const imgPath = custom[2];

    if (hoverImg) {{
      hoverImg.src = imgPath || '';
    }}
    if (hoverMeta) {{
      hoverMeta.innerHTML = [
        'Frame order: ' + (localIdx != null ? localIdx : ''),
        'Frame offset: ' + (offset != null ? offset : ''),
        '{label_js}: ' + (point.y != null ? Number(point.y).toFixed(4) : ''),
      ].join('<br>');
    }}

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


def _build_rank_overlay_script(div_id: str) -> str:
    return f"""
(function() {{
  const plotDiv = document.getElementById('{div_id}');
  if (!plotDiv) {{
    return;
  }}

  let hoverDiv = document.getElementById('{div_id}-hover');
  if (!hoverDiv) {{
    hoverDiv = document.createElement('div');
    hoverDiv.id = '{div_id}-hover';
    hoverDiv.style.cssText = 'position:fixed; display:none; pointer-events:none; border:1px solid #999; background:rgba(255,255,255,0.98); padding:6px; z-index:1000; max-width:220px;';
    hoverDiv.innerHTML = `
      <img id="{div_id}-hover-img" src="" style="width:100%; height:auto; border-bottom:1px solid #ccc; margin-bottom:4px;">
      <div id="{div_id}-hover-meta" style="font-size:12px; line-height:1.35;"></div>
    `;
    document.body.appendChild(hoverDiv);
  }}

  const hoverImg = document.getElementById('{div_id}-hover-img');
  const hoverMeta = document.getElementById('{div_id}-hover-meta');

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
    const traj = custom[0] || '';
    const frameIdx = custom[1] != null ? custom[1] : '';
    const imgPath = custom[2];

    if (hoverImg) {{
      hoverImg.src = imgPath || '';
    }}
    if (hoverMeta) {{
      const rank = (point.x != null && !isNaN(point.x)) ? Math.round(point.x) : '';
      hoverMeta.innerHTML = [
        '<strong>' + traj + '</strong>',
        'Frame index: ' + frameIdx,
        'Axis value: ' + (point.x != null ? Number(point.x).toFixed(4) : ''),
      ].join('<br>');
    }}

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


def _write_overlay_image(
    src_path: Path,
    dest_path: Path,
    value: float,
    vmax: float,
    cmap,
) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path).convert("RGB") as img:
        base = np.array(img, dtype=np.float32)
    if vmax <= 0:
        norm = 0.5
    else:
        norm = 0.5 + 0.5 * np.clip(value / vmax, -1.0, 1.0)
    color = np.array(cmap(norm)[:3]) * 255.0
    overlay = np.ones_like(base) * color
    blended = (0.6 * base + 0.4 * overlay).astype(np.uint8)
    Image.fromarray(blended).save(dest_path)


def _create_overlay_series(
    frame_paths: List[Path],
    values: np.ndarray,
    dest_dir: Path,
    out_dir: Path,
    cmap,
) -> List[str]:
    vmax = float(np.max(np.abs(values))) if values.size else 0.0
    if vmax <= 0:
        vmax = 1.0
    rel_paths: List[str] = []
    for frame_path, value in zip(frame_paths, values):
        dest_path = dest_dir / frame_path.name
        _write_overlay_image(frame_path, dest_path, float(value), vmax, cmap)
        rel_paths.append(dest_path.relative_to(out_dir).as_posix())
    return rel_paths

OVERLAY_POST_SCRIPT = _build_scatter_overlay_script(SCATTER_DIV_ID)


@dataclass
class Args:
    traj_dir: str
    out_dir: str = DEFAULT_OUT_DIR_TEMPLATE
    max_trajs: Optional[int] = None
    device: Optional[str] = None
    n_components: int = 2
    ica_components: Optional[int] = None
    progress: bool = True
    overlay_outputs: bool = False


def build_plot_data(
    copied_paths: List[List[Path]],
    out_dir: Path,
    l2_arrays: List[torch.Tensor],
    cos_arrays: List[torch.Tensor],
    embeddings: List[torch.Tensor],
    traj_names: List[str],
    relative_dirs: List[Path],
):
    records = []
    embed_list: List[torch.Tensor] = []

    for traj_name, rel_dir, paths, l2, cos, emb in zip(
        traj_names, relative_dirs, copied_paths, l2_arrays, cos_arrays, embeddings
    ):
        display_name = rel_dir.as_posix()
        for idx, (dst_path, d_l2, d_cos, vec) in enumerate(
            zip(paths, l2.tolist(), cos.tolist(), emb)
        ):
            rel_path = dst_path.relative_to(out_dir).as_posix()
            records.append(
                {
                    "traj": display_name,
                    "safe_traj": traj_name,
                    "frame_index": idx,
                    "l2": d_l2,
                    "cos": d_cos,
                    "image_path": rel_path,
                }
            )
            embed_list.append(vec.cpu())

    return records, embed_list


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


def compute_pca(
    embeds: torch.Tensor,
    n_components: int,
) -> Tuple[np.ndarray, np.ndarray]:
    data = embeds.cpu().numpy()
    if data.ndim != 2:
        raise ValueError("Embeddings must be a 2D tensor")
    count, dim = data.shape
    if count == 0:
        raise ValueError("No embeddings provided for PCA")
    k = max(1, min(n_components, dim, count))
    centered = data - data.mean(axis=0, keepdims=True)
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vh[:k].T
    if count > 1:
        eigenvalues = (s ** 2) / (count - 1)
        total_variance = eigenvalues.sum()
        if total_variance > 0:
            variance_ratio = eigenvalues[:k] / total_variance
        else:
            variance_ratio = np.zeros(k, dtype=float)
    else:
        variance_ratio = np.zeros(k, dtype=float)
    return coords, variance_ratio


def compute_ica(
    embeds: torch.Tensor,
    n_components: int,
    random_state: int = 0,
) -> np.ndarray:
    data = embeds.cpu().numpy()
    if data.ndim != 2:
        raise ValueError("Embeddings must be a 2D tensor")
    count, dim = data.shape
    if count == 0:
        raise ValueError("No embeddings provided for ICA")
    k = max(1, min(n_components, dim, count))
    ica = FastICA(n_components=k, random_state=random_state, max_iter=1000)
    coords = ica.fit_transform(data)
    return coords


def build_scatter_fig(
    records: List[dict],
    pca_coords: np.ndarray,
    variance_ratio: np.ndarray,
    ica_coords: Optional[np.ndarray] = None,
) -> go.Figure:
    offsets = np.array([rec["frame_index"] for rec in records])
    l2_values = np.array([rec["l2"] for rec in records])
    cos_values = np.array([rec["cos"] for rec in records])
    image_paths = np.array([rec["image_path"] for rec in records])
    trajs = np.array([rec["traj"] for rec in records])

    scatter_custom = np.array(
        list(
            zip(
                image_paths,
                trajs,
                offsets.astype(float),
                l2_values,
                cos_values,
            )
        ),
        dtype=object,
    )

    include_ica = ica_coords is not None
    subplot_titles = ["PCA axis 1 vs PCA axis 2"]
    if include_ica:
        subplot_titles.append("ICA axis 1 vs ICA axis 2")
    fig = make_subplots(rows=1, cols=len(subplot_titles), subplot_titles=subplot_titles)
    hover_template = "<extra></extra>"

    unique_trajs = np.unique(trajs)
    palette = px.colors.qualitative.Plotly
    if len(unique_trajs) > len(palette):
        repeats = len(unique_trajs) // len(palette) + 1
        palette = (palette * repeats)[: len(unique_trajs)]
    color_map = {traj: palette[i] for i, traj in enumerate(unique_trajs)}

    for traj_name in unique_trajs:
        mask = trajs == traj_name
        fig.add_trace(
            go.Scatter(
                x=pca_coords[mask, 0],
                y=pca_coords[mask, 1] if pca_coords.shape[1] > 1 else np.zeros(mask.sum()),
                mode='markers',
                name=traj_name,
                legendgroup=traj_name,
                marker=dict(size=6, opacity=0.75, color=color_map[traj_name]),
                customdata=scatter_custom[mask],
                hovertemplate=hover_template,
            ),
            row=1,
            col=1,
        )

    axis_1_var = variance_ratio[0] if variance_ratio.size > 0 else 0.0
    axis_2_var = variance_ratio[1] if variance_ratio.size > 1 else 0.0
    fig.update_xaxes(
        title_text=f'PCA axis 1 ({axis_1_var * 100:.2f}% var explained)',
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text=f'PCA axis 2 ({axis_2_var * 100:.2f}% var explained)',
        row=1,
        col=1,
    )

    if include_ica and ica_coords is not None:
        for traj_name in unique_trajs:
            mask = trajs == traj_name
            fig.add_trace(
                go.Scatter(
                    x=ica_coords[mask, 0],
                    y=ica_coords[mask, 1] if ica_coords.shape[1] > 1 else np.zeros(mask.sum()),
                    mode='markers',
                    name=f"{traj_name} (ICA)",
                    legendgroup=traj_name,
                    marker=dict(size=6, opacity=0.75, color=color_map[traj_name]),
                    customdata=scatter_custom[mask],
                    hovertemplate=hover_template,
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
        fig.update_xaxes(title_text='ICA axis 1', row=1, col=2)
        fig.update_yaxes(title_text='ICA axis 2', row=1, col=2)

    fig.update_layout(
        title='Self-distance scatter projections',
        hovermode='closest',
        template='plotly_white',
        legend=dict(groupclick='togglegroup'),
        hoverlabel=dict(
            bgcolor='rgba(255,255,255,0.95)',
            font=dict(color='#1a1a1a', size=12),
            align='left',
        ),
    )

    return fig


def build_axes_fig_for_traj(
    records: List[dict],
    coords: np.ndarray,
    traj_name: str,
    method_label: str,
    max_axes: int = 5,
    values_matrix: Optional[np.ndarray] = None,
    image_matrix: Optional[np.ndarray] = None,
    value_axis_suffix: str = "values",
) -> Optional[go.Figure]:
    if coords.size == 0:
        return None

    trajs = np.array([rec["traj"] for rec in records])
    offsets = np.array([rec["frame_index"] for rec in records], dtype=float)
    image_paths = np.array([rec["image_path"] for rec in records], dtype=object)
    mask = trajs == traj_name
    if not np.any(mask):
        return None

    traj_indices = np.where(mask)[0]
    order = traj_indices[np.argsort(offsets[traj_indices])]
    ordered_offsets = offsets[order]
    axis_source = values_matrix if values_matrix is not None else coords
    axis_count = min(max_axes, axis_source.shape[1])
    if axis_count == 0:
        return None

    rel_image_paths = np.array([rec["image_path"] for rec in records], dtype=object)
    base_image_matrix = np.tile(rel_image_paths.reshape(-1, 1), (1, axis_count))
    image_source = image_matrix if image_matrix is not None else base_image_matrix

    fig = make_subplots(
        rows=axis_count,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=[
            f"{method_label} axis {idx + 1} {value_axis_suffix} vs frame order"
            for idx in range(axis_count)
        ],
    )

    axis_palette = px.colors.qualitative.Safe
    local_frame_indices = np.arange(len(order), dtype=float)

    for axis_idx in range(axis_count):
        axis_values = axis_source[order, axis_idx]
        ordered_rel_images = np.array(
            [f"../{path}" for path in image_source[order, axis_idx]], dtype=object
        )
        axis_custom = np.array(
            list(zip(ordered_offsets, local_frame_indices, ordered_rel_images)),
            dtype=object,
        )
        color = axis_palette[axis_idx % len(axis_palette)]
        fig.add_trace(
            go.Scatter(
                x=local_frame_indices,
                y=axis_values,
                mode='lines+markers',
                name=f"{method_label} axis {axis_idx + 1}",
                marker=dict(size=4, color=color),
                line=dict(color=color, width=2),
                customdata=axis_custom,
                hovertemplate=(
                    "Trajectory frame %{customdata[1]}<br>"
                    "Frame offset %{customdata[0]}<br>"
                    f"{method_label} axis {axis_idx + 1} value %{{y:.4f}}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=axis_idx + 1,
            col=1,
        )
        fig.update_xaxes(
            title_text='Frame order within trajectory',
            row=axis_idx + 1,
            col=1,
        )
        fig.update_yaxes(
            title_text=f'{method_label} axis {axis_idx + 1} {value_axis_suffix}',
            row=axis_idx + 1,
            col=1,
        )

    fig.update_layout(
        title=f'{traj_name} {method_label} axis {value_axis_suffix}',
        hovermode='closest',
        template='plotly_white',
        hoverlabel=dict(
            bgcolor='rgba(255,255,255,0.95)',
            font=dict(color='#1a1a1a', size=12),
            align='left',
        ),
    )

    return fig


def build_axis_rank_fig(
    records: List[dict],
    coords: np.ndarray,
    method_label: str,
    max_axes: int = 5,
) -> Optional[go.Figure]:
    if coords.size == 0:
        return None

    axis_count = min(max_axes, coords.shape[1])
    if axis_count == 0:
        return None

    trajs = np.array([rec["traj"] for rec in records], dtype=object)
    frame_indices = np.array([rec["frame_index"] for rec in records], dtype=int)
    image_paths = np.array([rec["image_path"] for rec in records], dtype=object)

    unique_trajs = np.unique(trajs)
    palette = px.colors.qualitative.Plotly
    if len(unique_trajs) > len(palette):
        repeats = len(unique_trajs) // len(palette) + 1
        palette = (palette * repeats)[: len(unique_trajs)]
    color_map = {traj: palette[i] for i, traj in enumerate(unique_trajs)}

    fig = make_subplots(
        rows=axis_count,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=[
            f"{method_label} axis {idx + 1} values" for idx in range(axis_count)
        ],
    )

    for axis_idx in range(axis_count):
        axis_values = coords[:, axis_idx]
        colors = [color_map[traj] for traj in trajs]
        custom = np.array(
            list(zip(trajs, frame_indices, image_paths)),
            dtype=object,
        )

        fig.add_trace(
            go.Scatter(
                x=axis_values,
                y=np.zeros_like(axis_values, dtype=float),
                mode='markers',
                name=f"{method_label} axis {axis_idx + 1}",
                marker=dict(size=5, opacity=0.8, color=colors),
                customdata=custom,
                hovertemplate=(
                    "Trajectory %{customdata[0]}<br>"
                    "Frame %{customdata[1]}<br>"
                    f"{method_label} axis {axis_idx + 1} value %{{x:.4f}}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=axis_idx + 1,
            col=1,
        )

        fig.update_xaxes(
            title_text=f'{method_label} axis {axis_idx + 1} value',
            row=axis_idx + 1,
            col=1,
        )
        fig.update_yaxes(
            title_text='',
            row=axis_idx + 1,
            col=1,
            showticklabels=False,
        )
        fig.update_yaxes(
            range=[-0.5, 0.5],
            row=axis_idx + 1,
            col=1,
        )

    fig.update_layout(
        title=f'{method_label} axis rankings across trajectories',
        hovermode='closest',
        template='plotly_white',
        hoverlabel=dict(
            bgcolor='rgba(255,255,255,0.95)',
            font=dict(color='#1a1a1a', size=12),
            align='left',
        ),
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

    records, embed_list = build_plot_data(
        copied_paths,
        out_dir,
        [res.l2_distances for res in results],
        [res.cosine_distances for res in results],
        [res.embeddings for res in results],
        [res.traj_name for res in results],
        [res.relative_dir for res in results],
    )

    if not records:
        print("No records generated for visualization")
        return

    traj_to_safe: dict[str, str] = {}
    traj_to_indices: defaultdict[str, List[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        traj_to_safe.setdefault(rec["traj"], rec["safe_traj"])
        traj_to_indices[rec["traj"]].append(idx)

    base_rel_paths = np.array([rec["image_path"] for rec in records], dtype=object)

    overlay_frames_dir = out_dir / "frames_overlay"
    derivative_frames_dir = out_dir / "frames_derivative"
    overlay_cmap = cm.get_cmap("coolwarm")

    write_metadata_csv(records, out_dir / "self_distance_records.csv")

    if not embed_list:
        print("No embeddings available to compute PCA coordinates")
        return

    embeds = torch.stack(embed_list)
    num_samples = len(records)
    pca_components_for_axes = max(5, args.n_components)
    pca_coords_full, variance_ratio_full = compute_pca(embeds, pca_components_for_axes)
    if pca_coords_full.shape[1] < 2:
        scatter_pca = np.pad(
            pca_coords_full,
            ((0, 0), (0, 2 - pca_coords_full.shape[1])),
            mode='constant',
        )
        scatter_variance = np.pad(
            variance_ratio_full,
            (0, 2 - variance_ratio_full.shape[0]),
            mode='constant',
        )
    else:
        scatter_pca = pca_coords_full[:, :2]
        scatter_variance = variance_ratio_full[:2]

    overlay_enabled = args.overlay_outputs
    pca_axis_count = min(5, pca_coords_full.shape[1])
    if overlay_enabled and pca_axis_count > 0:
        pca_overlay_rel = np.tile(base_rel_paths.reshape(-1, 1), (1, pca_axis_count))
        pca_deriv_rel = np.tile(base_rel_paths.reshape(-1, 1), (1, pca_axis_count))
        pca_deriv_matrix = np.zeros((num_samples, pca_axis_count), dtype=float)
    else:
        pca_overlay_rel = np.empty((num_samples, 0), dtype=object)
        pca_deriv_rel = np.empty((num_samples, 0), dtype=object)
        pca_deriv_matrix = np.empty((num_samples, 0), dtype=float)

    ica_coords_full = None
    ica_coords_for_scatter = None
    ica_components_requested = args.ica_components if args.ica_components is not None else args.n_components
    ica_components_for_axes = max(5, ica_components_requested)
    if ica_components_for_axes > 0:
        try:
            ica_coords_full = compute_ica(embeds, ica_components_for_axes)
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"Failed to compute ICA coordinates: {exc}")
            ica_coords_full = None
    if ica_coords_full is not None:
        if ica_coords_full.shape[1] < 2:
            ica_coords_for_scatter = np.pad(
                ica_coords_full,
                ((0, 0), (0, 2 - ica_coords_full.shape[1])),
                mode='constant',
            )
        else:
            ica_coords_for_scatter = ica_coords_full[:, :2]

    if ica_coords_full is not None:
        ica_axis_count = min(5, ica_coords_full.shape[1])
        if overlay_enabled and ica_axis_count > 0:
            ica_overlay_rel = np.tile(base_rel_paths.reshape(-1, 1), (1, ica_axis_count))
            ica_deriv_rel = np.tile(base_rel_paths.reshape(-1, 1), (1, ica_axis_count))
            ica_deriv_matrix = np.zeros((num_samples, ica_axis_count), dtype=float)
        else:
            ica_overlay_rel = np.empty((num_samples, 0), dtype=object)
            ica_deriv_rel = np.empty((num_samples, 0), dtype=object)
            ica_deriv_matrix = np.empty((num_samples, 0), dtype=float)
    else:
        ica_axis_count = 0
        ica_overlay_rel = np.empty((num_samples, 0), dtype=object)
        ica_deriv_rel = np.empty((num_samples, 0), dtype=object)
        ica_deriv_matrix = np.empty((num_samples, 0), dtype=float)

    scatter_fig = build_scatter_fig(
        records,
        scatter_pca,
        scatter_variance,
        ica_coords=ica_coords_for_scatter,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    scatter_html_path = out_dir / "self_distance_pca_scatter.html"

    scatter_fig.write_html(
        scatter_html_path,
        include_plotlyjs='cdn',
        full_html=True,
        default_height='100vh',
        default_width='100%',
        div_id=SCATTER_DIV_ID,
        post_script=OVERLAY_POST_SCRIPT,
    )
    print(f"Saved PCA and ICA scatter visualization to {scatter_html_path}")

    pca_rank_fig = build_axis_rank_fig(records, pca_coords_full, "PCA")
    if pca_rank_fig is not None:
        pca_rank_path = out_dir / "pca_axis_rankings.html"
        pca_rank_div = "self-distance-pca-rank-plot"
        pca_rank_fig.write_html(
            pca_rank_path,
            include_plotlyjs='cdn',
            full_html=True,
            default_height='100vh',
            default_width='100%',
            div_id=pca_rank_div,
            post_script=_build_rank_overlay_script(pca_rank_div),
        )
        print(f"Saved PCA axis rankings to {pca_rank_path}")

    if ica_coords_full is not None:
        ica_rank_fig = build_axis_rank_fig(records, ica_coords_full, "ICA")
        if ica_rank_fig is not None:
            ica_rank_path = out_dir / "ica_axis_rankings.html"
            ica_rank_div = "self-distance-ica-rank-plot"
            ica_rank_fig.write_html(
                ica_rank_path,
                include_plotlyjs='cdn',
                full_html=True,
                default_height='100vh',
                default_width='100%',
                div_id=ica_rank_div,
                post_script=_build_rank_overlay_script(ica_rank_div),
            )
            print(f"Saved ICA axis rankings to {ica_rank_path}")

    axes_dir = out_dir / "axes_by_traj"
    axes_dir.mkdir(exist_ok=True)

    unique_trajs = sorted(traj_to_safe.keys())
    for traj_name in unique_trajs:
        indices = np.array(traj_to_indices.get(traj_name, []), dtype=int)
        if indices.size == 0:
            continue
        safe_name = traj_to_safe[traj_name]
        order = np.argsort([records[i]["frame_index"] for i in indices])
        sorted_indices = indices[order]
        frame_paths_abs = [out_dir / base_rel_paths[idx] for idx in sorted_indices]

        if pca_axis_count > 0:
            values_matrix = pca_coords_full[sorted_indices, :pca_axis_count]
            overlay_root = overlay_frames_dir / "pca" / safe_name
            deriv_root = derivative_frames_dir / "pca" / safe_name
            for axis_idx in range(pca_axis_count):
                axis_values = values_matrix[:, axis_idx]
                axis_overlay_dir = overlay_root / f"axis_{axis_idx + 1}"
                overlay_rel_list = _create_overlay_series(
                    frame_paths_abs,
                    axis_values,
                    axis_overlay_dir,
                    out_dir,
                    overlay_cmap,
                )
                for idx_local, record_idx in enumerate(sorted_indices):
                    pca_overlay_rel[record_idx, axis_idx] = overlay_rel_list[idx_local]

                deriv_values = np.zeros_like(axis_values)
                if axis_values.size > 1:
                    deriv_values[1:] = np.diff(axis_values)
                pca_deriv_matrix[sorted_indices, axis_idx] = deriv_values
                axis_deriv_dir = deriv_root / f"axis_{axis_idx + 1}"
                deriv_rel_list = _create_overlay_series(
                    frame_paths_abs,
                    deriv_values,
                    axis_deriv_dir,
                    out_dir,
                    overlay_cmap,
                )
                for idx_local, record_idx in enumerate(sorted_indices):
                    pca_deriv_rel[record_idx, axis_idx] = deriv_rel_list[idx_local]

        if ica_axis_count > 0:
            values_matrix = ica_coords_full[sorted_indices, :ica_axis_count]
            overlay_root = overlay_frames_dir / "ica" / safe_name
            deriv_root = derivative_frames_dir / "ica" / safe_name
            for axis_idx in range(ica_axis_count):
                axis_values = values_matrix[:, axis_idx]
                axis_overlay_dir = overlay_root / f"axis_{axis_idx + 1}"
                overlay_rel_list = _create_overlay_series(
                    frame_paths_abs,
                    axis_values,
                    axis_overlay_dir,
                    out_dir,
                    overlay_cmap,
                )
                for idx_local, record_idx in enumerate(sorted_indices):
                    ica_overlay_rel[record_idx, axis_idx] = overlay_rel_list[idx_local]

                deriv_values = np.zeros_like(axis_values)
                if axis_values.size > 1:
                    deriv_values[1:] = np.diff(axis_values)
                ica_deriv_matrix[sorted_indices, axis_idx] = deriv_values
                axis_deriv_dir = deriv_root / f"axis_{axis_idx + 1}"
                deriv_rel_list = _create_overlay_series(
                    frame_paths_abs,
                    deriv_values,
                    axis_deriv_dir,
                    out_dir,
                    overlay_cmap,
                )
                for idx_local, record_idx in enumerate(sorted_indices):
                    ica_deriv_rel[record_idx, axis_idx] = deriv_rel_list[idx_local]

        pca_fig = build_axes_fig_for_traj(
            records,
            pca_coords_full,
            traj_name,
            "PCA",
        )
        if pca_fig is not None:
            pca_path = axes_dir / f"{safe_name}_pca_axes.html"
            pca_div_id = f"{AXES_DIV_ID}-{safe_name}-pca"
            pca_fig.write_html(
                pca_path,
                include_plotlyjs='cdn',
                full_html=True,
                default_height='90vh',
                default_width='100%',
                div_id=pca_div_id,
                post_script=_build_axes_overlay_script(pca_div_id),
            )
            print(f"Saved {traj_name} PCA axes visualization to {pca_path}")

        if pca_axis_count > 0:
            pca_overlay_fig = build_axes_fig_for_traj(
                records,
                pca_coords_full,
                traj_name,
                "PCA",
                image_matrix=pca_overlay_rel,
            )
            if pca_overlay_fig is not None:
                pca_overlay_path = axes_dir / f"{safe_name}_pca_overlay_axes.html"
                pca_overlay_div = f"{AXES_DIV_ID}-{safe_name}-pca-overlay"
                pca_overlay_fig.write_html(
                    pca_overlay_path,
                    include_plotlyjs='cdn',
                    full_html=True,
                    default_height='90vh',
                    default_width='100%',
                    div_id=pca_overlay_div,
                    post_script=_build_axes_overlay_script(pca_overlay_div),
                )
                print(f"Saved {traj_name} PCA overlay axes visualization to {pca_overlay_path}")

            pca_deriv_fig = build_axes_fig_for_traj(
                records,
                pca_coords_full,
                traj_name,
                "PCA",
                values_matrix=pca_deriv_matrix,
                image_matrix=pca_deriv_rel,
                value_axis_suffix="Δ values",
            )
            if pca_deriv_fig is not None:
                pca_deriv_path = axes_dir / f"{safe_name}_pca_derivative_axes.html"
                pca_deriv_div = f"{AXES_DIV_ID}-{safe_name}-pca-derivative"
                pca_deriv_fig.write_html(
                    pca_deriv_path,
                    include_plotlyjs='cdn',
                    full_html=True,
                    default_height='90vh',
                    default_width='100%',
                    div_id=pca_deriv_div,
                    post_script=_build_axes_overlay_script(pca_deriv_div, value_label="Δ axis value"),
                )
                print(f"Saved {traj_name} PCA derivative axes visualization to {pca_deriv_path}")

        if ica_coords_full is not None:
            ica_fig = build_axes_fig_for_traj(
                records,
                ica_coords_full,
                traj_name,
                "ICA",
            )
            if ica_fig is not None:
                ica_path = axes_dir / f"{safe_name}_ica_axes.html"
                ica_div_id = f"{AXES_DIV_ID}-{safe_name}-ica"
                ica_fig.write_html(
                    ica_path,
                    include_plotlyjs='cdn',
                    full_html=True,
                    default_height='90vh',
                    default_width='100%',
                    div_id=ica_div_id,
                    post_script=_build_axes_overlay_script(ica_div_id),
                )
                print(f"Saved {traj_name} ICA axes visualization to {ica_path}")

            if ica_axis_count > 0:
                ica_overlay_fig = build_axes_fig_for_traj(
                    records,
                    ica_coords_full,
                    traj_name,
                    "ICA",
                    image_matrix=ica_overlay_rel,
                )
                if ica_overlay_fig is not None:
                    ica_overlay_path = axes_dir / f"{safe_name}_ica_overlay_axes.html"
                    ica_overlay_div = f"{AXES_DIV_ID}-{safe_name}-ica-overlay"
                    ica_overlay_fig.write_html(
                        ica_overlay_path,
                        include_plotlyjs='cdn',
                        full_html=True,
                        default_height='90vh',
                        default_width='100%',
                        div_id=ica_overlay_div,
                        post_script=_build_axes_overlay_script(ica_overlay_div),
                    )
                    print(f"Saved {traj_name} ICA overlay axes visualization to {ica_overlay_path}")

                ica_deriv_fig = build_axes_fig_for_traj(
                    records,
                    ica_coords_full,
                    traj_name,
                    "ICA",
                    values_matrix=ica_deriv_matrix,
                    image_matrix=ica_deriv_rel,
                    value_axis_suffix="Δ values",
                )
                if ica_deriv_fig is not None:
                    ica_deriv_path = axes_dir / f"{safe_name}_ica_derivative_axes.html"
                    ica_deriv_div = f"{AXES_DIV_ID}-{safe_name}-ica-derivative"
                    ica_deriv_fig.write_html(
                        ica_deriv_path,
                        include_plotlyjs='cdn',
                        full_html=True,
                        default_height='90vh',
                        default_width='100%',
                        div_id=ica_deriv_div,
                        post_script=_build_axes_overlay_script(ica_deriv_div, value_label="Δ axis value"),
                    )
                    print(f"Saved {traj_name} ICA derivative axes visualization to {ica_deriv_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
