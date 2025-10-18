#!/usr/bin/env python3
"""Interactive PCA-based self-distance visualizer with Plotly hover images."""
from __future__ import annotations

import csv
import os
import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


def _build_axes_overlay_script(div_id: str) -> str:
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
        'Axis value: ' + (point.y != null ? Number(point.y).toFixed(4) : ''),
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

OVERLAY_POST_SCRIPT = f"""
(function() {{
  const plotDiv = document.getElementById('{SCATTER_DIV_ID}');
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
    n_components: int = 2
    ica_components: Optional[int] = None
    progress: bool = True


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
    ordered_images = image_paths[order]
    rel_image_paths = np.array([f"../{img}" for img in ordered_images], dtype=object)
    axis_count = min(max_axes, coords.shape[1])
    if axis_count == 0:
        return None

    fig = make_subplots(
        rows=axis_count,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=[
            f"{method_label} axis {idx + 1} distance vs frame order" for idx in range(axis_count)
        ],
    )

    axis_palette = px.colors.qualitative.Safe
    local_frame_indices = np.arange(len(order), dtype=float)
    axis_custom = np.array(
        list(zip(ordered_offsets, local_frame_indices, rel_image_paths)),
        dtype=object,
    )

    for axis_idx in range(axis_count):
        axis_values = coords[order, axis_idx]
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
            title_text=f'{method_label} axis {axis_idx + 1} value',
            row=axis_idx + 1,
            col=1,
        )

    fig.update_layout(
        title=f'{traj_name} {method_label} axis distances',
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

    write_metadata_csv(records, out_dir / "self_distance_records.csv")

    if not embed_list:
        print("No embeddings available to compute PCA coordinates")
        return

    embeds = torch.stack(embed_list)
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

    trajs = [rec["traj"] for rec in records]
    safe_trajs = [rec["safe_traj"] for rec in records]
    traj_to_safe = {}
    for traj_name, safe_name in zip(trajs, safe_trajs):
        traj_to_safe.setdefault(traj_name, safe_name)

    axes_dir = out_dir / "axes_by_traj"
    axes_dir.mkdir(exist_ok=True)

    unique_trajs = sorted(traj_to_safe.keys())
    for traj_name in unique_trajs:
        safe_name = traj_to_safe[traj_name]
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


if __name__ == "__main__":
    main(tyro.cli(Args))
