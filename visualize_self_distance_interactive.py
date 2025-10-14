#!/usr/bin/env python3
"""Interactive self-distance visualizer with Plotly hover images."""
from __future__ import annotations

import csv
import os
import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from string import Template

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import torch
import tyro

try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover
    umap = None

from predict_mario_ms_ssim import pick_device
from self_distance_utils import (
    compute_self_distance_results,
    copy_frames_for_visualization,
)

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


@dataclass
class Args:
    traj_dir: str
    out_dir: str = "out.self_distance_interactive"
    max_trajs: Optional[int] = None
    device: Optional[str] = None
    umap_neighbors: int = 30
    umap_min_dist: float = 0.0
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


def build_fig(records: List[dict], umap_coords: Optional[np.ndarray]) -> go.Figure:
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

    titles = ["Frame offset vs L2 distance"]
    if umap_coords is not None:
        titles.append("UMAP of embeddings")

    specs = [[{"type": "scatter"} for _ in titles]]
    fig = make_subplots(rows=1, cols=len(titles), subplot_titles=titles, specs=specs)

    hover_template = "<extra></extra>"

    unique_trajs = np.unique(trajs)
    for traj_name in unique_trajs:
        mask = trajs == traj_name
        fig.add_trace(
            go.Scatter(
                x=offsets[mask],
                y=l2_values[mask],
                mode='markers',
                name=traj_name,
                legendgroup=traj_name,
                marker=dict(
                    size=6,
                    opacity=0.7,
                    color=offsets[mask],
                    colorscale='Viridis',
                    showscale=False,
                ),
                customdata=scatter_custom[mask],
                hovertemplate=hover_template,
            ),
            row=1,
            col=1,
        )
    fig.update_xaxes(title_text='Frame offset from reference (t)', row=1, col=1)
    fig.update_yaxes(title_text='L2 distance vs frame 0', row=1, col=1)

    if umap_coords is not None:
        for traj_name in unique_trajs:
            mask = trajs == traj_name
            fig.add_trace(
                go.Scatter(
                    x=umap_coords[mask, 0],
                    y=umap_coords[mask, 1],
                    mode='markers',
                    name=f"{traj_name} (UMAP)",
                    legendgroup=traj_name,
                    marker=dict(
                        size=6,
                        opacity=0.7,
                        color=offsets[mask],
                        colorscale='Viridis',
                        showscale=False,
                    ),
                    customdata=scatter_custom[mask],
                    hovertemplate=hover_template,
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
        fig.update_xaxes(title_text='UMAP-1', row=1, col=2)
        fig.update_yaxes(title_text='UMAP-2', row=1, col=2)

    fig.update_layout(
        title='Self-distance interactive visualization',
        hovermode='closest',
        template='plotly_white',
    )

    return fig


def main(args: Args) -> None:
    _set_max_active_levels(1)
    device = pick_device(args.device)
    traj_dir = Path(args.traj_dir)
    out_dir = Path(args.out_dir)
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

    write_metadata_csv(records, out_dir / "self_distance_records.csv")

    umap_coords = None
    if umap is not None and embed_list:
        embeds = torch.stack(embed_list).numpy()
        reducer = umap.UMAP(
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric="euclidean",
        )
        umap_coords = reducer.fit_transform(embeds)
    elif umap is None:
        print("[warn] umap-learn not installed; skipping UMAP projection")

    fig = build_fig(records, umap_coords)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "self_distance_interactive.html"

    plot_html = pio.to_html(
        fig,
        include_plotlyjs='cdn',
        full_html=False,
        div_id='self-distance-plot',
    )

    overlay_html = """
<div id=\"hover-image\" style=\"position:fixed; display:none; pointer-events:none; border:1px solid #999; background:rgba(255,255,255,0.98); padding:4px; z-index:1000; max-width:260px;\">
  <img id=\"hover-image-img\" src=\"\" style=\"width:100%; height:auto; display:block; border-bottom:1px solid #ccc; margin-bottom:4px;\">
  <div id=\"hover-image-details\" style=\"font-size:12px; line-height:1.4;\"></div>
</div>
"""

    script = """
<script>
  (function() {
    var plotDiv = document.getElementById('self-distance-plot');
    if (!plotDiv) { return; }
    var hoverDiv = document.getElementById('hover-image');
    var hoverImg = document.getElementById('hover-image-img');
    var hoverDetails = document.getElementById('hover-image-details');

    function positionHover(event) {
      if (!event) { return; }
      var x = (event.clientX || 0) + 16;
      var y = (event.clientY || 0) + 16;
      hoverDiv.style.left = x + 'px';
      hoverDiv.style.top = y + 'px';
    }

    plotDiv.on('plotly_hover', function(data) {
      if (!data || !data.points || !data.points.length) { return; }
      var point = data.points[0];
      var custom = point.customdata || [];
      var imgPath = custom[0];
      var trajName = custom[1] || '';
      var frameIdx = custom[2] != null ? custom[2] : '';
      var l2 = custom[3] != null ? custom[3].toFixed(4) : '';
      var cos = custom[4] != null ? custom[4].toFixed(4) : '';
      hoverImg.src = imgPath;
      hoverDetails.innerHTML =
        '<strong>' + trajName + '</strong><br>' +
        'Frame offset: ' + frameIdx + '<br>' +
        'L2 distance: ' + l2 + '<br>' +
        'Cosine distance: ' + cos;
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
</script>
"""

    template = Template("""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Self-distance Interactive Visualization</title>
  <style>
    body { font-family: sans-serif; margin: 0; padding: 0; }
    #container { padding: 12px; }
  </style>
</head>
<body>
  <div id="container">
    $plot
  </div>
  $overlay
  $script
</body>
</html>
""")

    html_path.write_text(
        template.substitute(plot=plot_html, overlay=overlay_html, script=script)
    )
    print(f"Saved interactive visualization to {html_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
