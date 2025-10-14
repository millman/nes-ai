#!/usr/bin/env python3
"""Interactive self-distance visualizer with Plotly hover images."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
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
        list(zip(image_paths, trajs, offsets.astype(float), cos_values)),
        dtype=object,
    )

    titles = ["Frame offset vs L2 distance"]
    if umap_coords is not None:
        titles.append("UMAP of embeddings")

    specs = [[{"type": "scatter"} for _ in titles]]
    fig = make_subplots(rows=1, cols=len(titles), subplot_titles=titles, specs=specs)

    hover_template = (
        "Trajectory: %{customdata[1]}<br>"
        "Frame offset: %{customdata[2]:.0f}<br>"
        "L2 distance: %{y:.4f}<br>"
        "Cosine distance: %{customdata[3]:.4f}<br>"
        "<img src='%{customdata[0]}' width='160'><extra></extra>"
    )

    fig.add_trace(
        go.Scatter(
            x=offsets,
            y=l2_values,
            mode='markers',
            marker=dict(size=6, opacity=0.7, color=offsets, colorscale='Viridis'),
            customdata=scatter_custom,
            hovertemplate=hover_template,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text='Frame offset from reference (t)', row=1, col=1)
    fig.update_yaxes(title_text='L2 distance vs frame 0', row=1, col=1)

    if umap_coords is not None:
        umap_custom = np.array(
            list(zip(image_paths, trajs, offsets.astype(float), cos_values)),
            dtype=object,
        )
        fig.add_trace(
            go.Scatter(
                x=umap_coords[:, 0],
                y=umap_coords[:, 1],
                mode='markers',
                marker=dict(size=6, opacity=0.7, color=offsets, colorscale='Viridis'),
                customdata=umap_custom,
                hovertemplate=hover_template,
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
    fig.write_html(html_path, include_plotlyjs='cdn')
    print(f"Saved interactive visualization to {html_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
