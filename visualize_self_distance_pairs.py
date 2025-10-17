#!/usr/bin/env python3
"""Visualize predicted distances for within- and cross-trajectory frame pairs."""
from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Annotated, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import torch
import tyro

from output_dir_utils import TIMESTAMP_PLACEHOLDER, resolve_output_dir
from predict_mario_ms_ssim import pick_device
from self_distance_utils import (
    TrajectorySelfDistance,
    compute_self_distance_results,
    copy_frames_for_visualization,
)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


DEFAULT_OUT_DIR_TEMPLATE = f"out.self_distance_pairs_{TIMESTAMP_PLACEHOLDER}"


@dataclass
class Args:
    traj_dir: str
    out_dir: Annotated[
        str,
        tyro.conf.arg(
            help=(
                "Output directory. Use the 'YYYY-MM-DD_HH-MM-SS' suffix to substitute the current"
                " timestamp automatically."
            )
        ),
    ] = DEFAULT_OUT_DIR_TEMPLATE
    num_within_pairs: int = 2000
    num_cross_pairs: int = 2000
    max_points: Optional[int] = None
    max_step_gap: Optional[int] = None
    max_trajs: Optional[int] = None
    device: Optional[str] = None
    seed: int = 0
    progress: bool = True


@dataclass
class FrameInfo:
    traj_name: str
    display_traj: str
    frame_idx: int
    embedding: torch.Tensor
    image_rel_path: str
    uid: str


@dataclass
class WithinPair:
    traj_name: str
    display_traj: str
    frame_a_idx: int
    frame_b_idx: int
    step_gap: int
    pred_distance: float
    image_a: str
    image_b: str


@dataclass
class CrossPair:
    traj_a: str
    display_traj_a: str
    frame_a_idx: int
    traj_b: str
    display_traj_b: str
    frame_b_idx: int
    pred_distance: float
    image_a: str
    image_b: str


def _pair_count(n: int, max_step_gap: Optional[int]) -> int:
    if n < 2:
        return 0
    if max_step_gap is None:
        return n * (n - 1) // 2
    gap = min(max_step_gap, n - 1)
    if gap <= 0:
        return 0
    return gap * n - gap * (gap + 1) // 2


def _build_frame_infos(
    results: Sequence[TrajectorySelfDistance],
    copied_paths: Sequence[Sequence[Path]],
    out_dir: Path,
) -> List[List[FrameInfo]]:
    all_traj_frames: List[List[FrameInfo]] = []
    for res, paths in zip(results, copied_paths):
        traj_frames: List[FrameInfo] = []
        for idx, (image_path, embedding) in enumerate(zip(paths, res.embeddings)):
            rel_path = image_path.relative_to(out_dir).as_posix()
            traj_frames.append(
                FrameInfo(
                    traj_name=res.traj_name,
                    display_traj=res.relative_dir.as_posix(),
                    frame_idx=idx,
                    embedding=embedding,
                    image_rel_path=rel_path,
                    uid=f"{res.traj_name}:{idx}",
                )
            )
        if traj_frames:
            all_traj_frames.append(traj_frames)
    return all_traj_frames


def _sample_within_pairs(
    traj_frames: Sequence[Sequence[FrameInfo]],
    num_pairs: int,
    max_step_gap: Optional[int],
    rng: random.Random,
) -> List[WithinPair]:
    candidates: List[WithinPair] = []
    traj_lists: List[Sequence[FrameInfo]] = []
    weights: List[int] = []
    for frames in traj_frames:
        count = _pair_count(len(frames), max_step_gap)
        if count > 0:
            traj_lists.append(frames)
            weights.append(count)
    if not traj_lists or num_pairs <= 0:
        return []

    total_pairs = sum(weights)
    # If the total number of possible pairs is small, enumerate all instead of sampling.
    if num_pairs >= total_pairs or total_pairs <= 5000:
        for frames in traj_lists:
            n = len(frames)
            limit = min(max_step_gap, n - 1) if max_step_gap is not None else n - 1
            for i in range(n - 1):
                max_gap = min(limit, n - 1 - i)
                if max_gap <= 0:
                    continue
                for gap in range(1, max_gap + 1):
                    j = i + gap
                    info_i = frames[i]
                    info_j = frames[j]
                    distance = float(torch.norm(info_i.embedding - info_j.embedding))
                    candidates.append(
                        WithinPair(
                            traj_name=info_i.traj_name,
                            display_traj=info_i.display_traj,
                            frame_a_idx=info_i.frame_idx,
                            frame_b_idx=info_j.frame_idx,
                            step_gap=gap,
                            pred_distance=distance,
                            image_a=info_i.image_rel_path,
                            image_b=info_j.image_rel_path,
                        )
                    )
        return candidates[:num_pairs]

    selected: List[WithinPair] = []
    seen: set[Tuple[str, int, int]] = set()
    weight_array = weights
    max_attempts = num_pairs * 10 + 1000
    attempts = 0
    while len(selected) < num_pairs and attempts < max_attempts:
        attempts += 1
        frames = rng.choices(traj_lists, weights=weight_array, k=1)[0]
        n = len(frames)
        if n < 2:
            continue
        max_gap = min(max_step_gap, n - 1) if max_step_gap is not None else n - 1
        if max_gap <= 0:
            continue
        i = rng.randrange(0, n - 1)
        upper = min(n, i + 1 + max_gap)
        if upper <= i + 1:
            continue
        j = rng.randrange(i + 1, upper)
        info_i = frames[i]
        info_j = frames[j]
        key = (info_i.traj_name, min(info_i.frame_idx, info_j.frame_idx), max(info_i.frame_idx, info_j.frame_idx))
        if key in seen:
            continue
        seen.add(key)
        distance = float(torch.norm(info_i.embedding - info_j.embedding))
        selected.append(
            WithinPair(
                traj_name=info_i.traj_name,
                display_traj=info_i.display_traj,
                frame_a_idx=info_i.frame_idx,
                frame_b_idx=info_j.frame_idx,
                step_gap=info_j.frame_idx - info_i.frame_idx,
                pred_distance=distance,
                image_a=info_i.image_rel_path,
                image_b=info_j.image_rel_path,
            )
        )
    if len(selected) < num_pairs:
        print(f"[warn] Requested {num_pairs} within pairs but only sampled {len(selected)}")
    return selected


def _sample_cross_pairs(
    traj_frames: Sequence[Sequence[FrameInfo]],
    num_pairs: int,
    rng: random.Random,
) -> List[CrossPair]:
    if num_pairs <= 0:
        return []
    all_frames: List[FrameInfo] = [frame for frames in traj_frames for frame in frames]
    if len(all_frames) < 2:
        return []
    selected: List[CrossPair] = []
    seen: set[Tuple[str, str]] = set()
    max_attempts = num_pairs * 20 + 2000
    attempts = 0
    while len(selected) < num_pairs and attempts < max_attempts:
        attempts += 1
        a, b = rng.sample(all_frames, 2)
        if a.traj_name == b.traj_name:
            continue
        key = tuple(sorted((a.uid, b.uid)))
        if key in seen:
            continue
        seen.add(key)
        distance = float(torch.norm(a.embedding - b.embedding))
        selected.append(
            CrossPair(
                traj_a=a.traj_name,
                display_traj_a=a.display_traj,
                frame_a_idx=a.frame_idx,
                traj_b=b.traj_name,
                display_traj_b=b.display_traj,
                frame_b_idx=b.frame_idx,
                pred_distance=distance,
                image_a=a.image_rel_path,
                image_b=b.image_rel_path,
            )
        )
    if len(selected) < num_pairs:
        print(f"[warn] Requested {num_pairs} cross pairs but only sampled {len(selected)}")
    return selected


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _build_figure(
    within_pairs: Sequence[WithinPair],
    cross_pairs_sorted: Sequence[Tuple[int, CrossPair]],
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Within-trajectory pairs",
            "Cross-trajectory pairs (sorted by predicted distance)",
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
    )

    if within_pairs:
        step_gaps = np.array([p.step_gap for p in within_pairs], dtype=float)
        distances = np.array([p.pred_distance for p in within_pairs], dtype=float)
        trajs = np.array([p.traj_name for p in within_pairs], dtype=object)
        custom = np.array(
            [
                [
                    p.image_a,
                    p.image_b,
                    p.display_traj,
                    p.frame_a_idx,
                    p.display_traj,
                    p.frame_b_idx,
                    p.step_gap,
                    p.pred_distance,
                    "within",
                    "",
                ]
                for p in within_pairs
            ],
            dtype=object,
        )
        unique_trajs = np.unique(trajs)
        palette = go.Figure().layout.template.layout.colorway or []
        if not palette:
            palette = [
                "#636EFA",
                "#EF553B",
                "#00CC96",
                "#AB63FA",
                "#FFA15A",
                "#19D3F3",
                "#FF6692",
                "#B6E880",
                "#FF97FF",
                "#FECB52",
            ]
        if len(unique_trajs) > len(palette):
            repeats = len(unique_trajs) // len(palette) + 1
            palette = (palette * repeats)[: len(unique_trajs)]
        color_map = {traj: palette[i] for i, traj in enumerate(unique_trajs)}
        for traj_name in unique_trajs:
            mask = trajs == traj_name
            fig.add_trace(
                go.Scatter(
                    x=step_gaps[mask],
                    y=distances[mask],
                    mode="markers",
                    name=traj_name,
                    legendgroup=traj_name,
                    marker=dict(size=6, opacity=0.7, color=color_map[traj_name]),
                    customdata=custom[mask],
                    hovertemplate="<extra></extra>",
                ),
                row=1,
                col=1,
            )
        fig.update_xaxes(title_text="Step gap |Δt|", row=1, col=1)
        fig.update_yaxes(title_text="Predicted distance", row=1, col=1)
    else:
        fig.update_xaxes(title_text="Step gap |Δt|", row=1, col=1)
        fig.update_yaxes(title_text="Predicted distance", row=1, col=1)

    if cross_pairs_sorted:
        indices = np.array([rank for rank, _ in cross_pairs_sorted], dtype=float)
        distances = np.array([pair.pred_distance for _, pair in cross_pairs_sorted], dtype=float)
        custom = np.array(
            [
                [
                    pair.image_a,
                    pair.image_b,
                    pair.display_traj_a,
                    pair.frame_a_idx,
                    pair.display_traj_b,
                    pair.frame_b_idx,
                    "",
                    pair.pred_distance,
                    "cross",
                    rank,
                ]
                for rank, pair in cross_pairs_sorted
            ],
            dtype=object,
        )
        fig.add_trace(
            go.Scatter(
                x=indices,
                y=distances,
                mode="markers",
                name="cross",
                marker=dict(size=6, opacity=0.7, color="#636EFA"),
                customdata=custom,
                hovertemplate="<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text="Pair rank", row=1, col=2)
    fig.update_yaxes(title_text="Predicted distance", row=1, col=2)

    fig.update_layout(
        title="Within vs cross-trajectory predicted distances",
        hovermode="closest",
        template="plotly_white",
        legend=dict(groupclick="togglegroup"),
    )
    return fig


def _build_overlay_html() -> str:
    return """
<div id=\"pair-hover\" style=\"position:fixed; display:none; pointer-events:none; border:1px solid #999; background:rgba(255,255,255,0.97); padding:6px; z-index:1000; max-width:520px; box-shadow:0 2px 8px rgba(0,0,0,0.15);\">
  <div id=\"pair-hover-title\" style=\"font-weight:600; margin-bottom:4px; font-size:13px;\"></div>
  <div style=\"display:flex; gap:6px;\">
    <div style=\"flex:1; min-width:0;\">
      <img id=\"pair-hover-img-a\" src=\"\" style=\"width:100%; height:auto; display:block; border:1px solid #ccc; background:#f8f8f8;\">
      <div id=\"pair-hover-meta-a\" style=\"font-size:12px; margin-top:2px; line-height:1.35;\"></div>
    </div>
    <div style=\"flex:1; min-width:0;\">
      <img id=\"pair-hover-img-b\" src=\"\" style=\"width:100%; height:auto; display:block; border:1px solid #ccc; background:#f8f8f8;\">
      <div id=\"pair-hover-meta-b\" style=\"font-size:12px; margin-top:2px; line-height:1.35;\"></div>
    </div>
  </div>
  <div id=\"pair-hover-info\" style=\"font-size:12px; margin-top:4px; line-height:1.4;\"></div>
</div>
"""


def _build_overlay_script() -> str:
    return """
<script>
  (function() {
    var plotDiv = document.getElementById('pair-distance-plot');
    if (!plotDiv) { return; }
    var hoverDiv = document.getElementById('pair-hover');
    var imgA = document.getElementById('pair-hover-img-a');
    var imgB = document.getElementById('pair-hover-img-b');
    var metaA = document.getElementById('pair-hover-meta-a');
    var metaB = document.getElementById('pair-hover-meta-b');
    var info = document.getElementById('pair-hover-info');
    var title = document.getElementById('pair-hover-title');

    function positionHover(event) {
      if (!event) { return; }
      var padding = 16;
      var viewportWidth = window.innerWidth || document.documentElement.clientWidth || 0;
      var viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;
      var pointX = event.clientX || 0;
      var pointY = event.clientY || 0;

      // Measure current overlay size; fall back to last known dimensions if needed.
      var rect = hoverDiv.getBoundingClientRect();
      var overlayWidth = rect.width || hoverDiv.offsetWidth || 0;
      var overlayHeight = rect.height || hoverDiv.offsetHeight || 0;

      var x = pointX + padding;
      var y = pointY + padding;

      // If the overlay would overflow the right edge, place it to the left of the cursor.
      if (overlayWidth && x + overlayWidth > viewportWidth - 4) {
        x = pointX - overlayWidth - padding;
      }
      // If still overflowing, clamp inside viewport.
      if (overlayWidth) {
        x = Math.min(x, viewportWidth - overlayWidth - 4);
      }
      if (x < 4) {
        x = 4;
      }

      // Adjust vertically if overlay would overflow the bottom edge.
      if (overlayHeight && y + overlayHeight > viewportHeight - 4) {
        y = pointY - overlayHeight - padding;
      }
      if (overlayHeight) {
        y = Math.min(y, viewportHeight - overlayHeight - 4);
      }
      if (y < 4) {
        y = 4;
      }

      hoverDiv.style.left = x + 'px';
      hoverDiv.style.top = y + 'px';
    }

    plotDiv.on('plotly_hover', function(data) {
      if (!data || !data.points || !data.points.length) { return; }
      var point = data.points[0];
      var custom = point.customdata || [];
      var imgPathA = custom[0] || '';
      var imgPathB = custom[1] || '';
      var trajA = custom[2] || '';
      var frameA = custom[3] != null ? custom[3] : '';
      var trajB = custom[4] || '';
      var frameB = custom[5] != null ? custom[5] : '';
      var stepGap = custom[6];
      var predDist = custom[7];
      var pairType = custom[8] || '';
      var extra = custom[9] || '';

      imgA.src = imgPathA;
      imgB.src = imgPathB;
      metaA.innerHTML = '<strong>' + trajA + '</strong><br>Frame ' + frameA;
      metaB.innerHTML = '<strong>' + trajB + '</strong><br>Frame ' + frameB;

      var distText = (predDist != null && !isNaN(predDist)) ? predDist.toFixed(4) : '';
      if (pairType === 'within') {
        title.textContent = 'Within-trajectory pair';
        info.innerHTML = 'Δt = ' + stepGap + '<br>Predicted distance = ' + distText;
      } else {
        title.textContent = 'Cross-trajectory pair';
        info.innerHTML = 'Rank = ' + extra + '<br>Predicted distance = ' + distText;
      }

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


def main(args: Args) -> None:
    rng = random.Random(args.seed)
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
    traj_frames = _build_frame_infos(results, copied_paths, out_dir)

    within_target = args.num_within_pairs
    cross_target = args.num_cross_pairs
    if args.max_points is not None:
        within_target = min(within_target, args.max_points)
        cross_target = min(cross_target, args.max_points)

    within_pairs = _sample_within_pairs(
        traj_frames,
        within_target,
        args.max_step_gap,
        rng,
    )
    cross_pairs = _sample_cross_pairs(
        traj_frames,
        cross_target,
        rng,
    )
    cross_pairs_sorted = list(
        enumerate(sorted(cross_pairs, key=lambda pair: pair.pred_distance), start=1)
    )

    _write_csv(
        out_dir / "pairs_within.csv",
        [
            "traj_name",
            "display_traj",
            "frame_a_idx",
            "frame_b_idx",
            "step_gap",
            "pred_distance",
            "image_a",
            "image_b",
        ],
        [
            [
                pair.traj_name,
                pair.display_traj,
                pair.frame_a_idx,
                pair.frame_b_idx,
                pair.step_gap,
                f"{pair.pred_distance:.6f}",
                pair.image_a,
                pair.image_b,
            ]
            for pair in within_pairs
        ],
    )

    _write_csv(
        out_dir / "pairs_cross.csv",
        [
            "traj_a",
            "display_traj_a",
            "frame_a_idx",
            "traj_b",
            "display_traj_b",
            "frame_b_idx",
            "pred_distance",
            "image_a",
            "image_b",
        ],
        [
            [
                pair.traj_a,
                pair.display_traj_a,
                pair.frame_a_idx,
                pair.traj_b,
                pair.display_traj_b,
                pair.frame_b_idx,
                f"{pair.pred_distance:.6f}",
                pair.image_a,
                pair.image_b,
            ]
            for pair in cross_pairs
        ],
    )

    fig = _build_figure(within_pairs, cross_pairs_sorted)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "pair_distance_visualization.html"
    plot_html = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        div_id="pair-distance-plot",
        default_height="100vh",
        default_width="100%",
    )

    template = Template(
        """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Within vs cross-trajectory distances</title>
  <style>
    html, body { height: 100%; margin: 0; padding: 0; font-family: sans-serif; }
    #container { height: 100vh; padding: 12px; box-sizing: border-box; }
    #pair-distance-plot { height: 100%; }
  </style>
</head>
<body>
  <div id=\"container\">
    $plot
  </div>
  $overlay
  $script
</body>
</html>
"""
    )

    html_path.write_text(
        template.substitute(
            plot=plot_html,
            overlay=_build_overlay_html(),
            script=_build_overlay_script(),
        )
    )
    print(f"Saved interactive visualization to {html_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
