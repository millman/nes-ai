"""Graph diagnostics components for composing outputs in the trainer."""
from __future__ import annotations

import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from jepa_world_model.plots.plot_edge_consistency_hist import (
    save_edge_consistency_hist_plot,
)
from jepa_world_model.plots.plot_graph_history import save_graph_history_plot
from jepa_world_model.plots.plot_in_degree_hist import save_in_degree_hist_plot
from jepa_world_model.plots.plot_neff_violin import save_neff_violin_plot
from jepa_world_model.plots.plot_rank_cdf import save_rank_cdf_plot

@dataclass
class GraphDiagnosticsConfig:
    enabled: bool = True
    sample_chunks: int = 32
    chunk_len: int = 12
    k_neighbors: int = 10
    temp: float = 0.1
    eps: float = 1e-8
    long_gap_window: int = 50
    top_m_candidates: int = 0
    block_size: int = 256
    normalize_latents: bool = True
    use_predictor_scores: bool = True
    use_ema_targets: bool = False
    mask_self_edges: bool = True
    include_edge_consistency: bool = True
    edge_consistency_samples: int = 1024


@dataclass
class GraphDiagnosticsStats:
    ranks1: np.ndarray
    ranks2: np.ndarray
    neff1: np.ndarray
    neff2: np.ndarray
    in_degree: np.ndarray
    edge_errors: np.ndarray
    k: int
    metrics: Dict[str, float]


def build_graph_diag_indices(batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len = batch.shape[:2]
    total = batch_size * seq_len
    next_index = torch.full((total,), -1, dtype=torch.long, device=batch.device)
    next2_index = torch.full_like(next_index, -1)
    chunk_ids = torch.arange(batch_size, device=batch.device).unsqueeze(1).expand(batch_size, seq_len).reshape(-1)
    for b in range(batch_size):
        base = b * seq_len
        next_index[base : base + seq_len - 1] = torch.arange(base + 1, base + seq_len, device=batch.device)
        if seq_len >= 3:
            next2_index[base : base + seq_len - 2] = torch.arange(base + 2, base + seq_len, device=batch.device)
    return next_index, next2_index, chunk_ids


def _build_graph_transition_matrix(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    cfg: GraphDiagnosticsConfig,
) -> torch.Tensor:
    if queries.shape[1] != candidates.shape[1]:
        raise ValueError("Query and candidate dimensions must match for graph diagnostics.")
    temp = max(cfg.temp, 1e-8)
    num_queries = queries.shape[0]
    num_candidates = candidates.shape[0]
    block = max(1, min(cfg.block_size, num_queries))
    probs = queries.new_zeros((num_queries, num_candidates))
    cand_norm = (candidates * candidates).sum(dim=1)
    for start in range(0, num_queries, block):
        end = min(start + block, num_queries)
        q = queries[start:end]
        row_ids = torch.arange(start, end, device=queries.device)
        if 0 < cfg.top_m_candidates < num_candidates:
            topk = min(cfg.top_m_candidates, num_candidates)
            cos = F.normalize(q, dim=-1) @ F.normalize(candidates, dim=-1).t()
            _, top_idx = torch.topk(cos, k=topk, dim=1)
            cand_sel = candidates[top_idx]
            cand_norm_sel = cand_sel.pow(2).sum(dim=2)
            q_norm = q.pow(2).sum(dim=1, keepdim=True)
            dist_sq = q_norm + cand_norm_sel - 2.0 * torch.einsum("bkd,bd->bk", cand_sel, q)
            scores = -dist_sq / temp
            if cfg.mask_self_edges:
                self_mask = top_idx == row_ids[:, None]
                scores = scores.masked_fill(self_mask, float("-inf"))
            scores = scores - scores.max(dim=1, keepdim=True).values
            probs_sel = torch.softmax(scores, dim=1)
            probs[start:end].scatter_(1, top_idx, probs_sel)
        else:
            q_norm = q.pow(2).sum(dim=1, keepdim=True)
            dist_sq = q_norm + cand_norm[None, :] - 2.0 * q @ candidates.t()
            dist_sq = dist_sq.clamp_min(0.0)
            scores = -dist_sq / temp
            if cfg.mask_self_edges:
                rel_rows = torch.arange(end - start, device=queries.device)
                scores[rel_rows, row_ids] = float("-inf")
            scores = scores - scores.max(dim=1, keepdim=True).values
            probs[start:end] = torch.softmax(scores, dim=1)
    probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
    row_sums = probs.sum(dim=1, keepdim=True)
    zero_rows = row_sums.squeeze(1) <= 0
    if torch.any(zero_rows):
        probs[zero_rows] = 1.0 / float(num_candidates)
    return probs


def _graph_rank_stats(probs: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    valid = targets >= 0
    if not torch.any(valid):
        return np.asarray([], dtype=np.float32)
    rows = probs[valid]
    tgt = targets[valid]
    row_indices = torch.arange(rows.shape[0], device=probs.device)
    target_probs = rows[row_indices, tgt]
    ranks = (rows > target_probs[:, None]).sum(dim=1) + 1
    return ranks.detach().cpu().numpy().astype(np.int64, copy=False)


def _graph_effective_neighborhood(probs: torch.Tensor, eps: float) -> torch.Tensor:
    probs_clamped = torch.clamp(probs, min=eps)
    entropy = -(probs_clamped * torch.log(probs_clamped)).sum(dim=1)
    return torch.exp(entropy)


def _graph_history_path(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "metrics_history.csv"


def _load_graph_history(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    rows: List[Dict[str, float]] = []
    try:
        with path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                parsed = {}
                for key, value in row.items():
                    try:
                        parsed[key] = float(value)
                    except (TypeError, ValueError):
                        continue
                rows.append(parsed)
    except (OSError, csv.Error):
        return []
    rows.sort(key=lambda r: r.get("step", float("inf")))
    return rows


def _write_graph_history(path: Path, metrics: Dict[str, float]) -> List[Dict[str, float]]:
    history = _load_graph_history(path)
    history = [row for row in history if row.get("step") != metrics.get("step")]
    history.append(metrics)
    history.sort(key=lambda r: r.get("step", float("inf")))
    headers = [
        "step",
        "hit1_at_k",
        "hit2_at_k",
        "median_neff1",
        "median_neff2",
        "neff_ratio",
        "long_gap_rate",
        "mutual_rate",
        "max_in_degree",
        "top1pct_mean_in_degree",
        "sample_size",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in headers})
    return history




def _compute_edge_errors(
    probs: torch.Tensor,
    top_idx: torch.Tensor,
    zhat_full: torch.Tensor,
    target_flat: torch.Tensor,
    k: int,
    sample_limit: int,
) -> np.ndarray:
    total_edges = probs.shape[0] * k
    sample = min(sample_limit, total_edges)
    if sample <= 0:
        raise AssertionError("Edge consistency sampling requires at least one edge sample.")
    perm = torch.randperm(total_edges, device=probs.device)[:sample]
    rows = perm // k
    cols = perm % k
    tgt_cols = top_idx[rows, cols]
    pred_vec = zhat_full[rows]
    tgt_vec = target_flat[tgt_cols]
    return ((pred_vec - tgt_vec) ** 2).sum(dim=1).detach().cpu().numpy()




def compute_graph_diagnostics_stats(
    queries: torch.Tensor,
    targets: torch.Tensor,
    zhat_full: torch.Tensor,
    next_index: torch.Tensor,
    next2_index: torch.Tensor,
    chunk_ids: torch.Tensor,
    cfg: GraphDiagnosticsConfig,
    global_step: int,
) -> GraphDiagnosticsStats:
    probs = _build_graph_transition_matrix(queries, targets, cfg)
    probs2 = probs @ probs

    ranks1 = _graph_rank_stats(probs, next_index)
    ranks2 = _graph_rank_stats(probs2, next2_index)
    k = max(1, min(cfg.k_neighbors, probs.shape[1]))
    hit1_at_k = float((ranks1 <= k).mean()) if ranks1.size else float("nan")
    hit2_at_k = float((ranks2 <= k).mean()) if ranks2.size else float("nan")

    neff1 = _graph_effective_neighborhood(probs, cfg.eps)
    neff2 = _graph_effective_neighborhood(probs2, cfg.eps)
    neff1_np = neff1.detach().cpu().numpy()
    neff2_np = neff2.detach().cpu().numpy()
    median_neff1 = float(np.median(neff1_np)) if neff1_np.size else float("nan")
    median_neff2 = float(np.median(neff2_np)) if neff2_np.size else float("nan")
    neff_ratio = median_neff2 / median_neff1 if median_neff1 > 0 else float("nan")

    top_vals, top_idx = torch.topk(probs, k=k, dim=1)
    idx_grid = torch.arange(probs.shape[0], device=probs.device).unsqueeze(1)
    gaps = (top_idx - idx_grid).abs()
    if chunk_ids.numel():
        chunk_ids_dev = chunk_ids.to(probs.device)
        same_chunk = chunk_ids_dev[top_idx] == chunk_ids_dev.unsqueeze(1)
        gaps = torch.where(same_chunk, gaps, gaps.new_full(gaps.shape, cfg.long_gap_window + 1))
    long_gap_rate = float((gaps > cfg.long_gap_window).float().mean()) if k > 0 else float("nan")

    neighbor_mask = torch.zeros_like(probs, dtype=torch.bool)
    neighbor_mask.scatter_(1, top_idx, True)
    mutual_counts = (neighbor_mask & neighbor_mask.t()).sum(dim=1)
    mutual_rate = float((mutual_counts.float() / k).mean()) if k > 0 else float("nan")
    in_degree = neighbor_mask.sum(dim=0).detach().cpu().numpy()
    max_in_degree = float(in_degree.max()) if in_degree.size else float("nan")
    if in_degree.size:
        top_count = max(1, int(math.ceil(0.01 * in_degree.size)))
        top_mean = float(np.mean(np.partition(in_degree, -top_count)[-top_count:]))
    else:
        top_mean = float("nan")

    if cfg.include_edge_consistency and k > 0:
        edge_errors = _compute_edge_errors(
            probs,
            top_idx,
            zhat_full,
            targets,
            k,
            cfg.edge_consistency_samples,
        )
    else:
        edge_errors = np.asarray([], dtype=np.float32)

    metrics: Dict[str, float] = {
        "step": float(global_step),
        "hit1_at_k": hit1_at_k,
        "hit2_at_k": hit2_at_k,
        "median_neff1": median_neff1,
        "median_neff2": median_neff2,
        "neff_ratio": neff_ratio,
        "long_gap_rate": long_gap_rate,
        "mutual_rate": mutual_rate,
        "max_in_degree": max_in_degree,
        "top1pct_mean_in_degree": top_mean,
        "sample_size": float(probs.shape[0]),
    }

    return GraphDiagnosticsStats(
        ranks1=ranks1,
        ranks2=ranks2,
        neff1=neff1_np,
        neff2=neff2_np,
        in_degree=in_degree,
        edge_errors=edge_errors,
        k=k,
        metrics=metrics,
    )


def update_graph_diagnostics_history(
    out_dir: Path,
    stats: GraphDiagnosticsStats,
    global_step: int,
    history_csv_path: Optional[Path] = None,
) -> List[Dict[str, float]]:
    history_path = _graph_history_path(out_dir)
    history = _write_graph_history(history_path, stats.metrics)
    if history_csv_path is not None:
        history_csv_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(history_path, history_csv_path)
    history_plot = out_dir / f"metrics_history_{global_step:07d}.png"
    save_graph_history_plot(history_plot, history, stats.k)
    latest_plot = out_dir / "metrics_history_latest.png"
    shutil.copy2(history_plot, latest_plot)
    return history
