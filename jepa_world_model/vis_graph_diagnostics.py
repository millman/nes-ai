#!/usr/bin/env python3
"""Graph diagnostics computation and plotting."""
from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from jepa_world_model_trainer import GraphDiagnosticsConfig, JEPAWorldModel


def _flatten_graph_diag_indices(batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _plot_rank_cdf(out_path: Path, ranks: np.ndarray, k: int, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if ranks.size == 0:
        ax.text(0.5, 0.5, "No valid transitions.", ha="center", va="center")
    else:
        ranks_sorted = np.sort(ranks)
        y = np.arange(1, len(ranks_sorted) + 1) / len(ranks_sorted)
        ax.plot(ranks_sorted, y, label="CDF", color="tab:blue")
        if k > 0:
            ax.axvline(k, color="tab:orange", linestyle="--", label=f"K={k}")
        ax.set_xscale("log")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Fraction ≤ rank")
        ax.grid(True, alpha=0.3)
        ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_neff_violin(out_path: Path, neff1: np.ndarray, neff2: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [neff1, neff2]
    labels = ["Neff1", "Neff2"]
    if all(arr.size == 0 for arr in data):
        ax.text(0.5, 0.5, "No neighborhood stats available.", ha="center", va="center")
    else:
        ax.violinplot(data, showmeans=True, showextrema=True)
        for idx, arr in enumerate(data):
            if arr.size == 0:
                continue
            ax.scatter(
                np.full_like(arr, idx + 1, dtype=np.float32),
                arr,
                s=8,
                alpha=0.15,
                color="tab:blue" if idx == 0 else "tab:green",
            )
        ax.set_xticks([1, 2])
        ax.set_xticklabels(labels)
        ax.set_ylabel("Effective neighborhood size")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    ax.set_title("Neighborhood size (exp entropy)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_in_degree_hist(out_path: Path, in_degree: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if in_degree.size == 0:
        ax.text(0.5, 0.5, "No edges to compute in-degree.", ha="center", va="center")
    else:
        bins = min(50, max(5, int(np.sqrt(in_degree.size))))
        ax.hist(in_degree, bins=bins, color="tab:purple", alpha=0.8)
        ax.set_xlabel("In-degree (top-K graph)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
    ax.set_title("Hubness / in-degree distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_graph_history(out_path: Path, history: List[Dict[str, float]], k: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    if not history:
        for ax in axes:
            ax.text(0.5, 0.5, "History unavailable.", ha="center", va="center")
            ax.axis("off")
    else:
        steps = [row.get("step", float("nan")) for row in history]
        hit1 = [row.get("hit1_at_k", float("nan")) for row in history]
        hit2 = [row.get("hit2_at_k", float("nan")) for row in history]
        median_neff1 = [row.get("median_neff1", float("nan")) for row in history]
        median_neff2 = [row.get("median_neff2", float("nan")) for row in history]
        ratio = [row.get("neff_ratio", float("nan")) for row in history]
        long_gap = [row.get("long_gap_rate", float("nan")) for row in history]
        mutual = [row.get("mutual_rate", float("nan")) for row in history]
        max_in = [row.get("max_in_degree", float("nan")) for row in history]

        axes[0].plot(steps, hit1, marker="o", label=f"hit1@{k}", color="tab:blue")
        axes[0].plot(steps, hit2, marker="o", label=f"hit2@{k}", color="tab:orange")
        axes[0].set_ylabel("Hit rate")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(steps, median_neff1, marker="o", label="median Neff1", color="tab:green")
        axes[1].plot(steps, median_neff2, marker="o", label="median Neff2", color="tab:red")
        axes[1].plot(steps, ratio, marker="o", label="Neff2/Neff1", color="tab:purple")
        axes[1].set_ylabel("Neighborhood size")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(steps, long_gap, marker="o", label="long-gap rate", color="tab:brown")
        axes[2].plot(steps, mutual, marker="o", label="mutual kNN rate", color="tab:cyan")
        axes[2].plot(steps, max_in, marker="o", label="max in-degree", color="tab:gray")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Graph health")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_graph_diagnostics(
    model: JEPAWorldModel,
    ema_model: Optional[JEPAWorldModel],
    frames_cpu: torch.Tensor,
    actions_cpu: torch.Tensor,
    device: torch.device,
    cfg: GraphDiagnosticsConfig,
    out_dir: Path,
    global_step: int,
) -> None:
    from jepa_world_model_trainer import _predictor_rollout
    if frames_cpu.ndim != 5:
        raise ValueError("Graph diagnostics requires frames shaped [B, T, C, H, W].")
    if frames_cpu.shape[1] < 3:
        return
    with torch.no_grad():
        frames = frames_cpu.to(device)
        actions = actions_cpu.to(device)
        embeddings = model.encode_sequence(frames)["embeddings"]
        targets = embeddings
        if cfg.use_ema_targets and ema_model is not None:
            targets = ema_model.encode_sequence(frames)["embeddings"]
        preds, _, _, _ = _predictor_rollout(model, embeddings, actions)

    batch_size, seq_len, latent_dim = embeddings.shape
    next_index, next2_index, chunk_ids = _flatten_graph_diag_indices(frames)

    z_flat = embeddings.reshape(-1, latent_dim)
    target_flat = targets.reshape(-1, latent_dim)
    zhat_full = torch.cat([preds, embeddings[:, -1:, :]], dim=1).reshape(-1, latent_dim)

    if cfg.normalize_latents:
        z_flat = F.normalize(z_flat, dim=-1)
        target_flat = F.normalize(target_flat, dim=-1)
        zhat_full = F.normalize(zhat_full, dim=-1)

    queries = zhat_full if cfg.use_predictor_scores else z_flat
    probs = _build_graph_transition_matrix(queries, target_flat, cfg)
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

    edge_errors: Optional[np.ndarray] = None
    if cfg.include_edge_consistency and k > 0:
        total_edges = probs.shape[0] * k
        sample = min(cfg.edge_consistency_samples, total_edges)
        if sample > 0:
            perm = torch.randperm(total_edges, device=probs.device)[:sample]
            rows = perm // k
            cols = perm % k
            tgt_cols = top_idx[rows, cols]
            pred_vec = zhat_full[rows]
            tgt_vec = target_flat[tgt_cols]
            edge_errors = ((pred_vec - tgt_vec) ** 2).sum(dim=1).detach().cpu().numpy()

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

    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_rank_cdf(out_dir / f"rank1_cdf_{global_step:07d}.png", ranks1, k, "1-step rank CDF")
    _plot_rank_cdf(out_dir / f"rank2_cdf_{global_step:07d}.png", ranks2, k, "2-hop rank CDF")
    _plot_neff_violin(out_dir / f"neff_violin_{global_step:07d}.png", neff1_np, neff2_np)
    _plot_in_degree_hist(out_dir / f"in_degree_hist_{global_step:07d}.png", in_degree)
    if edge_errors is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(edge_errors, bins=30, color="tab:gray", alpha=0.85)
        ax.set_title("Predictor-edge consistency (||zhat - zT||^2)")
        ax.set_xlabel("error")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(out_dir / f"edge_consistency_{global_step:07d}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    history_path = _graph_history_path(out_dir)
    history = _write_graph_history(history_path, metrics)
    history_plot = out_dir / f"metrics_history_{global_step:07d}.png"
    _plot_graph_history(history_plot, history, k)
    latest_plot = out_dir / "metrics_history_latest.png"
    shutil.copy2(history_plot, latest_plot)
