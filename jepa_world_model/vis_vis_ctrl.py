#!/usr/bin/env python3
"""Vis vs Ctrl diagnostics helpers."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from jepa_world_model.actions import compress_actions_to_ids
from jepa_world_model.plot_raincloud import plot_raincloud


@dataclass
class VisCtrlMetrics:
    knn_mean_distances: Dict[int, float]
    knn_distance_samples: Dict[int, np.ndarray]
    eigenvalues: np.ndarray
    global_variance: float
    composition_error_mean: float
    composition_errors: np.ndarray
    jaccard_means: Dict[int, float]
    jaccard_samples: Dict[int, np.ndarray]


def _compute_global_variance(embeddings: torch.Tensor) -> float:
    if embeddings.numel() == 0:
        return float("nan")
    var = embeddings.var(dim=0, unbiased=False)
    return float(var.sum().item())


def _compute_knn_indices(
    embeddings: torch.Tensor,
    k_max: int,
    chunk_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    embeddings = embeddings.to(dtype=torch.float32)
    num = embeddings.shape[0]
    if num == 0 or k_max <= 0:
        return np.zeros((0, 0), dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    k = min(k_max, max(1, num - 1))
    knn_idx = np.zeros((num, k), dtype=np.int64)
    knn_dist = np.zeros((num, k), dtype=np.float32)
    for start in range(0, num, chunk_size):
        end = min(num, start + chunk_size)
        chunk = embeddings[start:end]
        dists = torch.cdist(chunk, embeddings)
        row_ids = torch.arange(start, end, device=dists.device)
        dists[torch.arange(end - start), row_ids] = float("inf")
        vals, idx = torch.topk(dists, k=k, largest=False)
        knn_idx[start:end] = idx.cpu().numpy()
        knn_dist[start:end] = vals.cpu().numpy()
    return knn_idx, knn_dist


def _compute_knn_distance_stats(
    knn_dist: np.ndarray, ks: Iterable[int]
) -> Tuple[Dict[int, float], Dict[int, np.ndarray]]:
    if knn_dist.size == 0:
        empty_means = {int(k): float("nan") for k in ks}
        empty_samples = {int(k): np.zeros(0, dtype=np.float32) for k in ks}
        return empty_means, empty_samples
    ks_list = [int(k) for k in ks]
    max_k = knn_dist.shape[1]
    means: Dict[int, float] = {}
    samples: Dict[int, np.ndarray] = {}
    cumulative = np.cumsum(knn_dist, axis=1)
    for k in ks_list:
        k_eff = min(k, max_k)
        if k_eff <= 0:
            means[k] = float("nan")
            samples[k] = np.zeros(0, dtype=np.float32)
            continue
        per_row = cumulative[:, k_eff - 1] / float(k_eff)
        means[k] = float(np.mean(per_row))
        samples[k] = per_row.astype(np.float32)
    return means, samples


def _compute_jaccard_over_time(
    knn_idx: np.ndarray,
    seq_len: int,
    batch_size: int,
    ks: Iterable[int],
    delta: int = 1,
) -> Tuple[Dict[int, float], Dict[int, np.ndarray]]:
    if knn_idx.size == 0 or seq_len <= delta:
        empty_means = {int(k): float("nan") for k in ks}
        empty_samples = {int(k): np.zeros(0, dtype=np.float32) for k in ks}
        return empty_means, empty_samples
    ks_list = [int(k) for k in ks]
    max_k = knn_idx.shape[1]
    totals = {k: 0.0 for k in ks_list}
    counts = {k: 0 for k in ks_list}
    samples: Dict[int, List[float]] = {k: [] for k in ks_list}
    for b in range(batch_size):
        base = b * seq_len
        for t in range(seq_len - delta):
            idx_a = base + t
            idx_b = base + t + delta
            neigh_a = knn_idx[idx_a]
            neigh_b = knn_idx[idx_b]
            for k in ks_list:
                k_eff = min(k, max_k)
                if k_eff <= 0:
                    continue
                set_a = set(neigh_a[:k_eff])
                set_b = set(neigh_b[:k_eff])
                inter = len(set_a & set_b)
                union = len(set_a | set_b)
                if union == 0:
                    continue
                score = inter / union
                totals[k] += score
                samples[k].append(score)
                counts[k] += 1
    means = {k: (totals[k] / counts[k]) if counts[k] else float("nan") for k in ks_list}
    sample_arrays = {k: np.asarray(samples[k], dtype=np.float32) for k in ks_list}
    return means, sample_arrays


def _compute_eigen_spectrum(embeddings: torch.Tensor, top_k: int = 10) -> Tuple[np.ndarray, float]:
    if embeddings.numel() == 0 or embeddings.shape[0] < 2:
        return np.zeros(0, dtype=np.float32), float("nan")
    emb_np = embeddings.detach().cpu().numpy().astype(np.float64)
    emb_np = emb_np - emb_np.mean(axis=0, keepdims=True)
    cov = np.cov(emb_np, rowvar=False, bias=True)
    if cov.ndim == 0:
        eigvals = np.asarray([float(cov)], dtype=np.float64)
    else:
        eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    global_var = float(np.sum(eigvals))
    return eigvals[:top_k].astype(np.float32), global_var


def _compute_action_means(
    deltas: torch.Tensor,
    action_ids: torch.Tensor,
    min_action_count: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    flat_deltas = deltas.reshape(-1, deltas.shape[-1]).cpu().numpy()
    flat_actions = action_ids.reshape(-1).cpu().numpy()
    means: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    for aid in np.unique(flat_actions):
        mask = flat_actions == aid
        count = int(mask.sum())
        if count < max(min_action_count, 1):
            continue
        mean = flat_deltas[mask].mean(axis=0)
        means[int(aid)] = mean
        counts[int(aid)] = count
    return means, counts


def _compute_two_step_composition_error(
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    min_action_count: int,
    warmup: int,
) -> Tuple[float, np.ndarray]:
    if embeddings.shape[1] < warmup + 3:
        return float("nan"), np.zeros(0, dtype=np.float32)
    emb = embeddings[:, warmup:].cpu()
    act = actions[:, warmup:].cpu()
    deltas = emb[:, 1:] - emb[:, :-1]
    action_ids = compress_actions_to_ids(act.reshape(-1, act.shape[-1])).reshape(act.shape[0], act.shape[1])
    action_ids = action_ids[:, :-1]
    means, _ = _compute_action_means(deltas, action_ids, min_action_count)
    if not means:
        return float("nan"), np.zeros(0, dtype=np.float32)
    errors: List[float] = []
    seq_len = emb.shape[1]
    for b in range(emb.shape[0]):
        for t in range(seq_len - 2):
            a0 = int(action_ids[b, t])
            a1 = int(action_ids[b, t + 1])
            if a0 not in means or a1 not in means:
                continue
            predicted = means[a0] + means[a1]
            actual = (emb[b, t + 2] - emb[b, t]).numpy()
            errors.append(float(np.linalg.norm(actual - predicted)))
    if not errors:
        return float("nan"), np.zeros(0, dtype=np.float32)
    errors_np = np.asarray(errors, dtype=np.float32)
    return float(errors_np.mean()), errors_np


def compute_vis_ctrl_metrics(
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    ks: Sequence[int],
    warmup: int,
    min_action_count: int,
    stability_delta: int,
    knn_chunk_size: int,
) -> VisCtrlMetrics:
    if embeddings.ndim != 3:
        raise ValueError("Embeddings must have shape [B, T, D].")
    if actions.ndim != 3:
        raise ValueError("Actions must have shape [B, T, action_dim].")
    if embeddings.shape[1] < max(warmup + 1, 1):
        return VisCtrlMetrics({}, float("nan"), float("nan"), np.zeros(0, dtype=np.float32), {})
    emb = embeddings[:, warmup:]
    b, t, _ = emb.shape
    flat = emb.reshape(b * t, emb.shape[-1]).cpu()
    k_max = max([int(k) for k in ks] + [1])
    knn_idx, knn_dist = _compute_knn_indices(flat, k_max, chunk_size=knn_chunk_size)
    knn_means, knn_samples = _compute_knn_distance_stats(knn_dist, ks)
    eigenvalues, variance = _compute_eigen_spectrum(flat)
    if not np.isfinite(variance):
        variance = _compute_global_variance(flat)
    comp_mean, comp_vals = _compute_two_step_composition_error(embeddings, actions, min_action_count, warmup)
    jaccard_means, jaccard_samples = _compute_jaccard_over_time(knn_idx, t, b, ks, delta=stability_delta)
    return VisCtrlMetrics(
        knn_mean_distances=knn_means,
        knn_distance_samples=knn_samples,
        eigenvalues=eigenvalues,
        global_variance=variance,
        composition_error_mean=comp_mean,
        composition_errors=comp_vals,
        jaccard_means=jaccard_means,
        jaccard_samples=jaccard_samples,
    )


def save_smoothness_plot(
    out_path: Path,
    metrics: VisCtrlMetrics,
    embedding_label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ks = sorted(metrics.knn_mean_distances.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    if ks:
        samples = [metrics.knn_distance_samples.get(k, np.zeros(0, dtype=np.float32)) for k in ks]
        labels = [f"k={k}" for k in ks]
        plot_raincloud(axes[0], samples, labels, "kNN distance")
        axes[0].set_title(f"kNN distance raincloud ({embedding_label})")
    else:
        axes[0].text(0.5, 0.5, "No kNN data.", ha="center", va="center")
        axes[0].axis("off")
    eigvals = metrics.eigenvalues if metrics.eigenvalues.size else np.zeros(0, dtype=np.float32)
    if eigvals.size:
        axes[1].bar(np.arange(1, eigvals.size + 1), eigvals, color="tab:green", alpha=0.8)
        axes[1].set_xlabel("eigenvalue rank")
        axes[1].set_ylabel("eigenvalue")
        variance_text = metrics.global_variance
        if np.isfinite(variance_text):
            title = f"Eigenvalue spectrum ({embedding_label})\n(global variance = {variance_text:.4f})"
        else:
            title = f"Eigenvalue spectrum ({embedding_label})"
        axes[1].set_title(title)
    else:
        axes[1].text(0.5, 0.5, "No eigenvalues.", ha="center", va="center")
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_composition_error_plot(
    out_path: Path,
    metrics: VisCtrlMetrics,
    embedding_label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if metrics.composition_errors.size:
        ax.hist(metrics.composition_errors, bins=20, color="tab:purple", alpha=0.8)
        ax.set_xlabel("two-step error")
        ax.set_ylabel("count")
        mean_val = metrics.composition_error_mean
        if np.isfinite(mean_val):
            ax.axvline(mean_val, color="tab:red", linestyle="--", linewidth=1.5)
            ax.set_title(f"Two-step composition error ({embedding_label})")
        else:
            ax.set_title(f"Two-step composition error ({embedding_label})")
    else:
        ax.text(0.5, 0.5, "No composition errors.", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_stability_plot(
    out_path: Path,
    metrics: VisCtrlMetrics,
    embedding_label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ks = sorted(metrics.jaccard_means.keys())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if ks:
        samples = [metrics.jaccard_samples.get(k, np.zeros(0, dtype=np.float32)) for k in ks]
        labels = [f"k={k}" for k in ks]
        plot_raincloud(ax, samples, labels, "Jaccard")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Neighborhood stability ({embedding_label})")
    else:
        ax.text(0.5, 0.5, "No stability data.", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_vis_ctrl_metrics_csv(
    csv_path: Path,
    global_step: int,
    metrics_z: VisCtrlMetrics,
    metrics_s: VisCtrlMetrics,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    knn_ks = sorted({int(k) for k in metrics_z.knn_mean_distances.keys()} | {int(k) for k in metrics_s.knn_mean_distances.keys()})
    jac_ks = sorted({int(k) for k in metrics_z.jaccard_means.keys()} | {int(k) for k in metrics_s.jaccard_means.keys()})
    fieldnames = ["step"]
    for prefix in ("z", "s"):
        for k in knn_ks:
            fieldnames.append(f"{prefix}_knn_mean_k{k}")
        fieldnames.append(f"{prefix}_global_variance")
        fieldnames.append(f"{prefix}_composition_error")
        for k in jac_ks:
            fieldnames.append(f"{prefix}_jaccard_k{k}")
    row: Dict[str, float | int] = {"step": int(global_step)}
    for prefix, metrics in (("z", metrics_z), ("s", metrics_s)):
        for k in knn_ks:
            row[f"{prefix}_knn_mean_k{k}"] = float(metrics.knn_mean_distances.get(k, float("nan")))
        row[f"{prefix}_global_variance"] = float(metrics.global_variance)
        row[f"{prefix}_composition_error"] = float(metrics.composition_error_mean)
        for k in jac_ks:
            row[f"{prefix}_jaccard_k{k}"] = float(metrics.jaccard_means.get(k, float("nan")))
    write_header = True
    if csv_path.exists():
        try:
            with csv_path.open("r", newline="") as handle:
                reader = csv.reader(handle)
                existing = next(reader, [])
                if existing == fieldnames:
                    write_header = False
        except OSError:
            write_header = True
    mode = "a" if csv_path.exists() and not write_header else "w"
    with csv_path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
