#!/usr/bin/env python3
"""Motion/action diagnostics visualization and export helpers."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from jepa_world_model.actions import compress_actions_to_ids, decode_action_id
from jepa_world_model.vis import tensor_to_uint8_image
from jepa_world_model.vis_action_alignment import save_action_alignment_detail_plot
from jepa_world_model.vis_cycle_error import compute_cycle_errors, save_cycle_error_plot

@dataclass
class DiagnosticsConfig:
    enabled: bool = True
    sample_sequences: int = 128
    top_k_components: int = 4
    min_action_count: int = 5
    max_actions_to_plot: int = 12
    cosine_high_threshold: float = 0.7
    synthesize_cycle_samples: bool = False


def _build_inverse_action_map(action_dim: int, observed_ids: Iterable[int]) -> Dict[int, int]:
    """Approximate inverse mapping by swapping up/down and left/right bits."""
    weights = (1 << np.arange(action_dim, dtype=np.int64))

    def _invert_bits(action_id: int) -> int:
        bits = np.array([(action_id >> idx) & 1 for idx in range(action_dim)], dtype=np.int64)
        if action_dim > 5:
            bits[4], bits[5] = bits[5], bits[4]
        if action_dim > 7:
            bits[6], bits[7] = bits[7], bits[6]
        return int((bits * weights).sum())

    mapping: Dict[int, int] = {}
    for aid in observed_ids:
        mapping[int(aid)] = _invert_bits(int(aid))
    return mapping


def _compute_pca(delta_z_centered: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if delta_z_centered.shape[0] < 2:
        raise ValueError("Need at least two delta steps to compute PCA.")
    try:
        _, s, vt = np.linalg.svd(delta_z_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        jitter = np.random.normal(scale=1e-6, size=delta_z_centered.shape)
        _, s, vt = np.linalg.svd(delta_z_centered + jitter, full_matrices=False)
    eigvals = (s ** 2) / max(1, delta_z_centered.shape[0] - 1)
    total_var = float(eigvals.sum()) if eigvals.size else 0.0
    if total_var <= 0:
        var_ratio = np.zeros_like(eigvals)
    else:
        var_ratio = eigvals / total_var
    return vt, var_ratio


def _compute_motion_subspace(
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    top_k: int,
    paths: Optional[List[List[str]]] = None,
) -> Optional[Dict[str, Any]]:
    if embeddings.shape[1] < 2:
        return None
    embed_np = embeddings.detach().cpu().numpy()
    action_np = actions.detach().cpu().numpy()
    batch, seq_len, latent_dim = embed_np.shape
    delta_list: List[np.ndarray] = []
    action_vecs: List[np.ndarray] = []
    for b in range(batch):
        delta_list.append(embed_np[b, 1:] - embed_np[b, :-1])
        action_vecs.append(action_np[b, :-1])
    delta_embed = np.concatenate(delta_list, axis=0)
    if delta_embed.shape[0] < 2:
        return None
    actions_flat = np.concatenate(action_vecs, axis=0)
    action_ids = compress_actions_to_ids(actions_flat)
    delta_mean = delta_embed.mean(axis=0, keepdims=True)
    delta_centered = delta_embed - delta_mean
    components, variance_ratio = _compute_pca(delta_centered)
    use_k = max(1, min(top_k, components.shape[0]))
    projection = components[:use_k].T
    flat_embed = embed_np.reshape(-1, latent_dim)
    embed_centered = flat_embed - flat_embed.mean(axis=0, keepdims=True)
    delta_proj = delta_centered @ projection
    proj_flat = embed_centered @ projection
    proj_sequences: List[np.ndarray] = []
    offset = 0
    for _ in range(batch):
        proj_sequences.append(proj_flat[offset : offset + seq_len])
        offset += seq_len
    return {
        "delta_proj": delta_proj,
        "proj_flat": proj_flat,
        "proj_sequences": proj_sequences,
        "variance_ratio": variance_ratio,
        "components": components,
        "action_ids": action_ids,
        "action_dim": action_np.shape[-1],
        "actions_seq": action_np,
        "paths": paths,
    }


def _save_delta_pca_plot(
    out_path: Path,
    variance_ratio: np.ndarray,
    delta_proj: np.ndarray,
    proj_flat: np.ndarray,
    action_ids: np.ndarray,
    action_dim: int,
    embedding_label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    num_var = min(10, variance_ratio.shape[0])
    axes[0, 0].bar(np.arange(num_var), variance_ratio[:num_var], color="tab:blue")
    axes[0, 0].set_title(f"Delta-{embedding_label} PCA variance ratio")
    axes[0, 0].set_xlabel("component")
    axes[0, 0].set_ylabel("explained variance")

    if delta_proj.shape[1] >= 2:
        unique_actions = sorted({int(a) for a in np.asarray(action_ids).reshape(-1)})
        action_to_index = {aid: idx for idx, aid in enumerate(unique_actions)}
        color_indices = (
            np.array([action_to_index.get(int(a), 0) for a in np.asarray(action_ids).reshape(-1)], dtype=np.float32)
            if unique_actions
            else np.asarray(action_ids, dtype=np.float32)
        )
        palette = plt.get_cmap("tab20").colors
        color_count = max(1, min(len(palette), len(unique_actions) if unique_actions else 1))
        color_list = list(palette[:color_count])
        cmap = mcolors.ListedColormap(color_list)
        bounds = np.arange(color_count + 1) - 0.5
        color_indices_mapped = np.mod(color_indices, color_count) if color_count else color_indices
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        scatter = axes[0, 1].scatter(
            delta_proj[:, 0],
            delta_proj[:, 1],
            c=color_indices_mapped,
            cmap=cmap,
            norm=norm,
            s=8,
            alpha=0.7,
        )
        axes[0, 1].set_xlabel(f"PC1 (delta {embedding_label})")
        axes[0, 1].set_ylabel(f"PC2 (delta {embedding_label})")
        cbar = fig.colorbar(scatter, ax=axes[0, 1], fraction=0.046, pad=0.04, boundaries=bounds)
        ticks = list(range(color_count))
        cbar.set_ticks(ticks)
        tick_labels = (
            [decode_action_id(aid, action_dim) for aid in unique_actions[:color_count]] if unique_actions else ["NOOP"]
        )
        cbar.set_ticklabels(tick_labels)
        cbar.set_label("action")
    else:
        axes[0, 1].plot(delta_proj[:, 0], np.zeros_like(delta_proj[:, 0]), ".", alpha=0.6)
        axes[0, 1].set_xlabel(f"PC1 (delta {embedding_label})")
        axes[0, 1].set_ylabel("density")
    axes[0, 1].set_title(f"Delta-{embedding_label} projections")

    cumulative = np.cumsum(variance_ratio)
    axes[1, 0].plot(np.arange(len(cumulative)), cumulative, marker="o", color="tab:green")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_xlabel("component")
    axes[1, 0].set_ylabel("cumulative variance")
    axes[1, 0].set_title("Cumulative explained variance")

    if proj_flat.shape[1] >= 2:
        t = np.linspace(0, 1, num=proj_flat.shape[0])
        sc2 = axes[1, 1].scatter(proj_flat[:, 0], proj_flat[:, 1], c=t, cmap="viridis", s=6, alpha=0.6)
        axes[1, 1].set_xlabel(f"PC1 ({embedding_label})")
        axes[1, 1].set_ylabel(f"PC2 ({embedding_label})")
        fig.colorbar(sc2, ax=axes[1, 1], fraction=0.046, pad=0.04, label="time (normalized)")
    else:
        axes[1, 1].plot(proj_flat[:, 0], np.zeros_like(proj_flat[:, 0]), ".", alpha=0.6)
        axes[1, 1].set_xlabel(f"PC1 ({embedding_label})")
        axes[1, 1].set_ylabel("density")
    upper_label = embedding_label.upper()
    axes[1, 1].set_title(f"PCA of {upper_label} on motion-defined Δ{upper_label} basis")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_variance_spectrum_plot(out_path: Path, variance_ratio: np.ndarray, max_bars: int = 32) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    k = min(len(variance_ratio), max_bars)
    if k == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(k)
    ax.bar(x, variance_ratio[:k], color="tab:blue", alpha=0.8, label="variance ratio")
    cumulative = np.cumsum(variance_ratio[:k])
    ax.plot(x, cumulative, color="tab:red", marker="o", linewidth=2, label="cumulative")
    ax.set_xlabel("component")
    ax.set_ylabel("variance ratio")
    ax.set_ylim(0, max(1.05, float(cumulative[-1]) + 0.05))
    ax.set_title("Motion PCA spectrum")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_variance_report(delta_dir: Path, global_step: int, variance_ratio: np.ndarray, embedding_label: str) -> None:
    delta_dir.mkdir(parents=True, exist_ok=True)
    report_path = delta_dir / f"delta_{embedding_label}_pca_report_{global_step:07d}.txt"
    with report_path.open("w") as handle:
        if variance_ratio.size == 0:
            handle.write("No variance ratios available.\n")
            return
        cumulative = np.cumsum(variance_ratio)
        targets = [1, 2, 4, 8, 16, 32]
        handle.write("Explained variance coverage by component count:\n")
        for t in targets:
            if t <= len(cumulative):
                handle.write(f"top_{t:02d}: {cumulative[t-1]:.4f}\n")
        handle.write("\nTop variance ratios:\n")
        top_k = min(10, len(variance_ratio))
        for i in range(top_k):
            handle.write(f"comp_{i:02d}: {variance_ratio[i]:.6f}\n")
        handle.write(f"\nTotal components: {len(variance_ratio)}\n")


def _compute_action_alignment_stats(
    delta_proj: np.ndarray,
    action_ids: np.ndarray,
    cfg: DiagnosticsConfig,
    max_actions: Optional[int] = None,
    include_mean_vectors: bool = False,
    include_norm_stats: bool = False,
) -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    unique_actions, counts = np.unique(action_ids, return_counts=True)
    if unique_actions.size == 0:
        return stats
    order = np.argsort(counts)[::-1]
    for idx in order:
        aid = int(unique_actions[idx])
        mask = action_ids == aid
        delta_a = delta_proj[mask]
        if delta_a.shape[0] < cfg.min_action_count:
            continue
        mean_dir = delta_a.mean(axis=0)
        norm = np.linalg.norm(mean_dir)
        if norm < 1e-8:
            continue
        v_unit = mean_dir / norm
        cosines: List[float] = []
        for vec in delta_a:
            denom = np.linalg.norm(vec)
            if denom < 1e-8:
                continue
            cosines.append(float(np.dot(vec / denom, v_unit)))
        if not cosines:
            continue
        cos_np = np.array(cosines, dtype=np.float32)
        entry: Dict[str, Any] = {
            "action_id": aid,
            "count": len(cos_np),
            "mean": float(cos_np.mean()),
            "median": float(np.median(cos_np)),
            "std": float(cos_np.std()),
            "pct_high": float((cos_np > cfg.cosine_high_threshold).mean()),
            "frac_neg": float((cos_np < 0).mean()),
            "cosines": cos_np,
            "mean_dir_norm": float(norm),
        }
        if include_norm_stats:
            delta_norms = np.linalg.norm(delta_a, axis=1)
            entry.update(
                {
                    "delta_norm_mean": float(delta_norms.mean()),
                    "delta_norm_median": float(np.median(delta_norms)),
                    "delta_norm_p10": float(np.percentile(delta_norms, 10)),
                    "delta_norm_p90": float(np.percentile(delta_norms, 90)),
                    "frac_low_delta_norm": float((delta_norms < 1e-8).mean()),
                }
            )
        if include_mean_vectors:
            entry["mean_dir"] = mean_dir
        stats.append(entry)
        if max_actions is not None and len(stats) >= max_actions:
            break
    return stats


def _build_action_alignment_debug(
    alignment_stats: List[Dict[str, Any]],
    delta_proj: np.ndarray,
    action_ids: np.ndarray,
) -> Dict[str, Any]:
    """Build auxiliary tensors for debugging alignment drift/degeneracy."""
    mean_units: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    for stat in alignment_stats:
        aid = int(stat.get("action_id", -1))
        mean_vec = stat.get("mean_dir")
        norm = stat.get("mean_dir_norm", 0.0) or 0.0
        if mean_vec is not None and norm >= 1e-8:
            mean_units[aid] = np.asarray(mean_vec, dtype=np.float32) / float(norm)
        counts[aid] = int(stat.get("count", 0))

    per_action_norms: Dict[int, np.ndarray] = {}
    per_action_cos: Dict[int, np.ndarray] = {}
    overall_cos_list: List[np.ndarray] = []
    overall_norm_list: List[np.ndarray] = []

    for aid, mean_unit in mean_units.items():
        mask = action_ids == aid
        vecs = delta_proj[mask]
        if vecs.shape[0] == 0:
            continue
        norms = np.linalg.norm(vecs, axis=1)
        valid = norms >= 1e-8
        if not np.any(valid):
            continue
        vec_unit = vecs[valid] / norms[valid, None]
        cos = vec_unit @ mean_unit
        per_action_norms[aid] = norms[valid]
        per_action_cos[aid] = cos
        overall_cos_list.append(cos)
        overall_norm_list.append(norms[valid])

    overall_cos = np.concatenate(overall_cos_list) if overall_cos_list else np.asarray([], dtype=np.float32)
    overall_norms = np.concatenate(overall_norm_list) if overall_norm_list else np.asarray([], dtype=np.float32)

    actions_sorted = sorted(mean_units.keys())
    pairwise = np.full((len(actions_sorted), len(actions_sorted)), np.nan, dtype=np.float32)
    for i, ai in enumerate(actions_sorted):
        for j, aj in enumerate(actions_sorted):
            a_vec = mean_units.get(ai)
            b_vec = mean_units.get(aj)
            if a_vec is None or b_vec is None:
                continue
            denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
            if denom < 1e-8:
                continue
            pairwise[i, j] = float(np.dot(a_vec, b_vec) / denom)

    return {
        "actions_sorted": actions_sorted,
        "counts": counts,
        "overall_cos": overall_cos,
        "overall_norms": overall_norms,
        "per_action_norms": per_action_norms,
        "per_action_cos": per_action_cos,
        "pairwise": pairwise,
    }


def _write_action_alignment_report(
    alignment_dir: Path,
    global_step: int,
    stats: List[Dict[str, Any]],
    action_dim: int,
    inverse_map: Dict[int, int],
) -> None:
    alignment_dir.mkdir(parents=True, exist_ok=True)
    report_path = alignment_dir / f"action_alignment_report_{global_step:07d}.txt"
    with report_path.open("w") as handle:
        handle.write("Action alignment diagnostics (per action)\n")
        if not stats:
            handle.write("No actions met alignment criteria.\n")
            return
        handle.write(
            "action_id\tlabel\tcount\tmean\tmedian\tstd\tfrac_neg\tpct_gt_thr\tv_norm"
            "\tdelta_norm_median\tdelta_norm_p90\tfrac_low_norm\tinverse_alignment\tnotes\n"
        )
        mean_vecs: Dict[int, np.ndarray] = {}
        for stat in stats:
            if "mean_dir" in stat:
                mean_vecs[int(stat["action_id"])] = stat["mean_dir"]
        for stat in stats:
            aid = int(stat["action_id"])
            label = decode_action_id(aid, action_dim)
            inv_align = ""
            inv_id = inverse_map.get(aid)
            if inv_id is not None:
                inv_vec = mean_vecs.get(inv_id)
                this_vec = mean_vecs.get(aid)
                if inv_vec is not None and this_vec is not None:
                    inv_norm = float(np.linalg.norm(inv_vec))
                    this_norm = float(np.linalg.norm(this_vec))
                    if inv_norm > 1e-8 and this_norm > 1e-8:
                        inv_align = float(np.dot(this_vec, -inv_vec) / (inv_norm * this_norm))
            note = ""
            if stat.get("mean", 0.0) < 0:
                if stat.get("mean_dir_norm", 0.0) < 1e-6 or stat.get("delta_norm_p90", 0.0) < 1e-6:
                    note = "degenerate mean/blocked"
                elif stat.get("frac_neg", 0.0) > 0.4:
                    note = "bimodal/aliasing suspected"
                else:
                    note = "check action mapping/PCA"
            elif stat.get("mean_dir_norm", 0.0) < 1e-6:
                note = "mean direction near zero"
            handle.write(
                f"{aid}\t{label}\t{stat.get('count', 0)}\t{stat.get('mean', float('nan')):.4f}"
                f"\t{stat.get('median', float('nan')):.4f}\t{stat.get('std', float('nan')):.4f}"
                f"\t{stat.get('frac_neg', float('nan')):.3f}\t{stat.get('pct_high', float('nan')):.3f}"
                f"\t{stat.get('mean_dir_norm', float('nan')):.4f}"
                f"\t{stat.get('delta_norm_median', float('nan')):.4f}\t{stat.get('delta_norm_p90', float('nan')):.4f}"
                f"\t{stat.get('frac_low_delta_norm', float('nan')):.3f}\t{inv_align}\t{note}\n"
            )


def _write_action_alignment_strength(
    alignment_dir: Path,
    global_step: int,
    stats: List[Dict[str, Any]],
    action_dim: int,
) -> None:
    """Summarize per-action directional strength relative to step magnitude."""
    path = alignment_dir / f"action_alignment_strength_{global_step:07d}.txt"
    with path.open("w") as handle:
        if not stats:
            handle.write("No actions met alignment criteria.\n")
            return
        handle.write(
            "Per-action directional strength (mean_dir_norm / delta_norm_median)\n"
            "Lower ratios imply the average direction is weak relative to per-step magnitude (possible aliasing/sign flips).\n\n"
        )
        handle.write(
            "action_id\tlabel\tcount\tmean_cos\tstd\tfrac_neg\tmean_dir_norm\tdelta_norm_median\tstrength_ratio\tnote\n"
        )
        for stat in stats:
            delta_med = float(stat.get("delta_norm_median", float("nan")))
            mean_norm = float(stat.get("mean_dir_norm", float("nan")))
            strength = float("nan")
            if np.isfinite(delta_med) and delta_med > 0 and np.isfinite(mean_norm):
                strength = mean_norm / delta_med
            note = ""
            if not np.isfinite(strength) or strength < 0.05:
                note = "weak mean vs magnitude"
            elif strength < 0.15:
                note = "moderate mean vs magnitude"
            handle.write(
                f"{stat.get('action_id')}\t{decode_action_id(stat.get('action_id', -1), action_dim)}"
                f"\t{stat.get('count', 0)}\t{stat.get('mean', float('nan')):.4f}"
                f"\t{stat.get('std', float('nan')):.4f}\t{stat.get('frac_neg', float('nan')):.3f}"
                f"\t{mean_norm:.4f}\t{delta_med:.4f}\t{strength:.4f}\t{note}\n"
            )


def _write_action_alignment_crosscheck(
    alignment_dir: Path,
    global_step: int,
    stats: List[Dict[str, Any]],
    action_dim: int,
    action_ids: np.ndarray,
    delta_proj: np.ndarray,
) -> None:
    alignment_dir.mkdir(parents=True, exist_ok=True)
    path = alignment_dir / f"action_alignment_crosscheck_{global_step:07d}.txt"
    mean_units: Dict[int, np.ndarray] = {}
    for stat in stats:
        mean_vec = stat.get("mean_dir")
        norm = stat.get("mean_dir_norm", 0.0)
        aid = int(stat["action_id"])
        if mean_vec is None or norm is None or norm < 1e-8:
            continue
        mean_units[aid] = mean_vec / norm
    if not mean_units:
        with path.open("w") as handle:
            handle.write("No usable mean directions for crosscheck.\n")
        return
    with path.open("w") as handle:
        handle.write(
            "Cross-check: sample cosines against own vs other mean directions\n"
            "action_id\tlabel\tcount_valid\tself_mean\tbest_other_id\tbest_other_label"
            "\tbest_other_mean\tgap_self_minus_best_other\tnote\n"
        )
        for aid, mean_unit in mean_units.items():
            mask = action_ids == aid
            vecs = delta_proj[mask]
            if vecs.shape[0] == 0:
                continue
            norms = np.linalg.norm(vecs, axis=1)
            valid_mask = norms >= 1e-8
            if not np.any(valid_mask):
                continue
            vecs_unit = vecs[valid_mask] / norms[valid_mask, None]
            self_mean = float(np.dot(vecs_unit, mean_unit).mean())
            best_other_id: Optional[int] = None
            best_other_mean = -float("inf")
            for bid, other_unit in mean_units.items():
                if bid == aid:
                    continue
                other_mean = float(np.dot(vecs_unit, other_unit).mean())
                if other_mean > best_other_mean:
                    best_other_mean = other_mean
                    best_other_id = bid
            gap = self_mean - best_other_mean if best_other_id is not None else float("nan")
            note = ""
            if best_other_id is not None and gap < 0.05:
                note = "samples align similarly to another action"
            handle.write(
                f"{aid}\t{decode_action_id(aid, action_dim)}\t{vecs_unit.shape[0]}"
                f"\t{self_mean:.4f}\t{best_other_id}"
                f"\t{decode_action_id(best_other_id, action_dim) if best_other_id is not None else ''}"
                f"\t{best_other_mean:.4f}\t{gap:.4f}\t{note}\n"
            )


def _write_diagnostics_csvs(
    delta_dir: Path,
    alignment_dir: Path,
    cycle_dir: Path,
    global_step: int,
    motion: Dict[str, Any],
    cfg: DiagnosticsConfig,
    alignment_stats: List[Dict[str, Any]],
    alignment_debug: Optional[Dict[str, Any]],
    cycle_errors: List[Tuple[int, float]],
    cycle_per_action: Dict[int, List[float]],
    embedding_label: str,
) -> None:
    delta_dir.mkdir(parents=True, exist_ok=True)
    alignment_dir.mkdir(parents=True, exist_ok=True)
    cycle_dir.mkdir(parents=True, exist_ok=True)

    delta_var_csv = delta_dir / f"delta_{embedding_label}_pca_variance_{global_step:07d}.csv"
    with delta_var_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component", "variance_ratio"])
        for idx, val in enumerate(motion["variance_ratio"][:64]):  # cap rows
            writer.writerow([idx, float(val)])

    delta_samples_csv = delta_dir / f"delta_{embedding_label}_pca_samples_{global_step:07d}.csv"
    with delta_samples_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "frame_index", "frame_path"])
        paths = motion.get("paths") or []
        if paths:
            for sample_idx, frame_list in enumerate(paths):
                if not frame_list:
                    continue
                writer.writerow([sample_idx, 0, frame_list[0]])

    align_csv = alignment_dir / f"action_alignment_{global_step:07d}.csv"
    with align_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["action_id", "action_label", "count", "mean_cos", "std_cos", "pct_high"])
        for stat in alignment_stats:
            writer.writerow(
                [
                    stat["action_id"],
                    decode_action_id(stat["action_id"], motion["action_dim"]),
                    stat["count"],
                    stat["mean"],
                    stat["std"],
                    stat["pct_high"],
                ]
            )

    align_full_csv = alignment_dir / f"action_alignment_full_{global_step:07d}.csv"
    with align_full_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "action_id",
                "action_label",
                "count",
                "mean_cos",
                "median_cos",
                "std_cos",
                "pct_high",
                "frac_negative",
                "mean_dir_norm",
                "delta_norm_mean",
                "delta_norm_median",
                "delta_norm_p10",
                "delta_norm_p90",
                "frac_low_delta_norm",
            ]
        )
        for stat in alignment_stats:
            writer.writerow(
                [
                    stat.get("action_id"),
                    decode_action_id(stat.get("action_id", -1), motion["action_dim"]),
                    stat.get("count", 0),
                    stat.get("mean", float("nan")),
                    stat.get("median", float("nan")),
                    stat.get("std", float("nan")),
                    stat.get("pct_high", float("nan")),
                    stat.get("frac_neg", float("nan")),
                    stat.get("mean_dir_norm", float("nan")),
                    stat.get("delta_norm_mean", float("nan")),
                    stat.get("delta_norm_median", float("nan")),
                    stat.get("delta_norm_p10", float("nan")),
                    stat.get("delta_norm_p90", float("nan")),
                    stat.get("frac_low_delta_norm", float("nan")),
                ]
            )

    if alignment_debug is not None:
        pairwise_csv = alignment_dir / f"action_alignment_pairwise_{global_step:07d}.csv"
        actions_sorted: List[int] = list(alignment_debug.get("actions_sorted") or [])
        pairwise_raw = alignment_debug.get("pairwise")
        pairwise = np.asarray([] if pairwise_raw is None else pairwise_raw, dtype=np.float32)
        with pairwise_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["action_id_a", "label_a", "action_id_b", "label_b", "cosine"])
            for i, aid in enumerate(actions_sorted):
                for j, bid in enumerate(actions_sorted):
                    if pairwise.size == 0 or i >= pairwise.shape[0] or j >= pairwise.shape[1]:
                        continue
                    writer.writerow(
                        [
                            aid,
                            decode_action_id(aid, motion["action_dim"]),
                            bid,
                            decode_action_id(bid, motion["action_dim"]),
                            float(pairwise[i, j]),
                        ]
                    )

        overview_txt = alignment_dir / f"action_alignment_overview_{global_step:07d}.txt"
        overall_cos_raw = alignment_debug.get("overall_cos")
        overall_norms_raw = alignment_debug.get("overall_norms")
        overall_cos = np.asarray([] if overall_cos_raw is None else overall_cos_raw, dtype=np.float32)
        overall_norms = np.asarray([] if overall_norms_raw is None else overall_norms_raw, dtype=np.float32)
        with overview_txt.open("w") as handle:
            handle.write("Global alignment summary (cosine vs per-action mean)\n")
            if overall_cos.size == 0:
                handle.write("No valid cosine samples.\n")
            else:
                handle.write(f"samples: {overall_cos.size}\n")
                handle.write(f"mean: {float(overall_cos.mean()):.4f}\n")
                handle.write(f"median: {float(np.median(overall_cos)):.4f}\n")
                handle.write(f"std: {float(overall_cos.std()):.4f}\n")
                handle.write(
                    f"pct_gt_thr({cfg.cosine_high_threshold}): {float((overall_cos > cfg.cosine_high_threshold).mean()):.4f}\n"
                )
                handle.write(f"frac_negative: {float((overall_cos < 0).mean()):.4f}\n")
                if overall_norms.size:
                    handle.write("\nDelta norm stats (all actions):\n")
                    handle.write(f"median: {float(np.median(overall_norms)):.6f}\n")
                    handle.write(f"p10: {float(np.percentile(overall_norms, 10)):.6f}\n")
                    handle.write(f"p90: {float(np.percentile(overall_norms, 90)):.6f}\n")

    cycle_values_csv = cycle_dir / f"cycle_error_values_{global_step:07d}.csv"
    with cycle_values_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["action_id", "action_label", "cycle_error"])
        for aid, val in cycle_errors:
            writer.writerow([aid, decode_action_id(aid, motion["action_dim"]), val])

    cycle_summary_csv = cycle_dir / f"cycle_error_summary_{global_step:07d}.csv"
    with cycle_summary_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["action_id", "action_label", "count", "mean_cycle_error"])
        for aid, vals in sorted(cycle_per_action.items(), key=lambda kv: len(kv[1]), reverse=True):
            if not vals:
                continue
            writer.writerow([aid, decode_action_id(aid, motion["action_dim"]), len(vals), float(np.mean(vals))])


def _save_diagnostics_frames(
    frames: torch.Tensor,
    paths: Optional[List[List[str]]],
    actions: Optional[torch.Tensor],
    frames_dir: Path,
    global_step: int,
) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    csv_path = frames_dir / f"frames_{global_step:07d}.csv"
    max_save = frames.shape[0]
    entries: List[Tuple[str, int]] = []

    def _natural_path_key(path_str: str) -> Tuple:
        parts = re.split(r"(\d+)", path_str)
        key: List = []
        for part in parts:
            if not part:
                continue
            key.append(int(part) if part.isdigit() else part.lower())
        return tuple(key)

    for idx in range(max_save):
        src_path = paths[idx][0] if paths and idx < len(paths) and paths[idx] else ""
        entries.append((src_path, idx))
    entries.sort(key=lambda t: _natural_path_key(t[0]))
    new_sources_sorted = [src for src, _ in entries]

    # Try to reuse existing frames if the sources match exactly.
    reuse_image_lookup: Optional[Dict[str, str]] = None
    for existing_csv in sorted(frames_dir.glob("frames_*.csv")):
        try:
            with existing_csv.open("r", newline="") as handle:
                reader = csv.DictReader(handle)
                existing_records = list(reader)
        except (OSError, csv.Error):
            continue
        if not existing_records:
            continue
        existing_sources = [row.get("source_path", "") for row in existing_records]
        existing_sources_sorted = sorted(existing_sources, key=_natural_path_key)
        if len(existing_sources_sorted) != len(new_sources_sorted):
            continue
        if existing_sources_sorted == new_sources_sorted:
            reuse_image_lookup = {}
            for row in existing_records:
                src = row.get("source_path", "")
                img_rel = row.get("image_path", "")
                if src and img_rel:
                    reuse_image_lookup[src] = img_rel
            break

    records: List[Tuple[int, str, str, Optional[int], str]] = []
    if reuse_image_lookup:
        for out_idx, (src, orig_idx) in enumerate(entries):
            img_rel = reuse_image_lookup.get(src, "")
            if not img_rel:
                reuse_image_lookup = None
                records.clear()
                break
            action_id: Optional[int] = None
            action_label = ""
            if actions is not None and actions.ndim >= 2 and orig_idx < actions.shape[0]:
                action_vec = actions[orig_idx, 0].detach().cpu().numpy()
                action_id = int(compress_actions_to_ids(action_vec[None, ...])[0])
                action_label = decode_action_id(action_id, actions.shape[-1])
            records.append((out_idx, img_rel, src, action_id, action_label))

    if not records:
        step_dir = frames_dir / f"frames_{global_step:07d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        for out_idx, (src, orig_idx) in enumerate(entries):
            frame_img = tensor_to_uint8_image(frames[orig_idx, 0])
            out_path = step_dir / f"frame_{out_idx:04d}.png"
            Image.fromarray(frame_img).save(out_path)
            action_id: Optional[int] = None
            action_label = ""
            if actions is not None and actions.ndim >= 2 and orig_idx < actions.shape[0]:
                action_vec = actions[orig_idx, 0].detach().cpu().numpy()
                action_id = int(compress_actions_to_ids(action_vec[None, ...])[0])
                action_label = decode_action_id(action_id, actions.shape[-1])
            records.append(
                (
                    out_idx,
                    out_path.relative_to(step_dir.parent).as_posix(),
                    src,
                    action_id,
                    action_label,
                )
            )

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "image_path", "source_path", "action_id", "action_label"])
        writer.writerows(records)


def _write_alignment_debug_csv(
    frames: torch.Tensor,
    actions: torch.Tensor,
    paths: Optional[List[List[str]]],
    out_dir: Path,
    global_step: int,
) -> None:
    """Log per-frame checksums and action context to spot indexing issues."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bsz, seq_len = frames.shape[0], frames.shape[1]
    action_ids = compress_actions_to_ids(actions.cpu().numpy().reshape(-1, actions.shape[-1])).reshape(bsz, seq_len)
    csv_path = out_dir / f"alignment_debug_{global_step:07d}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "batch_index",
                "time_index",
                "frame_mean",
                "frame_std",
                "same_as_prev_frame",
                "action_to_this_id",
                "action_to_this_label",
                "action_from_this_id",
                "action_from_this_label",
                "frame_path",
            ]
        )
        for b in range(bsz):
            for t in range(seq_len):
                frame = frames[b, t]
                mean = float(frame.mean().item())
                std = float(frame.std().item())
                same_prev = False
                if t > 0:
                    same_prev = bool(torch.equal(frame, frames[b, t - 1]))
                action_to_this_id = action_ids[b, t - 1] if t > 0 else None
                action_from_this_id = action_ids[b, t] if t < seq_len - 1 else None
                frame_path = paths[b][t] if paths and b < len(paths) and t < len(paths[b]) else ""
                writer.writerow(
                    [
                        b,
                        t,
                        mean,
                        std,
                        int(same_prev),
                        "" if action_to_this_id is None else int(action_to_this_id),
                        "" if action_to_this_id is None else decode_action_id(int(action_to_this_id), actions.shape[-1]),
                        "" if action_from_this_id is None else int(action_from_this_id),
                        "" if action_from_this_id is None else decode_action_id(int(action_from_this_id), actions.shape[-1]),
                        frame_path,
                    ]
                )


def _save_motion_diagnostics_outputs(
    motion: Dict[str, Any],
    cfg: DiagnosticsConfig,
    delta_dir: Path,
    alignment_dir: Path,
    cycle_dir: Path,
    global_step: int,
    embedding_label: str,
    inverse_map: Dict[int, int],
) -> None:
    delta_path = delta_dir / f"delta_{embedding_label}_pca_{global_step:07d}.png"
    _save_delta_pca_plot(
        delta_path,
        motion["variance_ratio"],
        motion["delta_proj"],
        motion["proj_flat"],
        motion["action_ids"],
        motion["action_dim"],
        embedding_label,
    )
    _save_variance_spectrum_plot(
        delta_dir / f"delta_{embedding_label}_variance_spectrum_{global_step:07d}.png",
        motion["variance_ratio"],
    )
    _write_variance_report(delta_dir, global_step, motion["variance_ratio"], embedding_label)
    alignment_stats_full = _compute_action_alignment_stats(
        motion["delta_proj"],
        motion["action_ids"],
        cfg,
        max_actions=None,
        include_mean_vectors=True,
        include_norm_stats=True,
    )
    alignment_debug = _build_action_alignment_debug(alignment_stats_full, motion["delta_proj"], motion["action_ids"])
    save_action_alignment_detail_plot(
        alignment_dir / f"action_alignment_detail_{global_step:07d}.png",
        alignment_debug,
        cfg.cosine_high_threshold,
        motion["action_dim"],
    )
    _write_action_alignment_report(alignment_dir, global_step, alignment_stats_full, motion["action_dim"], inverse_map)
    _write_action_alignment_strength(alignment_dir, global_step, alignment_stats_full, motion["action_dim"])
    _write_action_alignment_crosscheck(
        alignment_dir,
        global_step,
        alignment_stats_full,
        motion["action_dim"],
        motion["action_ids"],
        motion["delta_proj"],
    )
    cycle_path = cycle_dir / f"cycle_error_{global_step:07d}.png"
    errors, per_action = compute_cycle_errors(
        motion["proj_sequences"],
        motion["actions_seq"],
        inverse_map,
        include_synthetic=cfg.synthesize_cycle_samples,
    )
    save_cycle_error_plot(cycle_path, [e[1] for e in errors], per_action, motion["action_dim"])
    _write_diagnostics_csvs(
        delta_dir,
        alignment_dir,
        cycle_dir,
        global_step,
        motion,
        cfg,
        alignment_stats_full,
        alignment_debug,
        errors,
        per_action,
        embedding_label,
    )


def save_diagnostics_outputs(
    embeddings: torch.Tensor,
    frames_cpu: torch.Tensor,
    actions_cpu: torch.Tensor,
    paths: Optional[List[List[str]]],
    cfg: DiagnosticsConfig,
    delta_dir: Path,
    alignment_dir: Path,
    cycle_dir: Path,
    frames_dir: Path,
    global_step: int,
    s_embeddings: Optional[torch.Tensor] = None,
    delta_s_dir: Optional[Path] = None,
    alignment_s_dir: Optional[Path] = None,
    cycle_s_dir: Optional[Path] = None,
) -> None:
    if frames_cpu.shape[0] == 0 or frames_cpu.shape[1] < 2:
        return
    motion = _compute_motion_subspace(embeddings, actions_cpu, cfg.top_k_components, paths)
    if motion is None:
        return
    inverse_map = _build_inverse_action_map(
        motion["action_dim"],
        np.unique(compress_actions_to_ids(motion["actions_seq"].reshape(-1, motion["actions_seq"].shape[-1]))),
    )
    _save_motion_diagnostics_outputs(
        motion,
        cfg,
        delta_dir,
        alignment_dir,
        cycle_dir,
        global_step,
        "z",
        inverse_map,
    )

    if (
        s_embeddings is not None
        and delta_s_dir is not None
        and alignment_s_dir is not None
        and cycle_s_dir is not None
    ):
        motion_s = _compute_motion_subspace(s_embeddings, actions_cpu, cfg.top_k_components, paths)
        if motion_s is not None:
            _save_motion_diagnostics_outputs(
                motion_s,
                cfg,
                delta_s_dir,
                alignment_s_dir,
                cycle_s_dir,
                global_step,
                "s",
                inverse_map,
            )

    _write_alignment_debug_csv(frames_cpu, actions_cpu, paths, frames_dir, global_step)
    _save_diagnostics_frames(frames_cpu, paths, actions_cpu, frames_dir, global_step)
