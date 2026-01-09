#!/usr/bin/env python3
"""Two-step composability diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from jepa_world_model.actions import compress_actions_to_ids, decode_action_id
from jepa_world_model.plots.plot_layout import DEFAULT_DPI, figsize_for_grid


@dataclass
class ComposabilitySeries:
    timesteps: np.ndarray
    actual_means: np.ndarray
    additive_means: np.ndarray
    fixed_actual: np.ndarray
    fixed_additive: np.ndarray
    pair_actual: Dict[str, np.ndarray]
    pair_additive: Dict[str, np.ndarray]
    pair_order: List[str]


def _pair_label(action_id: int, action_dim: int) -> str:
    return decode_action_id(int(action_id), action_dim)


def _build_pair_maps(
    action_ids: np.ndarray,
    action_ids_next: np.ndarray,
    actual_errors: np.ndarray,
    additive_errors: np.ndarray,
    action_dim: int,
    min_action_count: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    pair_actual: Dict[str, List[float]] = {}
    pair_add: Dict[str, List[float]] = {}
    flat_pairs = np.stack([action_ids.reshape(-1), action_ids_next.reshape(-1)], axis=1)
    flat_actual = actual_errors.reshape(-1)
    flat_add = additive_errors.reshape(-1)
    counts: Dict[Tuple[int, int], int] = {}
    for a0, a1 in flat_pairs:
        key = (int(a0), int(a1))
        counts[key] = counts.get(key, 0) + 1
    for idx, (a0, a1) in enumerate(flat_pairs):
        key = (int(a0), int(a1))
        if counts.get(key, 0) < max(min_action_count, 1):
            continue
        label = f"{_pair_label(a0, action_dim)}->{_pair_label(a1, action_dim)}"
        pair_actual.setdefault(label, []).append(float(flat_actual[idx]))
        pair_add.setdefault(label, []).append(float(flat_add[idx]))
    ordered_pairs = sorted(
        (key for key, count in counts.items() if count >= max(min_action_count, 1)),
        key=lambda pair: (pair[0], pair[1]),
    )
    pair_order = [f"{_pair_label(a0, action_dim)}->{_pair_label(a1, action_dim)}" for a0, a1 in ordered_pairs]
    pair_actual_np = {k: np.asarray(v, dtype=np.float32) for k, v in pair_actual.items()}
    pair_add_np = {k: np.asarray(v, dtype=np.float32) for k, v in pair_add.items()}
    return pair_actual_np, pair_add_np, pair_order


def _mean_by_timestep(errors: np.ndarray) -> np.ndarray:
    if errors.size == 0:
        return np.zeros(0, dtype=np.float32)
    return errors.mean(axis=0).astype(np.float32)


def _plot_strip(
    ax: plt.Axes,
    pair_values: Dict[str, np.ndarray],
    title: str,
    pair_order: List[str] | None = None,
) -> None:
    ax.set_title(title)
    if not pair_values:
        ax.text(0.5, 0.5, "No pair samples.", ha="center", va="center")
        ax.set_xticks([])
        return
    if pair_order:
        ordered = [(label, pair_values[label]) for label in pair_order if label in pair_values]
    else:
        ordered = sorted(pair_values.items(), key=lambda kv: float(np.mean(kv[1])) if kv[1].size else 0.0)
    labels = [k for k, _ in ordered]
    rng = np.random.default_rng(0)
    for idx, (_, values) in enumerate(ordered):
        if values.size == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=values.size)
        ax.scatter(np.full(values.shape, idx) + jitter, values, s=10, alpha=0.6)
        mean_value = float(np.mean(values))
        ax.plot([idx - 0.25, idx + 0.25], [mean_value, mean_value], color="red", linewidth=1.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("error")


def save_composability_plot(
    out_path,
    series: ComposabilitySeries,
    embedding_label: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=figsize_for_grid(2, 2), constrained_layout=True)
    axes = np.asarray(axes)
    if series.fixed_actual.size:
        axes[0, 0].hist(series.fixed_actual, bins=40, color="tab:blue", alpha=0.75)
        axes[0, 0].set_title(f"{embedding_label}: actual rollout (hist)")
        axes[0, 1].set_xlabel("L2 error")
        axes[0, 0].set_ylabel("count")
    else:
        axes[0, 0].text(0.5, 0.5, "No composability data.", ha="center", va="center")
        axes[0, 0].set_xticks([])
    if series.fixed_additive.size:
        axes[0, 1].hist(series.fixed_additive, bins=40, color="tab:orange", alpha=0.75)
        axes[0, 1].set_title(f"{embedding_label}: additive deltas (hist)")
        axes[0, 1].set_xlabel("L2 error")
        axes[0, 1].set_ylabel("count")
    else:
        axes[0, 1].text(0.5, 0.5, "No composability data.", ha="center", va="center")
        axes[0, 1].set_xticks([])
    _plot_strip(
        axes[1, 0],
        series.pair_actual,
        "action-pair strip (actual rollout)",
        pair_order=series.pair_order,
    )
    _plot_strip(
        axes[1, 1],
        series.pair_additive,
        "action-pair strip (additive deltas)",
        pair_order=series.pair_order,
    )
    fig.suptitle(f"Two-step composability ({embedding_label})", fontsize=12)
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def compute_composability_series(
    model,
    z_embeddings: torch.Tensor,
    h_states: torch.Tensor,
    actions: torch.Tensor,
    warmup: int,
    min_action_count: int,
) -> Dict[str, ComposabilitySeries]:
    def _empty_series() -> ComposabilitySeries:
        return ComposabilitySeries(
            timesteps=np.zeros(0, dtype=np.float32),
            actual_means=np.zeros(0, dtype=np.float32),
            additive_means=np.zeros(0, dtype=np.float32),
            fixed_actual=np.zeros(0, dtype=np.float32),
            fixed_additive=np.zeros(0, dtype=np.float32),
            pair_actual={},
            pair_additive={},
            pair_order=[],
        )

    if z_embeddings.shape[1] < warmup + 3:
        empty = _empty_series()
        return {"z": empty, "h": empty, "p": empty, "f": empty, "s": empty}
    z = z_embeddings[:, warmup:]
    h = h_states[:, warmup:]
    a = actions[:, warmup:]
    b, t, _ = z.shape
    steps = t - 2
    if steps <= 0:
        empty = _empty_series()
        return {"z": empty, "h": empty, "p": empty, "f": empty, "s": empty}

    z_t = z[:, :steps]
    z_tp1 = z[:, 1 : steps + 1]
    z_tp2 = z[:, 2 : steps + 2]
    h_t = h[:, :steps]
    h_tp2 = h[:, 2 : steps + 2]
    p = model.h2p(h)
    f = model.h2f(h)
    p_tp2 = p[:, 2 : steps + 2]
    f_tp2 = f[:, 2 : steps + 2]
    a_t = a[:, :steps]
    a_tp1 = a[:, 1 : steps + 1]

    flat_z_t = z_t.reshape(-1, z_t.shape[-1])
    flat_h_t = h_t.reshape(-1, h_t.shape[-1])
    flat_a_t = a_t.reshape(-1, a_t.shape[-1])
    h1 = model.predictor(flat_z_t, flat_h_t, flat_a_t)
    pred1 = model.h_to_z(h1).view(b, steps, -1)
    h1 = h1.view(b, steps, -1)

    flat_z_tp1 = z_tp1.reshape(-1, z_tp1.shape[-1])
    flat_h1 = h1.reshape(-1, h1.shape[-1])
    flat_a_tp1 = a_tp1.reshape(-1, a_tp1.shape[-1])
    h2 = model.predictor(flat_z_tp1, flat_h1, flat_a_tp1)
    pred2 = model.h_to_z(h2).view(b, steps, -1)
    h2 = h2.view(b, steps, -1)
    h2_to_z = model.h_to_z(h2)
    p2_pred = model.h2p(h2)
    f2_pred = model.h2f(h2)

    actual_z = torch.norm(pred2 - z_tp2, dim=-1)
    actual_h = torch.norm(h2_to_z - z_tp2, dim=-1)
    actual_p = torch.norm(p2_pred - p_tp2, dim=-1)
    actual_f = torch.norm(f2_pred - f_tp2, dim=-1)

    delta_z1 = model.action_delta_projector(flat_a_t).view(b, steps, -1)
    delta_z2 = model.action_delta_projector(flat_a_tp1).view(b, steps, -1)
    z_add = z_t + delta_z1 + delta_z2
    add_z = torch.norm(z_add - z_tp2, dim=-1)

    dh1 = model.h_action_delta_projector(flat_h_t, flat_a_t).view(b, steps, -1)
    flat_h_t2 = (h_t + dh1).reshape(-1, h_t.shape[-1])
    dh2 = model.h_action_delta_projector(flat_h_t2, flat_a_tp1).view(b, steps, -1)
    h_add = h_t + dh1 + dh2
    add_h = torch.norm(h_add - h_tp2, dim=-1)
    p_add = model.h2p(h_add)
    add_p = torch.norm(p_add - p_tp2, dim=-1)
    f_add = model.h2f(h_add)
    add_f = torch.norm(f_add - f_tp2, dim=-1)

    action_ids = compress_actions_to_ids(a_t.reshape(-1, a_t.shape[-1])).view(b, steps).cpu().numpy()
    action_ids_next = compress_actions_to_ids(a_tp1.reshape(-1, a_tp1.shape[-1])).view(b, steps).cpu().numpy()
    action_dim = a.shape[-1]
    timesteps = np.arange(warmup, warmup + steps, dtype=np.int32)

    def select_non_overlapping_triplets(errors: torch.Tensor) -> np.ndarray:
        if errors.numel() == 0:
            return np.zeros(0, dtype=np.float32)
        steps_local = errors.shape[1]
        start_idx = torch.arange(0, steps_local, 3, device=errors.device)
        if start_idx.numel() == 0:
            return np.zeros(0, dtype=np.float32)
        sampled = errors[:, start_idx]
        return sampled.detach().cpu().numpy().reshape(-1).astype(np.float32)

    def build_series(actual: torch.Tensor, additive: torch.Tensor) -> ComposabilitySeries:
        actual_np = actual.detach().cpu().numpy().astype(np.float32)
        additive_np = additive.detach().cpu().numpy().astype(np.float32)
        pair_actual, pair_additive, pair_order = _build_pair_maps(
            action_ids,
            action_ids_next,
            actual_np,
            additive_np,
            action_dim,
            min_action_count,
        )
        return ComposabilitySeries(
            timesteps=timesteps,
            actual_means=_mean_by_timestep(actual_np),
            additive_means=_mean_by_timestep(additive_np),
            fixed_actual=select_non_overlapping_triplets(actual),
            fixed_additive=select_non_overlapping_triplets(additive),
            pair_actual=pair_actual,
            pair_additive=pair_additive,
            pair_order=pair_order,
        )

    pose_series = build_series(actual_p, add_p)
    desc_series = build_series(actual_f, add_f)
    return {
        "z": build_series(actual_z, add_z),
        "h": build_series(actual_h, add_h),
        "p": pose_series,
        "f": desc_series,
        "s": pose_series,
    }
