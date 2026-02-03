#!/usr/bin/env python3
"""State embedding diagnostics outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from jepa_world_model.vis_self_distance import write_self_distance_outputs_from_embeddings
from jepa_world_model.plots.plot_layout import DEFAULT_DPI, apply_square_axes, figsize_for_grid
from jepa_world_model.rollout import (
    rollout_self_fed,
    rollout_teacher_forced_z,
    rollout_teacher_forced,
)
from jepa_world_model.vis_odometry import (
    _plot_latent_prediction_comparison,
    _plot_odometry_embeddings,
)
from jepa_world_model.pose_rollout import rollout_pose


def write_state_embedding_histogram(
    plot_dir: Path,
    p_norm_np,
    global_step: int,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_hist, ax = plt.subplots(figsize=figsize_for_grid(1, 1), constrained_layout=True)
    if p_norm_np.size:
        finite = np.asarray(p_norm_np)[np.isfinite(p_norm_np)]
        if finite.size:
            min_val = float(finite.min())
            max_val = float(finite.max())
            if finite.size < 2 or np.isclose(min_val, max_val):
                ax.hist(finite, bins=1, color="tab:purple", alpha=0.8)
            else:
                ax.hist(finite, bins=40, color="tab:purple", alpha=0.8)
        else:
            ax.text(0.5, 0.5, "No finite state norms.", ha="center", va="center")
            ax.axis("off")
        ax.set_xlabel("||p||")
        ax.set_ylabel("count")
    else:
        ax.text(0.5, 0.5, "No state embeddings.", ha="center", va="center")
        ax.axis("off")
    ax.set_title("Pose embedding norm distribution")
    apply_square_axes(ax)
    fig_hist.savefig(
        plot_dir / f"state_embedding_hist_{global_step:07d}.png",
        dpi=DEFAULT_DPI,
    )
    plt.close(fig_hist)


def write_state_embedding_outputs(
    model,
    traj_inputs,
    device: torch.device,
    csv_dir: Path,
    plot_dir: Path,
    hist_plot_dir: Optional[Path],
    odometry_plot_dir: Optional[Path],
    global_step: int,
    *,
    use_z2h_init: bool = False,
    force_h_zero: bool = False,
    hist_frames_cpu: Optional[torch.Tensor] = None,
    hist_actions_cpu: Optional[torch.Tensor] = None,
    write_self_distance: bool = True,
    write_state_hist: bool = True,
    write_odometry_current_z: bool = True,
    write_odometry_current_p: bool = True,
    write_odometry_current_h: bool = True,
    write_odometry_z_vs_z_hat: bool = True,
    write_odometry_p_vs_p_hat: bool = True,
    write_odometry_h_vs_h_hat: bool = True,
) -> None:
    if traj_inputs.frames.shape[1] < 2:
        return
    frames = traj_inputs.frames.to(device)
    with torch.no_grad():
        embeddings = model.encode_sequence(frames)["embeddings"]
        actions = torch.from_numpy(traj_inputs.actions).to(device).unsqueeze(0)
        _, _, h_states = rollout_teacher_forced(
            model,
            embeddings,
            actions,
            use_z2h_init=use_z2h_init,
            force_h_zero=force_h_zero,
        )
        if embeddings is None:
            raise AssertionError("State embedding outputs require embeddings.")
        _, p, _ = rollout_pose(model, h_states, actions, z_embeddings=embeddings)
        p = p[0]

    warmup_frames = max(model.cfg.warmup_frames_h, 0)
    warmup = max(min(warmup_frames, p.shape[0] - 1), 0)
    p = p[warmup:]
    if p.shape[0] < 1:
        return

    if write_self_distance:
        write_self_distance_outputs_from_embeddings(
            p,
            traj_inputs,
            csv_dir,
            plot_dir,
            global_step,
            embedding_label="p",
            title_prefix="Self-distance (P)",
            file_prefix="self_distance_p",
            cosine_prefix="self_distance_cosine",
            start_index=warmup,
        )

    if odometry_plot_dir is not None:
        z = embeddings[0].detach().cpu().numpy()
        z_seq = z[warmup:]
        if write_odometry_current_z and z_seq.shape[0] >= 2:
            delta_z = z_seq[1:] - z_seq[:-1]
            current_z = np.cumsum(delta_z, axis=0)
            _plot_odometry_embeddings(
                odometry_plot_dir / f"odometry_z_{global_step:07d}.png",
                current_z,
                "Cumulative sum of Δz",
            )

        p_np = p.detach().cpu().numpy()
        if write_odometry_current_p and p_np.shape[0] >= 2:
            delta_p = p_np[1:] - p_np[:-1]
            current_p = np.cumsum(delta_p, axis=0)
            _plot_odometry_embeddings(
                odometry_plot_dir / f"odometry_p_{global_step:07d}.png",
                current_p,
                "Cumulative sum of Δp",
            )

        h_np = h_states[0].detach().cpu().numpy()
        h_seq = h_np[warmup:] if warmup > 0 else h_np
        if write_odometry_current_h and h_seq.shape[0] >= 2:
            delta_h = h_seq[1:] - h_seq[:-1]
            current_h = np.cumsum(delta_h, axis=0)
            _plot_odometry_embeddings(
                odometry_plot_dir / f"odometry_h_{global_step:07d}.png",
                current_h,
                "Cumulative sum of Δh",
            )

        z_hat = rollout_teacher_forced_z(
            model,
            embeddings,
            actions,
            use_z2h_init=use_z2h_init,
        )[0].detach().cpu().numpy()
        if z_hat.shape[0] >= 1:
            z_next = z[1 + warmup :]
            z_hat_trim = z_hat[warmup:]
            min_len = min(z_next.shape[0], z_hat_trim.shape[0])
            if write_odometry_z_vs_z_hat and min_len >= 2:
                _plot_latent_prediction_comparison(
                    odometry_plot_dir / f"z_vs_z_hat_{global_step:07d}.png",
                    z_next[:min_len],
                    z_hat_trim[:min_len],
                    "z",
                )

        h_hat_open = rollout_self_fed(
            model,
            embeddings,
            actions,
            use_z2h_init=use_z2h_init,
        )
        if h_hat_open.shape[1] > 0:
            if embeddings is None:
                raise AssertionError("State embedding outputs require embeddings.")
            z_hat_len = h_hat_open.shape[1]
            if z_hat_len <= 0:
                raise AssertionError("State embedding outputs require non-empty h_hat_open.")
            z_hat = embeddings[:, :z_hat_len]
            _, p_hat, _ = rollout_pose(model, h_hat_open, actions[:, :z_hat_len], z_embeddings=z_hat)
            p_hat = p_hat[0].detach().cpu().numpy()
            p_next = p_np[1:]
            p_hat_trim = p_hat[warmup:]
            p_next_trim = p_next[warmup:] if warmup < p_next.shape[0] else p_next[:0]
            min_len = min(p_next_trim.shape[0], p_hat_trim.shape[0])
            if write_odometry_p_vs_p_hat and min_len >= 2:
                _plot_latent_prediction_comparison(
                    odometry_plot_dir / f"p_vs_p_hat_{global_step:07d}.png",
                    p_next_trim[:min_len],
                    p_hat_trim[:min_len],
                    "p",
                )
            h_hat = h_hat_open[0].detach().cpu().numpy()
            h_next = h_seq[1:]
            h_hat_trim = h_hat[warmup:] if warmup > 0 else h_hat
            min_len = min(h_next.shape[0], h_hat_trim.shape[0])
            if write_odometry_h_vs_h_hat and min_len >= 2:
                _plot_latent_prediction_comparison(
                    odometry_plot_dir / f"h_vs_h_hat_{global_step:07d}.png",
                    h_next[:min_len],
                    h_hat_trim[:min_len],
                    "h",
                )

    if not write_state_hist:
        return
    hist_frames = hist_frames_cpu if hist_frames_cpu is not None else traj_inputs.frames
    if hist_frames.ndim != 5 or hist_frames.shape[1] < 1:
        return
    if hist_actions_cpu is None:
        return
    hist_frames = hist_frames.to(device)
    with torch.no_grad():
        hist_embeddings = model.encode_sequence(hist_frames)["embeddings"]
        hist_actions = hist_actions_cpu.to(device)
        _, _, hist_h_states = rollout_teacher_forced(
            model,
            hist_embeddings,
            hist_actions,
            use_z2h_init=use_z2h_init,
        )
        if hist_embeddings is None:
            raise AssertionError("State embedding outputs require hist_embeddings.")
        _, hist_p, _ = rollout_pose(model, hist_h_states, hist_actions, z_embeddings=hist_embeddings)
        hist_p = hist_p[0]
    hist_warmup = max(min(warmup_frames, hist_p.shape[1] - 1), 0)
    hist_p = hist_p[:, hist_warmup:]
    if hist_p.shape[1] < 1:
        return
    hist_flat = hist_p.reshape(-1, hist_p.shape[-1])
    hist_norm = torch.norm(hist_flat, dim=-1).detach().cpu().numpy()
    hist_dir = hist_plot_dir if hist_plot_dir is not None else plot_dir
    write_state_embedding_histogram(hist_dir, hist_norm, global_step)
