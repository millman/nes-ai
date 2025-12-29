#!/usr/bin/env python3
"""State embedding diagnostics outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from jepa_world_model.vis_self_distance import write_self_distance_outputs_from_embeddings


def _pair_actions(actions: torch.Tensor) -> torch.Tensor:
    """Concatenate current and prior actions for predictor conditioning."""
    if actions.ndim != 3:
        raise ValueError("Actions must have shape [B, T, action_dim].")
    if actions.shape[1] == 0:
        return actions.new_zeros((actions.shape[0], 0, actions.shape[2] * 2))
    zeros = actions.new_zeros((actions.shape[0], 1, actions.shape[2]))
    prev = torch.cat([zeros, actions[:, :-1]], dim=1)
    return torch.cat([actions, prev], dim=-1)


def _rollout_hidden_states(
    model,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Roll hidden states forward with the predictor to build h_t for each step."""
    b, t, _ = embeddings.shape
    h_states = embeddings.new_zeros((b, t, model.state_dim))
    if t < 2:
        return h_states
    paired_actions = _pair_actions(actions)
    h_t = embeddings.new_zeros((b, model.state_dim))
    for step in range(t - 1):
        z_t = embeddings[:, step]
        act_t = paired_actions[:, step]
        _, _, h_next = model.predictor(z_t, h_t, act_t)
        h_states[:, step + 1] = h_next
        h_t = h_next
    return h_states


def write_state_embedding_histogram(
    plot_dir: Path,
    s_norm_np,
    global_step: int,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_hist, ax = plt.subplots(figsize=(7, 4))
    if s_norm_np.size:
        finite = np.asarray(s_norm_np)[np.isfinite(s_norm_np)]
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
        ax.set_xlabel("||s||")
        ax.set_ylabel("count")
    else:
        ax.text(0.5, 0.5, "No state embeddings.", ha="center", va="center")
        ax.axis("off")
    ax.set_title("State embedding norm distribution")
    fig_hist.tight_layout()
    fig_hist.savefig(plot_dir / f"state_embedding_hist_{global_step:07d}.png", dpi=200, bbox_inches="tight")
    plt.close(fig_hist)


def write_state_embedding_outputs(
    model,
    traj_inputs,
    device: torch.device,
    csv_dir: Path,
    plot_dir: Path,
    hist_plot_dir: Optional[Path],
    global_step: int,
    hist_frames_cpu: Optional[torch.Tensor] = None,
    hist_actions_cpu: Optional[torch.Tensor] = None,
) -> None:
    if traj_inputs.frames.shape[1] < 2:
        return
    frames = traj_inputs.frames.to(device)
    with torch.no_grad():
        embeddings = model.encode_sequence(frames)["embeddings"]
        actions = torch.from_numpy(traj_inputs.actions).to(device).unsqueeze(0)
        h_states = _rollout_hidden_states(model, embeddings, actions)
        s = model.state_head(h_states)[0]

    warmup_frames = max(getattr(model.cfg, "state_warmup_frames", 0), 0)
    warmup = max(min(warmup_frames, s.shape[0] - 1), 0)
    s = s[warmup:]
    if s.shape[0] < 1:
        return

    write_self_distance_outputs_from_embeddings(
        s,
        traj_inputs,
        csv_dir,
        plot_dir,
        global_step,
        embedding_label="s",
        title_prefix="Self-distance (S)",
        file_prefix="self_distance_s",
        cosine_prefix="self_distance_cosine",
        start_index=warmup,
    )

    hist_frames = hist_frames_cpu if hist_frames_cpu is not None else traj_inputs.frames
    if hist_frames.ndim != 5 or hist_frames.shape[1] < 1:
        return
    if hist_actions_cpu is None:
        return
    hist_frames = hist_frames.to(device)
    with torch.no_grad():
        hist_embeddings = model.encode_sequence(hist_frames)["embeddings"]
        hist_actions = hist_actions_cpu.to(device)
        hist_h_states = _rollout_hidden_states(model, hist_embeddings, hist_actions)
        hist_s = model.state_head(hist_h_states)
    hist_warmup = max(min(warmup_frames, hist_s.shape[1] - 1), 0)
    hist_s = hist_s[:, hist_warmup:]
    if hist_s.shape[1] < 1:
        return
    hist_flat = hist_s.reshape(-1, hist_s.shape[-1])
    hist_norm = torch.norm(hist_flat, dim=-1).detach().cpu().numpy()
    hist_dir = hist_plot_dir if hist_plot_dir is not None else plot_dir
    write_state_embedding_histogram(hist_dir, hist_norm, global_step)
