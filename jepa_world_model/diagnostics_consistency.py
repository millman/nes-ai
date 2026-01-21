from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from jepa_world_model.diagnostics_prepare import DiagnosticsBatchState
from jepa_world_model.diagnostics_utils import write_step_csv
from jepa_world_model.plots.plot_diagnostics_extra import save_monotonicity_plot, save_path_independence_plot, save_z_consistency_plot


def _shift_frame(frame: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    if dx == 0 and dy == 0:
        return frame
    c, h, w = frame.shape
    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)
    pad_top = max(dy, 0)
    pad_bottom = max(-dy, 0)
    padded = F.pad(frame.unsqueeze(0), (pad_left, pad_right, pad_top, pad_bottom))
    return padded[:, :, pad_bottom : pad_bottom + h, pad_right : pad_right + w].squeeze(0)


def compute_z_monotonicity_distances(
    *,
    model,
    diag_frames_device: torch.Tensor,
    max_shift: int,
    monotonicity_samples: int,
    diagnostics_generator: torch.Generator,
) -> Tuple[List[int], List[float]]:
    if monotonicity_samples <= 0:
        raise AssertionError("z monotonicity requires at least one sample.")
    frame_count = diag_frames_device.shape[0] * diag_frames_device.shape[1]
    if frame_count <= 0:
        raise AssertionError("z monotonicity requires at least one frame.")
    perm = torch.randperm(frame_count, generator=diagnostics_generator)[:monotonicity_samples]
    frames_sel = []
    for flat_idx in perm.tolist():
        b = flat_idx // diag_frames_device.shape[1]
        t0 = flat_idx % diag_frames_device.shape[1]
        frames_sel.append(diag_frames_device[b, t0])
    frames_batch = torch.stack(frames_sel, dim=0)
    shifts = list(range(0, max_shift + 1))
    distances: List[float] = []
    with torch.no_grad():
        z_refs = model.encoder(frames_batch)
        for shift in shifts:
            shifted_batch = torch.stack(
                [_shift_frame(frame, shift, 0) for frame in frames_batch],
                dim=0,
            )
            z_shift = model.encoder(shifted_batch)
            distances.append(float((z_shift - z_refs).norm(dim=-1).mean().item()))
    return shifts, distances


def compute_path_independence_diffs(
    *,
    model,
    diag_embeddings: torch.Tensor,
    diag_h_states: torch.Tensor,
    diag_p_embeddings: torch.Tensor,
    action_a: torch.Tensor,
    action_b: torch.Tensor,
    action_b_first: torch.Tensor,
    action_b_second: torch.Tensor,
    start_frame: int,
    max_starts: int,
    path_independence_steps: int,
    diagnostics_generator: torch.Generator,
) -> Tuple[List[float], List[float]]:
    if path_independence_steps <= 0:
        raise AssertionError("path_independence_steps must be positive.")
    if max_starts <= 0:
        raise AssertionError("path_independence requires at least one start.")
    if diag_embeddings.shape[1] <= start_frame:
        raise AssertionError("path independence start_frame must be within the sequence.")
    perm = torch.randperm(diag_embeddings.shape[0], generator=diagnostics_generator)[:max_starts]
    z_diffs: List[float] = []
    p_diffs: List[float] = []
    for b_idx in perm:
        b = int(b_idx.item())
        z_a = diag_embeddings[b, start_frame]
        h_a = diag_h_states[b, start_frame]
        z_b = diag_embeddings[b, start_frame]
        h_b = diag_h_states[b, start_frame]
        p_a = diag_p_embeddings[b, start_frame]
        p_b = diag_p_embeddings[b, start_frame]
        h_a_in = h_a.detach() if model.cfg.pose_delta_detach_h else h_a
        h_b_in = h_b.detach() if model.cfg.pose_delta_detach_h else h_b
        delta_a = model.p_action_delta_projector(
            p_a.unsqueeze(0),
            h_a_in.unsqueeze(0),
            action_a.unsqueeze(0),
        ).squeeze(0)
        delta_b = model.p_action_delta_projector(
            p_b.unsqueeze(0),
            h_b_in.unsqueeze(0),
            action_b_first.unsqueeze(0),
        ).squeeze(0)
        p_a = p_a + delta_a
        p_b = p_b + delta_b
        h_a_next = model.predictor(
            z_a.unsqueeze(0),
            h_a.unsqueeze(0),
            action_a.unsqueeze(0),
        )
        h_b_next = model.predictor(
            z_b.unsqueeze(0),
            h_b.unsqueeze(0),
            action_b_first.unsqueeze(0),
        )
        z_a = model.h_to_z(h_a_next).squeeze(0)
        h_a = h_a_next.squeeze(0)
        z_b = model.h_to_z(h_b_next).squeeze(0)
        h_b = h_b_next.squeeze(0)
        for _ in range(path_independence_steps):
            h_a_in = h_a.detach() if model.cfg.pose_delta_detach_h else h_a
            h_b_in = h_b.detach() if model.cfg.pose_delta_detach_h else h_b
            delta_a = model.p_action_delta_projector(
                p_a.unsqueeze(0),
                h_a_in.unsqueeze(0),
                action_b.unsqueeze(0),
            ).squeeze(0)
            delta_b = model.p_action_delta_projector(
                p_b.unsqueeze(0),
                h_b_in.unsqueeze(0),
                action_b_second.unsqueeze(0),
            ).squeeze(0)
            p_a = p_a + delta_a
            p_b = p_b + delta_b
            h_a_next = model.predictor(
                z_a.unsqueeze(0),
                h_a.unsqueeze(0),
                action_b.unsqueeze(0),
            )
            h_b_next = model.predictor(
                z_b.unsqueeze(0),
                h_b.unsqueeze(0),
                action_b_second.unsqueeze(0),
            )
            z_a = model.h_to_z(h_a_next).squeeze(0)
            h_a = h_a_next.squeeze(0)
            z_b = model.h_to_z(h_b_next).squeeze(0)
            h_b = h_b_next.squeeze(0)
        z_diffs.append(float((z_a - z_b).norm().item()))
        p_diffs.append(float((p_a - p_b).norm().item()))
    return z_diffs, p_diffs


def run_z_consistency(
    *,
    diagnostics_cfg,
    model,
    state: DiagnosticsBatchState,
    global_step: int,
    diagnostics_generator: torch.Generator,
    diagnostics_z_consistency_dir: Path,
) -> None:
    z_consistency_samples = min(
        diagnostics_cfg.z_consistency_samples,
        state.frames.shape[0] * state.frames.shape[1],
    )
    if z_consistency_samples <= 0:
        raise AssertionError("Z-consistency diagnostics require at least one sample.")
    frame_count = state.frames.shape[0] * state.frames.shape[1]
    perm = torch.randperm(frame_count, generator=diagnostics_generator)[:z_consistency_samples]
    distances: List[float] = []
    cosines: List[float] = []
    for flat_idx in perm.tolist():
        b = flat_idx // state.frames.shape[1]
        t0 = flat_idx % state.frames.shape[1]
        frame = state.frames_device[b, t0]
        repeats = diagnostics_cfg.z_consistency_repeats
        noise = torch.randn(
            (repeats, *frame.shape),
            device=frame.device,
        ) * diagnostics_cfg.z_consistency_noise_std
        noisy = (frame.unsqueeze(0) + noise).clamp(0, 1)
        z_samples = model.encoder(noisy)
        z_mean = z_samples.mean(dim=0, keepdim=True)
        dist = (z_samples - z_mean).norm(dim=-1)
        cos = F.cosine_similarity(z_samples, z_mean, dim=-1)
        distances.extend(dist.detach().cpu().numpy().tolist())
        cosines.extend(cos.detach().cpu().numpy().tolist())
    if distances and cosines:
        save_z_consistency_plot(
            diagnostics_z_consistency_dir / f"z_consistency_{global_step:07d}.png",
            distances,
            cosines,
        )
        write_step_csv(
            diagnostics_z_consistency_dir,
            f"z_consistency_{global_step:07d}.csv",
            ["idx", "distance", "cosine"],
            [(idx, d, c) for idx, (d, c) in enumerate(zip(distances, cosines))],
        )


def run_z_monotonicity(
    *,
    diagnostics_cfg,
    model,
    state: DiagnosticsBatchState,
    global_step: int,
    diagnostics_generator: torch.Generator,
    diagnostics_z_monotonicity_dir: Path,
) -> None:
    max_shift = max(1, diagnostics_cfg.z_monotonicity_max_shift)
    monotonicity_samples = min(
        diagnostics_cfg.z_monotonicity_samples,
        state.frames.shape[0] * state.frames.shape[1],
    )
    if monotonicity_samples <= 0:
        raise AssertionError("Z-monotonicity diagnostics require at least one sample.")
    shifts, distances = compute_z_monotonicity_distances(
        model=model,
        diag_frames_device=state.frames_device,
        max_shift=max_shift,
        monotonicity_samples=monotonicity_samples,
        diagnostics_generator=diagnostics_generator,
    )
    save_monotonicity_plot(
        diagnostics_z_monotonicity_dir / f"z_monotonicity_{global_step:07d}.png",
        shifts,
        distances,
    )
    write_step_csv(
        diagnostics_z_monotonicity_dir,
        f"z_monotonicity_{global_step:07d}.csv",
        ["shift", "distance"],
        zip(shifts, distances),
    )


def run_path_independence(
    *,
    diagnostics_cfg,
    model,
    state: DiagnosticsBatchState,
    global_step: int,
    diagnostics_generator: torch.Generator,
    diagnostics_path_independence_dir: Path,
) -> None:
    if not (state.has_p and state.embeddings.shape[1] >= 2 and state.p_embeddings is not None):
        raise AssertionError("Path independence diagnostics require pose embeddings with at least two timesteps.")
    action_labels = state.action_metadata.action_labels
    action_vectors = state.action_metadata.action_vectors
    unique_actions = state.action_metadata.unique_actions
    action_counts = state.action_metadata.action_counts

    def _find_action_id(keyword: str) -> Optional[int]:
        for aid, label in action_labels.items():
            if keyword in label:
                return aid
        return None

    right_id = _find_action_id("RIGHT")
    left_id = _find_action_id("LEFT")
    up_id = _find_action_id("UP")
    down_id = _find_action_id("DOWN")

    sorted_ids = [
        int(aid)
        for aid, _ in sorted(
            zip(unique_actions.tolist(), action_counts.tolist()),
            key=lambda pair: pair[1],
            reverse=True,
        )
    ]
    straightline_ids: List[int]
    if right_id is not None and left_id is not None:
        straightline_ids = [right_id, left_id]
    elif up_id is not None and down_id is not None:
        straightline_ids = [up_id, down_id]
    else:
        straightline_ids = sorted_ids[:2]
    if len(straightline_ids) < 2:
        raise AssertionError("Path independence diagnostics require at least two action IDs.")
    a_id = straightline_ids[0]
    b_id = straightline_ids[1] if len(straightline_ids) > 1 else straightline_ids[0]
    action_a = action_vectors.get(a_id)
    action_b = action_vectors.get(b_id)
    if action_a is None or action_b is None:
        raise AssertionError("Path independence diagnostics require action vectors for both actions.")

    seq_len = state.embeddings.shape[1]
    start_frame = max(min(state.warmup_frames, seq_len - 2), 0)
    max_starts = min(diagnostics_cfg.path_independence_samples, state.embeddings.shape[0])
    if max_starts <= 0:
        raise AssertionError("Path independence diagnostics require at least one start.")
    use_noop = state.action_metadata.noop_id is not None and state.action_metadata.noop_id in action_vectors
    action_b_first = action_vectors.get(state.action_metadata.noop_id) if use_noop else action_b
    action_b_second = action_vectors.get(state.action_metadata.noop_id) if use_noop else action_a
    z_diffs, p_diffs = compute_path_independence_diffs(
        model=model,
        diag_embeddings=state.embeddings,
        diag_h_states=state.h_states,
        diag_p_embeddings=state.p_embeddings,
        action_a=action_a,
        action_b=action_b,
        action_b_first=action_b_first,
        action_b_second=action_b_second,
        start_frame=start_frame,
        max_starts=max_starts,
        path_independence_steps=diagnostics_cfg.path_independence_steps,
        diagnostics_generator=diagnostics_generator,
    )
    if z_diffs and p_diffs:
        label_a = action_labels.get(a_id, f"action {a_id}")
        label_b = action_labels.get(b_id, f"action {b_id}")
        if use_noop:
            label_b_path = action_labels.get(state.action_metadata.noop_id, "NOOP")
        else:
            label_b_path = f"{label_b}+{label_a}"
        labels = [f"{label_a}+{label_b} vs {label_b_path}"]
        save_path_independence_plot(
            diagnostics_path_independence_dir / f"path_independence_{global_step:07d}.png",
            labels,
            [float(np.mean(z_diffs))],
            [float(np.mean(p_diffs))],
        )
        write_step_csv(
            diagnostics_path_independence_dir,
            f"path_independence_{global_step:07d}.csv",
            ["label", "z_distance", "p_distance"],
            [(labels[0], float(np.mean(z_diffs)), float(np.mean(p_diffs)))],
        )
