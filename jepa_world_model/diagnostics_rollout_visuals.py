from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from jepa_world_model.vis import describe_action_tensor
from jepa_world_model.vis_rollout import VisualizationSelection, VisualizationSequence, render_rollout_batch
from jepa_world_model.vis_rollout_batch import save_rollout_sequence_batch
from jepa_world_model.vis_visualization_batch import _render_visualization_batch
from jepa_world_model.plots.plot_off_manifold_error import save_off_manifold_visualization
from jepa_world_model.plots.plot_recon_heatmaps import (
    loss_to_heatmap,
    prediction_gradient_heatmap,
)
from recon.data import short_traj_state_label

def _extract_frame_labels(
    batch_paths: List[List[str]],
    sample_idx: int,
    start_idx: int,
    length: int,
) -> List[str]:
    if sample_idx >= len(batch_paths):
        raise AssertionError("Frame paths are required for visualization labels.")
    sample_paths = batch_paths[sample_idx]
    end = min(start_idx + length, len(sample_paths))
    slice_paths = sample_paths[start_idx:end]
    if len(slice_paths) < length:
        slice_paths = sample_paths[-length:]
    return [short_traj_state_label(path) for path in slice_paths]


def build_visualization_sequences(
    *,
    batch_cpu: Tuple[torch.Tensor, torch.Tensor, List[List[str]]],
    selection: Optional[VisualizationSelection],
    model,
    decoder,
    device: torch.device,
    vis_cfg,
    vis_selection_generator: torch.Generator,
    use_z2h_init: bool,
) -> Tuple[List[VisualizationSequence], str]:
    vis_frames = batch_cpu[0].to(device)
    vis_actions = batch_cpu[1].to(device)
    frame_paths = batch_cpu[2]
    assert torch.is_grad_enabled()
    with torch.no_grad():
        vis_embeddings = model.encode_sequence(vis_frames)["embeddings"]
        decoded_frames = decoder(vis_embeddings)
    assert torch.is_grad_enabled()
    if vis_frames.shape[0] == 0:
        raise ValueError("Visualization batch must include at least one sequence.")
    if vis_frames.shape[1] < 2:
        raise ValueError("Visualization batch must include at least two frames.")
    batch_size = vis_frames.shape[0]
    min_start = 0
    target_window = max(2, vis_cfg.rollout + 1)
    if vis_cfg.columns is not None:
        target_window = max(target_window, max(2, vis_cfg.columns))
    max_window = min(target_window, vis_frames.shape[1] - min_start)
    if max_window < 2:
        raise ValueError("Visualization window must be at least two frames wide.")
    max_start = max(min_start, vis_frames.shape[1] - max_window)
    if selection is not None and selection.row_indices.numel() > 0:
        num_rows = min(vis_cfg.rows, selection.row_indices.numel())
        row_indices = selection.row_indices[:num_rows].to(device=vis_frames.device)
        base_starts = selection.time_indices[:num_rows].to(device=vis_frames.device)
    else:
        num_rows = min(vis_cfg.rows, batch_size)
        row_indices = torch.randperm(batch_size, generator=vis_selection_generator, device=vis_frames.device)[:num_rows]
        base_starts = torch.randint(
            min_start,
            max_start + 1,
            (num_rows,),
            device=vis_frames.device,
            generator=vis_selection_generator,
        )
    action_texts: List[List[str]] = []
    rollout_frames: List[List[Optional[torch.Tensor]]] = []
    reencoded_frames: List[List[Optional[torch.Tensor]]] = []
    kept_row_indices: List[torch.Tensor] = []
    kept_start_indices: List[int] = []
    debug_lines: List[str] = []
    for row_offset, idx in enumerate(row_indices):
        start_idx = int(base_starts[row_offset].item())
        start_idx = max(min_start, min(start_idx, max_start))
        gt_slice = vis_frames[idx, start_idx : start_idx + max_window]
        if gt_slice.shape[0] < max_window:
            continue
        row_actions: List[str] = []
        for offset in range(max_window):
            action_idx = min(start_idx + offset, vis_actions.shape[1] - 1)
            row_actions.append(describe_action_tensor(vis_actions[idx, action_idx]))
        row_rollout: List[Optional[torch.Tensor]] = [None for _ in range(max_window)]
        row_reencoded: List[Optional[torch.Tensor]] = [None for _ in range(max_window)]
        assert torch.is_grad_enabled()
        with torch.no_grad():
            current_embed = vis_embeddings[idx, start_idx].unsqueeze(0)
            if use_z2h_init:
                current_hidden = model.z_to_h(current_embed.detach())
            else:
                current_hidden = current_embed.new_zeros(1, model.state_dim)
            prev_pred_frame = decoded_frames[idx, start_idx].detach()
            current_frame = prev_pred_frame
            for step in range(1, max_window):
                action = vis_actions[idx, start_idx + step - 1].unsqueeze(0)
                prev_embed = current_embed
                h_next = model.predictor(current_embed, current_hidden, action)
                next_embed = model.h_to_z(h_next)
                decoded_next = decoder(next_embed)[0]
                current_frame = decoded_next.clamp(0, 1)
                row_rollout[step] = current_frame.detach().cpu()
                reenc_embed = model.encoder(current_frame.unsqueeze(0))
                row_reencoded[step] = decoder(reenc_embed)[0].clamp(0, 1).detach().cpu()
                if vis_cfg.log_deltas and row_offset < 2:
                    latent_norm = (next_embed - prev_embed).norm().item()
                    pixel_delta = (current_frame - prev_pred_frame).abs().mean().item()
                    frame_mse = F.mse_loss(current_frame, gt_slice[step]).item()
                    debug_lines.append(
                        (
                            f"[viz] row={int(idx)} step={step} "
                            f"latent_norm={latent_norm:.4f} pixel_delta={pixel_delta:.4f} "
                            f"frame_mse={frame_mse:.4f}"
                        )
                    )
                prev_pred_frame = current_frame.detach()
                current_embed = next_embed
                current_hidden = h_next
        assert torch.is_grad_enabled()
        action_texts.append(row_actions)
        rollout_frames.append(row_rollout)
        reencoded_frames.append(row_reencoded)
        kept_row_indices.append(idx)
        kept_start_indices.append(start_idx)
    if not kept_row_indices:
        raise AssertionError("Failed to build any visualization sequences.")
    kept_rows = torch.stack(kept_row_indices)
    kept_starts = torch.tensor(kept_start_indices, device=vis_frames.device)
    items = render_rollout_batch(
        vis_frames=vis_frames,
        decoded_frames=decoded_frames,
        row_indices=kept_rows,
        start_indices=kept_starts,
        max_window=max_window,
        action_texts=action_texts,
        rollout_frames=rollout_frames,
        reencoded_frames=reencoded_frames,
    )
    if debug_lines:
        print("\n".join(debug_lines))
    labels: List[List[str]] = []
    gradients: List[List[Optional[np.ndarray]]] = []
    for item in items:
        labels.append(
            _extract_frame_labels(
                frame_paths,
                item.row_index,
                item.start_idx,
                len(item.ground_truth),
            )
        )
        item_gradients: List[Optional[np.ndarray]] = [None for _ in range(len(item.rollout))]
        for step in range(1, len(item.rollout)):
            current_frame = item.rollout[step]
            if current_frame is None:
                continue
            target_frame = item.ground_truth[step]
            if vis_cfg.gradient_norms:
                item_gradients[step] = prediction_gradient_heatmap(current_frame, target_frame)
            else:
                item_gradients[step] = loss_to_heatmap(target_frame, current_frame)
        gradients.append(item_gradients)
    return _render_visualization_batch(
        items=items,
        labels=labels,
        gradients=gradients,
        show_gradients=vis_cfg.gradient_norms,
    )


def compute_off_manifold_errors(
    *,
    model,
    decoder,
    batch_cpu: Tuple[torch.Tensor, torch.Tensor, List[List[str]]],
    device: torch.device,
    rollout_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    frames = batch_cpu[0].to(device)
    actions = batch_cpu[1].to(device)
    if frames.shape[0] == 0:
        raise AssertionError("Off-manifold batch requires at least one sequence.")
    if frames.shape[1] < rollout_steps + 1:
        raise AssertionError("Off-manifold batch does not include enough frames for the requested rollout.")
    if actions.shape[1] < rollout_steps:
        raise AssertionError("Off-manifold batch does not include enough actions for the requested rollout.")
    start_frames = frames[:, 0]
    action_seq = actions[:, :rollout_steps]
    z_t = model.encoder(start_frames)
    h_t = z_t.new_zeros(z_t.shape[0], model.state_dim)
    step_indices: List[np.ndarray] = []
    errors: List[np.ndarray] = []
    for step in range(rollout_steps + 1):
        decoded = decoder(z_t).clamp(0, 1)
        z_back = model.encoder(decoded)
        err = (z_back - z_t).norm(dim=-1)
        errors.append(err.detach().cpu().numpy())
        step_indices.append(np.full(err.shape, step, dtype=np.int64))
        if step < rollout_steps:
            act = action_seq[:, step]
            h_next = model.predictor(z_t, h_t, act)
            z_t = model.h_to_z(h_next)
            h_t = h_next
    return np.concatenate(step_indices), np.concatenate(errors)


def run_rollout_visualizations(
    *,
    vis_cfg,
    model,
    decoder,
    device: torch.device,
    weights,
    global_step: int,
    fixed_batch_cpu: Tuple[torch.Tensor, torch.Tensor, List[List[str]]],
    fixed_selection: Optional[VisualizationSelection],
    rolling_batch_cpu: Tuple[torch.Tensor, torch.Tensor, List[List[str]]],
    off_manifold_batch_cpu: Optional[Tuple[torch.Tensor, torch.Tensor, List[List[str]]]],
    off_manifold_steps: int,
    vis_selection_generator: torch.Generator,
    fixed_vis_dir: Path,
    rolling_vis_dir: Path,
    vis_off_manifold_dir: Path,
) -> None:
    sequences, grad_label = build_visualization_sequences(
        batch_cpu=fixed_batch_cpu,
        selection=fixed_selection,
        model=model,
        decoder=decoder,
        device=device,
        vis_cfg=vis_cfg,
        vis_selection_generator=vis_selection_generator,
        use_z2h_init=weights.z2h > 0,
    )
    save_rollout_sequence_batch(
        fixed_vis_dir,
        sequences,
        grad_label,
        global_step,
        include_pixel_delta=(weights.pixel_delta > 0 or weights.pixel_delta_multi_box > 0),
    )

    sequences, grad_label = build_visualization_sequences(
        batch_cpu=rolling_batch_cpu,
        selection=None,
        model=model,
        decoder=decoder,
        device=device,
        vis_cfg=vis_cfg,
        vis_selection_generator=vis_selection_generator,
        use_z2h_init=weights.z2h > 0,
    )
    save_rollout_sequence_batch(
        rolling_vis_dir,
        sequences,
        grad_label,
        global_step,
        include_pixel_delta=(weights.pixel_delta > 0 or weights.pixel_delta_multi_box > 0),
    )

    if off_manifold_batch_cpu is not None:
        assert torch.is_grad_enabled()
        with torch.no_grad():
            step_indices, errors = compute_off_manifold_errors(
                model=model,
                decoder=decoder,
                batch_cpu=off_manifold_batch_cpu,
                device=device,
                rollout_steps=off_manifold_steps,
            )
        assert torch.is_grad_enabled()
        save_off_manifold_visualization(
            vis_off_manifold_dir,
            step_indices,
            errors,
            global_step,
        )
