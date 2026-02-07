from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from jepa_world_model.diagnostics_prepare import DiagnosticsBatchState
from jepa_world_model.diagnostics_utils import write_step_csv
from jepa_world_model.plots.plot_diagnostics_extra import save_ablation_divergence_plot, save_rollout_divergence_plot

RECON_LOSS = nn.MSELoss()


def compute_rollout_divergence_metrics(
    *,
    model,
    decoder,
    diag_embeddings: torch.Tensor,
    diag_h_states: torch.Tensor,
    diag_p_embeddings: Optional[torch.Tensor],
    diag_actions_device: torch.Tensor,
    diag_frames_device: torch.Tensor,
    rollout_horizon: int,
    warmup_frames: int,
    start_span: int,
    rollout_divergence_samples: int,
    diagnostics_generator: torch.Generator,
    force_h_zero: bool = False,
) -> Tuple[List[int], List[float], List[float], List[float], List[float], List[float]]:
    if rollout_horizon <= 0:
        raise AssertionError("rollout_horizon must be positive for rollout divergence.")
    if start_span <= 0:
        raise AssertionError("start_span must be positive for rollout divergence.")
    if diag_frames_device.shape[1] < 2:
        raise AssertionError("rollout divergence requires at least two timesteps.")
    total_positions = diag_frames_device.shape[0] * start_span
    if total_positions <= 0:
        raise AssertionError("rollout divergence requires at least one valid start position.")
    with torch.no_grad():
        sample_count = min(rollout_divergence_samples, total_positions)
        perm = torch.randperm(total_positions, generator=diagnostics_generator)[:sample_count]
        has_p = diag_p_embeddings is not None and model.p_action_delta_projector is not None
        pixel_errors = torch.zeros(rollout_horizon, device=diag_embeddings.device)
        pixel_teacher_errors = torch.zeros(rollout_horizon, device=diag_embeddings.device)
        z_errors = torch.zeros(rollout_horizon, device=diag_embeddings.device)
        h_errors = torch.zeros(rollout_horizon, device=diag_embeddings.device)
        p_errors = torch.zeros(rollout_horizon, device=diag_embeddings.device)
        counts = torch.zeros(rollout_horizon, device=diag_embeddings.device)
        for flat_idx in perm.tolist():
            b = flat_idx // start_span
            t0 = (flat_idx % start_span) + warmup_frames
            z_t = diag_embeddings[b, t0]
            h_t = diag_h_states[b, t0]
            if force_h_zero:
                h_t = torch.zeros_like(h_t)
            p_t = diag_p_embeddings[b, t0] if has_p else None
            for k in range(rollout_horizon):
                if t0 + k >= diag_frames_device.shape[1] - 1:
                    break
                act = diag_actions_device[b, t0 + k]
                if has_p:
                    if force_h_zero:
                        h_in = torch.zeros_like(h_t)
                    else:
                        h_in = h_t.detach() if model.cfg.pose_delta_detach_h else h_t
                    delta = model.p_action_delta_projector(
                        p_t.unsqueeze(0),
                        h_in.unsqueeze(0),
                        act.unsqueeze(0),
                    ).squeeze(0)
                    p_t = p_t + delta
                h_next = model.predictor(
                    z_t.unsqueeze(0),
                    h_t.unsqueeze(0),
                    act.unsqueeze(0),
                )
                z_t = model.h_to_z(h_next).squeeze(0)
                if force_h_zero:
                    h_t = torch.zeros_like(h_t)
                else:
                    h_t = h_next.squeeze(0)
                decoded = decoder(z_t.unsqueeze(0))
                pixel_errors[k] += RECON_LOSS(decoded, diag_frames_device[b, t0 + k + 1].unsqueeze(0))
                z_gt = diag_embeddings[b, t0 + k + 1]
                h_gt = diag_h_states[b, t0 + k + 1]
                decoded_teacher = decoder(z_gt.unsqueeze(0))
                pixel_teacher_errors[k] += RECON_LOSS(decoded_teacher, diag_frames_device[b, t0 + k + 1].unsqueeze(0))
                z_errors[k] += (z_t - z_gt).norm()
                h_errors[k] += (h_t - h_gt).norm()
                if has_p:
                    p_gt = diag_p_embeddings[b, t0 + k + 1]
                    p_errors[k] += (p_t - p_gt).norm()
                counts[k] += 1
        counts = torch.clamp(counts, min=1.0)
        pixel_mean = (pixel_errors / counts).detach().cpu().numpy().tolist()
        pixel_teacher_mean = (pixel_teacher_errors / counts).detach().cpu().numpy().tolist()
        z_mean = (z_errors / counts).detach().cpu().numpy().tolist()
        h_mean = (h_errors / counts).detach().cpu().numpy().tolist()
        if has_p:
            p_mean = (p_errors / counts).detach().cpu().numpy().tolist()
        else:
            p_mean = [0.0 for _ in range(rollout_horizon)]
    horizons = list(range(1, rollout_horizon + 1))
    return horizons, pixel_mean, pixel_teacher_mean, z_mean, h_mean, p_mean


def compute_h_ablation_divergence(
    *,
    model,
    decoder,
    diag_embeddings: torch.Tensor,
    diag_h_states: torch.Tensor,
    diag_p_embeddings: torch.Tensor,
    diag_actions_device: torch.Tensor,
    diag_frames_device: torch.Tensor,
    rollout_horizon: int,
    warmup_frames: int,
    start_span: int,
    rollout_divergence_samples: int,
    diagnostics_generator: torch.Generator,
    force_h_zero: bool = False,
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    if force_h_zero:
        raise AssertionError("h-ablation divergence is undefined when force_h_zero is enabled.")
    if rollout_horizon <= 0:
        raise AssertionError("rollout_horizon must be positive for h-ablation divergence.")
    if start_span <= 0:
        raise AssertionError("start_span must be positive for h-ablation divergence.")
    if diag_embeddings.shape[1] < 2:
        raise AssertionError("h-ablation divergence requires at least two timesteps.")
    total_positions = diag_embeddings.shape[0] * start_span
    if total_positions <= 0:
        raise AssertionError("h-ablation divergence requires at least one valid start position.")
    with torch.no_grad():
        sample_count = min(rollout_divergence_samples, total_positions)
        perm = torch.randperm(total_positions, generator=diagnostics_generator)[:sample_count]
        pixel_errors = torch.zeros(rollout_horizon, device=diag_embeddings.device)
        pixel_errors_zero = torch.zeros(rollout_horizon, device=diag_embeddings.device)
        latent_errors = torch.zeros(rollout_horizon, device=diag_embeddings.device)
        latent_errors_zero = torch.zeros(rollout_horizon, device=diag_embeddings.device)
        counts = torch.zeros(rollout_horizon, device=diag_embeddings.device)
        for flat_idx in perm.tolist():
            b = flat_idx // start_span
            t0 = (flat_idx % start_span) + warmup_frames
            z_norm = diag_embeddings[b, t0]
            h_norm = diag_h_states[b, t0]
            z_zero = z_norm.clone()
            p_norm = diag_p_embeddings[b, t0]
            p_zero = diag_p_embeddings[b, t0]
            for k in range(rollout_horizon):
                if t0 + k >= diag_embeddings.shape[1] - 1:
                    break
                act = diag_actions_device[b, t0 + k]
                h_norm_in = h_norm.detach() if model.cfg.pose_delta_detach_h else h_norm
                delta_norm = model.p_action_delta_projector(
                    p_norm.unsqueeze(0),
                    h_norm_in.unsqueeze(0),
                    act.unsqueeze(0),
                ).squeeze(0)
                p_norm = p_norm + delta_norm
                h_zero = torch.zeros_like(h_norm)
                h_zero_in = h_zero.detach() if model.cfg.pose_delta_detach_h else h_zero
                delta_zero = model.p_action_delta_projector(
                    p_zero.unsqueeze(0),
                    h_zero_in.unsqueeze(0),
                    act.unsqueeze(0),
                ).squeeze(0)
                p_zero = p_zero + delta_zero
                h_norm_next = model.predictor(
                    z_norm.unsqueeze(0),
                    h_norm.unsqueeze(0),
                    act.unsqueeze(0),
                )
                z_norm = model.h_to_z(h_norm_next).squeeze(0)
                h_norm = h_norm_next.squeeze(0)
                h_zero_next = model.predictor(
                    z_zero.unsqueeze(0),
                    h_zero.unsqueeze(0),
                    act.unsqueeze(0),
                )
                z_zero = model.h_to_z(h_zero_next).squeeze(0)
                h_zero = h_zero_next.squeeze(0)
                decoded_norm = decoder(z_norm.unsqueeze(0))
                decoded_zero = decoder(z_zero.unsqueeze(0))
                target = diag_frames_device[b, t0 + k + 1].unsqueeze(0)
                pixel_errors[k] += RECON_LOSS(decoded_norm, target)
                pixel_errors_zero[k] += RECON_LOSS(decoded_zero, target)
                p_gt = diag_p_embeddings[b, t0 + k + 1]
                latent_errors[k] += (p_norm - p_gt).norm()
                latent_errors_zero[k] += (p_zero - p_gt).norm()
                counts[k] += 1
        counts = torch.clamp(counts, min=1.0)
        pixel_mean = (pixel_errors / counts).detach().cpu().numpy().tolist()
        pixel_zero_mean = (pixel_errors_zero / counts).detach().cpu().numpy().tolist()
        latent_mean = (latent_errors / counts).detach().cpu().numpy().tolist()
        latent_zero_mean = (latent_errors_zero / counts).detach().cpu().numpy().tolist()
    horizons = list(range(1, rollout_horizon + 1))
    return horizons, pixel_mean, pixel_zero_mean, latent_mean, latent_zero_mean
