#!/usr/bin/env python3
"""FLOPs estimation utilities for the JEPA world model."""
from __future__ import annotations

from typing import Tuple

from jepa_world_model.model_config import ModelConfig


def _conv2d_flops(in_ch: int, out_ch: int, kernel_size: int, h: int, w: int, stride: int = 1) -> Tuple[int, int, int]:
    """Calculate FLOPs for Conv2d (multiply-adds counted as 2 ops). Returns (flops, out_h, out_w)."""
    padding = (kernel_size - 1) // 2
    out_h = (h + 2 * padding - kernel_size) // stride + 1
    out_w = (w + 2 * padding - kernel_size) // stride + 1
    flops_per_pixel = kernel_size * kernel_size * in_ch * 2  # *2 for multiply-add
    total_flops = flops_per_pixel * out_ch * out_h * out_w
    return total_flops, out_h, out_w


def _conv_transpose2d_flops(
    in_ch: int,
    out_ch: int,
    kernel_size: int,
    h: int,
    w: int,
    stride: int = 2,
) -> Tuple[int, int, int]:
    """Calculate FLOPs for ConvTranspose2d. Returns (flops, out_h, out_w)."""
    out_h = (h - 1) * stride + kernel_size
    out_w = (w - 1) * stride + kernel_size
    flops_per_pixel = kernel_size * kernel_size * in_ch * 2
    total_flops = flops_per_pixel * out_ch * out_h * out_w
    return total_flops, out_h, out_w


def _linear_flops(in_features: int, out_features: int) -> int:
    """Calculate FLOPs for Linear layer."""
    return in_features * out_features * 2  # multiply-add


def calculate_flops_per_step(cfg: ModelConfig, batch_size: int, seq_len: int) -> int:
    """Calculate estimated FLOPs per training step (forward + backward).

    Returns total FLOPs including forward pass and backward pass (estimated as 2x forward).
    """
    h, w = cfg.image_size, cfg.image_size

    # --- Encoder FLOPs (per frame) ---
    encoder_flops = 0
    curr_h, curr_w = h, w
    in_ch = cfg.in_channels

    for i, out_ch in enumerate(cfg.encoder_schedule):
        # First conv (stride 2) - first layer has CoordConv (+2 channels)
        actual_in_ch = in_ch + 2 if i == 0 else in_ch
        flops1, curr_h, curr_w = _conv2d_flops(actual_in_ch, out_ch, 3, curr_h, curr_w, stride=2)
        # Second conv (stride 1)
        flops2, _, _ = _conv2d_flops(out_ch, out_ch, 3, curr_h, curr_w, stride=1)
        encoder_flops += flops1 + flops2
        in_ch = out_ch

    encoder_total = encoder_flops * batch_size * seq_len

    # --- Predictor FLOPs (per prediction) ---
    predictor_flops = 0
    emb_dim = cfg.embedding_dim
    hidden_dim = cfg.hidden_dim
    action_dim = cfg.action_dim

    # in_proj: (emb + h + action) -> hidden_dim
    predictor_flops += _linear_flops(emb_dim + action_dim + cfg.state_dim, hidden_dim)
    # hidden_proj: hidden -> hidden
    predictor_flops += _linear_flops(hidden_dim, hidden_dim)
    # out_proj: hidden -> emb
    predictor_flops += _linear_flops(hidden_dim, emb_dim)
    # h_out projection
    predictor_flops += _linear_flops(hidden_dim, cfg.state_dim)

    num_predictions = batch_size * (seq_len - 1)
    predictor_total = predictor_flops * num_predictions

    # --- Action-from-pair head (per transition) ---
    delta_head_flops = 0
    delta_head_flops += _linear_flops(emb_dim * 2, hidden_dim)
    delta_head_flops += _linear_flops(hidden_dim, hidden_dim)
    delta_head_flops += _linear_flops(hidden_dim, action_dim)
    delta_head_total = delta_head_flops * num_predictions

    # --- Action-from-s-pair head (per transition) ---
    s_dim = cfg.state_embed_dim if cfg.state_embed_dim is not None else cfg.state_dim
    s_delta_head_flops = 0
    s_delta_head_flops += _linear_flops(s_dim * 2, hidden_dim)
    s_delta_head_flops += _linear_flops(hidden_dim, hidden_dim)
    s_delta_head_flops += _linear_flops(hidden_dim, action_dim)
    s_delta_head_total = s_delta_head_flops * num_predictions

    # --- Decoder FLOPs (per frame) ---
    decoder_schedule = cfg.decoder_schedule if cfg.decoder_schedule is not None else cfg.encoder_schedule
    num_layers = len(decoder_schedule)
    start_hw = cfg.image_size // (2 ** num_layers)
    start_ch = decoder_schedule[-1]

    decoder_flops = 0
    # Linear projection
    decoder_flops += _linear_flops(emb_dim, start_ch * start_hw * start_hw)

    curr_h, curr_w = start_hw, start_hw
    in_ch = start_ch

    for out_ch in reversed(decoder_schedule):
        # Upsample (ConvTranspose2d kernel=2, stride=2)
        flops1, curr_h, curr_w = _conv_transpose2d_flops(in_ch, out_ch, 2, curr_h, curr_w, stride=2)
        # Conv refinement
        flops2, _, _ = _conv2d_flops(out_ch, out_ch, 3, curr_h, curr_w, stride=1)
        decoder_flops += flops1 + flops2
        in_ch = out_ch

    # Head convs
    head_hidden = max(in_ch // 2, 1)
    flops_head1, _, _ = _conv2d_flops(in_ch, head_hidden, 3, curr_h, curr_w)
    flops_head2, _, _ = _conv2d_flops(head_hidden, cfg.in_channels, 1, curr_h, curr_w)
    decoder_flops += flops_head1 + flops_head2

    decoder_total = decoder_flops * batch_size * seq_len

    # --- Total ---
    forward_total = encoder_total + predictor_total + decoder_total + delta_head_total + s_delta_head_total
    backward_total = forward_total * 2  # Backward is roughly 2x forward
    total_per_step = forward_total + backward_total

    return total_per_step
