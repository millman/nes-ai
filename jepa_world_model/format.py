#!/usr/bin/env python3
"""Formatting helpers for summaries and diagnostics."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


def _format_elapsed_time(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_hwc(height: int, width: int, channels: int) -> str:
    return f"{height}×{width}×{channels}"


def _format_param_count(count: int) -> str:
    if count < 0:
        raise ValueError("Parameter count cannot be negative.")
    if count < 1_000:
        return str(count)
    for divisor, suffix in (
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "k"),
    ):
        if count >= divisor:
            value = count / divisor
            if value >= 100:
                formatted = f"{value:.0f}"
            elif value >= 10:
                formatted = f"{value:.1f}".rstrip("0").rstrip(".")
            else:
                formatted = f"{value:.2f}".rstrip("0").rstrip(".")
            return f"{formatted}{suffix}"
    return str(count)


def _format_flops(flops: int) -> str:
    """Format FLOPs count in human-readable form."""
    if flops < 0:
        raise ValueError("FLOPs count cannot be negative.")
    if flops >= 1_000_000_000_000:
        value = flops / 1_000_000_000_000
        suffix = "TFLOPs"
    elif flops >= 1_000_000_000:
        value = flops / 1_000_000_000
        suffix = "GFLOPs"
    elif flops >= 1_000_000:
        value = flops / 1_000_000
        suffix = "MFLOPs"
    elif flops >= 1_000:
        value = flops / 1_000
        suffix = "KFLOPs"
    else:
        return f"{flops} FLOPs"

    if value >= 100:
        formatted = f"{value:.0f}"
    elif value >= 10:
        formatted = f"{value:.1f}".rstrip("0").rstrip(".")
    else:
        formatted = f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{formatted} {suffix}"


def format_shape_summary(
    raw_shape: Tuple[int, int, int],
    encoder_info: Dict[str, Any],
    predictor_info: Dict[str, Any],
    decoder_info: Dict[str, Any],
    encoder_schedule: Sequence[int],
    total_param_text: Optional[str] = None,
    flops_per_step_text: Optional[str] = None,
) -> str:
    """Summarize model shapes for input, encoder, predictor, and decoder stages.

    Expected argument contents:
    - raw_shape: (height, width, channels) of the raw input frame.
    - encoder_info: dict with keys "input" (H, W, C), "stages" (list of dicts with "stage", "in", "out"),
      and "latent_dim" (int).
    - predictor_info: dict with keys "latent_dim", "hidden_dim", "action_dim", and optional "conditioning".
    - decoder_info: dict with keys "latent_dim", "projection" (H, W, C), "upsample" (list of dicts with
      "stage", "in", "out"), "pre_resize" (H, W, C), "final_target" (H, W, C), and "needs_resize" (bool).
    - encoder_schedule: sequence of encoder channel sizes (ints).
    - total_param_text/flops_per_step_text: optional preformatted strings for totals.
    """
    lines: List[str] = []
    lines.append("Model Shape Summary (H×W×C)")
    lines.append(f"Raw frame {_format_hwc(*raw_shape)}")
    lines.append(f"  └─ Data loader resize → {_format_hwc(*encoder_info['input'])}")
    lines.append("")
    lines.append(f"Encoder schedule: {tuple(encoder_schedule)}")
    lines.append("Encoder:")
    for stage in encoder_info["stages"]:
        lines.append(f"  • Stage {stage['stage']}: {_format_hwc(*stage['in'])} → {_format_hwc(*stage['out'])}")
    # Show pooling
    latent_dim = encoder_info["latent_dim"]
    lines.append(f"  AdaptiveAvgPool → 1×1×{latent_dim} (latent)")
    lines.append("")
    lines.append("Predictor:")
    conditioning = predictor_info.get("conditioning")
    cond_text = f"conditioning={conditioning}" if conditioning is not None else "conditioning=unknown"
    lines.append(
        f"  latent {predictor_info['latent_dim']} → hidden {predictor_info['hidden_dim']} "
        f"(action_dim={predictor_info['action_dim']}, {cond_text})"
    )
    lines.append("")
    lines.append("Decoder:")
    lines.append(f"  latent_dim={decoder_info.get('latent_dim', 'N/A')}")
    lines.append(f"  Projection reshape → {_format_hwc(*decoder_info['projection'])}")
    for stage in decoder_info["upsample"]:
        lines.append(f"  • UpStage {stage['stage']}: {_format_hwc(*stage['in'])} → {_format_hwc(*stage['out'])}")
    # No more detail_skip in decoder
    pre_resize = decoder_info["pre_resize"]
    target = decoder_info["final_target"]
    if decoder_info["needs_resize"]:
        lines.append(f"  Final conv output {_format_hwc(*pre_resize)} → bilinear resize → {_format_hwc(*target)}")
    else:
        lines.append(f"  Final output {_format_hwc(*pre_resize)}")
    if total_param_text or flops_per_step_text:
        lines.append("")
    if total_param_text:
        lines.append(f"Total parameters: {total_param_text}")
    if flops_per_step_text:
        lines.append(f"FLOPs per step: {flops_per_step_text}")
    return "\n".join(lines)
