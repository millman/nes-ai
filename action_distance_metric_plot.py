#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot per-trajectory action-distance metrics across full rollouts.

Outputs:
  - Metric curves (MS-SSIM, patch MS-SSIM, MSE, focal losses) vs. action distance
  - Gradient scatter plot coloured by frame index
  - Per-frame gradient-difference panels (dx / dy) saved under out/gradients/<traj>

Run:
  python action_distance_metric_plot.py --traj_root traj_dumps --out_dir out.action_distance_metrics
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from recon import load_frame_as_tensor, set_seed


STATE_RE = re.compile(r"state_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)


@dataclass
class PlotCfg:
    traj_root: Path
    out_dir: Path
    seed: int = 0
    device: Optional[str] = None
    max_traj: Optional[int] = None
    max_frames: Optional[int] = None
    patch_size: int = 16
    patch_stride: int = 8
    patch_focal_gamma: float = 1.5
    patch_eps: float = 1e-6


def parse_args() -> PlotCfg:
    ap = argparse.ArgumentParser(description="Plot MS-SSIM/MSE/Focal losses vs. action distance")
    ap.add_argument("--traj_root", type=Path, required=True, help="Directory containing trajectories")
    ap.add_argument("--out_dir", type=Path, default=Path("out.action_distance_metrics"), help="Destination for plots")
    ap.add_argument("--seed", type=int, default=0, help="Seed for deterministic dataloading order")
    ap.add_argument("--device", type=str, default=None, help="Computation device (cpu, cuda, cuda:0, ...)")
    ap.add_argument("--max_traj", type=int, default=None, help="Optional limit on number of trajectories to plot")
    ap.add_argument("--max_frames", type=int, default=None, help="Optional limit on frames per trajectory")
    ap.add_argument("--patch_size", type=int, default=16, help="Patch size for focal L1 metric")
    ap.add_argument("--patch_stride", type=int, default=8, help="Stride for focal L1 metric")
    ap.add_argument("--patch_focal_gamma", type=float, default=1.5, help="Gamma parameter for focal weighting")
    ap.add_argument("--patch_eps", type=float, default=1e-6, help="Numerical epsilon for focal weighting")
    args = ap.parse_args()
    return PlotCfg(
        traj_root=args.traj_root,
        out_dir=args.out_dir,
        seed=args.seed,
        device=args.device,
        max_traj=args.max_traj,
        max_frames=args.max_frames,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        patch_focal_gamma=args.patch_focal_gamma,
        patch_eps=args.patch_eps,
    )


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_trajectories(root: Path) -> Dict[str, List[Path]]:
    if not root.exists():
        raise FileNotFoundError(f"Trajectory root {root} does not exist")

    trajectories: Dict[str, List[Path]] = {}
    for states_dir in sorted(root.rglob("states")):
        if not states_dir.is_dir():
            continue
        frame_paths = sorted(states_dir.glob("state_*"), key=_frame_sort_key)
        # The first frame in trajectories seems to be bugged, it is very different from the next frame.
        frame_paths = frame_paths[1:]
        if not frame_paths:
            continue
        traj_rel = states_dir.parent.relative_to(root)
        trajectories[str(traj_rel)] = frame_paths

    if not trajectories:
        raise RuntimeError(
            "No trajectories found. Expected directories containing a 'states' subdirectory with state_*.png images."
        )
    return trajectories


def _frame_sort_key(path: Path) -> Tuple[int, str]:
    match = STATE_RE.match(path.name)
    if match:
        return int(match.group(1)), path.name
    return (10**9, path.name)


def pick_device(pref: Optional[str]) -> torch.device:
    if pref:
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ms_ssim(x: torch.Tensor, y: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weights is None:
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=x.device, dtype=x.dtype)
    window_size = 11
    sigma = 1.5
    channels = x.shape[1]
    window = _gaussian_window(window_size, sigma, channels, x.device)
    levels = weights.shape[0]
    mssim: List[torch.Tensor] = []
    mcs: List[torch.Tensor] = []

    x_scaled, y_scaled = x, y
    for level_idx in range(levels):
        ssim_val, cs_val = _ssim_components(x_scaled, y_scaled, window)
        mssim.append(ssim_val)
        mcs.append(cs_val)
        if min(x_scaled.shape[-2], x_scaled.shape[-1], y_scaled.shape[-2], y_scaled.shape[-1]) < 2:
            break
        x_scaled = F.avg_pool2d(x_scaled, kernel_size=2, stride=2, padding=0, ceil_mode=False)
        y_scaled = F.avg_pool2d(y_scaled, kernel_size=2, stride=2, padding=0, ceil_mode=False)

    if not mssim:
        return torch.tensor(1.0, device=x.device, dtype=x.dtype)

    level_weights = weights[: len(mssim)]
    mssim_tensor = torch.stack(mssim, dim=0)

    if len(mssim) == 1:
        return mssim_tensor[0]

    mcs_tensor = torch.stack(mcs[:-1], dim=0)

    pow1 = level_weights[:-1].unsqueeze(1)
    pow2 = level_weights[-1]
    ms_prod = torch.prod(mcs_tensor ** pow1, dim=0) * (mssim_tensor[-1] ** pow2)
    return ms_prod.mean()


def ms_ssim_patch(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    patch_size: int,
    stride: int,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute MS-SSIM across sliding patches."""

    stride = max(1, stride)
    patch_size = min(patch_size, x.shape[-2], x.shape[-1])
    if patch_size <= 0:
        raise ValueError("patch_size must be >= 1")

    patches_x = F.unfold(x, kernel_size=patch_size, stride=stride)
    patches_y = F.unfold(y, kernel_size=patch_size, stride=stride)
    if patches_x.numel() == 0 or patches_y.numel() == 0:
        return torch.tensor(1.0, device=x.device, dtype=x.dtype)

    patches_x = patches_x.transpose(1, 2).reshape(-1, x.shape[1], patch_size, patch_size)
    patches_y = patches_y.transpose(1, 2).reshape(-1, y.shape[1], patch_size, patch_size)
    return ms_ssim(patches_x, patches_y, weights=weights)


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = (gauss / gauss.sum()).unsqueeze(0)
    window_2d = (gauss.t() @ gauss).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def _ssim_components(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tuple[torch.Tensor, torch.Tensor]:
    padding = window.shape[-1] // 2
    mu_x = F.conv2d(x, window, padding=padding, groups=x.shape[1])
    mu_y = F.conv2d(y, window, padding=padding, groups=y.shape[1])

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=padding, groups=x.shape[1]) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=padding, groups=y.shape[1]) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=x.shape[1]) - mu_xy

    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    ssim_map = numerator / denominator
    cs_map = (2 * sigma_xy + C2) / (sigma_x_sq + sigma_y_sq + C2)

    ssim_val = ssim_map.mean(dim=(1, 2, 3))
    cs_val = cs_map.mean(dim=(1, 2, 3))
    return ssim_val, cs_val


def focal_patch_l1_loss(
    x_hat: torch.Tensor,
    x_true: torch.Tensor,
    *,
    patch_size: int,
    stride: int,
    gamma: float,
    eps: float,
) -> torch.Tensor:
    err = (x_hat - x_true).abs().mean(dim=1, keepdim=True)
    pooled = F.avg_pool2d(err, kernel_size=patch_size, stride=max(1, stride), padding=0)
    if gamma > 0:
        weights = (pooled + eps).pow(gamma)
        weighted = (weights * pooled).sum(dim=(1, 2, 3))
        norm = weights.sum(dim=(1, 2, 3)).clamp_min(eps)
        loss = weighted / norm
    else:
        loss = pooled.mean(dim=(1, 2, 3))
    return loss.mean()


def spatial_gradient(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (grad_y, grad_x) using forward differences with replicate padding."""
    grad_y = img[:, :, 1:, :] - img[:, :, :-1, :]
    grad_y = F.pad(grad_y, (0, 0, 0, 1))
    grad_x = img[:, :, :, 1:] - img[:, :, :, :-1]
    grad_x = F.pad(grad_x, (0, 1, 0, 0))
    return grad_y, grad_x


def gradient_difference_maps(
    x_hat: torch.Tensor,
    x_true: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return channel-wise gradient difference maps (dx, dy)."""
    grad_y_hat, grad_x_hat = spatial_gradient(x_hat)
    grad_y_true, grad_x_true = spatial_gradient(x_true)
    diff_x = grad_x_hat - grad_x_true
    diff_y = grad_y_hat - grad_y_true
    return diff_x.squeeze(0).cpu(), diff_y.squeeze(0).cpu()


def focal_l1_loss(
    x_hat: torch.Tensor,
    x_true: torch.Tensor,
    *,
    gamma: float,
    eps: float,
) -> torch.Tensor:
    err = (x_hat - x_true).abs()
    if gamma > 0:
        weights = (err + eps).pow(gamma)
        weighted = (weights * err).sum(dim=(1, 2, 3))
        norm = weights.sum(dim=(1, 2, 3)).clamp_min(eps)
        loss = weighted / norm
    else:
        loss = err.mean(dim=(1, 2, 3))
    return loss.mean()


def compute_metrics_for_traj(
    paths: List[Path], *, cfg: PlotCfg, device: torch.device
) -> Tuple[
    Dict[str, List[float]],
    Dict[str, List[float]],
    List[Tuple[torch.Tensor, torch.Tensor]],
]:
    metrics = {
        "ms_ssim": [],
        "mse": [],
        "focal_l1": [],
        "patch_focal_l1": [],
        "patch_ms_ssim": [],
    }
    gradients = {
        "dx": [],
        "dy": [],
        "mag": [],
    }
    gradient_maps: List[Tuple[torch.Tensor, torch.Tensor]] = []

    if not paths:
        return metrics, gradients, gradient_maps

    with torch.no_grad():
        ref = load_frame_as_tensor(paths[0]).to(device)
        ref_batch = ref.unsqueeze(0)

        for idx, path in enumerate(paths):
            if cfg.max_frames is not None and idx >= cfg.max_frames:
                break
            frame = load_frame_as_tensor(path).to(device)
            frame_batch = frame.unsqueeze(0)

            ms_val = float(ms_ssim(ref_batch, frame_batch).item())
            mse_val = float(F.mse_loss(frame, ref, reduction="mean").item())
            focal_val = float(
                focal_l1_loss(
                    frame_batch,
                    ref_batch,
                    gamma=cfg.patch_focal_gamma,
                    eps=cfg.patch_eps,
                ).item()
            )
            patch_focal_val = float(
                focal_patch_l1_loss(
                    frame_batch,
                    ref_batch,
                    patch_size=cfg.patch_size,
                    stride=cfg.patch_stride,
                    gamma=cfg.patch_focal_gamma,
                    eps=cfg.patch_eps,
                ).item()
            )
            patch_ms_val = float(
                ms_ssim_patch(
                    frame_batch,
                    ref_batch,
                    patch_size=cfg.patch_size,
                    stride=cfg.patch_stride,
                ).item()
            )

            diff_map_x, diff_map_y = gradient_difference_maps(frame_batch, ref_batch)
            dx = float(diff_map_x.mean().item())
            dy = float(diff_map_y.mean().item())
            mag = float((dx ** 2 + dy ** 2) ** 0.5)

            metrics["ms_ssim"].append(ms_val)
            metrics["mse"].append(mse_val)
            metrics["focal_l1"].append(focal_val)
            metrics["patch_focal_l1"].append(patch_focal_val)
            metrics["patch_ms_ssim"].append(patch_ms_val)

            gradients["dx"].append(dx)
            gradients["dy"].append(dy)
            gradients["mag"].append(mag)

            gradient_maps.append((diff_map_x, diff_map_y))

    return metrics, gradients, gradient_maps


def plot_metrics(
    traj_name: str,
    steps: List[int],
    metrics: Dict[str, List[float]],
    out_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))
    has_any = False
    for label, values in metrics.items():
        if not values:
            continue
        plt.plot(steps[: len(values)], values, label=label)
        has_any = True

    if not has_any:
        plt.close()
        return

    plt.xlabel("Action distance (frames since start)")
    plt.ylabel("Metric distance vs. first frame")
    plt.title(traj_name)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_gradient_scatter(
    traj_name: str,
    steps: List[int],
    gradients: Dict[str, List[float]],
    out_path: Path,
) -> None:
    if not gradients["dx"]:
        return

    dx = gradients["dx"]
    dy = gradients["dy"]
    mag = gradients["mag"]
    max_step = max(steps) if steps else 1
    max_step = max(max_step, 1)
    norm_steps = [s / max_step for s in steps]

    max_mag = max(mag) if mag else 0.0
    size_scale = 240.0 if max_mag > 0 else 0.0
    sizes = [60.0 + size_scale * (m / max_mag if max_mag > 0 else 0.0) for m in mag]

    plt.figure(figsize=(7, 7))
    scatter = plt.scatter(
        dx,
        dy,
        c=norm_steps,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        # s=sizes,
        edgecolors="black",
        linewidths=0.4,
        alpha=0.9,
    )
    plt.colorbar(scatter, label="Normalized action distance")
    plt.axhline(0.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
    plt.axvline(0.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
    plt.xlabel("Gradient L1 (x direction)")
    plt.ylabel("Gradient L1 (y direction)")
    plt.title(f"{traj_name} – gradient L1 scatter")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _diff_map_to_rgb(diff: torch.Tensor) -> Tuple[np.ndarray, float, float]:
    arr = diff.permute(1, 2, 0).numpy()
    max_abs = float(np.max(np.abs(arr)))
    scale = max(max_abs, 1e-6)
    norm = (arr / (2.0 * scale)) + 0.5
    signed = float(arr.sum())
    abs_sum = float(np.abs(arr).sum())
    return np.clip(norm, 0.0, 1.0), signed, abs_sum


def save_gradient_panels(
    traj_name: str,
    step_idx: int,
    diff_x: torch.Tensor,
    diff_y: torch.Tensor,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x_rgb, sum_x, sum_x_abs = _diff_map_to_rgb(diff_x)
    y_rgb, sum_y, sum_y_abs = _diff_map_to_rgb(diff_y)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(x_rgb)
    axes[0].set_title(f"dx Σ={sum_x:.2f} |Σ|={sum_x_abs:.2f} (step {step_idx})")
    axes[0].axis("off")
    axes[1].imshow(y_rgb)
    axes[1].set_title(f"dy Σ={sum_y:.2f} |Σ|={sum_y_abs:.2f} (step {step_idx})")
    axes[1].axis("off")
    fig.suptitle(traj_name, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def slugify(name: str, used: Dict[str, int]) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_") or "traj"
    count = used.get(base, 0)
    used[base] = count + 1
    if count == 0:
        return base
    return f"{base}_{count}"


def run(cfg: PlotCfg) -> None:
    set_seed(cfg.seed)
    ensure_out_dir(cfg.out_dir)
    device = pick_device(cfg.device)

    trajectories = collect_trajectories(cfg.traj_root)
    names = sorted(trajectories.keys())
    if cfg.max_traj is not None:
        names = names[: cfg.max_traj]

    slug_counts: Dict[str, int] = {}
    for idx, name in enumerate(names, 1):
        paths = trajectories[name]
        metrics, gradients, gradient_maps = compute_metrics_for_traj(paths, cfg=cfg, device=device)
        steps = list(range(len(metrics["mse"])))
        if not steps:
            continue
        slug = slugify(name, slug_counts)
        out_path = cfg.out_dir / f"{slug}.png"
        grad_path = cfg.out_dir / f"{slug}_grad.png"
        print(f"[{idx}/{len(names)}] {name}: frames={len(paths)} -> {out_path}, {grad_path}")
        plot_metrics(name, steps, metrics, out_path)
        plot_gradient_scatter(name, steps, gradients, grad_path)

        grad_img_dir = cfg.out_dir / "gradients" / slug
        for step_idx, (diff_x, diff_y) in enumerate(gradient_maps):
            img_path = grad_img_dir / f"{step_idx:04d}.png"
            save_gradient_panels(name, step_idx, diff_x, diff_y, img_path)


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
