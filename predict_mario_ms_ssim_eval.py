#!/usr/bin/env python3
"""
Extended visualizer for predict_mario_ms_ssim models.

Adds pairwise frame-difference visualizations on top of the original
predict_mario_ms_ssim_eval.py script, including:
  1) Feature-map L2 difference heatmap (from ResNet18 layer4 features)
  2) Grad-CAM on a feature-space distance score (cosine distance)
  3) Occlusion sensitivity map on the distance score
  4) Future divergence map (difference between *predicted next frames* from two contexts)
  5) Local SSIM map between the two input frames

Output images are saved under out_dir/vis_pairs/ as both per-map PNGs and a
combined panel per pair. Self-distance CSV/plots from the base script remain.

Notes:
- This script keeps the base behavior for rollouts and self-distance. New
  visualizations are computed for automatically chosen *pairs* within each
  sample's mini-trajectory: we compare the last context frame (t=3) vs a later
  frame t=j (configurable via --pair_offset, default 4 steps ahead if present).
- For the “future divergence” map we build two 4-frame contexts: one ending at
  the A frame and one ending at the B frame (if enough frames exist inside the
  sample window). We then predict the next frame from each context and compare
  the two predicted futures with local SSIM.
- Feature-space distance and Grad-CAM use a pretrained ResNet18 as a stable
  perceptual encoder. This avoids assumptions about the internal structure of
  your UNet predictor while remaining informative. You can swap in your own
  encoder by editing `PerceptualEncoder`.

Author: ChatGPT
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import tyro
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from predict_mario_ms_ssim import (
    Mario4to1Dataset,
    UNetPredictor,
    default_transform,
    pick_device,
    unnormalize,
)
from trajectory_utils import list_state_frames, list_traj_dirs

# -----------------------------
# Args
# -----------------------------

@dataclass
class Args:
    traj_dir: str = "data.image_distance.train_levels_1_2"
    out_dir: str = "out.predict_mario_ms_ssim_eval_plus"
    num_samples: int = 8
    max_trajs: Optional[int] = None
    self_distance_max_trajs: Optional[int] = None
    seed: int = 0
    device: Optional[str] = None
    save_name: Optional[str] = None
    rollout_steps: int = 12
    checkpoint: str = "out.predict_mario_ms_ssim/run__2025-10-08_16-44-53/last.pt"

    # New visualization controls
    pair_offset: int = 4  # compare context last frame (t=3) vs frame t=3+offset
    occl_kernel: int = 16
    occl_stride: int = 8
    occl_fill: float = 0.0
    ssim_window: int = 11
    ssim_sigma: float = 1.5
    max_pairs_per_sample: int = 1

# -----------------------------
# Base rollout
# -----------------------------

@torch.no_grad()
def rollout(model: UNetPredictor, context: torch.Tensor, steps: int) -> List[torch.Tensor]:
    """Runs autoregressive prediction starting from 4-frame context.

    Args:
        model: trained predictor in eval mode.
        context: (4, 3, H, W), normalized like the training data.
        steps: number of future frames to predict.
    Returns:
        List of predicted frames (each (3, H, W), normalized).
    """
    preds: List[torch.Tensor] = []
    window = context.clone()
    H, W = context.shape[-2:]
    for _ in range(steps):
        model_input = window.reshape(1, 12, H, W)
        pred = model(model_input)[0]
        preds.append(pred)
        window = torch.cat([window[1:], pred.unsqueeze(0)], dim=0)
    return preds

# -----------------------------
# Perceptual encoder (ResNet18) and utilities
# -----------------------------

class PerceptualEncoder(nn.Module):
    """ResNet18 encoder exposing both spatial features (layer4) and pooled embeddings."""
    def __init__(self, device: torch.device):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Extract layer4 feature map and the final pooled embedding
        self.feat_extractor = create_feature_extractor(
            base,
            return_nodes={
                'layer4': 'feat',
                'avgpool': 'pool',
            }
        ).to(device).eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (feat_map [B,C,Hf,Wf], embedding [B,D])."""
        out = self.feat_extractor(x)
        feat: torch.Tensor = out['feat']
        pool: torch.Tensor = out['pool']  # [B,512,1,1]
        emb = torch.flatten(pool, 1)
        return feat, emb

@torch.no_grad()
def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(a, b)

# -----------------------------
# SSIM helpers (local/patch map and scalar)
# -----------------------------

def _gaussian_window(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = (g / g.sum()).unsqueeze(0)
    window = (g.t() @ g).unsqueeze(0).unsqueeze(0)  # [1,1,ks,ks]
    return window

@torch.no_grad()
def ssim_map(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Return per-pixel SSIM map in [0,1] for grayscale x,y in [0,1].
    x,y: [1,1,H,W]
    """
    device = x.device
    K1, K2 = 0.01, 0.03
    L = 1.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    w = _gaussian_window(window_size, sigma, device)
    pad = window_size // 2
    mu_x = F.conv2d(x, w, padding=pad)
    mu_y = F.conv2d(y, w, padding=pad)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, w, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(y * y, w, padding=pad) - mu_y2
    sigma_xy = F.conv2d(x * y, w, padding=pad) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim = torch.clamp(num / (den + 1e-8), 0.0, 1.0)
    return ssim  # [1,1,H,W]

# -----------------------------
# Heatmaps
# -----------------------------

@torch.no_grad()
def feature_diff_heatmap(encoder: PerceptualEncoder, a: torch.Tensor, b: torch.Tensor, out_hw: Tuple[int,int]) -> torch.Tensor:
    Fa, _ = encoder(a)
    Fb, _ = encoder(b)
    diff = torch.norm(Fa - Fb, dim=1, keepdim=True)  # [1,1,Hf,Wf]
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    heat = F.interpolate(diff, size=out_hw, mode='bilinear', align_corners=False)
    return heat[0,0]

class _GradHook:
    def __init__(self):
        self.activ = None
        self.grads = None
    def fwd(self, m, i, o):
        self.activ = o
    def bwd(self, m, gi, go):
        self.grads = go[0]

@torch.no_grad()
def _prep_for_grad(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()  # enable grads

def gradcam_distance_heatmap(base: resnet18, a: torch.Tensor, b: torch.Tensor, out_hw: Tuple[int,int]) -> Tuple[torch.Tensor, float]:
    """Grad-CAM on cosine distance between pooled features of ResNet18.
    Returns (heatmap [H,W], scalar distance).
    """
    device = a.device
    model = base
    model.fc = nn.Identity()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(True)

    with torch.no_grad():
        emb_a = model(a)
        emb_b = model(b)

    layer4 = model.layer4
    hook = _GradHook()
    h1 = layer4.register_forward_hook(hook.fwd)
    h2 = layer4.register_full_backward_hook(hook.bwd)

    def cam_for_input(input_tensor: torch.Tensor, ref_embed: torch.Tensor) -> Tuple[torch.Tensor, float]:
        model.zero_grad(set_to_none=True)
        hook.activ = None
        hook.grads = None
        out = model(input_tensor)
        score = 1.0 - F.cosine_similarity(out, ref_embed.detach())
        score.mean().backward()
        A = hook.activ
        G = hook.grads
        if A is None or G is None:
            H, W = out_hw
            return torch.zeros((H, W), device=device), float(score.item())
        weights = G.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * A).sum(dim=1, keepdim=True))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = F.interpolate(cam, size=out_hw, mode='bilinear', align_corners=False)[0, 0]
        return cam, float(score.item())

    try:
        cam_a, score_ab = cam_for_input(a, emb_b)
        cam_b, score_ba = cam_for_input(b, emb_a)
    finally:
        h1.remove()
        h2.remove()

    cam = 0.5 * (cam_a + cam_b)
    dist = float(((score_ab + score_ba) * 0.5))
    return cam, dist

@torch.no_grad()
def occlusion_sensitivity_map(encoder: PerceptualEncoder, a: torch.Tensor, b: torch.Tensor, k: int = 16, stride: int = 8, fill: float = 0.0) -> torch.Tensor:
    """Occlusion map of signed cosine-distance change when occluding A's patches."""
    device = a.device
    _, za = encoder(a)
    _, zb = encoder(b)
    base = cosine_distance(za, zb).item()
    _, _, H, W = a.shape
    heat = torch.zeros((H, W), device=device)
    for y in range(0, H - k + 1, stride):
        for x in range(0, W - k + 1, stride):
            a_occ = a.clone()
            a_occ[..., y:y+k, x:x+k] = fill
            _, za_occ = encoder(a_occ)
            d = cosine_distance(za_occ, zb).item()
            delta = base - d
            heat[y:y+k, x:x+k] += delta
    max_abs = heat.abs().max().item()
    if max_abs > 0:
        heat = heat / max_abs
    return heat

@torch.no_grad()
def local_ssim_heat(x: torch.Tensor, y: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """1-SSIM map in [0,1] for RGB x,y in [0,1]."""
    # Convert to luminance
    def to_gray(z: torch.Tensor) -> torch.Tensor:
        r, g, b = z[:,0:1], z[:,1:2], z[:,2:3]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b
    gx = to_gray(x)
    gy = to_gray(y)
    ssim = ssim_map(gx, gy, window, sigma)  # [1,1,H,W]
    heat = 1.0 - ssim
    return heat[0,0]

@torch.no_grad()
def future_divergence_heat(model: UNetPredictor, ctx_a: torch.Tensor, ctx_b: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    ya = rollout(model, ctx_a, steps=1)[0].unsqueeze(0)  # [1,3,H,W]
    yb = rollout(model, ctx_b, steps=1)[0].unsqueeze(0)
    # Unnormalize to [0,1] for SSIM computation
    ya = unnormalize(ya).clamp(0,1)
    yb = unnormalize(yb).clamp(0,1)
    return local_ssim_heat(ya, yb, window, sigma)

# -----------------------------
# Drawing utilities
# -----------------------------

def _to_pil01(frame: torch.Tensor) -> Image.Image:
    # frame: (3,H,W) normalized. Convert to [0,1] then to PIL
    vis = unnormalize(frame.unsqueeze(0)).clamp(0.0, 1.0)[0]
    return TF.to_pil_image(vis)

@torch.no_grad()
def overlay_heatmap(
    rgb: Image.Image,
    heat: torch.Tensor,
    alpha: float = 0.55,
    cmap: str = 'jet',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Image.Image:
    H, W = heat.shape[-2:]
    heat_np = heat.detach().cpu().numpy()
    fig = Figure(figsize=(W / 100, H / 100), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(rgb)
    ax.imshow(heat_np, cmap=cmap, alpha=alpha, interpolation='bilinear', vmin=vmin, vmax=vmax)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    data = buf.reshape(H, W, 4)[..., :3]
    plt.close(fig)
    return Image.fromarray(data)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _placeholder_tile(size: Tuple[int, int], text: str) -> Image.Image:
    tile = Image.new("RGB", size, (28, 28, 28))
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    w, h = _text_size(draw, text, font)
    draw.text(((size[0] - w) / 2, (size[1] - h) / 2), text, fill="white", font=font)
    return tile


@torch.no_grad()
def save_trajectory_panels(
    model: UNetPredictor,
    encoder: PerceptualEncoder,
    seq: torch.Tensor,
    out_dir: Path,
    base_name: str,
    chunk_size: int = 16,
) -> None:
    """Render trajectory visualization panels for an entire sequence."""
    T, C, H, W = seq.shape
    if T <= 0:
        return

    seq_vis = unnormalize(seq).clamp(0, 1)
    gt_imgs: List[Image.Image] = [TF.to_pil_image(seq_vis[i].cpu()) for i in range(T)]

    tf_preds: List[Optional[torch.Tensor]] = [None] * T
    for t in range(4, T):
        ctx = seq[t-4:t].reshape(1, 12, H, W)
        tf_preds[t] = model(ctx)[0]

    auto_list = rollout(model, seq[:4], steps=max(T - 4, 0)) if T > 4 else []
    auto_preds: List[Optional[torch.Tensor]] = [None] * T
    for idx, pred in enumerate(auto_list, start=4):
        if idx < T:
            auto_preds[idx] = pred

    tf_imgs: List[Image.Image] = []
    auto_imgs: List[Image.Image] = []
    size = gt_imgs[0].size
    context_tile = _placeholder_tile(size, "context")
    for t in range(T):
        if tf_preds[t] is None:
            tf_imgs.append(context_tile.copy())
        else:
            vis = unnormalize(tf_preds[t].unsqueeze(0)).clamp(0, 1)[0]
            tf_imgs.append(TF.to_pil_image(vis.cpu()))
        if auto_preds[t] is None:
            auto_imgs.append(context_tile.copy())
        else:
            vis = unnormalize(auto_preds[t].unsqueeze(0)).clamp(0, 1)[0]
            auto_imgs.append(TF.to_pil_image(vis.cpu()))

    diff_global: List[torch.Tensor] = []
    diff_prev: List[torch.Tensor] = []
    first_frame = seq_vis[0].unsqueeze(0)
    zero_heat = torch.zeros((H, W), device=seq.device)
    for t in range(T):
        if t == 0:
            diff_global.append(zero_heat)
            diff_prev.append(zero_heat)
            continue
        diff_global.append(feature_diff_heatmap(encoder, seq_vis[t:t+1], first_frame, (H, W)))
        prev_frame = seq_vis[t-1:t]
        diff_prev.append(feature_diff_heatmap(encoder, seq_vis[t:t+1], prev_frame, (H, W)))

    diff_global_imgs = [overlay_heatmap(gt_imgs[t], diff_global[t]) for t in range(T)]
    diff_prev_imgs = [overlay_heatmap(gt_imgs[t], diff_prev[t]) for t in range(T)]

    rows: List[Tuple[str, List[Image.Image]]] = [
        ("Ground Truth", gt_imgs),
        ("Pred (GT ctx)", tf_imgs),
        ("Pred (pred ctx)", auto_imgs),
        ("FeatΔ vs t0", diff_global_imgs),
        ("FeatΔ vs prev", diff_prev_imgs),
    ]

    label_w = 140
    gap = 6
    header_h = 30
    rows_n = len(rows)
    tile_w, tile_h = size

    font = ImageFont.load_default()

    for chunk_idx, start in enumerate(range(0, T, chunk_size)):
        end = min(start + chunk_size, T)
        cols = end - start
        panel_w = label_w + gap + cols * tile_w + (cols - 1) * gap + gap
        panel_h = header_h + rows_n * tile_h + (rows_n + 1) * gap
        panel = Image.new("RGB", (panel_w, panel_h), (10, 10, 10))
        draw = ImageDraw.Draw(panel)
        title = f"{base_name} frames {start}-{end-1}"
        title_w, title_h = _text_size(draw, title, font)
        draw.text(((panel_w - title_w) / 2, (header_h - title_h) / 2), title, fill="white", font=font)

        for r, (label, imgs) in enumerate(rows):
            row_y = header_h + gap + r * (tile_h + gap)
            label_wt, label_ht = _text_size(draw, label, font)
            draw.text((max(0, (label_w - label_wt) / 2), row_y + (tile_h - label_ht) / 2), label, fill="white", font=font)
            for c, frame_idx in enumerate(range(start, end)):
                img = imgs[frame_idx]
                x = label_w + gap + c * (tile_w + gap)
                panel.paste(img, (x, row_y))

        out_path = out_dir / f"{base_name}__traj_{chunk_idx:02d}.png"
        panel.save(out_path)

# -----------------------------
# Pair builder from a single dataset sample
# -----------------------------

@torch.no_grad()
def build_pair_contexts(frames4: torch.Tensor, targets: torch.Tensor, offset: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return (img_a, img_b, ctx_a, ctx_b) where:
        - img_a = frames4[-1]
        - img_b = frame at time t = 3 + offset if available
        - ctx_a = frames4
        - ctx_b = the 4-frame context ending at img_b, if available
    """
    # frames4: (4,3,H,W), targets: (S,3,H,W)
    img_a = frames4[-1]
    t_b = 3 + offset
    avail = 3 + targets.shape[0]  # last available absolute t index inside sample window
    if t_b > avail:
        return None

    # Construct img_b
    if t_b == 3:
        img_b = frames4[-1]
    else:
        j = t_b - 4  # targets[t] starts at absolute time t=4
        if j >= targets.shape[0]:
            return None
        img_b = targets[j]

    # Build ctx_b (needs previous 3 frames before img_b)
    # We can compose from frames4 and targets sequence if offset >= 1
    if offset >= 1:
        # We need frames at times t_b-3, t_b-2, t_b-1, t_b
        seq: List[torch.Tensor] = []
        for tb in range(t_b - 3, t_b + 1):
            if tb <= 3:
                seq.append(frames4[tb])
            else:
                seq.append(targets[tb - 4])
        ctx_b = torch.stack(seq, dim=0)
    else:
        # offset <= 0 edge cases (use same ctx for both)
        ctx_b = frames4.clone()

    ctx_a = frames4
    return img_a, img_b, ctx_a, ctx_b

# -----------------------------
# Original self-distance computation (kept, with path tweak)
# -----------------------------

@torch.no_grad()
def compute_self_distance_metrics(
    traj_dir: Path,
    device: torch.device,
    out_dir: Path,
    max_trajs: Optional[int] = None,
) -> None:
    transform = default_transform()
    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    backbone.eval().to(device)

    traj_dirs = list_traj_dirs(traj_dir)
    if max_trajs is not None:
        traj_dirs = traj_dirs[:max_trajs]
    if not traj_dirs:
        print(f"[self-distance] no trajectory directories found in {traj_dir}")
        return

    print(f"[self-distance] found {len(traj_dirs)} trajectories in {traj_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    traj_bar = tqdm(
        traj_dirs,
        desc="[self-distance] trajectories",
        unit="traj",
        total=len(traj_dirs),
        position=0,
    )
    for traj_path in traj_bar:
        states_dir = traj_path / "states"
        if not states_dir.is_dir():
            continue
        frame_paths = list_state_frames(states_dir)
        if len(frame_paths) < 2:
            continue

        embeddings: List[torch.Tensor] = []
        frame_bar = tqdm(
            frame_paths,
            desc=f"Embedding {traj_path.name}",
            unit="frame",
            leave=False,
            total=len(frame_paths),
            position=1,
        )
        for frame_path in frame_bar:
            with Image.open(frame_path).convert("RGB") as img:
                frame_tensor = transform(img).unsqueeze(0).to(device)
            feat = backbone(frame_tensor).squeeze(0).cpu()
            embeddings.append(feat)
        frame_bar.close()

        h0 = embeddings[0]
        l2_vals: List[float] = []
        cos_vals: List[float] = []
        for feat in embeddings:
            diff = feat - h0
            l2_vals.append(float(diff.norm().item()))
            cos_vals.append(float(1.0 - F.cosine_similarity(feat.unsqueeze(0), h0.unsqueeze(0)).item()))

        rel_name = traj_path.relative_to(traj_dir)
        traj_name = rel_name.as_posix().replace('/', '_')

        if False:
            csv_path = out_dir / f"{traj_name}_self_distance.csv"
            with csv_path.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["frame_index", "l2_distance", "cosine_distance"])
                for idx, (d_l2, d_cos) in enumerate(zip(l2_vals, cos_vals)):
                    writer.writerow([idx, d_l2, d_cos])

        if len(l2_vals) > 1:
            fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
            indices = list(range(len(l2_vals)))
            axes[0].plot(indices, l2_vals, marker='o')
            axes[0].set_ylabel('L2 distance')
            axes[0].set_title(f'{rel_name}: frame 0 vs t')
            axes[0].grid(True, linestyle='--', linewidth=0.4)

            axes[1].plot(indices, cos_vals, marker='o')
            axes[1].set_ylabel('Cosine distance')
            axes[1].set_xlabel('Frame index t')
            axes[1].grid(True, linestyle='--', linewidth=0.4)

            fig.tight_layout()
            fig.savefig(out_dir / f"{traj_name}_self_distance.png", dpi=150)
            plt.close(fig)

        traj_bar.write(f"[self-distance] processed trajectory {traj_name} ({len(frame_paths)} frames)")
    traj_bar.close()

# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = tyro.cli(Args)

    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if args.rollout_steps < 1:
        raise ValueError("rollout_steps must be >= 1")

    device = pick_device(args.device)
    print(f"[device] using {device}")

    ds = Mario4to1Dataset(args.traj_dir, max_trajs=args.max_trajs, rollout=args.rollout_steps)
    if len(ds) == 0:
        raise RuntimeError(f"No samples found in trajectory directory: {args.traj_dir}")
    eval_n = min(args.num_samples, len(ds))
    print(f"Dataset size: {len(ds)} | evaluating {eval_n} samples")

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(ds), generator=generator)[:eval_n].tolist()

    # Load predictor
    model = UNetPredictor().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Perceptual encoder
    enc = PerceptualEncoder(device)

    # Output dirs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_dir = out_dir / "vis_pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = out_dir / "traj_chunks"
    traj_dir.mkdir(parents=True, exist_ok=True)

    # Iterate samples and build pairs
    pair_count = 0
    sample_total = len(indices)
    sample_bar = tqdm(indices, desc="Evaluating samples", unit="sample", total=sample_total)
    for row_idx, ds_idx in enumerate(sample_bar):
        sample_bar.set_description(
            f"Evaluating sample {row_idx + 1}/{sample_total} (ds_idx={ds_idx})"
        )
        x_stack, targets = ds[ds_idx]
        frames4 = x_stack.view(4, 3, *x_stack.shape[-2:]).to(device)
        targets = targets.to(device)

        built = build_pair_contexts(frames4, targets, args.pair_offset)
        if built is None:
            print(f"[warn] sample {ds_idx}: not enough frames for offset={args.pair_offset}")
            continue
        img_a, img_b, ctx_a, ctx_b = built

        # Prepare 1x batches for encoders
        A = unnormalize(img_a.unsqueeze(0)).clamp(0,1).to(device)
        B = unnormalize(img_b.unsqueeze(0)).clamp(0,1).to(device)

        H, W = A.shape[-2:]

        # 1) Feature-map difference heatmap
        feat_diff = feature_diff_heatmap(enc, A, B, (H, W))

        # 2) Grad-CAM on distance
        base_resnet = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
        cam, dist_scalar = gradcam_distance_heatmap(base_resnet, A, B, (H, W))

        # 3) Occlusion sensitivity map
        occl = occlusion_sensitivity_map(enc, A, B, k=args.occl_kernel, stride=args.occl_stride, fill=args.occl_fill)

        # 4) Future divergence heatmap
        fut = future_divergence_heat(model, ctx_a, ctx_b, window=args.ssim_window, sigma=args.ssim_sigma)

        # 5) Local SSIM map on the pair itself
        pair_ssim = local_ssim_heat(A, B, window=args.ssim_window, sigma=args.ssim_sigma)

        # Compose and save visuals
        a_pil = _to_pil01(img_a.cpu())
        b_pil = _to_pil01(img_b.cpu())
        feat_img = overlay_heatmap(a_pil, feat_diff)
        cam_img = overlay_heatmap(a_pil, cam)
        occl_img = overlay_heatmap(a_pil, occl, cmap='coolwarm', vmin=-1.0, vmax=1.0)
        fut_img = overlay_heatmap(a_pil, fut)
        ssim_img = overlay_heatmap(a_pil, pair_ssim)

        pair_base = f"sample{row_idx:03d}_idx{ds_idx}_off{args.pair_offset}"
        # Combined panel only (skip saving individual tiles per request)
        tiles = [a_pil, b_pil, feat_img, cam_img, occl_img, fut_img, ssim_img]
        labels = [
            "A (t=3)",
            f"B (t=3+{args.pair_offset})",
            "Feature-map Δ",
            f"Grad-CAM (dist={dist_scalar:.3f})",
            "Occlusion Δdist",
            "Future divergence",
            "1-SSIM (A vs B)",
        ]
        cols = 4
        rows = 2
        tile_w, tile_h = a_pil.size
        gap = 4
        panel_w = cols*tile_w + (cols-1)*gap
        panel_h = rows*tile_h + (rows-1)*gap + 24  # header band
        panel = Image.new("RGB", (panel_w, panel_h), (12,12,12))
        # draw tiles
        for i, img in enumerate(tiles[:cols*rows]):
            r = i // cols
            c = i % cols
            x = c*(tile_w+gap)
            y = r*(tile_h+gap) + 24
            panel.paste(img, (x,y))
        # put labels using matplotlib (quick overlay)
        fig = Figure(figsize=(panel_w / 100, panel_h / 100), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(panel)
        for i, text in enumerate(labels[:cols*rows]):
            r = i // cols
            c = i % cols
            x = c*(tile_w+gap) + 6
            y = r*(tile_h+gap) + 18
            ax.text(x, y, text, color='white', fontsize=10, weight='bold', va='top', ha='left')
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        data = buf.reshape(panel_h, panel_w, 4)[..., :3]
        plt.close(fig)
        panel_img = Image.fromarray(data)
        panel_img.save(pair_dir / f"{pair_base}__panel.png")

        # Trajectory chunk visualization
        full_seq = torch.cat([frames4, targets], dim=0)
        traj_base = f"sample{row_idx:03d}_idx{ds_idx}"
        save_trajectory_panels(model, enc, full_seq, traj_dir, traj_base)

        pair_count += 1
        if pair_count >= args.max_pairs_per_sample * eval_n:
            break

    # Also run original self-distance analysis for convenience
    compute_self_distance_metrics(
        Path(args.traj_dir),
        device,
        out_dir / "self_distance",
        max_trajs=args.self_distance_max_trajs,
    )

    print(f"Saved {pair_count} pair panels to {pair_dir}")


if __name__ == "__main__":
    main()
