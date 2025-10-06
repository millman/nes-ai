#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latent Intervention Learning trainer with masked residual dynamics.

Run:
  python latent_intervention_learning.py --data_root traj_dumps --out_dir out.latent_intervention
"""

from __future__ import annotations

import argparse
import contextlib
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from recon import (
    H,
    W,
    PairFromTrajDataset,
    TileSpec,
    load_frame_as_tensor as base_load_frame_as_tensor,
    render_image_grid,
    set_seed,
    short_traj_state_label,
    to_float01,
)
from recon.utils import grad_norm, psnr_01, tensor_to_pil

def _normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    return t

def _denormalize_tensor(t: torch.Tensor) -> torch.Tensor:
    return t

def load_frame_as_tensor(path: Path) -> torch.Tensor:
    return base_load_frame_as_tensor(path, normalize=_normalize_tensor)

def _to_pil(t: torch.Tensor) -> Image.Image:
    return tensor_to_pil(t, denormalize=_denormalize_tensor)

def _psnr_01(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return psnr_01(_denormalize_tensor(x), _denormalize_tensor(y), eps)

def _grad_norm(module: nn.Module) -> float:
    return grad_norm(module)

# ---------------------------
# Helpers
# ---------------------------

def tv_loss(img: Tensor) -> Tensor:
    # total variation for smooth/compact masks (expects Bx1xH xW or BxCxH xW)
    dh = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean()
    dw = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean()
    return dh + dw

def kl_gaussian_standard(mu: Tensor, logvar: Tensor) -> Tensor:
    # KL( N(mu, diag(exp(logvar))) || N(0, I) )
    return 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar).sum(dim=-1).mean()

def cosine_sim(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    a = a.flatten(1); b = b.flatten(1)
    return (F.normalize(a, dim=1) * F.normalize(b, dim=1)).sum(dim=1).mean()

# ---------------------------
# Small VAE (latent H,W ~ 240/8 = 30 if stride total=8)
# You can swap this with your LDM VAE if you have one.
# ---------------------------

class Encoder(nn.Module):
    def __init__(self, in_ch=3, z_ch=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),   # 240->120->60
            nn.Conv2d(64, z_ch, 4, 2, 1),           # 60 -> 30
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)  # (B, z_ch, H=30, W=30 for 240x240 input)

class Decoder(nn.Module):
    def __init__(self, z_ch=64, out_ch=3):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(z_ch, 64, 4, 2, 1), nn.SiLU(),   # 30->60
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),     # 60->120
            nn.ConvTranspose2d(32, out_ch, 4, 2, 1),            # 120->240
        )
    def forward(self, z: Tensor) -> Tensor:
        return torch.sigmoid(self.deconv(z))  # images in [0,1]

# (Optional) add VAE stochasticity; here we keep it deterministic for clarity.

# ---------------------------
# Amortized inference for latent intervention u_t
# q(u | z_t, z_{t+1}, z_{t-1}...) -> N(mu, diag(exp(logvar)))
# ---------------------------

class UInference(nn.Module):
    def __init__(self, z_ch=64, du=4, k_hist: int = 1):
        """
        k_hist = number of past frames to include (z_{t-k_hist+1: t}) in addition to z_{t+1}
        Input channels ~ (k_hist + 1) * z_ch
        """
        super().__init__()
        # When z_hist is provided: [z_hist, z_t, z_tp1, delta] = (k_hist + 3) * z_ch
        # When z_hist is empty: [z_t, z_tp1, delta] = 3 * z_ch
        # Use the larger value to support both cases, or fix to 3*z_ch if never using history
        in_ch = 3 * z_ch  # z_t, z_tp1, delta (when no history provided)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(128, 128, 3, 2, 1), nn.SiLU(),  # down a bit
            nn.Conv2d(128, 128, 3, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mu = nn.Linear(128, du)
        self.logvar = nn.Linear(128, du)

        # simple "no-op" / exogenous change detector w_t \in [0,1]
        self.noop_head = nn.Sequential(nn.Linear(128, 1))

    def forward(self, z_hist: Tensor, z_t: Tensor, z_tp1: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        z_hist: (B, k_hist*z_ch, H, W) or zeros if none
        z_t, z_tp1: (B, z_ch, H, W)
        """
        delta = z_tp1 - z_t
        x = torch.cat([z_t, z_tp1, delta], dim=1)  # Always use this version since training doesn't provide history
        h = self.net(x).flatten(1)
        mu, logvar = self.mu(h), self.logvar(h)
        w_noop = torch.sigmoid(self.noop_head(h))  # (B,1)
        return mu, logvar, w_noop

# ---------------------------
# Dynamics: base + masked residual (driven by u)
# Mask produced via cross-attention: u (query) -> z tokens (keys)
# ---------------------------

class CrossAttnMask(nn.Module):
    def __init__(self, z_ch=64, d_q=32):
        super().__init__()
        self.to_q = nn.Linear(d_q, z_ch)   # project u to query (dim ~ z_ch)
        self.to_k = nn.Conv2d(z_ch, z_ch, 1)
        # We use attention weights (softmax over spatial tokens) as the mask itself.

    def forward(self, z: Tensor, u: Tensor) -> Tensor:
        """
        z: (B, C, H, W); u: (B, du)
        returns M: (B, 1, H, W) in [0,1]
        """
        B, C, H, W = z.shape
        K = self.to_k(z).flatten(2)                       # (B, C, H*W)
        q = self.to_q(u).unsqueeze(1)                     # (B, 1, C)
        attn_logits = torch.bmm(q, K) / (C ** 0.5)        # (B, 1, H*W)
        attn = torch.softmax(attn_logits, dim=-1)
        M = attn.view(B, 1, H, W)
        return M

class BaseResidualDynamics(nn.Module):
    def __init__(self, z_ch=64, du=4):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(z_ch, 128, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(128, z_ch, 3, 1, 1),
        )
        self.u_embed = nn.Sequential(nn.Linear(du, 32), nn.SiLU())
        self.mask = CrossAttnMask(z_ch=z_ch, d_q=32)
        # residual uses depthwise + pointwise conv; light bottleneck
        self.resid = nn.Sequential(
            nn.Conv2d(z_ch, z_ch, 3, 1, 1, groups=z_ch),  # depthwise
            nn.SiLU(),
            nn.Conv2d(z_ch, z_ch, 1, 1, 0),               # pointwise
        )

    def forward(self, z_t: Tensor, u: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        returns: z_pred, Δz_base, (M * Δz_resid)
        """
        Δz_base  = self.base(z_t)
        u_e      = self.u_embed(u)
        M        = self.mask(z_t, u_e)                    # (B,1,H,W)
        Δz_resid = self.resid(z_t)
        z_pred   = z_t + Δz_base + M * Δz_resid
        return z_pred, Δz_base, M * Δz_resid, M

# ---------------------------
# Full model wrapper
# ---------------------------

class LatentInterventionModel(nn.Module):
    def __init__(self, img_ch=3, z_ch=64, du=4, k_hist=1):
        super().__init__()
        self.enc = Encoder(img_ch, z_ch)
        self.dec = Decoder(z_ch, img_ch)
        self.inf_u = UInference(z_ch, du, k_hist)
        self.dyn = BaseResidualDynamics(z_ch, du)
        self.du = du
        self.k_hist = k_hist

    def infer_u(self, z_hist, z_t, z_tp1):
        mu, logvar, w_noop = self.inf_u(z_hist, z_t, z_tp1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        u = mu + eps * std  # reparameterization
        return u, mu, logvar, w_noop

    def forward(self, x_hist: Tensor, x_t: Tensor, x_tp1: Tensor) -> Dict[str, Tensor]:
        """
        x_hist: optional history tensor cat along channel dim: (B, k_hist*3, H, W) or zeros
        """
        # Encode to latent
        z_hist = self.enc(x_hist) if x_hist.numel() else x_hist   # allow empty
        z_t = self.enc(x_t)
        z_tp1 = self.enc(x_tp1)

        # Infer intervention u and no-op weight
        u, mu, logvar, w_noop = self.infer_u(z_hist, z_t, z_tp1)  # (B,du), (B,1)

        # Factual prediction (with intervention)
        z_pred, dB, dR_m, M = self.dyn(z_t, u)
        x_pred = self.dec(z_pred)

        # Counterfactual (no intervention): zero residual path
        z_pred_off = z_t + dB                      # same base, M*dR disabled
        x_pred_off = self.dec(z_pred_off)

        return {
            "x_pred": x_pred, "x_pred_off": x_pred_off,
            "z_t": z_t, "z_tp1": z_tp1, "z_pred": z_pred, "z_pred_off": z_pred_off,
            "Δz_base": dB, "Δz_resid_masked": dR_m, "M": M,
            "u": u, "mu": mu, "logvar": logvar, "w_noop": w_noop
        }

# ---------------------------
# Loss computation
# ---------------------------

def compute_losses(batch: Dict[str, Tensor],
                   out: Dict[str, Tensor],
                   λ_per: float = 0.0,           # set >0 if you plug an LPIPS/feature loss
                   λ_cf: float = 1.0,
                   λ_u: float = 1e-3,
                   λ_M: float = 1e-3,
                   λ_tv: float = 1e-3,
                   λ_ortho: float = 1e-1) -> Dict[str, Tensor]:
    x_tp1 = batch["x_tp1"]    # (B,3,240,240)
    x_pred = out["x_pred"]
    x_pred_off = out["x_pred_off"]

    # Recon (L1; add LPIPS/SSIM if desired)
    L_rec = F.l1_loss(x_pred, x_tp1)

    # Counterfactual weighted by no-op detector
    w = out["w_noop"].detach()      # stopgrad
    L_cf = (w * (x_pred_off - x_tp1).abs().mean(dim=(1,2,3), keepdim=True)).mean()

    # KL on u
    L_kl = kl_gaussian_standard(out["mu"], out["logvar"])

    # Sparsity & locality
    L_u = out["u"].abs().mean()
    L_M = out["M"].abs().mean()
    L_tv = tv_loss(out["M"])

    # Orthogonality between base and masked residual deltas in latent space
    L_ortho = cosine_sim(out["Δz_base"].detach(), out["Δz_resid_masked"]).pow(2)

    # Total
    loss = (L_rec
            + λ_cf * L_cf
            + L_kl
            + λ_u * L_u
            + λ_M * L_M
            + λ_tv * L_tv
            + λ_ortho * L_ortho)

    return dict(
        loss=loss, L_rec=L_rec, L_cf=L_cf, L_kl=L_kl,
        L_u=L_u, L_M=L_M, L_tv=L_tv, L_ortho=L_ortho,
        infl=((x_pred - x_pred_off).pow(2).mean(dim=(1,2,3)) / (x_tp1 - batch["x_t"]).pow(2).mean(dim=(1,2,3)).clamp_min(1e-6)).mean()
    )

# ---------------------------
# One training step (sketch)
# ---------------------------

def train_step(model: LatentInterventionModel, optimizer, batch: Dict[str, Tensor], scaler=None) -> Dict[str, float]:
    model.train()
    x_t, x_tp1 = batch["x_t"], batch["x_tp1"]        # (B,3,240,240)
    x_hist = batch.get("x_hist", torch.zeros(0, device=x_t.device))  # optional

    with torch.autocast(device_type=x_t.device.type, dtype=torch.bfloat16 if x_t.device.type=="cuda" else torch.float32):
        out = model(x_hist, x_t, x_tp1)
        losses = compute_losses(batch, out)

    optimizer.zero_grad(set_to_none=True)
    if scaler:
        scaler.scale(losses["loss"]).grad_scaler_step = None
        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer); scaler.update()
    else:
        losses["loss"].backward()
        optimizer.step()

    return {k: float(v.detach().item()) for k, v in losses.items()}

# ---------------------------
# Visualization hooks (minimal)
# ---------------------------

@torch.no_grad()
def make_viz_panel(model: LatentInterventionModel, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
    model.eval()
    x_t, x_tp1 = batch["x_t"], batch["x_tp1"]
    x_hist = batch.get("x_hist", torch.zeros(0, device=x_t.device))
    out = model(x_hist, x_t, x_tp1)
    # key maps
    M_up = F.interpolate(out["M"], size=x_t.shape[-2:], mode="bilinear", align_corners=False)
    delta_u = (out["x_pred"] - out["x_pred_off"]).abs()  # what intervention changed
    panel = {
        "x_t": x_t,
        "x_tp1": x_tp1,
        "x_pred": out["x_pred"],
        "x_pred_off": out["x_pred_off"],
        "M_overlay": (x_t * 0.5 + M_up.expand_as(x_t) * 0.5).clamp(0,1),
        "delta_u": delta_u.clamp(0,1),
    }
    return panel

# ---------------------------
# Visualization
# ---------------------------
@torch.no_grad()
def save_viz_grid(
    model: LatentInterventionModel,
    x_t: torch.Tensor, x_tp1: torch.Tensor,
    pathsA: List[str], pathsB: List[str],
    out_path: Path, device: torch.device
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Bsz = min(x_t.shape[0], 8)
    x_t = x_t[:Bsz].contiguous()
    x_tp1 = x_tp1[:Bsz].contiguous()
    pA = pathsA[:Bsz]
    pB = pathsB[:Bsz]

    x_t_dev = to_float01(x_t, device, non_blocking=False)
    x_tp1_dev = to_float01(x_tp1, device, non_blocking=False)

    x_hist = torch.zeros(0, device=device)
    out = model(x_hist, x_t_dev, x_tp1_dev)

    M_up = F.interpolate(out["M"], size=x_t_dev.shape[-2:], mode="bilinear", align_corners=False)
    delta_u = (out["x_pred"] - out["x_pred_off"]).abs()

    rows: List[List[TileSpec]] = []
    for idx in range(Bsz):
        labelA = short_traj_state_label(pA[idx])
        labelB = short_traj_state_label(pB[idx])

        w_noop_val = out["w_noop"][idx, 0].item()

        row: List[TileSpec] = [
            TileSpec(
                image=_to_pil(x_t[idx]),
                top_label=f"{labelA} (t)",
            ),
            TileSpec(
                image=_to_pil(out["x_pred"][idx]),
                top_label=f"{labelB} pred (w/ u)",
                top_color=(200, 255, 200),
            ),
            TileSpec(
                image=_to_pil(out["x_pred_off"][idx]),
                top_label=f"{labelB} pred (no u)",
                top_color=(255, 200, 200),
            ),
            TileSpec(
                image=_to_pil(x_tp1[idx]),
                top_label=f"{labelB} (t+1)",
            ),
            TileSpec(
                image=_to_pil((x_t[idx] * 0.5 + M_up[idx].cpu() * 0.5).clamp(0,1)),
                top_label=f"mask overlay",
                top_color=(255, 255, 200),
            ),
            TileSpec(
                image=_to_pil(delta_u[idx].cpu().clamp(0,1)),
                top_label=f"Δ(u) w={w_noop_val:.2f}",
                top_color=(220, 220, 255),
            ),
        ]
        rows.append(row)

    render_image_grid(rows, out_path, tile_size=(W, H))

# ---------------------------
# Training configuration
# ---------------------------
@dataclass
class TrainCfg:
    data_root: Path
    out_dir: Path = Path("out.latent_intervention")
    img_ch: int = 3
    z_ch: int = 64
    du: int = 4
    k_hist: int = 1
    batch_size: int = 32
    epochs: int = 5
    lr: float = 3e-4
    seed: int = 0
    num_workers: int = 4
    device: Optional[str] = None
    max_step_gap: int = 20
    allow_cross_traj: bool = False
    p_cross_traj: float = 0.0
    use_foreach_optim: bool = True

    # Loss weights
    lambda_cf: float = 1.0
    lambda_u: float = 1e-3
    lambda_M: float = 1e-3
    lambda_tv: float = 1e-3
    lambda_ortho: float = 1e-1

    # Viz/debug
    viz_every: int = 500
    log_every: int = 50

# ---------------------------
# Device picker
# ---------------------------
def pick_device(pref: Optional[str]) -> torch.device:
    if pref:
        dev = torch.device(pref)
        if dev.type == "cuda":
            raise ValueError("CUDA is not supported by this trainer; use 'cpu' or 'mps'.")
        return dev
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ---------------------------
# Training loop
# ---------------------------
def train(cfg: TrainCfg):
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "checkpoints").mkdir(exist_ok=True, parents=True)
    (cfg.out_dir / "viz").mkdir(exist_ok=True, parents=True)

    ds_tr = PairFromTrajDataset(
        cfg.data_root,
        "train",
        0.95,
        cfg.seed,
        cfg.max_step_gap,
        cfg.allow_cross_traj,
        cfg.p_cross_traj,
        load_frame=load_frame_as_tensor,
    )
    ds_va = PairFromTrajDataset(
        cfg.data_root,
        "val",
        0.95,
        cfg.seed,
        cfg.max_step_gap,
        cfg.allow_cross_traj,
        cfg.p_cross_traj,
        load_frame=load_frame_as_tensor,
    )

    use_pin = False
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers, pin_memory=use_pin, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=use_pin, drop_last=False)

    print(f"[Device] {device} | [Data] train pairs≈{len(ds_tr)}  val pairs≈{len(ds_va)}")

    model = LatentInterventionModel(
        img_ch=cfg.img_ch,
        z_ch=cfg.z_ch,
        du=cfg.du,
        k_hist=cfg.k_hist
    ).to(device)

    optim_kwargs = {"lr": cfg.lr, "weight_decay": 1e-4}
    if cfg.use_foreach_optim:
        optim_kwargs["foreach"] = True
    try:
        opt = torch.optim.AdamW(model.parameters(), **optim_kwargs)
    except TypeError:
        if optim_kwargs.pop("foreach", None) is not None:
            opt = torch.optim.AdamW(model.parameters(), **optim_kwargs)
        else:
            raise

    # AMP only on CUDA
    amp_autocast = contextlib.nullcontext
    scaler = GradScaler(enabled=False)

    global_step = 0
    best_val = float("inf")

    # Running windows
    win = 50
    q_rec, q_cf, q_kl, q_u, q_M, q_tv, q_ortho, q_ltot = [deque(maxlen=win) for _ in range(8)]
    q_psnr, q_infl = deque(maxlen=win), deque(maxlen=win)
    q_gmodel = deque(maxlen=win)
    q_data_time, q_forward_time, q_backward_time = deque(maxlen=win), deque(maxlen=win), deque(maxlen=win)
    step_time_accum = 0.0
    step_time_count = 0

    start_time = time.monotonic()

    for ep in range(1, cfg.epochs+1):
        model.train()
        run_loss = run_n = 0.0

        for A, B, pathsA, pathsB in dl_tr:
            step_start = time.perf_counter()
            need_viz = (cfg.viz_every > 0) and (global_step % cfg.viz_every == 0)
            A_cpu = A.detach().clone() if need_viz else None
            B_cpu = B.detach().clone() if need_viz else None

            data_start = time.perf_counter()
            A = to_float01(A, device)
            B = to_float01(B, device)
            data_time = time.perf_counter() - data_start

            forward_start = time.perf_counter()
            with amp_autocast():
                x_hist = torch.zeros(0, device=device)
                batch = {"x_t": A, "x_tp1": B, "x_hist": x_hist}
                out = model(x_hist, A, B)
                losses = compute_losses(
                    batch, out,
                    λ_cf=cfg.lambda_cf,
                    λ_u=cfg.lambda_u,
                    λ_M=cfg.lambda_M,
                    λ_tv=cfg.lambda_tv,
                    λ_ortho=cfg.lambda_ortho
                )
                loss = losses["loss"]
            forward_time = time.perf_counter() - forward_start

            backward_start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            g_model = _grad_norm(model)

            scaler.step(opt)
            scaler.update()
            backward_time = time.perf_counter() - backward_start

            step_time = time.perf_counter() - step_start
            step_time_accum += step_time
            step_time_count += 1

            run_loss += loss.item() * A.shape[0]
            run_n += A.shape[0]

            with torch.no_grad():
                psnr_val = _psnr_01(out["x_pred"], B).item()

                q_rec.append(float(losses["L_rec"].item()))
                q_cf.append(float(losses["L_cf"].item()))
                q_kl.append(float(losses["L_kl"].item()))
                q_u.append(float(losses["L_u"].item()))
                q_M.append(float(losses["L_M"].item()))
                q_tv.append(float(losses["L_tv"].item()))
                q_ortho.append(float(losses["L_ortho"].item()))
                q_ltot.append(float(loss.item()))
                q_psnr.append(psnr_val)
                q_infl.append(float(losses["infl"].item()))
                q_gmodel.append(g_model)
                q_data_time.append(data_time * 1000)
                q_forward_time.append(forward_time * 1000)
                q_backward_time.append(backward_time * 1000)

            if (cfg.log_every > 0) and (global_step % cfg.log_every == 0):
                avg = lambda q: (sum(q)/len(q)) if len(q) else 0.0
                avg_step_time = (step_time_accum / step_time_count) if step_time_count else 0.0
                step_time_accum = 0.0
                step_time_count = 0
                if avg_step_time > 0:
                    throughput = (cfg.batch_size / avg_step_time)
                else:
                    throughput = 0.0
                elapsed = int(time.monotonic() - start_time)
                h = elapsed // 3600
                m = (elapsed % 3600) // 60
                s = elapsed % 60
                print(
                    f"[{h:02d}:{m:02d}:{s:02d}] "
                    f"ep {ep:02d} step {global_step:06d} | "
                    f"loss {avg(q_ltot):.4f} | "
                    f"Lrec {avg(q_rec):.4f} Lcf {avg(q_cf):.4f} Lkl {avg(q_kl):.5f} "
                    f"Lu {avg(q_u):.5f} LM {avg(q_M):.5f} Ltv {avg(q_tv):.5f} Lortho {avg(q_ortho):.4f} | "
                    f"PSNR {avg(q_psnr):.2f}dB infl {avg(q_infl):.3f} | "
                    f"∥g∥ {avg(q_gmodel):.3e} | "
                    f"timing: data {avg(q_data_time):.1f}ms fwd {avg(q_forward_time):.1f}ms bwd {avg(q_backward_time):.1f}ms | "
                    f"step_time {avg_step_time*1000:.1f} ms  ({throughput:.1f} samples/s)"
                )

            if global_step % cfg.viz_every == 0:
                model.eval()
                out_path = cfg.out_dir / "viz" / f"ep{ep:02d}_step{global_step:06d}.png"
                save_viz_grid(
                    model,
                    A_cpu if A_cpu is not None else A.detach().cpu(),
                    B_cpu if B_cpu is not None else B.detach().cpu(),
                    list(pathsA), list(pathsB),
                    out_path,
                    device=device,
                )
                model.train()

            global_step += 1

        tr_loss = run_loss / max(1, run_n)
        print(f"[ep {ep:02d}] train: loss={tr_loss:.4f}")

        # ---- Validation ----
        model.eval()
        va_loss = va_n = 0.0
        with torch.no_grad():
            for A, B, _, _ in dl_va:
                A = to_float01(A, device, non_blocking=False)
                B = to_float01(B, device, non_blocking=False)
                x_hist = torch.zeros(0, device=device)
                batch = {"x_t": A, "x_tp1": B, "x_hist": x_hist}
                out = model(x_hist, A, B)
                losses = compute_losses(
                    batch, out,
                    λ_cf=cfg.lambda_cf,
                    λ_u=cfg.lambda_u,
                    λ_M=cfg.lambda_M,
                    λ_tv=cfg.lambda_tv,
                    λ_ortho=cfg.lambda_ortho
                )
                va_loss += losses["loss"].item() * A.shape[0]
                va_n += A.shape[0]
        va_loss /= max(1, va_n)
        print(f"[ep {ep:02d}]   val: loss={va_loss:.4f}")

        # Save checkpoint
        ckpt = {
            "epoch": ep,
            "model": model.state_dict(),
            "val_loss": va_loss,
            "cfg": vars(cfg)
        }
        torch.save(ckpt, cfg.out_dir / "last.ckpt")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, cfg.out_dir / "best.ckpt")
            print(f"[ep {ep:02d}] saved best (val loss={best_val:.4f})")
        torch.save(ckpt, cfg.out_dir / "checkpoints" / f"ep{ep:02d}.ckpt")

    print("[done]")

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("out.latent_intervention"))
    ap.add_argument("--img_ch", type=int, default=3)
    ap.add_argument("--z_ch", type=int, default=64)
    ap.add_argument("--du", type=int, default=4)
    ap.add_argument("--k_hist", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max_step_gap", type=int, default=20)
    ap.add_argument("--allow_cross_traj", action="store_true")
    ap.add_argument("--p_cross_traj", type=float, default=0.0)

    ap.add_argument("--lambda_cf", type=float, default=1.0)
    ap.add_argument("--lambda_u", type=float, default=1e-3)
    ap.add_argument("--lambda_M", type=float, default=1e-3)
    ap.add_argument("--lambda_tv", type=float, default=1e-3)
    ap.add_argument("--lambda_ortho", type=float, default=1e-1)

    ap.add_argument("--viz_every", type=int, default=50)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--no_foreach_optim", dest="use_foreach_optim", action="store_false",
                    help="disable foreach AdamW updates (use if PyTorch build lacks foreach support)")
    ap.add_argument("--foreach_optim", dest="use_foreach_optim", action="store_true",
                    help="force-enable foreach AdamW updates")
    ap.set_defaults(use_foreach_optim=True)

    return ap.parse_args()

def main():
    args = parse_args()
    cfg = TrainCfg(
        data_root=args.data_root, out_dir=args.out_dir,
        img_ch=args.img_ch, z_ch=args.z_ch, du=args.du, k_hist=args.k_hist,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        seed=args.seed, num_workers=args.num_workers, device=args.device,
        max_step_gap=args.max_step_gap,
        allow_cross_traj=args.allow_cross_traj, p_cross_traj=args.p_cross_traj,
        lambda_cf=args.lambda_cf, lambda_u=args.lambda_u, lambda_M=args.lambda_M,
        lambda_tv=args.lambda_tv, lambda_ortho=args.lambda_ortho,
        viz_every=args.viz_every, log_every=args.log_every,
        use_foreach_optim=args.use_foreach_optim,
    )
    train(cfg)

if __name__ == "__main__":
    main()