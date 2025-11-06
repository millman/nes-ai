from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from predict_mario_ms_ssim import unnormalize


def _build_gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = (gauss / gauss.sum()).unsqueeze(0)
    window_2d = (gauss.t() @ gauss).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def _ssim_components_full(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    *,
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
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / denominator
    cs_map = (2 * sigma_xy + c2) / (sigma_x_sq + sigma_y_sq + c2)
    ssim_val = ssim_map.mean(dim=(1, 2, 3))
    cs_val = cs_map.mean(dim=(1, 2, 3))
    return ssim_val, cs_val


def ms_ssim_per_sample(
    x_hat_norm: torch.Tensor,
    x_true_norm: torch.Tensor,
    *,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if weights is None:
        weights = torch.tensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
            device=x_hat_norm.device,
            dtype=x_hat_norm.dtype,
        )
    xh = unnormalize(x_hat_norm)
    xt = unnormalize(x_true_norm)
    window_size = 11
    sigma = 1.5
    channels = xh.shape[1]
    window = _build_gaussian_window(window_size, sigma, channels, xh.device, xh.dtype)
    levels = weights.shape[0]
    mssim: List[torch.Tensor] = []
    mcs: List[torch.Tensor] = []
    x_scaled = xh
    y_scaled = xt
    for _ in range(levels):
        ssim_val, cs_val = _ssim_components_full(x_scaled, y_scaled, window)
        mssim.append(ssim_val)
        mcs.append(cs_val)
        x_scaled = F.avg_pool2d(x_scaled, kernel_size=2, stride=2)
        y_scaled = F.avg_pool2d(y_scaled, kernel_size=2, stride=2)
    mssim_tensor = torch.stack(mssim, dim=0)
    mcs_tensor = torch.stack(mcs[:-1], dim=0)
    eps = torch.finfo(mssim_tensor.dtype).eps
    mssim_tensor = mssim_tensor.clamp(min=eps, max=1.0)
    mcs_tensor = mcs_tensor.clamp(min=eps, max=1.0)
    pow1 = weights[:-1].unsqueeze(1)
    pow2 = weights[-1]
    ms_prod = torch.prod(mcs_tensor ** pow1, dim=0) * (mssim_tensor[-1] ** pow2)
    return ms_prod


def compute_shared_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        l1 = F.l1_loss(pred, target).item()
        ms = float(ms_ssim_per_sample(pred, target).mean().item())
    return {"l1": l1, "ms_ssim": ms}


__all__ = ["ms_ssim_per_sample", "compute_shared_metrics"]
