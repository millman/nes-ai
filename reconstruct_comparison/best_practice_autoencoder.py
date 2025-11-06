from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAutoencoderTrainer


def _norm_groups(channels: int) -> int:
    return 8 if channels % 8 == 0 else 1


class SqueezeExcite(nn.Module):
    """Channel attention to re-weight features after convolutions."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced, kernel_size=1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        return x * self.gate(scale)


class ResidualBlock(nn.Module):
    """Pre-activation residual block with squeeze-excite channel recalibration."""

    def __init__(self, channels: int, dropout: float = 0.05, expansion: int = 2) -> None:
        super().__init__()
        hidden_channels = channels * expansion
        self.norm1 = nn.GroupNorm(_norm_groups(channels), channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_norm_groups(hidden_channels), hidden_channels)
        self.act2 = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1)
        self.se = SqueezeExcite(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.se(x)
        return x + residual


class SelfAttention2d(nn.Module):
    """Multi-head self-attention operating over flattened spatial tokens."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(_norm_groups(channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        flat = x.view(b, c, h * w)
        qkv = self.qkv(flat)
        q, k, v = qkv.chunk(3, dim=1)
        head_dim = c // self.num_heads
        if head_dim == 0:
            return residual
        q = q.view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)
        k = k.view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)
        v = v.view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)
        scale = head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h * w)
        out = self.proj(out)
        out = out.view(b, c, h, w)
        return out + residual


class EncoderStage(nn.Module):
    """Down-sampling stage that halves spatial size while deepening features."""

    def __init__(self, in_ch: int, out_ch: int, *, depth: int, use_attention: bool) -> None:
        super().__init__()
        layers = [
            # [B, in_ch, H, W] -> [B, out_ch, H/2, W/2]; strided conv reduces
            # resolution and expands channels for richer representations.
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_norm_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
        ]
        for _ in range(depth):
            layers.append(ResidualBlock(out_ch))
            if use_attention:
                layers.append(SelfAttention2d(out_ch))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderStage(nn.Module):
    """Up-sampling stage that doubles spatial size while refining details."""

    def __init__(self, in_ch: int, out_ch: int, *, depth: int, use_attention: bool) -> None:
        super().__init__()
        layers = [
            # [B, in_ch, H, W] -> [B, out_ch, 2H, 2W]; transpose conv undoes
            # the encoder stride while introducing learnable up-sampling.
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_norm_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
        ]
        for _ in range(depth):
            layers.append(ResidualBlock(out_ch))
            if use_attention:
                layers.append(SelfAttention2d(out_ch))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BestPracticeAutoencoder(nn.Module):
    """Autoencoder with residual blocks, attention, and squeeze-excite regularisation.

    Rationale:
    - GroupNorm + SiLU provide stable gradients across devices and small batch
      sizes typical for reconstruction tasks.
    - Residual blocks with channel expansion and squeeze-excite capture fine
      detail while keeping the network deep and expressive.
    - Lightweight self-attention augments global context so small objects are
      not lost during aggressive down-sampling.

    Total parameters: ≈2.89e7 learnable weights.
    """

    def __init__(self, base_channels: int = 64, latent_channels: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            # [B, 3, 224, 224] -> [B, base_channels, 224, 224]; shallow features.
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(base_channels), base_channels),
            nn.SiLU(inplace=True),
        )
        self.encoder = nn.ModuleList(
            [
                # Stage 1: [B, 64, 224, 224] -> [B, 128, 112, 112]; no attention.
                EncoderStage(base_channels, base_channels * 2, depth=2, use_attention=False),
                # Stage 2: [B, 128, 112, 112] -> [B, 256, 56, 56]; introduce attention.
                EncoderStage(base_channels * 2, base_channels * 4, depth=2, use_attention=True),
                # Stage 3: [B, 256, 56, 56] -> [B, latent_channels, 28, 28]; deepest stage.
                EncoderStage(base_channels * 4, latent_channels, depth=3, use_attention=True),
            ]
        )
        self.bottleneck = nn.Sequential(
            # Keeps latent resolution while mixing information globally.
            ResidualBlock(latent_channels, dropout=0.1, expansion=2),
            SelfAttention2d(latent_channels, num_heads=8),
            ResidualBlock(latent_channels, dropout=0.1, expansion=2),
        )
        self.decoder = nn.ModuleList(
            [
                # Stage 1: [B, latent_channels, 28, 28] -> [B, 256, 56, 56]; attention aids global detail.
                DecoderStage(latent_channels, base_channels * 4, depth=2, use_attention=True),
                # Stage 2: [B, 256, 56, 56] -> [B, 128, 112, 112].
                DecoderStage(base_channels * 4, base_channels * 2, depth=2, use_attention=True),
                # Stage 3: [B, 128, 112, 112] -> [B, 64, 224, 224]; no attention to save compute.
                DecoderStage(base_channels * 2, base_channels, depth=2, use_attention=False),
            ]
        )
        self.head = nn.Sequential(
            # Final refinement: [B, 64, 224, 224] -> [B, 3, 224, 224].
            nn.GroupNorm(_norm_groups(base_channels), base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels // 2, 3, kernel_size=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for stage in self.encoder:
            h = stage(h)
        return self.bottleneck(h)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        h = latent
        for stage in self.decoder:
            h = stage(h)
        out = self.head(h)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        recon = self.decode(latent)
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return recon


class BestPracticeAutoencoderTrainer(BaseAutoencoderTrainer):
    """Trainer for the best-practices autoencoder."""

    def __init__(
        self,
        *,
        device: torch.device,
        lr: float,
        loss_fn: Optional[nn.Module] = None,
        base_channels: int = 64,
        latent_channels: int = 256,
        weight_decay: float = 1e-4,
        name: str = "best_practice_autoencoder",
    ) -> None:
        model = BestPracticeAutoencoder(
            base_channels=base_channels,
            latent_channels=latent_channels,
        )
        super().__init__(
            name,
            model,
            device=device,
            lr=lr,
            loss_fn=loss_fn,
            weight_decay=weight_decay,
        )


__all__ = ["BestPracticeAutoencoder", "BestPracticeAutoencoderTrainer"]
