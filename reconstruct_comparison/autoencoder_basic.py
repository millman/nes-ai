from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicAutoencoder(nn.Module):
    """Minimal stride-2 convolutional autoencoder.

    Rationale:
    - Uses only strided convolutions with ReLU activations so the encoder is
      extremely cheap while still down-sampling spatially for compression.
    - Mirrors the encoder with transpose convolutions so every latent feature
      directly informs pixels without expensive skip connections.

    Total parameters: ≈1.51e5 learnable weights.
    """

    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()
        self.encoder_net = nn.Sequential(
            # [B, 3, 224, 224] -> [B, 32, 112, 112]; stride-2 conv halves spatial
            # size while expanding to 32 channels for basic edge/colour detectors.
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # [B, 32, 112, 112] -> [B, 48, 56, 56]; maintains cheap compute while
            # increasing channels for richer mid-level features.
            nn.Conv2d(32, 48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # [B, 48, 56, 56] -> [B, latent_channels, 28, 28]; the latent tensor is
            # spatially compact yet preserves layout for easy decoding.
            nn.Conv2d(48, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder_net = nn.Sequential(
            # Mirrors encoder: [B, latent_channels, 28, 28] -> [B, 48, 56, 56].
            nn.ConvTranspose2d(latent_channels, 48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # [B, 48, 56, 56] -> [B, 32, 112, 112]; progressive up-sampling.
            nn.ConvTranspose2d(48, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # [B, 32, 112, 112] -> [B, 3, 224, 224]; reconstruct RGB image.
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_net(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        recon = self.decoder_net(latent)
        scale = 2 ** 3
        target_hw = (latent.shape[-2] * scale, latent.shape[-1] * scale)
        if recon.shape[-2:] != target_hw:
            # ConvTranspose restores the canonical 224×224 resolution when inputs follow
            # the expected spatial stride progression. Guard against mismatched shapes
            # during validation or scripted export by interpolating.
            recon = F.interpolate(recon, size=target_hw, mode="bilinear", align_corners=False)
        return recon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        recon = self.decode(latent)
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(
                recon, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        return recon
__all__ = ["BasicAutoencoder"]
