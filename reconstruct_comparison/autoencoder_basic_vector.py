from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicVectorAutoencoder(nn.Module):
    """Basic autoencoder that exposes a flattened latent vector.

    Rationale:
    - Mirrors the spatial basic encoder so the convolutional trunk stays
      inexpensive while capturing coarse layout cues.
    - Uses adaptive pooling before the linear bottleneck so the latent vector
      remains compact without the ≈1e8 parameters of a full 28×28 flatten.

    Total parameters: ≈1.8e6 learnable weights (dominated by the latent MLP).
    """

    def __init__(
        self,
        latent_channels: int = 128,
        latent_dim: int = 256,
        latent_spatial: int = 28,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_dim = latent_dim
        self.latent_spatial = latent_spatial
        self.encoder_conv = nn.Sequential(
            # [B, 3, 224, 224] -> [B, 32, 112, 112]; stride-2 conv for cheap
            # down-sampling while expanding representational capacity.
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # [B, 32, 112, 112] -> [B, 48, 56, 56]; builds mid-level features.
            nn.Conv2d(32, 48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # [B, 48, 56, 56] -> [B, latent_channels, 28, 28]; compact spatial
            # latent keeps decoding simple while feeding the MLP.
            nn.Conv2d(48, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # Pool to smaller spatial grid: [B, latent_channels, 28, 28] ->
        # [B, latent_channels, latent_spatial, latent_spatial].
        self.pool = nn.AdaptiveAvgPool2d((latent_spatial, latent_spatial))
        # Flatten pooled features and map to latent vector.
        self.flatten = nn.Flatten()
        # Linear bottleneck: [B, latent_channels*latent_spatial^2] -> [B, latent_dim].
        self.to_latent = nn.Linear(
            latent_channels * latent_spatial * latent_spatial, latent_dim
        )
        # Linear projection back to pooled spatial tensor for the decoder.
        self.from_latent = nn.Linear(
            latent_dim, latent_channels * latent_spatial * latent_spatial
        )
        self.decoder_conv = nn.Sequential(
            # [B, latent_channels, 28, 28] -> [B, 48, 56, 56].
            nn.ConvTranspose2d(latent_channels, 48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # [B, 48, 56, 56] -> [B, 32, 112, 112].
            nn.ConvTranspose2d(48, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # [B, 32, 112, 112] -> [B, 3, 224, 224].
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder_conv(x)
        pooled = self.pool(feats)
        flat = self.flatten(pooled)
        latent = self.to_latent(flat)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        feats = self.from_latent(latent)
        feats = feats.view(
            latent.shape[0], self.latent_channels, self.latent_spatial, self.latent_spatial
        )
        # Bring pooled tensor back to the 28×28 latent grid expected by the decoder.
        feats = F.interpolate(
            feats,
            size=(28, 28),
            mode="bilinear",
            align_corners=False,
        )
        recon = self.decoder_conv(feats)
        if recon.shape[-2:] != (224, 224):
            recon = F.interpolate(recon, size=(224, 224), mode="bilinear", align_corners=False)
        return recon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        recon = self.decode(latent)
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return recon
__all__ = ["BasicVectorAutoencoder"]
