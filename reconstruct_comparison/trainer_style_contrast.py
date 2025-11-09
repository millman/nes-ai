from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16

from .metrics import compute_shared_metrics


def _gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feat.shape
    feature = feat.view(b, c, h * w)
    gram = torch.bmm(feature, feature.transpose(1, 2))
    return gram / (c * h * w)


def _style_loss(
    pred_feats: Sequence[torch.Tensor],
    target_feats: Sequence[torch.Tensor],
) -> torch.Tensor:
    losses = []
    for pred, target in zip(pred_feats, target_feats):
        gram_pred = _gram_matrix(pred)
        gram_target = _gram_matrix(target)
        losses.append(F.l1_loss(gram_pred, gram_target))
    if not losses:
        raise ValueError("No feature maps provided to style loss.")
    return torch.stack(losses).mean()


def _patch_contrastive_loss(
    pred_feat: torch.Tensor,
    target_feat: torch.Tensor,
    *,
    temperature: float,
    max_patches: int,
) -> torch.Tensor:
    b, c, h, w = pred_feat.shape
    pred_flat = pred_feat.permute(0, 2, 3, 1).reshape(-1, c)
    target_flat = target_feat.permute(0, 2, 3, 1).reshape(-1, c)
    total_patches = pred_flat.shape[0]
    if total_patches == 0:
        raise ValueError("Feature map contains no patches for contrastive loss.")
    if max_patches <= 0:
        raise ValueError("max_patches must be positive.")
    if total_patches <= max_patches:
        indices = torch.arange(total_patches, device=pred_feat.device)
    else:
        indices = torch.randperm(total_patches, device=pred_feat.device)[:max_patches]
    pred_emb = F.normalize(pred_flat[indices], dim=-1)
    target_emb = F.normalize(target_flat[indices], dim=-1)
    logits = pred_emb @ target_emb.t()
    logits = logits / temperature
    labels = torch.arange(pred_emb.shape[0], device=pred_feat.device)
    loss_fw = F.cross_entropy(logits, labels)
    loss_bw = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_fw + loss_bw)


class StyleFeatureExtractor(nn.Module):
    """Frozen VGG16 feature pyramid used for style and contrastive objectives."""

    def __init__(self, layers: Sequence[int]) -> None:
        super().__init__()
        if not layers:
            raise ValueError("At least one layer index must be specified for feature extraction.")
        max_idx = max(layers)
        backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features
        self.layers = nn.ModuleList(backbone[: max_idx + 1])
        self.selected_layers = set(layers)
        for param in self.layers.parameters():
            param.requires_grad_(False)
        self.eval()

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        features: Dict[int, torch.Tensor] = {}
        out = x
        for idx, layer in enumerate(self.layers):
            out = layer(out)
            if idx in self.selected_layers:
                features[idx] = out
            if len(features) == len(self.selected_layers):
                break
        return features


class StyleContrastTrainer:
    """Autoencoder trained with Gram style loss and patchwise contrastive loss."""

    def __init__(
        self,
        model: nn.Module,
        feature_extractor: StyleFeatureExtractor,
        *,
        device: torch.device,
        lr: float,
        style_layers: Sequence[int],
        patch_layer: int,
        style_weight: float,
        contrast_weight: float,
        contrast_temperature: float,
        contrast_patches: int,
        reconstruction_weight: float = 0.0,
        reconstruction_loss: Optional[nn.Module] = None,
    ) -> None:
        self.device = device
        self.model = model.to(device)
        self.feature_extractor = feature_extractor.to(device)
        self.style_layers = list(style_layers)
        self.patch_layer = patch_layer
        missing = set(self.style_layers + [self.patch_layer]) - feature_extractor.selected_layers
        if missing:
            raise ValueError(f"Feature extractor does not provide layers: {sorted(missing)}")
        self.feature_extractor.eval()
        self.style_weight = style_weight
        self.contrast_weight = contrast_weight
        self.contrast_temperature = contrast_temperature
        self.contrast_patches = contrast_patches
        if self.contrast_temperature <= 0:
            raise ValueError("contrast_temperature must be positive.")
        if self.contrast_patches <= 0:
            raise ValueError("contrast_patches must be positive.")
        self.reconstruction_weight = reconstruction_weight
        self.reconstruction_loss = reconstruction_loss or nn.L1Loss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.history: List[Tuple[int, float]] = []
        self.shared_history: List[Tuple[int, Dict[str, float]]] = []
        self.global_step = 0
        self.best_loss: Optional[float] = None

    def step(self, batch: torch.Tensor) -> Tuple[float, bool, Dict[str, float]]:
        self.model.train()
        recon = self.model(batch)
        pred_feats = self.feature_extractor(recon)
        with torch.no_grad():
            target_feats = self.feature_extractor(batch)
        style_pred = [pred_feats[idx] for idx in self.style_layers]
        style_target = [target_feats[idx] for idx in self.style_layers]
        style_loss = _style_loss(style_pred, style_target) * self.style_weight
        contrast_loss = _patch_contrastive_loss(
            pred_feats[self.patch_layer],
            target_feats[self.patch_layer],
            temperature=self.contrast_temperature,
            max_patches=self.contrast_patches,
        ) * self.contrast_weight
        total_loss = style_loss + contrast_loss
        if self.reconstruction_weight > 0.0:
            recon_loss = self.reconstruction_loss(recon, batch) * self.reconstruction_weight
            total_loss = total_loss + recon_loss
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()
        self.global_step += 1
        loss_val = float(total_loss.detach().item())
        self.history.append((self.global_step, loss_val))
        metrics = compute_shared_metrics(recon.detach(), batch)
        self.shared_history.append((self.global_step, metrics))
        improved = self.best_loss is None or loss_val < self.best_loss
        if improved:
            self.best_loss = loss_val
        return loss_val, improved, metrics

    @torch.no_grad()
    def reconstruct(self, batch: torch.Tensor) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        recon = self.model(batch.to(self.device))
        if was_training:
            self.model.train()
        return recon.cpu()

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "history": self.history,
            "shared_history": self.shared_history,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
        }

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_state_dict(self, state: dict, *, lr: Optional[float] = None) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if lr is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = lr
        self.history = state.get("history", [])
        self.shared_history = state.get("shared_history", [])
        self.global_step = state.get("global_step", 0)
        self.best_loss = state.get("best_loss")


__all__ = ["StyleContrastTrainer", "StyleFeatureExtractor"]
