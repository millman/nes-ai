from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .metrics import compute_shared_metrics

OptimizerFactory = Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]


class ReconstructionTrainer:
    """Wraps a frozen encoder and trainable decoder with a unified step API."""

    def __init__(
        self,
        name: str,
        encoder: nn.Module,
        decoder: nn.Module,
        *,
        device: torch.device,
        lr: float,
        loss_fn: Optional[nn.Module] = None,
        weight_decay: float = 0.0,
        optimizer_factory: Optional[OptimizerFactory] = None,
    ) -> None:
        self.name = name
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.model = self.decoder  # align with AutoencoderTrainer interface
        if optimizer_factory is not None:
            self.optimizer = optimizer_factory(self.decoder.parameters())
        else:
            self.optimizer = torch.optim.Adam(
                self.decoder.parameters(), lr=lr, weight_decay=weight_decay
            )
        self.loss_fn = loss_fn or nn.MSELoss()
        self.history: List[Tuple[int, float]] = []
        self.shared_history: List[Tuple[int, Dict[str, float]]] = []
        self.global_step = 0
        self.best_loss: Optional[float] = None

    def step(self, batch: torch.Tensor) -> Tuple[float, bool, Dict[str, float]]:
        batch = batch.to(self.device)
        self.decoder.train()
        with torch.no_grad():
            feats = self.encoder(batch)
        recon = self.decoder(feats)
        loss = self.loss_fn(recon, batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.global_step += 1
        loss_val = float(loss.detach().item())
        self.history.append((self.global_step, loss_val))
        metrics = compute_shared_metrics(recon.detach(), batch)
        self.shared_history.append((self.global_step, metrics))
        improved = self.best_loss is None or loss_val < self.best_loss
        if improved:
            self.best_loss = loss_val
        return loss_val, improved, metrics

    @torch.no_grad()
    def reconstruct(self, batch: torch.Tensor) -> torch.Tensor:
        was_training = self.decoder.training
        self.encoder.eval()
        self.decoder.eval()
        feats = self.encoder(batch.to(self.device))
        recon = self.decoder(feats)
        if was_training:
            self.decoder.train()
        return recon.cpu()

    @torch.no_grad()
    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        was_training = self.encoder.training
        self.encoder.eval()
        feats = self.encoder(batch.to(self.device))
        if was_training:
            self.encoder.train()
        return feats.cpu()

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        was_training = self.decoder.training
        self.decoder.eval()
        recon = self.decoder(latent.to(self.device))
        if was_training:
            self.decoder.train()
        return recon.cpu()

    def state_dict(self) -> dict:
        decoder_state = self.decoder.state_dict()
        return {
            "model": decoder_state,  # mirrors AutoencoderTrainer
            "decoder": decoder_state,  # backward compatibility
            "optimizer": self.optimizer.state_dict(),
            "history": self.history,
            "shared_history": self.shared_history,
            "global_step": self.global_step,
            "name": self.name,
            "best_loss": self.best_loss,
        }

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_state_dict(self, state: dict, *, lr: Optional[float] = None) -> None:
        decoder_state = state.get("model", state.get("decoder"))
        if decoder_state is None:
            raise KeyError("State dict missing both 'model' and 'decoder' keys.")
        self.decoder.load_state_dict(decoder_state)
        self.optimizer.load_state_dict(state["optimizer"])
        if lr is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = lr
        self.history = state.get("history", [])
        self.shared_history = state.get("shared_history", [])
        self.global_step = state.get("global_step", 0)
        self.best_loss = state.get("best_loss")


__all__ = ["ReconstructionTrainer"]
