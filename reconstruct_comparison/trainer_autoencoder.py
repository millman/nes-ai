from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from .metrics import compute_shared_metrics

OptimizerFactory = Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]


class AutoencoderTrainer:
    """Shared training harness for autoencoders matching comparison script style."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device,
        lr: float,
        loss_fn: nn.Module,
        weight_decay: float = 1e-4,
        optimizer_factory: Optional[OptimizerFactory] = None,
    ) -> None:
        self.device = device
        self.model = model.to(device)
        if optimizer_factory is not None:
            self.optimizer = optimizer_factory(self.model.parameters())
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        if loss_fn is None:
            raise ValueError("loss_fn must be provided.")
        self.loss_fn = loss_fn
        self.history: list[Tuple[int, float]] = []
        self.shared_history: list[Tuple[int, Dict[str, float]]] = []
        self.global_step = 0
        self.best_loss: Optional[float] = None

    def step(self, batch: torch.Tensor) -> Tuple[float, bool, Dict[str, float]]:
        self.model.train()
        recon = self.model(batch)
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
        was_training = self.model.training
        self.model.eval()
        recon = self.model(batch.to(self.device))
        if was_training:
            self.model.train()
        return recon

    @torch.no_grad()
    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.model, "encode"):
            raise AttributeError("Model does not implement encode().")
        was_training = self.model.training
        self.model.eval()
        latent = self.model.encode(batch.to(self.device))
        if was_training:
            self.model.train()
        return latent

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.model, "decode"):
            raise AttributeError("Model does not implement decode().")
        was_training = self.model.training
        self.model.eval()
        recon = self.model.decode(latent.to(self.device))
        if was_training:
            self.model.train()
        return recon

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


__all__ = ["AutoencoderTrainer"]
