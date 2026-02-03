from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa_world_model.conv_encoder_decoder import Encoder as ConvEncoder
from jepa_world_model.model_config import ModelConfig

Encoder = ConvEncoder
LegacyEncoder = ConvEncoder


class PredictorNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        action_dim: int,
        state_dim: int,
        use_layer_norm: bool = True,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embedding_dim + action_dim + state_dim, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.h_out = nn.Linear(hidden_dim, state_dim)
        if use_spectral_norm:
            self.in_proj = nn.utils.spectral_norm(self.in_proj)
            self.hidden_proj = nn.utils.spectral_norm(self.hidden_proj)
            self.h_out = nn.utils.spectral_norm(self.h_out)
        self.h_norm = nn.LayerNorm(state_dim) if use_layer_norm else None
        self.activation = nn.SiLU(inplace=True)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.use_layer_norm = use_layer_norm
        self.use_spectral_norm = use_spectral_norm

    def forward(
        self, embeddings: torch.Tensor, hidden_state: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if not (embeddings.shape[:-1] == actions.shape[:-1] == hidden_state.shape[:-1]):
            raise ValueError("Embeddings, hidden state, and actions must share leading dimensions for predictor conditioning.")
        original_shape = embeddings.shape[:-1]
        emb_flat = embeddings.reshape(-1, embeddings.shape[-1])
        act_flat = actions.reshape(-1, actions.shape[-1])
        h_flat = hidden_state.reshape(-1, hidden_state.shape[-1])
        hidden = self.activation(self.in_proj(torch.cat([emb_flat, h_flat, act_flat], dim=-1)))
        hidden = self.activation(self.hidden_proj(hidden))
        h_next = self.h_out(hidden)
        if self.h_norm is not None:
            h_next = self.h_norm(h_next)
        h_next = h_next.view(*original_shape, h_next.shape[-1])
        return h_next

    def shape_info(self) -> Dict[str, Any]:
        return {
            "module": "Predictor",
            "latent_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "conditioning": "concat(z,h,action)",
        }


class HiddenToZProjector(nn.Module):
    """Project hidden state to an image-anchored embedding prediction."""

    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        hidden_dim: int,
        use_layer_norm: bool = True,
        normalize_output: bool = False,
    ) -> None:
        super().__init__()
        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(h_dim))
        layers.extend(
            [
                nn.Linear(h_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, z_dim),
            ]
        )
        self.net = nn.Sequential(*layers)
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        self.normalize_output = normalize_output

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        original_shape = h.shape[:-1]
        h_flat = h.reshape(-1, h.shape[-1])
        z_hat = self.net(h_flat)
        z_hat = z_hat.view(*original_shape, z_hat.shape[-1])
        if self.normalize_output:
            z_hat = F.normalize(z_hat, dim=-1)
        return z_hat


class ZToHProjector(nn.Module):
    """Project image-anchored embedding to a hidden state seed."""

    def __init__(self, z_dim: int, h_dim: int, hidden_dim: int, use_layer_norm: bool = True) -> None:
        super().__init__()
        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(z_dim))
        layers.extend(
            [
                nn.Linear(z_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, h_dim),
            ]
        )
        self.net = nn.Sequential(*layers)
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        original_shape = z.shape[:-1]
        z_flat = z.reshape(-1, z.shape[-1])
        h_seed = self.net(z_flat)
        return h_seed.view(*original_shape, h_seed.shape[-1])


class HiddenToDeltaProjector(nn.Module):
    """Project hidden state to a delta in the target latent space."""

    def __init__(
        self,
        h_dim: int,
        target_dim: int,
        hidden_dim: int,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(h_dim))
        layers.extend(
            [
                nn.Linear(h_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, target_dim),
            ]
        )
        self.net = nn.Sequential(*layers)
        self.h_dim = h_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        original_shape = h.shape[:-1]
        h_flat = h.reshape(-1, h.shape[-1])
        delta = self.net(h_flat)
        return delta.view(*original_shape, delta.shape[-1])


class HiddenActionDeltaProjector(nn.Module):
    """Predict state delta from hidden state and action (no pose input)."""

    def __init__(
        self,
        h_dim: int,
        action_dim: int,
        hidden_dim: int,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(h_dim + action_dim))
        layers.extend(
            [
                nn.Linear(h_dim + action_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, h_dim),
            ]
        )
        self.net = nn.Sequential(*layers)
        self.h_dim = h_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm

    def forward(self, h: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if h.shape[:-1] != actions.shape[:-1]:
            raise ValueError("Hidden state and actions must share leading dimensions.")
        original_shape = h.shape[:-1]
        h_flat = h.reshape(-1, h.shape[-1])
        act_flat = actions.reshape(-1, actions.shape[-1])
        delta = self.net(torch.cat([h_flat, act_flat], dim=-1))
        return delta.view(*original_shape, delta.shape[-1])


class PoseActionDeltaProjector(nn.Module):
    """Predict pose delta from pose, hidden state, and action (adds pose context vs HiddenActionDeltaProjector)."""

    def __init__(
        self,
        pose_dim: int,
        h_dim: int,
        action_dim: int,
        hidden_dim: int,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        input_dim = pose_dim + h_dim + action_dim
        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(input_dim))
        layers.extend(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, pose_dim),
            ]
        )
        self.net = nn.Sequential(*layers)
        self.pose_dim = pose_dim
        self.h_dim = h_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm

    def forward(
        self,
        pose: torch.Tensor,
        h: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if not (pose.shape[:-1] == h.shape[:-1] == actions.shape[:-1]):
            raise ValueError("Pose, hidden state, and actions must share leading dimensions.")
        original_shape = pose.shape[:-1]
        pose_flat = pose.reshape(-1, pose.shape[-1])
        h_flat = h.reshape(-1, h.shape[-1])
        act_flat = actions.reshape(-1, actions.shape[-1])
        delta = self.net(torch.cat([pose_flat, h_flat, act_flat], dim=-1))
        return delta.view(*original_shape, delta.shape[-1])


class InverseActionHead(nn.Module):
    """Predict inverse-action logits from current state and action."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        input_dim = state_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

    def forward(self, h: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if h.shape[:-1] != actions.shape[:-1]:
            raise ValueError("Hidden state and actions must share leading dimensions.")
        h_flat = h.reshape(-1, h.shape[-1])
        act_flat = actions.reshape(-1, actions.shape[-1])
        logits = self.net(torch.cat([h_flat, act_flat], dim=-1))
        return logits.view(*h.shape[:-1], logits.shape[-1])


class ActionDeltaProjector(nn.Module):
    """Project action vectors into latent delta prototypes (no state inputs)."""

    def __init__(self, action_dim: int, target_dim: int, use_layer_norm: bool = True) -> None:
        super().__init__()
        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(action_dim))
        layers.append(nn.Linear(action_dim, target_dim, bias=False))
        self.net = nn.Sequential(*layers)
        self.action_dim = action_dim
        self.target_dim = target_dim
        self.use_layer_norm = use_layer_norm

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        original_shape = actions.shape[:-1]
        flat = actions.reshape(-1, actions.shape[-1])
        projected = self.net(flat)
        return projected.view(*original_shape, projected.shape[-1])


class DeltaToDeltaProjector(nn.Module):
    """Project one delta space into another (e.g., Δz -> Δp)."""

    def __init__(self, input_dim: int, target_dim: int, use_layer_norm: bool = True) -> None:
        super().__init__()
        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(input_dim))
        layers.append(nn.Linear(input_dim, target_dim, bias=False))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.use_layer_norm = use_layer_norm

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        original_shape = delta.shape[:-1]
        flat = delta.reshape(-1, delta.shape[-1])
        projected = self.net(flat)
        return projected.view(*original_shape, projected.shape[-1])


class InverseDynamicsHead(nn.Module):
    """Predict action from consecutive observation embeddings."""

    def __init__(self, z_dim: int, hidden_dim: int, action_dim: int, use_layer_norm: bool = True) -> None:
        super().__init__()
        layers = []
        input_dim = z_dim
        if use_layer_norm:
            layers.append(nn.LayerNorm(input_dim))
        layers.extend(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, action_dim),
            ]
        )
        self.net = nn.Sequential(*layers)
        self.use_layer_norm = use_layer_norm

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        if z_t.shape != z_next.shape:
            raise ValueError("z_t and z_next must have matching shapes.")
        delta = z_next - z_t
        original_shape = delta.shape[:-1]
        flat = delta.reshape(-1, delta.shape[-1])
        logits = self.net(flat)
        return logits.view(*original_shape, logits.shape[-1])


class InverseDynamicsDeltaHead(nn.Module):
    """Predict action directly from delta embeddings."""

    def __init__(self, delta_dim: int, hidden_dim: int, action_dim: int, use_layer_norm: bool = True) -> None:
        super().__init__()
        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(delta_dim))
        layers.extend(
            [
                nn.Linear(delta_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, action_dim),
            ]
        )
        self.net = nn.Sequential(*layers)
        self.use_layer_norm = use_layer_norm

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        original_shape = delta.shape[:-1]
        flat = delta.reshape(-1, delta.shape[-1])
        logits = self.net(flat)
        return logits.view(*original_shape, logits.shape[-1])


class ActionEmbedding(nn.Module):
    """Embed controller actions for FiLM modulation."""

    def __init__(self, action_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        original_shape = actions.shape[:-1]
        flat = actions.reshape(-1, actions.shape[-1])
        embedded = self.net(flat)
        return embedded.view(*original_shape, embedded.shape[-1])


class FiLMLayer(nn.Module):
    """Single FiLM modulation layer."""

    def __init__(self, hidden_dim: int, action_dim: int, use_layer_norm: bool = True) -> None:
        super().__init__()
        self.gamma = nn.Linear(action_dim, hidden_dim)
        self.beta = nn.Linear(action_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim) if use_layer_norm else None
        self.use_layer_norm = use_layer_norm

    def forward(self, hidden: torch.Tensor, action_embed: torch.Tensor) -> torch.Tensor:
        conditioned = self.norm(hidden) if self.norm is not None else hidden
        gamma = self.gamma(action_embed)
        beta = self.beta(action_embed)
        return conditioned * (1.0 + gamma) + beta


class ActionFiLM(nn.Module):
    """Stack multiple FiLM layers for action conditioning."""

    def __init__(self, hidden_dim: int, action_dim: int, layers: int, use_layer_norm: bool = True) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [FiLMLayer(hidden_dim, action_dim, use_layer_norm=use_layer_norm) for _ in range(layers)]
        )
        self.act = nn.GELU()

    def forward(self, hidden: torch.Tensor, action_embed: torch.Tensor) -> torch.Tensor:
        out = hidden
        for layer in self.layers:
            out = layer(out, action_embed)
            out = self.act(out)
        return out


class JEPAWorldModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        for name in (
            "enable_inverse_dynamics_z",
            "enable_inverse_dynamics_h",
            "enable_inverse_dynamics_p",
            "enable_action_delta_z",
            "enable_action_delta_h",
            "enable_action_delta_p",
            "enable_dz_to_dp_projector",
            "enable_h2z_delta",
        ):
            if getattr(cfg, name) is None:
                raise AssertionError(f"{name} must be resolved before JEPAWorldModel initialization.")
        self.encoder = Encoder(
            cfg.in_channels,
            cfg.encoder_schedule,
            cfg.image_size,
        )
        # latent_dim is encoder_schedule[-1]
        self.embedding_dim = self.encoder.latent_dim
        self.predictor = PredictorNetwork(
            self.embedding_dim,
            cfg.hidden_dim,
            cfg.action_dim,
            cfg.state_dim,
            use_layer_norm=cfg.layer_norms.h_next,
            use_spectral_norm=cfg.predictor_spectral_norm,
        )
        self.state_dim = cfg.state_dim
        pose_dim = cfg.pose_dim if cfg.pose_dim is not None else cfg.state_dim
        # Backwards-compatible aliases for legacy names.
        self.h_to_z = HiddenToZProjector(
            cfg.state_dim,
            self.embedding_dim,
            cfg.hidden_dim,
            use_layer_norm=cfg.layer_norms.h2z_projector,
            normalize_output=cfg.z_norm,
        )
        self.z_to_h = ZToHProjector(
            self.embedding_dim,
            cfg.state_dim,
            cfg.hidden_dim,
            use_layer_norm=cfg.layer_norms.z2h_projector,
        )
        self.h2z_delta = (
            HiddenToDeltaProjector(
                cfg.state_dim,
                self.embedding_dim,
                cfg.hidden_dim,
                use_layer_norm=cfg.layer_norms.h2z_delta_projector,
            )
            if cfg.enable_h2z_delta
            else None
        )
        self.inverse_dynamics_z = (
            InverseDynamicsHead(
                self.embedding_dim,
                cfg.hidden_dim,
                cfg.action_dim,
                use_layer_norm=cfg.layer_norms.inverse_dynamics_z,
            )
            if cfg.enable_inverse_dynamics_z
            else None
        )
        self.inverse_dynamics_h = (
            InverseDynamicsHead(
                cfg.state_dim,
                cfg.hidden_dim,
                cfg.action_dim,
                use_layer_norm=cfg.layer_norms.inverse_dynamics_h,
            )
            if cfg.enable_inverse_dynamics_h
            else None
        )
        self.inverse_dynamics_p = (
            InverseDynamicsHead(
                pose_dim if pose_dim is not None else cfg.state_dim,
                cfg.hidden_dim,
                cfg.action_dim,
                use_layer_norm=cfg.layer_norms.inverse_dynamics_p,
            )
            if cfg.enable_inverse_dynamics_p
            else None
        )
        self.inverse_dynamics_dp = (
            InverseDynamicsDeltaHead(
                pose_dim if pose_dim is not None else cfg.state_dim,
                cfg.hidden_dim,
                cfg.action_dim,
                use_layer_norm=cfg.layer_norms.inverse_dynamics_dp,
            )
            if cfg.enable_inverse_dynamics_dp
            else None
        )
        self.inverse_action_head = InverseActionHead(
            cfg.state_dim,
            cfg.action_dim,
            cfg.hidden_dim,
        )
        self.z_action_delta_projector = (
            # Action-only prototypes for z; no state/pose conditioning.
            ActionDeltaProjector(
                cfg.action_dim,
                self.embedding_dim,
                use_layer_norm=cfg.layer_norms.action_delta_projector_z,
            )
            if cfg.enable_action_delta_z
            else None
        )
        self.h_action_delta_projector = (
            HiddenActionDeltaProjector(
                cfg.state_dim,
                cfg.action_dim,
                cfg.hidden_dim,
                use_layer_norm=cfg.layer_norms.action_delta_projector_h,
            )
            if cfg.enable_action_delta_h
            else None
        )
        self.p_action_delta_projector = (
            PoseActionDeltaProjector(
                pose_dim,
                cfg.state_dim,
                cfg.action_dim,
                cfg.hidden_dim,
                use_layer_norm=cfg.layer_norms.action_delta_projector_p,
            )
            if cfg.enable_action_delta_p
            else None
        )
        self.dp_action_delta_projector = (
            ActionDeltaProjector(
                cfg.action_dim,
                pose_dim if pose_dim is not None else cfg.state_dim,
                use_layer_norm=cfg.layer_norms.action_delta_projector_dp,
            )
            if cfg.enable_action_delta_p
            else None
        )
        self.dz_to_dp_projector = (
            DeltaToDeltaProjector(
                self.embedding_dim,
                pose_dim if pose_dim is not None else cfg.state_dim,
                use_layer_norm=cfg.layer_norms.dz_to_dp_projector,
            )
            if cfg.enable_dz_to_dp_projector
            else None
        )

    def encode_sequence(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, t, _, _, _ = images.shape
        embeddings: list[torch.Tensor] = []
        for step in range(t):
            current = images[:, step]
            pooled = self.encoder(current)
            embeddings.append(pooled)
        z_raw = torch.stack(embeddings, dim=1)
        z = F.normalize(z_raw, dim=-1) if self.cfg.z_norm else z_raw
        return {
            "embeddings": z,
            "embeddings_raw": z_raw,
        }
