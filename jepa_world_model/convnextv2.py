"""ConvNeXtV2-inspired encoder/decoder blocks for the JEPA world model."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormChannelsLast(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x_perm = x.permute(0, 2, 3, 1)
        x_norm = self.norm(x_perm)
        return x_norm.permute(0, 3, 1, 2)


class GlobalResponseNorm(nn.Module):
    """Global Response Normalization as used in ConvNeXtV2."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=-1, keepdim=True)
        nx = gx / (gx.mean(dim=(1, 2, 3), keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x.div(keep_prob) * random_tensor


def _make_blur_kernel() -> torch.Tensor:
    kernel = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
    kernel = kernel / kernel.sum()
    return kernel


class Blur(nn.Module):
    """Depthwise blur filter for anti-aliased sampling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        kernel = _make_blur_kernel().unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel.expand(self.channels, 1, 3, 3)
        return F.conv2d(x, kernel, stride=1, padding=1, groups=self.channels)


class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: int = 4,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-2,
        dilation: int = 1,
        use_grn: bool = True,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=7,
            padding=3 * dilation,
            dilation=dilation,
            groups=dim,
        )
        hidden_dim = dim * mlp_ratio
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.grn = GlobalResponseNorm(hidden_dim) if use_grn else nn.Identity()
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x * self.gamma
        x = x.permute(0, 3, 1, 2)
        x = self.drop_path(x)
        return shortcut + x


class PatchStem(nn.Module):
    """Simplified stem to mirror the legacy stride-2 entry."""

    def __init__(self, in_ch: int, out_ch: int, use_blur: bool = False) -> None:
        super().__init__()
        _ = use_blur  # kept for signature compatibility
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class ConvNeXtV2DownStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        depth: int,
        drop_paths: Sequence[float],
        dilated: bool = False,
        use_blur: bool = False,
    ) -> None:
        super().__init__()
        dilation = 2 if dilated else 1
        self.blocks = nn.Sequential(
            *[
                ConvNeXtV2Block(
                    in_ch,
                    drop_path=drop_paths[i],
                    dilation=dilation,
                )
                for i in range(depth)
            ]
        )
        self.down = nn.Sequential(
            Blur(in_ch) if use_blur else nn.Identity(),
            nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.blocks(x)
        detail = x
        x = self.down(x)
        return x, detail


class ConvNeXtV2UpStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        depth: int,
        drop_paths: Sequence[float],
        use_blur: bool = False,
        skip_ch: int | None = None,
    ) -> None:
        super().__init__()
        self.skip_adapter = None
        if skip_ch is not None:
            self.skip_adapter = nn.Sequential(
                LayerNormChannelsLast(skip_ch),
                nn.Conv2d(skip_ch, out_ch, kernel_size=1),
            )
            self.skip_gate = nn.Parameter(torch.zeros(1))
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            Blur(in_ch) if use_blur else nn.Identity(),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
        )
        self.blocks = nn.Sequential(
            *[
                ConvNeXtV2Block(out_ch, drop_path=drop_paths[i])
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.upsample(x)
        if self.skip_adapter is not None and skip is not None:
            adapted = self.skip_adapter(skip)
            x = x + torch.tanh(self.skip_gate) * adapted
        x = self.blocks(x)
        return x


class ConvNeXtV2Encoder(nn.Module):
    """Encoder producing pooled embeddings and a detail skip map."""

    def __init__(self, in_channels: int, channel_schedule: Tuple[int, ...], input_hw: int) -> None:
        super().__init__()
        if not channel_schedule:
            raise ValueError("channel_schedule must contain at least one stage for the encoder.")
        self.stem = PatchStem(in_channels, channel_schedule[0])
        self.num_stages = len(channel_schedule)
        block_depths = self._default_block_depths(self.num_stages)
        drop_paths = self._drop_path_schedule(sum(block_depths), 0.05)
        stages: List[nn.Module] = []
        offset = 0
        for i in range(self.num_stages - 1):
            depth = block_depths[i]
            stage_drop = drop_paths[offset : offset + depth]
            stages.append(
                ConvNeXtV2DownStage(
                    channel_schedule[i],
                    channel_schedule[i + 1],
                    depth=depth,
                    drop_paths=stage_drop,
                    dilated=False,
                )
            )
            offset += depth
        self.stages = nn.ModuleList(stages)
        final_depth = block_depths[-1]
        final_drop = drop_paths[offset : offset + final_depth]
        self.final_blocks = nn.Sequential(
            *[ConvNeXtV2Block(channel_schedule[-1], drop_path=final_drop[i]) for i in range(final_depth)]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.channel_schedule = channel_schedule
        self.input_hw = input_hw
        self.in_channels = in_channels

    @staticmethod
    def _default_block_depths(num_stages: int) -> List[int]:
        if num_stages >= 4:
            return [2, 3, 3, 2][:num_stages] + [2] * max(0, num_stages - 4)
        if num_stages == 3:
            return [2, 3, 2]
        if num_stages == 2:
            return [2, 2]
        return [2]

    @staticmethod
    def _drop_path_schedule(num_blocks: int, max_rate: float) -> List[float]:
        return [0.0] * num_blocks

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        detail_skip = None
        for idx, stage in enumerate(self.stages):
            x, detail = stage(x)
            if idx == 0:
                detail_skip = detail
        x = self.final_blocks(x)
        pooled = self.pool(x).flatten(1)
        if detail_skip is None:
            raise RuntimeError("detail_skip should be captured from the first stage.")
        return pooled, detail_skip

    def shape_info(self) -> Dict[str, object]:
        h = self.input_hw
        stages: List[Dict[str, object]] = []
        prev_ch = self.channel_schedule[0]
        h = h // 2
        for idx, ch in enumerate(self.channel_schedule[1:], start=1):
            prev_h = h
            h = max(1, h // 2)
            stages.append({"stage": idx, "in": (prev_h * 2, prev_h * 2, prev_ch), "out": (h, h, ch)})
            prev_ch = ch
        return {
            "module": "ConvNeXtV2Encoder",
            "input": (self.input_hw, self.input_hw, self.in_channels),
            "detail_skip": (self.input_hw // 2, self.input_hw // 2, self.channel_schedule[0]),
            "stages": stages,
            "latent_dim": self.channel_schedule[-1],
        }


class EdgeSharpenHead(nn.Module):
    def __init__(self, channels: int, out_ch: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.act = nn.SiLU(inplace=True)
        self.proj = nn.Conv2d(channels, out_ch, kernel_size=1)
        laplacian = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=torch.float32)
        laplacian = laplacian.unsqueeze(0).unsqueeze(0)
        self.register_buffer("laplacian", laplacian)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.act(x)
        x = self.proj(x)
        lap = self.laplacian.to(dtype=x.dtype, device=x.device)
        lap = lap.expand(x.shape[1], 1, 3, 3)
        sharpen = F.conv2d(x, lap, padding=1, groups=x.shape[1])
        return x + 0.1 * sharpen


class ConvNeXtV2Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        image_channels: int,
        image_size: int,
        channel_schedule: Tuple[int, ...],
        latent_hw: int,
    ) -> None:
        super().__init__()
        if not channel_schedule:
            raise ValueError("channel_schedule must be non-empty for decoder construction.")
        scale = 2 ** len(channel_schedule)
        if image_size % scale != 0:
            raise ValueError("image_size must be divisible by 2**len(channel_schedule) for upsampling alignment.")
        if latent_hw != image_size // scale:
            raise ValueError("latent_hw must equal image_size // 2**len(channel_schedule) to mirror encoder.")
        self.start_hw = latent_hw
        self.start_ch = channel_schedule[-1]
        self.channel_schedule = channel_schedule
        block_depths = ConvNeXtV2Encoder._default_block_depths(len(channel_schedule))
        drop_paths = ConvNeXtV2Encoder._drop_path_schedule(sum(block_depths), 0.05)
        self.project = nn.Sequential(
            nn.Linear(latent_dim, self.start_ch * self.start_hw * self.start_hw),
            nn.SiLU(inplace=True),
        )
        stages: List[ConvNeXtV2UpStage] = []
        offset = 0
        reversed_schedule = list(reversed(channel_schedule))
        reversed_depths = list(reversed(block_depths))
        for idx in range(len(reversed_schedule) - 1):
            in_ch = reversed_schedule[idx]
            out_ch = reversed_schedule[idx + 1]
            depth = reversed_depths[idx]
            stage_drop = drop_paths[offset : offset + depth]
            offset += depth
            skip_ch = channel_schedule[0] if idx == len(reversed_schedule) - 2 else None
            stages.append(
                ConvNeXtV2UpStage(
                    in_ch,
                    out_ch,
                    depth=depth,
                    drop_paths=stage_drop,
                    skip_ch=skip_ch,
                )
            )
        self.up_stages = nn.ModuleList(stages)
        final_drop = drop_paths[-1] if drop_paths else 0.0
        self.final_upsample = ConvNeXtV2UpStage(
            channel_schedule[0],
            channel_schedule[0],
            depth=1,
            drop_paths=[final_drop],
            skip_ch=None,
        )
        self.head = EdgeSharpenHead(channel_schedule[0], image_channels)
        self.image_channels = image_channels
        self.image_size = image_size

    def forward(self, latent: torch.Tensor, detail_skip: torch.Tensor | None = None) -> torch.Tensor:
        original_shape = latent.shape[:-1]
        latent = latent.reshape(-1, latent.shape[-1])
        x = self.project(latent)
        x = x.view(-1, self.start_ch, self.start_hw, self.start_hw)
        skip = None
        if detail_skip is not None:
            skip = detail_skip.reshape(-1, detail_skip.shape[2], detail_skip.shape[3], detail_skip.shape[4])
        for idx, stage in enumerate(self.up_stages):
            if idx == len(self.up_stages) - 1:
                x = stage(x, skip)
            else:
                x = stage(x)
        x = self.final_upsample(x)
        frame = self.head(x)
        frame = frame.view(*original_shape, self.image_channels, self.image_size, self.image_size)
        return frame

    def shape_info(self) -> Dict[str, object]:
        hw = self.start_hw
        ch = self.start_ch
        up_info: List[Dict[str, object]] = []
        for idx, out_ch in enumerate(reversed(self.channel_schedule[:-1]), start=1):
            prev_hw = hw
            hw = max(1, hw * 2)
            up_info.append({"stage": idx, "in": (prev_hw, prev_hw, ch), "out": (hw, hw, out_ch)})
            ch = out_ch
        final_hw = max(1, hw * 2)
        up_info.append({"stage": len(up_info) + 1, "in": (hw, hw, ch), "out": (final_hw, final_hw, ch)})
        return {
            "module": "ConvNeXtV2Decoder",
            "projection": (self.start_hw, self.start_hw, self.start_ch),
            "upsample": up_info,
            "final_target": (self.image_size, self.image_size, self.image_channels),
            "detail_skip": (self.image_size // 2, self.image_size // 2, self.channel_schedule[0]),
            "pre_resize": (final_hw, final_hw, self.image_channels),
            "needs_resize": final_hw != self.image_size,
        }
