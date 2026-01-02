from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from jepa_world_model.actions import describe_action_tensor


TEXT_FONT = ImageFont.load_default()
BASE_FONT_SIZE = 10
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def _get_font(font_size: int) -> ImageFont.FreeTypeFont:
    cached = _FONT_CACHE.get(font_size)
    if cached is not None:
        return cached
    font = TEXT_FONT
    _FONT_CACHE[font_size] = font
    return font


def tensor_to_uint8_image(frame: torch.Tensor) -> np.ndarray:
    array = frame.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (array * 255.0).round().astype(np.uint8)


def delta_to_uint8_image(frame: torch.Tensor) -> np.ndarray:
    scaled = frame.detach().cpu().clamp(-1, 1)
    scaled = (scaled * 0.5 + 0.5).permute(1, 2, 0).numpy()
    return (scaled * 255.0).round().astype(np.uint8)


def _annotate_with_text(image: np.ndarray, text: str) -> np.ndarray:
    if not text:
        return image
    height = image.shape[0]
    scale = max(0.25, height / 128.0)
    font_size = max(6, int(round(BASE_FONT_SIZE * scale)))
    font = _get_font(font_size)
    padding = max(1, int(round(2 * scale)))
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    text = text.strip()
    bbox = draw.textbbox((padding, padding), text, font=font)
    rect = (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding)
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.text((padding, padding), text, fill=(255, 255, 255), font=font)
    return np.array(pil_image)


__all__ = [
    "describe_action_tensor",
    "tensor_to_uint8_image",
]
