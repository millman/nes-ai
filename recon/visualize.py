"""Common helpers for rendering annotated image grids."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


RGBColor = Tuple[int, int, int]


@dataclass
class TileSpec:
    """Specification for a single tile in a rendered grid."""

    image: Image.Image
    top_label: Optional[str] = None
    top_color: RGBColor = (255, 255, 255)
    bottom_label: Optional[str] = None
    bottom_color: RGBColor = (255, 255, 255)


def _ensure_rgb(img: Image.Image, tile_size: Tuple[int, int]) -> Image.Image:
    """Resize image to the target tile size and ensure RGB mode."""

    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != tile_size:
        img = img.resize(tile_size, Image.NEAREST)
    return img


@lru_cache(maxsize=8)
def _load_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a truetype font, falling back to PIL's default font."""

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=font_size)
    except Exception:
        return ImageFont.load_default()


def _draw_label(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    *,
    tile_x: int,
    tile_y: int,
    tile_h: int,
    text: str,
    color: RGBColor,
    anchor: str,
    pad: int = 4,
    bg: RGBColor = (0, 0, 0),
) -> None:
    """Draw a label either at the top or bottom of a tile."""

    if not text:
        return

    # Compute text extents with a small guard for non-TrueType fonts.
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        text_w = len(text) * 6
        text_h = 10

    if anchor == "top":
        label_x = tile_x + 2
        label_y = tile_y + 2
    elif anchor == "bottom":
        label_x = tile_x + 2
        label_y = tile_y + tile_h - (text_h + pad * 2) - 2
    else:
        raise ValueError(f"Unsupported anchor '{anchor}' (expected 'top' or 'bottom')")

    rect = [label_x, label_y, label_x + text_w + pad * 2, label_y + text_h + pad * 2]
    draw.rectangle(rect, fill=bg)
    draw.text((label_x + pad, label_y + pad), text, font=font, fill=color)


def render_image_grid(
    rows: Sequence[Sequence[TileSpec]],
    out_path: Path,
    *,
    tile_size: Tuple[int, int],
    background: RGBColor = (0, 0, 0),
    left_border: RGBColor = (255, 255, 255),
    right_border: RGBColor = (0, 0, 0),
    font_size: int = 18,
) -> None:
    """Render a grid of tiles with annotations and save it to ``out_path``.

    Parameters
    ----------
    rows:
        A sequence of rows where each row is a sequence of :class:`TileSpec`.
    out_path:
        Location where the rendered image will be saved. Parent directories are
        created automatically.
    tile_size:
        Target ``(width, height)`` for each tile. Images are resized with
        ``Image.NEAREST`` if they do not already match the desired size.
    background:
        Background colour for the canvas.
    left_border / right_border:
        Colours used for the left and right borders (top/bottom edges are not
        drawn to preserve the original grid aesthetic).
    font_size:
        Size (in points) of the annotation font.
    """

    if not rows:
        raise ValueError("render_image_grid requires at least one row")

    max_cols = max(len(row) for row in rows)
    if max_cols == 0:
        raise ValueError("render_image_grid requires rows with at least one tile")

    tile_w, tile_h = tile_size
    canvas_w = tile_w * max_cols
    canvas_h = tile_h * len(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (canvas_w, canvas_h), background)
    draw = ImageDraw.Draw(canvas)
    font = _load_font(font_size)

    for row_idx, row in enumerate(rows):
        y = row_idx * tile_h
        for col_idx, tile in enumerate(row):
            x = col_idx * tile_w

            img = _ensure_rgb(tile.image, (tile_w, tile_h))
            canvas.paste(img, (x, y))

            # Draw left and right borders for each tile.
            draw.line([(x, y), (x, y + tile_h - 1)], fill=left_border)
            draw.line([(x + tile_w - 1, y), (x + tile_w - 1, y + tile_h - 1)], fill=right_border)

            if tile.top_label:
                _draw_label(draw, font, tile_x=x, tile_y=y, tile_h=tile_h, text=tile.top_label, color=tile.top_color, anchor="top")
            if tile.bottom_label:
                _draw_label(draw, font, tile_x=x, tile_y=y, tile_h=tile_h, text=tile.bottom_label, color=tile.bottom_color, anchor="bottom")

    canvas.save(out_path)

