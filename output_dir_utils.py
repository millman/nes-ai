"""Helpers for resolving CLI output directories with timestamp placeholders."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

TIMESTAMP_PLACEHOLDER = "YYYY-MM-DD_HH-MM-SS"


def resolve_output_dir(
    value: Optional[str],
    *,
    default_template: str,
    placeholder: str = TIMESTAMP_PLACEHOLDER,
) -> Path:
    """Return a concrete output directory from an optional template string."""
    raw = (value or default_template).strip()
    if not raw:
        raise ValueError("Output directory template must not be empty.")

    path = Path(raw).expanduser()
    if path.name.endswith(placeholder):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        new_name = path.name[: -len(placeholder)] + timestamp
        path = path.with_name(new_name)
    return path
