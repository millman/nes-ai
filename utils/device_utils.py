"""Device selection helpers."""

from __future__ import annotations

from typing import Optional

import torch


def pick_device(preferred: Optional[str]) -> torch.device:
    """Choose a torch.device honoring an optional user preference."""
    if preferred:
        pref = preferred.lower()
        if pref.startswith("cuda") and torch.cuda.is_available():
            return torch.device(preferred)
        if pref in {"mps", "metal"} and torch.backends.mps.is_available():
            return torch.device("mps")
        if pref == "cpu":
            return torch.device("cpu")
        # Fall back to requested device object; caller is responsible for validity.
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
