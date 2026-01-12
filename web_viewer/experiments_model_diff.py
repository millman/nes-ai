from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


_MODEL_DIFF_SHORTNAMES: Dict[str, str] = {
    "loss_normalization_enabled": "norm",
    "image_size": "img",
}


def _parse_model_diff_items(text: str) -> List[Tuple[str, str, bool]]:
    """Return display, full text, shortened? tuples for model diff entries."""
    stripped = text.strip()
    if not stripped or stripped.startswith("model_diff.txt missing"):
        return []
    items: List[Tuple[str, str, bool]] = []
    for line in text.splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue
        normalized = trimmed.lstrip("+- ").lower()
        if normalized.startswith("data_root"):
            continue
        display = trimmed
        shortened = False
        sep = None
        key = None
        for candidate in ("=", ":"):
            if candidate in trimmed:
                key, rest = trimmed.split(candidate, 1)
                sep = candidate
                break
        if key:
            key_clean = key.strip()
            short = _MODEL_DIFF_SHORTNAMES.get(key_clean)
            if short and sep is not None:
                display = f"{short}{sep}{rest.strip()}"
                shortened = True
        items.append((display, trimmed, shortened))
    return items


def _render_model_diff(text: str) -> str:
    items = _parse_model_diff_items(text)
    if not text.strip():
        return "—"
    if text.strip().startswith("model_diff.txt missing"):
        return text.strip()
    if not items:
        return "—"
    return ", ".join(display for display, _, _ in items)


def _ensure_model_diff(path: Path) -> str:
    cache_path = path / "server_cache" / "model_diff.txt"
    if cache_path.exists():
        try:
            return cache_path.read_text()
        except OSError:
            pass

    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "web_viewer" / "model_diff.py"
    cmd = [
        "uv",
        "run",
        "python",
        str(script_path),
        "--experiment",
        str(path),
        "--repo-root",
        str(repo_root),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
    if result.returncode != 0:
        failure = result.stderr.strip() or result.stdout.strip() or "unknown failure"
        return f"model diff generation failed: {failure}"
    try:
        return cache_path.read_text()
    except OSError as exc:
        return f"model diff cache missing after generation: {exc}"
