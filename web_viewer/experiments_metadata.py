from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import tomli
import tomli_w


def write_notes(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = text.replace("\r\n", "\n")
    path.write_text(normalized)


def write_title(path: Path, title: str) -> None:
    _write_metadata(path, title=title)


def write_tags(path: Path, tags: str) -> None:
    _write_metadata(path, tags=tags)


def _read_or_create_notes(path: Path) -> str:
    if not path.exists():
        path.write_text("")
        return ""
    return path.read_text()


def _read_title(path: Path) -> str:
    title, _, _, _ = _read_metadata(path)
    return title


def _read_metadata(path: Path) -> tuple[str, str, bool, bool]:
    """Read custom metadata (title, tags, starred, archived) with sane defaults."""
    if not path.exists():
        return "Untitled", "", False, False
    try:
        data = tomli.loads(path.read_text())
    except (tomli.TOMLDecodeError, OSError) as exc:
        raise ValueError(f"Invalid metadata file: {path}") from exc
    raw_title = data.get("title")
    raw_tags = data.get("tags")
    title = raw_title.strip() if isinstance(raw_title, str) and raw_title.strip() else "Untitled"
    tags = _normalize_tags(raw_tags)
    starred = _coerce_bool(data.get("starred"))
    archived = _coerce_bool(data.get("archived"))
    return title, tags, starred, archived


def _normalize_tags(raw_tags) -> str:
    """Normalize tags from string or list to a single string."""
    if isinstance(raw_tags, str):
        return raw_tags.strip()
    if isinstance(raw_tags, list):
        cleaned = []
        for item in raw_tags:
            if isinstance(item, str) and item.strip():
                cleaned.append(item.strip())
        return ", ".join(cleaned)
    return ""


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, int):
        return value != 0
    return False


def _write_metadata(
    path: Path,
    title: Optional[str] = None,
    tags: Optional[str] = None,
    starred: Optional[bool] = None,
    archived: Optional[bool] = None,
) -> None:
    """Write combined metadata, preserving existing values."""
    current_title, current_tags, current_starred, current_archived = _read_metadata(path)
    existing_data: Dict[str, object] = {}
    if path.exists():
        try:
            parsed = tomli.loads(path.read_text())
            if isinstance(parsed, dict):
                existing_data = dict(parsed)
        except (tomli.TOMLDecodeError, OSError) as exc:
            raise ValueError(f"Invalid metadata file: {path}") from exc
    next_title = title.strip() if isinstance(title, str) else None
    next_tags = tags.strip() if isinstance(tags, str) else None
    next_starred = current_starred if starred is None else _coerce_bool(starred)
    next_archived = current_archived if archived is None else _coerce_bool(archived)
    payload = dict(existing_data)
    payload.update(
        {
            "title": (next_title if next_title is not None and next_title else current_title or "Untitled"),
            "tags": (next_tags if next_tags is not None else current_tags),
            "starred": next_starred,
            "archived": next_archived,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(payload))


def write_starred(path: Path, starred: bool) -> None:
    _write_metadata(path, starred=starred)


def write_archived(path: Path, archived: bool) -> None:
    _write_metadata(path, archived=archived)


def _extract_data_root_from_metadata(metadata_text: str) -> Optional[str]:
    """Return the first data_root value found within a TOML metadata blob."""
    try:
        parsed = tomli.loads(metadata_text)
    except (tomli.TOMLDecodeError, OSError) as exc:
        raise ValueError("Invalid metadata.txt TOML") from exc

    def _walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "data_root" and isinstance(value, str) and value.strip():
                    return value.strip()
                found = _walk(value)
                if found:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = _walk(item)
                if found:
                    return found
        return None

    return _walk(parsed)


def _read_model_metadata(path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Read total_params and flops_per_step from metadata_model.txt."""
    if not path.exists():
        return None, None
    try:
        data = tomli.loads(path.read_text())
    except (tomli.TOMLDecodeError, OSError) as exc:
        raise ValueError(f"Invalid metadata_model.txt TOML: {path}") from exc
    total_params = None
    flops_per_step = None
    params_section = data.get("parameters")
    if isinstance(params_section, dict):
        total = params_section.get("total")
        if isinstance(total, int):
            total_params = total
    flops_section = data.get("flops")
    if isinstance(flops_section, dict):
        per_step = flops_section.get("per_step")
        if isinstance(per_step, int):
            flops_per_step = per_step
    return total_params, flops_per_step
