from __future__ import annotations

from difflib import unified_diff
from typing import Dict, Iterable, List, Tuple


def diff_metadata(base: str, other: str) -> str:
    """Return a unified diff (or placeholder text) comparing metadata blobs."""
    diff_lines = list(
        unified_diff(
            base.splitlines(),
            other.splitlines(),
            fromfile="A",
            tofile="B",
            lineterm="",
        )
    )
    if not diff_lines:
        return "(identical metadata)"
    return "\n".join(diff_lines)


def build_metadata_diff_views(metadata_texts: List[str]) -> List[str]:
    assert metadata_texts, "metadata_texts must not be empty"
    entries_by_exp: List[Dict[Tuple[str, str], List[str]]] = []
    ordered_entry_ids: List[Tuple[str, str]] = []

    for text in metadata_texts:
        entry_map: Dict[Tuple[str, str], List[str]] = {}
        for section, key, lines in _parse_metadata_entries(text):
            entry_id = (section, key)
            if entry_id not in entry_map:
                entry_map[entry_id] = lines
            if entry_id not in ordered_entry_ids:
                ordered_entry_ids.append(entry_id)
        entries_by_exp.append(entry_map)

    if not ordered_entry_ids:
        return ["(no metadata differences)" for _ in metadata_texts]

    diff_entry_ids: List[Tuple[str, str]] = []
    for entry_id in ordered_entry_ids:
        values = [
            _normalize_entry_lines(entry_map.get(entry_id)) for entry_map in entries_by_exp
        ]
        baseline = next((value for value in values if value is not None), None)
        if baseline is None:
            continue
        if any(value is None or value != baseline for value in values):
            diff_entry_ids.append(entry_id)

    if not diff_entry_ids:
        return ["(no metadata differences)" for _ in metadata_texts]

    section_order = _ordered_sections(ordered_entry_ids)
    outputs: List[str] = []
    for entry_map in entries_by_exp:
        lines = _render_metadata_diff(entry_map, diff_entry_ids, section_order)
        outputs.append("\n".join(lines) if lines else "(no metadata differences)")
    return outputs


def _ordered_sections(entry_ids: Iterable[Tuple[str, str]]) -> List[str]:
    seen = set()
    ordered = []
    for section, _ in entry_ids:
        if section not in seen:
            seen.add(section)
            ordered.append(section)
    return ordered


def _normalize_entry_lines(lines: List[str] | None) -> str | None:
    if lines is None:
        return None
    return "\n".join(lines).rstrip()


def _parse_metadata_entries(metadata_text: str) -> List[Tuple[str, str, List[str]]]:
    entries: List[Tuple[str, str, List[str]]] = []
    section = ""
    lines = metadata_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip()
            i += 1
            continue
        if "=" in line and line.lstrip() == line:
            key = line.split("=", 1)[0].strip()
            entry_lines = [line]
            value_part = line.split("=", 1)[1]
            bracket_balance = value_part.count("[") - value_part.count("]")
            if bracket_balance > 0 and not value_part.strip().endswith("]"):
                i += 1
                while i < len(lines):
                    entry_lines.append(lines[i])
                    bracket_balance += lines[i].count("[") - lines[i].count("]")
                    if bracket_balance <= 0 and lines[i].strip().endswith("]"):
                        break
                    i += 1
            entries.append((section, key, entry_lines))
        i += 1
    return entries


def _render_metadata_diff(
    entry_map: Dict[Tuple[str, str], List[str]],
    diff_entry_ids: Iterable[Tuple[str, str]],
    section_order: Iterable[str],
) -> List[str]:
    lines: List[str] = []
    for section in section_order:
        section_entries = [entry_id for entry_id in diff_entry_ids if entry_id[0] == section]
        if not section_entries:
            continue
        if lines:
            lines.append("")
        if section:
            lines.append(f"[{section}]")
        for _, key in section_entries:
            entry_id = (section, key)
            if entry_id in entry_map:
                lines.extend(entry_map[entry_id])
            else:
                lines.append(f"{key} = doesn't exist")
    return lines
