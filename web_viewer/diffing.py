from __future__ import annotations

from difflib import unified_diff


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
