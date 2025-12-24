from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import tomli


NULL_PATTERN = re.compile(r'(?<!["\'])\bnull\b')
INLINE_COLON_PATTERN = re.compile(r'("?[A-Za-z0-9_\-]+"?)\s*:\s*')


def _iter_metadata_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    for exp_dir in root.iterdir():
        if not exp_dir.is_dir():
            continue
        meta_path = exp_dir / "metadata.txt"
        if meta_path.exists():
            yield meta_path


def _can_parse(path: Path) -> bool:
    try:
        tomli.loads(path.read_text())
        return True
    except tomli.TOMLDecodeError:
        return False


def _rewrite_nulls(text: str) -> str:
    # Replace bare null tokens (not already quoted) with the string "null".
    return NULL_PATTERN.sub('"null"', text)


def _rewrite_inline_table_colons(text: str) -> str:
    # TOML inline tables require '=' rather than ':'; convert common JSON-style syntax.
    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)} = "

    return INLINE_COLON_PATTERN.sub(repl, text)


def fix_metadata(root: Path) -> None:
    for meta_path in _iter_metadata_files(root):
        if _can_parse(meta_path):
            continue
        original = meta_path.read_text()
        fixed = _rewrite_inline_table_colons(_rewrite_nulls(original))
        if fixed == original:
            # Nothing changed; surface the original parse error.
            try:
                tomli.loads(fixed)
            except tomli.TOMLDecodeError as exc:
                print(f"Failed to fix {meta_path}: {exc}")
            else:
                print(f"Failed to fix {meta_path}: unknown parse issue")
            continue

        meta_path.write_text(fixed)
        try:
            tomli.loads(fixed)
            print(f"Rewrote {meta_path} (fixed null)")
        except tomli.TOMLDecodeError as exc:
            print(f"Rewrote {meta_path} but still unparseable: {exc}")


def _can_parse_text(text: str) -> bool:
    try:
        tomli.loads(text)
        return True
    except tomli.TOMLDecodeError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix metadata.txt files with bare null values (e.g., max_trajectories = null)."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("out.jepa_world_model_trainer"),
        help="Root directory containing experiment subdirectories with metadata.txt files.",
    )
    args = parser.parse_args()
    fix_metadata(args.root)


if __name__ == "__main__":
    main()
