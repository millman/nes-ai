from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import subprocess
import tomli_w

__all__ = ["write_run_metadata"]

_TOML_NULL_SENTINEL = "__TOML_NULL__3d67a1c3a66a4d62ad0f0dec1d18454b__"


def _run_git_command(args: List[str]) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return "git executable not found."
    if result.returncode != 0:
        stderr = result.stderr.strip()
        return f"Command {' '.join(args)} failed (code {result.returncode}): {stderr}"
    return result.stdout.strip()


def _serialize_for_json(value: Any):
    if value is None:
        return _TOML_NULL_SENTINEL
    if isinstance(value, dict):
        return {k: _serialize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_for_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_run_metadata(run_dir: Path, cfg: Any, model_cfg: Any) -> None:
    commit_sha = _run_git_command(["git", "rev-parse", "HEAD"])
    diff_output = _run_git_command(["git", "diff", "--patch"])
    if not diff_output.strip():
        diff_output = "No uncommitted changes."
    git_metadata = "\n".join(
        [
            "Git commit:",
            commit_sha or "Unavailable",
            "",
            "Git diff (uncommitted changes):",
            diff_output,
            "",
        ]
    )
    (run_dir / "metadata_git.txt").write_text(git_metadata)
    train_config = _serialize_for_json(asdict(cfg))
    model_config = _serialize_for_json(asdict(model_cfg))
    toml_payload: Dict[str, Any] = {
        "train_config": train_config,
        "model_config": model_config,
        "data_root": str(cfg.data_root),
    }
    metadata_text = tomli_w.dumps(toml_payload)
    metadata_text = metadata_text.replace(f'"{_TOML_NULL_SENTINEL}"', "null")
    (run_dir / "metadata.txt").write_text(metadata_text)
