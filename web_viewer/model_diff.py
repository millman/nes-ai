from __future__ import annotations

import argparse
import importlib.util
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tomli
import tomli_w

from jepa_world_model.metadata import write_run_metadata


def _run_git(args: list[str], cwd: Optional[Path] = None) -> Tuple[str, int]:
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(cwd) if cwd else None,
    )
    output = result.stdout.strip() or result.stderr.strip()
    return output, result.returncode


def _extract_commit(metadata_git_text: str) -> Optional[str]:
    lines = metadata_git_text.splitlines()
    seen_header = False
    for line in lines:
        stripped = line.strip()
        if not stripped and not seen_header:
            continue
        if not seen_header:
            if stripped.lower().startswith("git commit"):
                seen_header = True
            continue
        if stripped:
            return stripped
    return None


def _load_configs_from_source(source: str) -> Tuple[Any, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        module_path = Path(tmpdir) / "trainer_checked_in.py"
        module_path.write_text(source)
        spec = importlib.util.spec_from_file_location("trainer_checked_in", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to build module spec for trainer source")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        if not hasattr(module, "TrainConfig") or not hasattr(module, "ModelConfig"):
            raise RuntimeError("Trainer module missing TrainConfig or ModelConfig")
        return module.TrainConfig(), module.ModelConfig()


def _write_base_metadata(commit: str, repo_root: Path, trainer_path: str, cache_root: Path) -> Tuple[Optional[Path], Optional[str]]:
    target_dir = cache_root / commit
    metadata_path = target_dir / "metadata.txt"
    if metadata_path.exists():
        return target_dir, None
    source, code = _run_git(["git", "show", f"{commit}:{trainer_path}"], cwd=repo_root)
    if code != 0 or not source:
        return None, f"git show failed for {commit}:{trainer_path}: {source}"
    try:
        train_cfg, model_cfg = _load_configs_from_source(source)
        target_dir.mkdir(parents=True, exist_ok=True)
        write_run_metadata(target_dir, train_cfg, model_cfg, exclude_fields={"title"})
    except Exception as exc:
        return None, f"Failed to write base metadata: {exc}"
    if not metadata_path.exists():
        return None, "metadata.txt missing after write_run_metadata"
    return target_dir, None


def _read_configs_from_metadata(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    try:
        data = tomli.loads(path.read_text())
        return data.get("train_config"), data.get("model_config"), None
    except Exception as exc:
        return None, None, f"Failed to parse {path.name}: {exc}"


def _diff_dicts(current: Any, base: Any) -> Optional[Any]:
    if isinstance(current, dict) and isinstance(base, dict):
        merged: Dict[str, Any] = {}
        for key in sorted(set(current.keys()) | set(base.keys())):
            diff = _diff_dicts(current.get(key), base.get(key))
            if diff is not None:
                merged[key] = diff
        return merged or None
    if current == base:
        return None
    return {"current": current, "base": base}


def generate_model_diff(exp_dir: Path, trainer_path: str = "jepa_world_model_trainer.py", repo_root: Optional[Path] = None) -> Path:
    cache_dir = exp_dir / "server_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "model_diff.txt"
    metadata_path = exp_dir / "metadata.txt"
    metadata_git_path = exp_dir / "metadata_git.txt"

    if cache_path.exists():
        return cache_path
    if not metadata_path.exists() or not metadata_git_path.exists():
        cache_path.write_text("error = \"metadata.txt or metadata_git.txt missing\"")
        return cache_path

    train_current, model_current, parse_err = _read_configs_from_metadata(metadata_path)
    if parse_err:
        cache_path.write_text(f"error = \"{parse_err}\"")
        return cache_path

    base_commit = _extract_commit(metadata_git_path.read_text())
    if base_commit is None:
        cache_path.write_text("error = \"Unable to extract base commit from metadata_git.txt\"")
        return cache_path

    git_root_text, code = _run_git(["git", "rev-parse", "--show-toplevel"], cwd=repo_root)
    if code != 0 or not git_root_text:
        cache_path.write_text(f"error = \"Failed to locate git root: {git_root_text}\"")
        return cache_path
    git_root = Path(git_root_text)
    trainer_rel = trainer_path
    try:
        trainer_rel = str((git_root / trainer_path).relative_to(git_root))
    except Exception:
        trainer_rel = trainer_path

    base_dir, base_err = _write_base_metadata(base_commit, git_root, trainer_rel, cache_dir)
    if base_err or base_dir is None:
        cache_path.write_text(f"error = \"{base_err}\"")
        return cache_path

    base_metadata_path = base_dir / "metadata.txt"
    train_base, model_base, base_parse_err = _read_configs_from_metadata(base_metadata_path)
    if base_parse_err:
        cache_path.write_text(f"error = \"{base_parse_err}\"")
        return cache_path

    payload: Dict[str, Any] = {}
    train_diff = _diff_dicts(train_current, train_base)
    model_diff = _diff_dicts(model_current, model_base)
    if train_diff:
        payload["train_config"] = train_diff
    if model_diff:
        payload["model_config"] = model_diff
    if not payload:
        payload["status"] = "No differences between current and base configs."
    cache_path.write_text(tomli_w.dumps(payload))
    return cache_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model diff for an experiment directory.")
    parser.add_argument("--experiment", type=Path, required=True, help="Path to experiment directory.")
    parser.add_argument("--trainer-path", type=str, default="jepa_world_model_trainer.py", help="Trainer path relative to repo root.")
    parser.add_argument("--repo-root", type=Path, default=None, help="Override git repo root.")
    args = parser.parse_args()
    cache_path = generate_model_diff(args.experiment, trainer_path=args.trainer_path, repo_root=args.repo_root)
    print(f"Wrote model diff to {cache_path}")


if __name__ == "__main__":
    main()
