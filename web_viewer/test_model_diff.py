from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from web_viewer.model_diff import generate_model_diff, _write_base_metadata, _read_configs_from_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test model diff generation.")
    parser.add_argument("--experiment", type=Path, required=True, help="Experiment directory with metadata.txt and metadata_git.txt.")
    parser.add_argument("--trainer-path", type=str, default="jepa_world_model_trainer.py", help="Trainer path relative to repo root.")
    parser.add_argument("--repo-root", type=Path, default=None, help="Override git repo root.")
    args = parser.parse_args()

    cache_path = generate_model_diff(args.experiment, trainer_path=args.trainer_path, repo_root=args.repo_root)
    print(f"Diff written to: {cache_path}")
    try:
        print(cache_path.read_text())
    except OSError as exc:
        print(f"Failed to read diff: {exc}")

    # Optional: show base metadata location
    base_commit = None
    metadata_git = args.experiment / "metadata_git.txt"
    if metadata_git.exists():
        from web_viewer.model_diff import _extract_commit

        base_commit = _extract_commit(metadata_git.read_text())
    if base_commit:
        from web_viewer.model_diff import _run_git

        git_root_text, code = _run_git(["git", "rev-parse", "--show-toplevel"], cwd=args.repo_root)
        if code == 0 and git_root_text:
            base_dir, err = _write_base_metadata(base_commit, Path(git_root_text), args.trainer_path, args.experiment / "server_cache")
            if base_dir and not err:
                train_base, model_base, _ = _read_configs_from_metadata(base_dir / "metadata.txt")
                print("Base metadata path:", base_dir)
                print("Base train keys:", list(train_base.keys())[:5] if train_base else None)
                print("Base model keys:", list(model_base.keys())[:5] if model_base else None)


if __name__ == "__main__":
    main()
