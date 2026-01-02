from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tyro
from jepa_world_model.metadata import write_run_metadata, write_git_metadata

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jepa_world_model.model_config import ModelConfig
from jepa_world_model_trainer import TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump metadata.txt using the trainer metadata path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where metadata.txt and metadata_git.txt will be written.",
    )
    # Parse TrainConfig from CLI (same as trainer) to capture overrides.
    args, remaining = parser.parse_known_args()
    cfg = tyro.cli(TrainConfig, args=remaining, config=(tyro.conf.HelptextFromCommentsOff,))
    model_cfg = ModelConfig()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(out_dir, cfg, model_cfg, exclude_fields={"title"})
    write_git_metadata(out_dir)
    print(f"Metadata written to {out_dir / 'metadata.txt'}")


if __name__ == "__main__":
    main()
