from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ViewerConfig:
    """Runtime configuration for the experiment viewer."""

    output_dir: Path = Path("out.jepa_world_model_trainer")
    host: str = "127.0.0.1"
    port: int = 5001
    debug: bool = False
    refresh_seconds: int = 0

    @staticmethod
    def from_env(
        output_dir: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        debug: Optional[bool] = None,
    ) -> "ViewerConfig":
        cfg = ViewerConfig()
        return ViewerConfig(
            output_dir=Path(output_dir) if output_dir is not None else cfg.output_dir,
            host=host or cfg.host,
            port=port or cfg.port,
            debug=debug if debug is not None else cfg.debug,
            refresh_seconds=cfg.refresh_seconds,
        )
