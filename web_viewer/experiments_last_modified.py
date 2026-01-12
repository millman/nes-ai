from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional


def _get_last_modified(
    path: Path,
    *,
    profile: Optional[Callable[..., None]] = None,
) -> Optional[datetime]:
    """Get the most recent modification time for files under metrics/."""
    if not path.is_dir():
        return None
    metrics_dir = path / "metrics"
    if not metrics_dir.is_dir():
        return None
    start_time = time.perf_counter()
    latest_mtime: Optional[float] = None
    files_scanned = 0
    try:
        for item in metrics_dir.rglob("*"):
            if item.is_file():
                try:
                    mtime = item.stat().st_mtime
                    files_scanned += 1
                    if latest_mtime is None or mtime > latest_mtime:
                        latest_mtime = mtime
                except OSError:
                    continue
    except OSError:
        return None
    if latest_mtime is None:
        try:
            return datetime.fromtimestamp(metrics_dir.stat().st_mtime)
        except OSError:
            return None
    if profile:
        profile("last_modified.rglob", start_time, metrics_dir, files=files_scanned)
    return datetime.fromtimestamp(latest_mtime)


def _quick_last_modified(path: Path) -> Optional[datetime]:
    """Fast last-modified using depth-1 files under metrics/."""
    metrics_dir = path / "metrics"
    if not metrics_dir.is_dir():
        return None
    latest_mtime: Optional[float] = None
    try:
        for child in metrics_dir.iterdir():
            try:
                mtime = child.stat().st_mtime
                if latest_mtime is None or mtime > latest_mtime:
                    latest_mtime = mtime
            except OSError:
                continue
    except OSError:
        return None
    if latest_mtime is None:
        try:
            return datetime.fromtimestamp(metrics_dir.stat().st_mtime)
        except OSError:
            return None
    return datetime.fromtimestamp(latest_mtime)
