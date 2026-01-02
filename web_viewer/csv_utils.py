from __future__ import annotations

import csv
import os
from pathlib import Path


def get_max_step(csv_path: Path) -> int:
    """Get the maximum step value from a CSV file with a step column."""
    with csv_path.open("r", newline="") as handle:
        header = handle.readline()
    if not header:
        raise ValueError(f"{csv_path} has no header row.")
    header_fields = next(csv.reader([header]))
    if "step" not in header_fields:
        raise ValueError(f"{csv_path} header missing 'step' column.")
    step_idx = header_fields.index("step")
    last_line = _read_last_non_empty_line(csv_path)
    if last_line:
        row = next(csv.reader([last_line]))
        if step_idx >= len(row):
            raise ValueError(f"{csv_path} last row missing step column.")
        return int(float(row[step_idx]))
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "step" not in reader.fieldnames:
            raise ValueError(f"{csv_path} header missing 'step' column.")
        max_step = None
        for row in reader:
            step = int(float(row.get("step", 0)))
            if max_step is None or step > max_step:
                max_step = step
        if max_step is None:
            raise ValueError(f"{csv_path} contains no step values.")
        return max_step


def _read_last_non_empty_line(path: Path, chunk_size: int = 8192) -> str:
    """Read the last non-empty line from a file without scanning it all."""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        end_pos = handle.tell()
        if end_pos == 0:
            return ""
        buffer = b""
        pos = end_pos
        while pos > 0:
            read_size = min(chunk_size, pos)
            pos -= read_size
            handle.seek(pos)
            buffer = handle.read(read_size) + buffer
            if b"\n" in buffer or pos == 0:
                lines = buffer.splitlines()
                for raw in reversed(lines):
                    if raw.strip():
                        return raw.decode("utf-8", errors="replace")
                if pos == 0:
                    return ""
        return ""
