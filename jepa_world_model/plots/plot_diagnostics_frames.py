"""Diagnostics frame output helper."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from jepa_world_model.actions import compress_actions_to_ids, decode_action_id
from jepa_world_model.vis import tensor_to_uint8_image


def save_diagnostics_frames(
    frames: torch.Tensor,
    paths: Optional[List[List[str]]],
    actions: Optional[torch.Tensor],
    out_dir: Path,
    global_step: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"frames_{global_step:07d}.csv"
    max_save = frames.shape[0]
    entries: List[Tuple[str, int]] = []

    def _natural_path_key(path_str: str) -> Tuple:
        parts = re.split(r"(\d+)", path_str)
        key: List = []
        for part in parts:
            if not part:
                continue
            key.append(int(part) if part.isdigit() else part.lower())
        return tuple(key)

    for idx in range(max_save):
        src_path = paths[idx][0] if paths and idx < len(paths) and paths[idx] else ""
        entries.append((src_path, idx))
    entries.sort(key=lambda t: _natural_path_key(t[0]))
    new_sources_sorted = [src for src, _ in entries]

    reuse_image_lookup: Optional[Dict[str, str]] = None
    for existing_csv in sorted(out_dir.glob("frames_*.csv")):
        try:
            with existing_csv.open("r", newline="") as handle:
                reader = csv.DictReader(handle)
                existing_records = list(reader)
        except (OSError, csv.Error):
            continue
        if not existing_records:
            continue
        existing_sources = [row.get("source_path", "") for row in existing_records]
        existing_sources_sorted = sorted(existing_sources, key=_natural_path_key)
        if len(existing_sources_sorted) != len(new_sources_sorted):
            continue
        if existing_sources_sorted == new_sources_sorted:
            reuse_image_lookup = {}
            for row in existing_records:
                src = row.get("source_path", "")
                img_rel = row.get("image_path", "")
                if src and img_rel:
                    reuse_image_lookup[src] = img_rel
            break

    records: List[Tuple[int, str, str, Optional[int], str]] = []
    if reuse_image_lookup:
        for out_idx, (src, orig_idx) in enumerate(entries):
            img_rel = reuse_image_lookup.get(src, "")
            if not img_rel:
                reuse_image_lookup = None
                records.clear()
                break
            action_id: Optional[int] = None
            action_label = ""
            if actions is not None and actions.ndim >= 2 and orig_idx < actions.shape[0]:
                action_vec = actions[orig_idx, 0].detach().cpu().numpy()
                action_id = int(compress_actions_to_ids(action_vec[None, ...])[0])
                action_label = decode_action_id(action_id, actions.shape[-1])
            records.append((out_idx, img_rel, src, action_id, action_label))

    if not records:
        step_dir = out_dir / f"frames_{global_step:07d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        for out_idx, (src, orig_idx) in enumerate(entries):
            frame_img = tensor_to_uint8_image(frames[orig_idx, 0])
            out_path = step_dir / f"frame_{out_idx:04d}.png"
            Image.fromarray(frame_img).save(out_path)
            action_id: Optional[int] = None
            action_label = ""
            if actions is not None and actions.ndim >= 2 and orig_idx < actions.shape[0]:
                action_vec = actions[orig_idx, 0].detach().cpu().numpy()
                action_id = int(compress_actions_to_ids(action_vec[None, ...])[0])
                action_label = decode_action_id(action_id, actions.shape[-1])
            records.append(
                (
                    out_idx,
                    out_path.relative_to(step_dir.parent).as_posix(),
                    src,
                    action_id,
                    action_label,
                )
            )

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "image_path", "source_path", "action_id", "action_label"])
        writer.writerows(records)
