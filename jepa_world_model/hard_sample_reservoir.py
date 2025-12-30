#!/usr/bin/env python3
"""Hard sample reservoir for mining difficult training sequences."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from jepa_world_model.data import TrajectorySequenceDataset


def _short_traj_state_label(frame_path: str) -> str:
    path = Path(frame_path)
    traj = next((part for part in path.parts if part.startswith("traj_")), path.parent.name)
    return f"{traj}/{path.stem}"


@dataclass
class HardSampleRecord:
    dataset_index: int
    score: float
    frame_path: str
    label: str
    sequence_paths: List[str]
    frame_index: int


class HardSampleReservoir:
    def __init__(self, capacity: int, sample_decay: float = 0.9, rng: random.Random = None) -> None:
        self.capacity = max(0, capacity)
        self.sample_decay = sample_decay
        self._samples: Dict[int, HardSampleRecord] = {}
        if rng is None:
            raise ValueError("HardSampleReservoir requires an explicit RNG; got None.")
        self.rng = rng

    def __len__(self) -> int:
        return len(self._samples)

    def update(
        self,
        indices: List[int],
        paths: List[List[str]],
        scores: List[float],
        frame_indices: List[int],
    ) -> None:
        if self.capacity <= 0:
            return
        for idx, seq_paths, score, frame_idx in zip(indices, paths, scores, frame_indices):
            if seq_paths is None or not seq_paths:
                continue
            score_val = float(score)
            if not math.isfinite(score_val):
                continue
            idx_int = int(idx)
            frame_list = list(seq_paths)
            frame_pos = max(0, min(int(frame_idx), len(frame_list) - 1))
            frame_path = frame_list[frame_pos]
            label = _short_traj_state_label(frame_path)
            record = self._samples.get(idx_int)
            if record is None:
                self._samples[idx_int] = HardSampleRecord(idx_int, score_val, frame_path, label, frame_list, frame_pos)
            else:
                if score_val >= record.score:
                    record.score = score_val
                    record.frame_path = frame_path
                    record.label = label
                    record.sequence_paths = frame_list
                    record.frame_index = frame_pos
                else:
                    record.score = (record.score * 0.75) + (score_val * 0.25)
        self._prune()

    def sample_records(self, count: int) -> List[HardSampleRecord]:
        if count <= 0 or not self._samples:
            return []
        population = list(self._samples.values())
        count = min(count, len(population))
        weights = [max(record.score, 1e-6) for record in population]
        chosen = self.rng.choices(population=population, weights=weights, k=count)
        return chosen

    def mark_sampled(self, dataset_index: int) -> None:
        record = self._samples.get(dataset_index)
        if record is None:
            return
        record.score *= self.sample_decay
        if record.score <= 1e-6:
            self._samples.pop(dataset_index, None)

    def topk(self, limit: int) -> List[HardSampleRecord]:
        if limit <= 0 or not self._samples:
            return []
        limit = min(limit, len(self._samples))
        return sorted(self._samples.values(), key=lambda rec: rec.score, reverse=True)[:limit]

    def _prune(self) -> None:
        if self.capacity <= 0 or len(self._samples) <= self.capacity:
            return
        ordered = sorted(self._samples.items(), key=lambda item: item[1].score, reverse=True)
        self._samples = dict(ordered[: self.capacity])


def inject_hard_examples_into_batch(
    batch: Tuple[torch.Tensor, torch.Tensor, List[List[str]], torch.Tensor],
    dataset: TrajectorySequenceDataset,
    reservoir: Optional[HardSampleReservoir],
    mix_ratio: float,
) -> None:
    if reservoir is None or mix_ratio <= 0:
        return
    images, actions, paths, indices = batch
    batch_size = images.shape[0]
    if batch_size == 0 or len(reservoir) == 0:
        return
    ratio = max(0.0, min(1.0, mix_ratio))
    desired = min(int(round(batch_size * ratio)), len(reservoir))
    if desired <= 0:
        return
    hard_records = reservoir.sample_records(desired)
    for slot, record in enumerate(hard_records):
        hard_obs, hard_actions, hard_paths, hard_index = dataset[record.dataset_index]
        images[slot].copy_(hard_obs)
        actions[slot].copy_(hard_actions)
        paths[slot] = list(hard_paths)
        indices[slot] = hard_index
        reservoir.mark_sampled(record.dataset_index)


__all__ = ["HardSampleRecord", "HardSampleReservoir", "inject_hard_examples_into_batch"]
