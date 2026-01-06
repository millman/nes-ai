"""Visualization batch rendering helpers."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from jepa_world_model.vis_rollout import RolloutBatchItem, VisualizationSequence


def _render_visualization_batch(
    *,
    items: List[RolloutBatchItem],
    labels: List[List[str]],
    gradients: List[List[Optional[np.ndarray]]],
    show_gradients: bool,
) -> Tuple[List[VisualizationSequence], str]:
    if len(items) != len(labels) or len(items) != len(gradients):
        raise ValueError("Visualization metadata must match the number of rollout items.")
    sequences: List[VisualizationSequence] = []
    grad_label = "Gradient Norm" if show_gradients else "Error Heatmap"
    for item, item_labels, item_gradients in zip(items, labels, gradients):
        sequences.append(
            VisualizationSequence(
                ground_truth=item.ground_truth,
                rollout=item.rollout,
                gradients=item_gradients,
                reconstructions=item.reconstructions,
                reencoded=item.reencoded,
                labels=item_labels,
                actions=item.actions,
            )
        )
    return sequences, grad_label


__all__ = ["_render_visualization_batch"]
