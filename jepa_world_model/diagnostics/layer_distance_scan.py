"""Layer distance scan: measures per-layer feature sensitivity to small frame changes."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class LayerDistanceResult:
    """Result of computing distance for a single layer."""

    layer_name: str
    mean_l2_distance: float
    mean_abs_distance: float
    num_elements: int


@dataclass
class LayerDistanceScanResult:
    """Aggregate results from a layer distance scan."""

    layer_results: List[LayerDistanceResult] = field(default_factory=list)
    num_pairs: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_pairs": self.num_pairs,
            "layers": [
                {
                    "layer_name": r.layer_name,
                    "mean_l2_distance": r.mean_l2_distance,
                    "mean_abs_distance": r.mean_abs_distance,
                    "num_elements": r.num_elements,
                }
                for r in self.layer_results
            ],
        }

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["layer_name", "mean_l2_distance", "mean_abs_distance", "num_elements"])
            for r in self.layer_results:
                writer.writerow([r.layer_name, r.mean_l2_distance, r.mean_abs_distance, r.num_elements])

    def print_summary(self) -> None:
        print(f"\nLayer Distance Scan Results ({self.num_pairs} pairs)")
        print("-" * 70)
        print(f"{'Layer':<40} {'L2 Dist':>12} {'Abs Dist':>12}")
        print("-" * 70)
        for r in self.layer_results:
            print(f"{r.layer_name:<40} {r.mean_l2_distance:>12.6f} {r.mean_abs_distance:>12.6f}")
        print("-" * 70)


class ActivationHookManager:
    """Manages forward hooks to capture intermediate activations from a model."""

    def __init__(self) -> None:
        self._activations: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def register_hooks(self, model: nn.Module, layer_names: Optional[List[str]] = None) -> None:
        """Register forward hooks on specified layers or all named modules.

        Args:
            model: The model to register hooks on.
            layer_names: Optional list of layer names to hook. If None, hooks all
                modules that have parameters (excluding container modules).
        """
        self.clear()

        for name, module in model.named_modules():
            # Skip the root module
            if not name:
                continue
            # Skip container modules (Sequential, ModuleList, etc.)
            if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                continue
            # If layer_names specified, only hook those
            if layer_names is not None and name not in layer_names:
                continue
            # Only hook modules that have children or are leaf modules
            if list(module.children()):
                continue

            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(module: nn.Module, input: Any, output: Any) -> None:
            if isinstance(output, torch.Tensor):
                self._activations[name] = output.detach()
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                # Some modules return tuples (e.g., encoder returns pooled, detail_skip)
                self._activations[name] = output[0].detach()

        return hook

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Return captured activations."""
        return dict(self._activations)

    def clear_activations(self) -> None:
        """Clear captured activations."""
        self._activations.clear()

    def clear(self) -> None:
        """Remove all hooks and clear activations."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._activations.clear()


def compute_layer_distances(
    activations_t: Dict[str, torch.Tensor],
    activations_t1: Dict[str, torch.Tensor],
) -> Dict[str, Tuple[float, float, int]]:
    """Compute L2 and absolute distances between two sets of activations.

    Args:
        activations_t: Activations from frame t.
        activations_t1: Activations from frame t+1.

    Returns:
        Dict mapping layer_name to (l2_distance, abs_distance, num_elements).
    """
    results: Dict[str, Tuple[float, float, int]] = {}

    common_layers = set(activations_t.keys()) & set(activations_t1.keys())

    for layer_name in common_layers:
        act_t = activations_t[layer_name]
        act_t1 = activations_t1[layer_name]

        if act_t.shape != act_t1.shape:
            continue

        # Flatten and compute distances
        flat_t = act_t.flatten()
        flat_t1 = act_t1.flatten()
        num_elements = flat_t.numel()

        # L2 distance normalized by number of elements
        l2_dist = (flat_t - flat_t1).pow(2).sum().sqrt().item() / num_elements

        # Mean absolute distance
        abs_dist = (flat_t - flat_t1).abs().mean().item()

        results[layer_name] = (l2_dist, abs_dist, num_elements)

    return results


def run_layer_distance_scan(
    encoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_pairs: int = 100,
    layer_names: Optional[List[str]] = None,
) -> LayerDistanceScanResult:
    """Run layer distance scan to measure per-layer sensitivity to small frame changes.

    Args:
        encoder: The encoder model to analyze.
        dataloader: DataLoader yielding batches of (images, actions, paths, indices).
            Images should be shape (B, T, C, H, W) where T >= 2.
        device: Device to run inference on.
        max_pairs: Maximum number of frame pairs to analyze.
        layer_names: Optional list of specific layer names to hook.

    Returns:
        LayerDistanceScanResult with per-layer distance statistics.
    """
    encoder.eval()
    hook_manager = ActivationHookManager()
    hook_manager.register_hooks(encoder, layer_names)

    # Accumulators for each layer
    layer_accumulators: Dict[str, List[Tuple[float, float, int]]] = {}
    num_pairs = 0

    try:
        with torch.no_grad():
            for batch in dataloader:
                images = batch[0].to(device)
                batch_size, seq_len = images.shape[0], images.shape[1]

                if seq_len < 2:
                    continue

                # Process consecutive frame pairs
                for b in range(batch_size):
                    for t in range(seq_len - 1):
                        if num_pairs >= max_pairs:
                            break

                        frame_t = images[b, t].unsqueeze(0)
                        frame_t1 = images[b, t + 1].unsqueeze(0)

                        # Forward pass for frame t
                        hook_manager.clear_activations()
                        encoder(frame_t)
                        activations_t = hook_manager.get_activations()

                        # Forward pass for frame t+1
                        hook_manager.clear_activations()
                        encoder(frame_t1)
                        activations_t1 = hook_manager.get_activations()

                        # Compute distances
                        distances = compute_layer_distances(activations_t, activations_t1)

                        for layer_name, (l2_dist, abs_dist, num_elem) in distances.items():
                            if layer_name not in layer_accumulators:
                                layer_accumulators[layer_name] = []
                            layer_accumulators[layer_name].append((l2_dist, abs_dist, num_elem))

                        num_pairs += 1

                    if num_pairs >= max_pairs:
                        break
                if num_pairs >= max_pairs:
                    break

    finally:
        hook_manager.clear()

    # Compute mean distances per layer
    layer_results: List[LayerDistanceResult] = []
    for layer_name, measurements in sorted(layer_accumulators.items()):
        if not measurements:
            continue
        mean_l2 = sum(m[0] for m in measurements) / len(measurements)
        mean_abs = sum(m[1] for m in measurements) / len(measurements)
        num_elem = measurements[0][2]  # Same for all measurements of this layer
        layer_results.append(
            LayerDistanceResult(
                layer_name=layer_name,
                mean_l2_distance=mean_l2,
                mean_abs_distance=mean_abs,
                num_elements=num_elem,
            )
        )

    return LayerDistanceScanResult(layer_results=layer_results, num_pairs=num_pairs)
