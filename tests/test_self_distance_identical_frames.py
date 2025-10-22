"""Regression helper for self-distance drift on repeated frames.

Creates a synthetic trajectory consisting of the *same* RGB image saved
multiple times, then mirrors the ResNet-based self-distance computation
from `predict_mario_ms_ssim_eval.compute_self_distance_metrics`.

The expectation is that identical inputs should yield zero distance from
frame 0. If this test fails, it surfaces whatever component introduces
drift (e.g. device-specific numerics or transforms).
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pytest
import torch
from PIL import Image

from predict_mario_ms_ssim import default_transform, pick_device


def _make_identical_traj(root: Path, count: int = 24, size: tuple[int, int] = (64, 64)) -> Path:
    """Populate `traj_000/states` under `root` with `count` copies of one image."""
    states_dir = root / "traj_000" / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, (123, 77, 45))
    for idx in range(count):
        img.save(states_dir / f"frame_{idx:04d}.png")
    return states_dir


def _load_backbone(device: torch.device) -> torch.nn.Module:
    """Mirror the pretrained ResNet18 setup from the evaluator."""
    try:
        pytest.importorskip("torchvision")
        from torchvision.models import ResNet18_Weights, resnet18  # type: ignore
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception as exc:  # pragma: no cover - helps diagnose missing weights offline
        pytest.skip(f"ResNet18 DEFAULT weights unavailable: {exc}")
    backbone.fc = torch.nn.Identity()
    backbone.eval().to(device)
    return backbone


def _measure_distances(stack: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return L2 and cosine self-distances computed on `device`."""
    backbone = _load_backbone(device)

    feats = []
    with torch.no_grad():
        for frame in stack:
            feat = backbone(frame.unsqueeze(0).to(device)).squeeze(0).cpu()
            feats.append(feat)
    feats_tensor = torch.stack(feats)

    base_feat = feats_tensor[0]
    l2_vals = torch.norm(feats_tensor - base_feat, dim=1)
    cos_vals = 1.0 - torch.nn.functional.cosine_similarity(
        feats_tensor, base_feat.unsqueeze(0), dim=1
    )
    return l2_vals, cos_vals


def test_self_distance_constant_for_identical_frames(tmp_path: Path) -> None:
    # Minimize OpenMP resource usage in sandboxed environments.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    states_dir = _make_identical_traj(tmp_path)
    transform = default_transform()

    tensors = []
    for frame_path in sorted(states_dir.iterdir()):
        with Image.open(frame_path).convert("RGB") as img:
            tensors.append(transform(img))
    stack = torch.stack(tensors)  # [N,3,H,W]

    # Sanity check: transforms should yield identical tensors.
    max_pixel_delta = (stack - stack[0]).abs().max().item()
    assert max_pixel_delta == pytest.approx(0.0, abs=1e-6)

    cpu_device = torch.device("cpu")
    cpu_l2, cpu_cos = _measure_distances(stack, cpu_device)
    assert cpu_l2.max().item() == pytest.approx(0.0, abs=1e-5)
    assert cpu_cos.max().item() == pytest.approx(0.0, abs=1e-5)

    test_device = pick_device(None)
    if test_device.type != "cpu":
        dev_l2, dev_cos = _measure_distances(stack, test_device)
        assert dev_l2.max().item() == pytest.approx(0.0, abs=1e-5)
        assert dev_cos.max().item() == pytest.approx(0.0, abs=1e-5)
