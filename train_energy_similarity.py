#!/usr/bin/env python3
"""Simple alternating grouping + embedding training for trajectory frames."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from torch.utils.data import DataLoader

from jepa_world_model.conv_encoder_decoder import Encoder
from jepa_world_model.data import TrajectorySequenceDataset
from jepa_world_model.loss_sigreg import sigreg_loss
from jepa_world_model.model_config import ModelConfig
from recon.data import list_trajectories, load_frame_as_tensor


@dataclass
class GroupStats:
    group_ids: Dict[str, int]
    group_sizes: Dict[int, int]


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, idx: int) -> int:
        while self.parent[idx] != idx:
            self.parent[idx] = self.parent[self.parent[idx]]
            idx = self.parent[idx]
        return idx

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


@dataclass
class Args:
    data_root: Path = Path("data.gridworldkey_wander_to_key")
    image_size: int = 64
    seq_len: int = 8
    batch_size: int = 8
    steps: int = 10_000
    lr: float = 1e-4
    weight_decay: float = 0.03
    device: str = "mps"
    seed: int = 0
    regroup_every: int = 250
    group_window: int = 2
    pixel_delta_threshold: float = 0.01
    embed_cos_threshold: float = 0.95
    sigreg_weight: float = 0.001
    sigreg_projections: int = 64
    group_weight: float = 50.0
    vis_every: int = 200
    num_workers: int = 0
    title: Annotated[
        str,
        tyro.conf.arg(aliases=["-m"]),
    ] = ""


def downsample_gray(frame: torch.Tensor, out_hw: int = 16) -> torch.Tensor:
    if frame.ndim != 3:
        raise ValueError("Expected CHW frame tensor.")
    gray = frame.mean(dim=0, keepdim=True)
    gray = gray.unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(gray, (out_hw, out_hw))
    return pooled.squeeze(0)


def mean_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def embed_frames(
    model: Encoder,
    frames: List[torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            chunk = torch.stack(frames[start : start + batch_size], dim=0).to(device)
            emb = model(chunk).detach().cpu().numpy()
            outputs.append(emb)
    if not outputs:
        return np.zeros((0, model.latent_dim), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def compute_group_stats(
    model: Encoder,
    data_root: Path,
    image_hw: Tuple[int, int],
    device: torch.device,
    group_window: int,
    pixel_delta_threshold: float,
    embed_cos_threshold: float,
    batch_size: int,
) -> GroupStats:
    trajectories = list_trajectories(data_root)
    path_list: List[Path] = []
    traj_spans: List[Tuple[int, int, List[Path]]] = []
    for _, paths in trajectories.items():
        start = len(path_list)
        path_list.extend(paths)
        traj_spans.append((start, len(paths), paths))

    uf = UnionFind(len(path_list))
    for offset, length, paths in traj_spans:
        frames: List[torch.Tensor] = [
            load_frame_as_tensor(path, size=image_hw) for path in paths
        ]
        small_frames = [downsample_gray(frame) for frame in frames]
        embeddings = embed_frames(model, frames, device, batch_size)
        for i in range(length):
            max_j = min(length, i + group_window + 1)
            for j in range(i + 1, max_j):
                delta = mean_abs_delta(small_frames[i], small_frames[j])
                if delta <= pixel_delta_threshold:
                    uf.union(offset + i, offset + j)
                    continue
                if embed_cos_threshold > 0.0:
                    if cosine_sim(embeddings[i], embeddings[j]) >= embed_cos_threshold:
                        uf.union(offset + i, offset + j)

    group_ids: Dict[str, int] = {}
    for idx, path in enumerate(path_list):
        group_ids[str(path)] = uf.find(idx)
    group_sizes: Dict[int, int] = {}
    for gid in group_ids.values():
        group_sizes[gid] = group_sizes.get(gid, 0) + 1
    return GroupStats(group_ids=group_ids, group_sizes=group_sizes)


def group_pull_loss(
    embeddings: torch.Tensor,
    group_ids: List[int],
) -> torch.Tensor:
    if embeddings.shape[0] != len(group_ids):
        raise ValueError("Embeddings and group_ids length mismatch.")
    if embeddings.numel() == 0:
        return embeddings.new_tensor(0.0)
    loss_terms: List[torch.Tensor] = []
    ids_tensor = torch.tensor(group_ids, device=embeddings.device)
    unique_ids = torch.unique(ids_tensor)
    for gid in unique_ids:
        mask = ids_tensor == gid
        if mask.sum() < 2:
            continue
        group_emb = embeddings[mask]
        mean_emb = group_emb.mean(dim=0, keepdim=True)
        loss_terms.append(((group_emb - mean_emb) ** 2).mean())
    if not loss_terms:
        return embeddings.new_tensor(0.0)
    return torch.stack(loss_terms).mean()


def plot_group_sizes(group_sizes: Dict[int, int], out_path: Path, step: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sizes = np.array(list(group_sizes.values()), dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6, 4))
    if sizes.size:
        ax.hist(sizes, bins=30, color="tab:blue", alpha=0.8)
        ax.set_xlabel("group size")
        ax.set_ylabel("count")
    else:
        ax.text(0.5, 0.5, "No groups.", ha="center", va="center")
        ax.axis("off")
    ax.set_title(f"Group size distribution (step {step})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_pca(embeddings: np.ndarray, out_path: Path, step: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if embeddings.shape[0] < 2:
        return
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        jitter = np.random.normal(scale=1e-6, size=centered.shape)
        _, _, vt = np.linalg.svd(centered + jitter, full_matrices=False)
    proj = centered @ vt[:2].T
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = np.linspace(0.0, 1.0, proj.shape[0])
    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=colors, cmap="viridis", s=10)
    fig.colorbar(scatter, ax=ax, label="frame index")
    ax.set_title(f"Embedding PCA (step {step})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_curves(
    steps: List[int],
    loss_total: List[float],
    loss_group: List[float],
    loss_sigreg: List[float],
    loss_group_weighted: List[float],
    loss_sigreg_weighted: List[float],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, loss_total, label="loss_total")
    ax.plot(steps, loss_group, label="loss_group")
    ax.plot(steps, loss_sigreg, label="loss_sigreg")
    ax.plot(steps, loss_group_weighted, label="loss_group_weighted")
    ax.plot(steps, loss_sigreg_weighted, label="loss_sigreg_weighted")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training losses")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_metrics_row(path: Path, row: Dict[str, float], header: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = tyro.cli(
        Args,
        config=(tyro.conf.HelptextFromCommentsOff,),
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    script_name = Path(__file__).stem
    output_root = Path(f"out.{script_name}")
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = run_dir / "plots"
    metrics_path = run_dir / "metrics.csv"
    metrics_plot_path = plot_dir / "metrics.png"

    device = torch.device(args.device)
    model_cfg = ModelConfig(image_size=args.image_size)
    encoder = Encoder(
        in_channels=model_cfg.in_channels,
        schedule=model_cfg.encoder_schedule,
        input_hw=model_cfg.image_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    dataset = TrajectorySequenceDataset(
        root=args.data_root,
        seq_len=args.seq_len,
        image_hw=(args.image_size, args.image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    loader_iter = iter(loader)

    group_stats = compute_group_stats(
        encoder,
        args.data_root,
        (args.image_size, args.image_size),
        device,
        args.group_window,
        args.pixel_delta_threshold,
        args.embed_cos_threshold,
        args.batch_size,
    )
    plot_group_sizes(
        group_stats.group_sizes,
        plot_dir / f"group_sizes_{0:07d}.png",
        0,
    )

    sample_paths = next(iter(list_trajectories(args.data_root).values()))
    sample_frames = [load_frame_as_tensor(path, size=(args.image_size, args.image_size)) for path in sample_paths]

    steps_hist: List[int] = []
    loss_total_hist: List[float] = []
    loss_group_hist: List[float] = []
    loss_sigreg_hist: List[float] = []
    loss_group_weighted_hist: List[float] = []
    loss_sigreg_weighted_hist: List[float] = []

    for step in range(args.steps + 1):
        if step > 0 and step % args.regroup_every == 0:
            group_stats = compute_group_stats(
                encoder,
                args.data_root,
                (args.image_size, args.image_size),
                device,
                args.group_window,
                args.pixel_delta_threshold,
                args.embed_cos_threshold,
                args.batch_size,
            )
            plot_group_sizes(
                group_stats.group_sizes,
                plot_dir / f"group_sizes_{step:07d}.png",
                step,
            )

        if step % args.vis_every == 0:
            encoder.eval()
            with torch.no_grad():
                emb = embed_frames(encoder, sample_frames, device, args.batch_size)
            plot_embedding_pca(emb, plot_dir / f"embedding_pca_{step:07d}.png", step)

        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        frames, _, paths, _ = batch
        frames = frames.to(device)
        batch_size, seq_len, _, _, _ = frames.shape
        flat_frames = frames.reshape(batch_size * seq_len, *frames.shape[2:])
        embeddings = encoder(flat_frames)
        group_ids: List[int] = []
        for seq_paths in paths:
            for path in seq_paths:
                group_ids.append(group_stats.group_ids.get(path, -1))
        loss_group = group_pull_loss(embeddings, group_ids)
        embeddings_seq = embeddings.view(batch_size, seq_len, -1)
        loss_sigreg = sigreg_loss(embeddings_seq, args.sigreg_projections)
        loss_group_weighted = args.group_weight * loss_group
        loss_sigreg_weighted = args.sigreg_weight * loss_sigreg
        loss_total = loss_group_weighted + loss_sigreg_weighted

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        loss_total_val = float(loss_total.item())
        loss_group_val = float(loss_group.item())
        loss_sigreg_val = float(loss_sigreg.item())
        row = {
            "step": float(step),
            "loss_total": loss_total_val,
            "loss_group": loss_group_val,
            "loss_sigreg": loss_sigreg_val,
            "loss_group_weighted": float(loss_group_weighted.item()),
            "loss_sigreg_weighted": float(loss_sigreg_weighted.item()),
        }
        write_metrics_row(metrics_path, row, row.keys())
        steps_hist.append(step)
        loss_total_hist.append(loss_total_val)
        loss_group_hist.append(loss_group_val)
        loss_sigreg_hist.append(loss_sigreg_val)
        loss_group_weighted_hist.append(float(loss_group_weighted.item()))
        loss_sigreg_weighted_hist.append(float(loss_sigreg_weighted.item()))
        if step % args.vis_every == 0:
            plot_metrics_curves(
                steps_hist,
                loss_total_hist,
                loss_group_hist,
                loss_sigreg_hist,
                loss_group_weighted_hist,
                loss_sigreg_weighted_hist,
                metrics_plot_path,
            )
        if step % 50 == 0:
            print(
                f"[{step:05d}] loss_total={row['loss_total']:.4f} "
                f"loss_group={row['loss_group']:.4f} loss_sigreg={row['loss_sigreg']:.4f} "
                f"loss_group_w={row['loss_group_weighted']:.4f} "
                f"loss_sigreg_w={row['loss_sigreg_weighted']:.4f}"
            )


if __name__ == "__main__":
    main()
