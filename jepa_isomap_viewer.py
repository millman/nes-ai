#!/usr/bin/env python3
"""Standalone JEPA-style world model outline with Isomap visualization."""
from __future__ import annotations

import argparse
import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
from sklearn.manifold import Isomap
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Model components
# ----------------------------


def _norm_groups(out_ch: int) -> int:
    # Anchor to architecture snippet in reconstruct_comparison/blocks.py
    return max(1, out_ch // 8)


class DownBlock(nn.Module):
    """Strided contraction block that preserves channel locality."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups = _norm_groups(out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, channel_schedule: Sequence[int]):
        super().__init__()
        blocks = []
        ch_prev = in_channels
        for ch in channel_schedule:
            blocks.append(DownBlock(ch_prev, ch))
            ch_prev = ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.blocks(x)
        return self.pool(feats).flatten(1)


class StochasticLatentHead(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.SiLU(inplace=True),
            nn.Linear(in_dim, 2 * latent_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_logvar = self.net(x)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar


class PredictorNetwork(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, action_dim: int, embedding_dim: int):
        super().__init__()
        inp_dim = latent_dim + hidden_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, z: torch.Tensor, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z, h, action], dim=-1)
        return self.net(inp)


class VisualizationDecoder(nn.Module):
    def __init__(self, latent_dim: int, image_size: int):
        super().__init__()
        out_dim = 3 * image_size * image_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.SiLU(inplace=True),
            nn.Linear(512, 512),
            nn.SiLU(inplace=True),
            nn.Linear(512, out_dim),
        )
        self.image_size = image_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img = torch.sigmoid(self.net(x))
        img = img.view(-1, 3, self.image_size, self.image_size)
        return img


@dataclass
class WorldModelConfig:
    in_channels: int = 3
    channel_schedule: Tuple[int, ...] = (32, 64, 128, 256)
    latent_dim: int = 64
    hidden_dim: int = 128
    h_local_dim: int = 64
    h_global_dim: int = 64
    embedding_dim: int = 96
    action_dim: int = 4
    proj_vis_dim: int = 128


class JEPAWorldModel(nn.Module):
    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        if cfg.h_local_dim + cfg.h_global_dim != cfg.hidden_dim:
            raise ValueError("h_local_dim + h_global_dim must match hidden_dim")
        self.cfg = cfg
        self.encoder = Encoder(cfg.in_channels, cfg.channel_schedule)
        enc_out = cfg.channel_schedule[-1]
        self.stochastic = StochasticLatentHead(enc_out, cfg.latent_dim)
        self.gru = nn.GRUCell(cfg.latent_dim + cfg.action_dim, cfg.hidden_dim)
        emb_in = cfg.latent_dim + cfg.hidden_dim
        self.proj_e = nn.Sequential(
            nn.Linear(emb_in, cfg.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.embedding_dim),
        )
        self.proj_vis = nn.Sequential(
            nn.Linear(cfg.embedding_dim, cfg.proj_vis_dim),
            nn.SiLU(inplace=True),
        )
        self.latent_adapter = nn.Linear(cfg.embedding_dim, cfg.latent_dim)

    @property
    def hidden_dim(self) -> int:
        return self.cfg.hidden_dim

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    def split_hidden(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_local = h[:, : self.cfg.h_local_dim]
        h_global = h[:, self.cfg.h_local_dim : self.cfg.h_local_dim + self.cfg.h_global_dim]
        return h_local, h_global

    def transition(self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        gru_input = torch.cat([z, action], dim=-1)
        return self.gru(gru_input, h)

    def project_for_decoder(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.proj_vis(embedding)

    def latent_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.latent_adapter(embedding)

    def encode_step(
        self, x: torch.Tensor, h: torch.Tensor, action: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        feats = self.encoder(x)
        mu, logvar = self.stochastic(feats)
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps
        h_next = self.transition(h, z, action)
        h_local, h_global = self.split_hidden(h_next)
        embedding = self.proj_e(torch.cat([z, h_local, h_global], dim=-1))
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "h_next": h_next,
            "embedding": embedding,
            "h_local": h_local,
            "h_global": h_global,
        }


# ----------------------------
# Utility helpers
# ----------------------------


def pil_to_tensor(img: Image.Image, image_size: int) -> torch.Tensor:
    img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    array = t.detach().cpu().clamp(0, 1).numpy()
    array = np.transpose(array, (1, 2, 0))
    array = (array * 255.0).clip(0, 255).astype(np.uint8)
    return array


def load_images(paths: Sequence[Path], image_size: int) -> Tuple[List[torch.Tensor], List[np.ndarray], List[str]]:
    tensors: List[torch.Tensor] = []
    previews: List[np.ndarray] = []
    encoded: List[str] = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        tensor = pil_to_tensor(img, image_size)
        preview = tensor_to_image(tensor)
        tensors.append(tensor)
        previews.append(preview)
        encoded.append(pil_to_base64(img))
    return tensors, previews, encoded


def synthetic_actions(num_frames: int, action_dim: int) -> torch.Tensor:
    idx = torch.linspace(0.0, 1.0, steps=num_frames).unsqueeze(1)
    actions = torch.zeros(num_frames, action_dim)
    if action_dim >= 1:
        actions[:, 0] = torch.sin(idx[:, 0] * np.pi)
    if action_dim >= 2:
        actions[:, 1] = torch.cos(idx[:, 0] * np.pi)
    if action_dim >= 3:
        actions[:, 2] = idx[:, 0]
    if action_dim >= 4:
        actions[:, 3] = 1.0 - idx[:, 0]
    return actions


def load_actions_npz(action_path: Path) -> np.ndarray:
    if not action_path.is_file():
        raise FileNotFoundError(f"Missing action file: {action_path}")
    with np.load(action_path) as data:
        key = "actions" if "actions" in data.files else data.files[0]
        arr = np.asarray(data[key])
    if arr.ndim == 1:
        arr = arr[:, None]
    arr = np.ascontiguousarray(arr.astype(np.float32))
    print(f"Loaded actions {arr.shape} from {action_path}")
    return arr


def rollout_future_images(
    model: JEPAWorldModel,
    predictor: PredictorNetwork,
    decoder: VisualizationDecoder,
    start_z: torch.Tensor,
    start_h: torch.Tensor,
    future_actions: Sequence[torch.Tensor],
    rollout_steps: int,
) -> List[np.ndarray]:
    device = start_z.device
    z = start_z
    h = start_h
    outputs: List[np.ndarray] = []
    zero_action = torch.zeros(1, model.action_dim, device=device)
    with torch.no_grad():
        for step in range(rollout_steps):
            if step < len(future_actions):
                action = future_actions[step].unsqueeze(0).to(device)
            else:
                action = zero_action
            embedding_pred = predictor(z, h, action)
            vis_latent = model.project_for_decoder(embedding_pred)
            decoded = decoder(vis_latent)[0]
            outputs.append(tensor_to_image(decoded))
            z = model.latent_from_embedding(embedding_pred)
            h = model.transition(h, z, action)
    return outputs


def process_sequence(
    tensors: Sequence[torch.Tensor],
    model: JEPAWorldModel,
    predictor: PredictorNetwork,
    decoder: VisualizationDecoder,
    actions: torch.Tensor,
    rollout_steps: int,
) -> Tuple[np.ndarray, List[np.ndarray], List[List[np.ndarray]], List[torch.Tensor], List[torch.Tensor]]:
    device = torch.device("cpu")
    model = model.to(device)
    predictor = predictor.to(device)
    decoder = decoder.to(device)
    h = torch.zeros(1, model.hidden_dim, device=device)
    embeddings: List[torch.Tensor] = []
    reconstructions: List[np.ndarray] = []
    rollouts: List[List[np.ndarray]] = []
    z_states: List[torch.Tensor] = []
    h_states: List[torch.Tensor] = []
    with torch.no_grad():
        for idx, tensor in enumerate(tensors):
            x = tensor.unsqueeze(0).to(device)
            action = actions[idx : idx + 1].to(device)
            step = model.encode_step(x, h, action)
            h = step["h_next"]
            embeddings.append(step["embedding"].cpu())
            z_states.append(step["z"].cpu())
            h_states.append(h.cpu())
            vis_latent = model.project_for_decoder(step["embedding"].detach())
            recon_img = decoder(vis_latent.cpu())[0]
            reconstructions.append(tensor_to_image(recon_img))
            future_idxs = range(idx + 1, min(len(actions), idx + 1 + rollout_steps))
            future_actions = [actions[j] for j in future_idxs]
            rollout_imgs = rollout_future_images(
                model, predictor, decoder, step["z"], step["h_next"], future_actions, rollout_steps
            )
            rollouts.append(rollout_imgs)
    embedding_array = torch.stack(embeddings).numpy()
    return embedding_array, reconstructions, rollouts, z_states, h_states


def run_isomap(embeddings: np.ndarray, n_neighbors: int, n_components: int = 2) -> np.ndarray:
    if embeddings.shape[0] <= n_neighbors:
        raise ValueError("Need more samples than n_neighbors for Isomap")
    model = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    coords = model.fit_transform(embeddings)
    return coords


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    min_vals = coords.min(axis=0, keepdims=True)
    coords = coords - min_vals
    max_vals = coords.max(axis=0, keepdims=True)
    max_vals[max_vals == 0] = 1.0
    return coords / max_vals


def build_grid(
    ground_truth: Sequence[np.ndarray],
    reconstructions: Sequence[np.ndarray],
    rollouts: Sequence[Sequence[np.ndarray]],
    rows: int,
    output_path: Path,
) -> None:
    if not ground_truth:
        return
    image_rows: List[np.ndarray] = []
    max_rows = min(rows, len(ground_truth))
    for idx in range(max_rows):
        stack = [ground_truth[idx], reconstructions[idx]] + list(rollouts[idx])
        while len(stack) < 6:
            stack.append(stack[-1])
        row_arrays = [img.astype(np.uint8) for img in stack[:6]]
        concat_row = np.concatenate(row_arrays, axis=1)
        image_rows.append(concat_row)
    grid = np.concatenate(image_rows, axis=0)
    Image.fromarray(grid).save(output_path)


def make_html(
    coords: np.ndarray,
    metadata: Sequence[Dict[str, str]],
    encoded_images: Sequence[str],
    output_path: Path,
) -> None:
    points = []
    for coord, meta, encoded in zip(coords, metadata, encoded_images):
        points.append(
            {
                "x": float(coord[0]),
                "y": float(coord[1]),
                "idx": meta["index"],
                "label": meta["label"],
                "color": meta["color"],
                "image": encoded,
            }
        )
    plot_data = json.dumps(points)
    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<title>JEPA Latent Isomap Viewer</title>
<script src=\"https://cdn.plot.ly/plotly-2.27.0.min.js\"></script>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; display: flex; height: 100vh; }}
#plot {{ flex: 2; }}
#preview {{ flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; background: #111; color: #fff; }}
#preview img {{ max-width: 90%; max-height: 80%; border: 2px solid #fff; }}
</style>
</head>
<body>
<div id=\"plot\"></div>
<div id=\"preview\">
  <img id=\"previewImage\" alt=\"Hover to view\" />
  <p id=\"previewLabel\"></p>
</div>
<script>
const points = {plot_data};
const xs = points.map(p => p.x);
const ys = points.map(p => p.y);
const colors = points.map(p => p.color);
const labels = points.map(p => p.label);
const custom = points.map(p => ({image: p.image, label: p.label}));
const scatter = {{
    x: xs,
    y: ys,
    mode: 'markers',
    type: 'scattergl',
    marker: {{
        size: 9,
        color: colors,
        colorscale: 'Turbo',
        showscale: true,
        colorbar: {{title: 'Time'}}
    }},
    text: labels,
    customdata: custom,
    hovertemplate: '%{text}<extra></extra>'
}};
const trajectory = {{
    x: xs,
    y: ys,
    mode: 'lines',
    line: {{color: 'rgba(255,255,255,0.2)', width: 1}},
    hoverinfo: 'skip'
}};
const layout = {{
    dragmode: 'pan',
    paper_bgcolor: '#000',
    plot_bgcolor: '#000',
    xaxis: {{visible: false}},
    yaxis: {{visible: false}},
    margin: {{l: 20, r: 20, t: 20, b: 20}},
}};
const plot = document.getElementById('plot');
Plotly.newPlot(plot, [trajectory, scatter], layout);
const previewImg = document.getElementById('previewImage');
const previewLabel = document.getElementById('previewLabel');
let locked = null;
function updatePreview(data) {{
    previewImg.src = 'data:image/png;base64,' + data.image;
    previewLabel.textContent = data.label;
}}
plot.on('plotly_hover', (event) => {{
    if (locked) return;
    const data = event.points[0].customdata;
    updatePreview(data);
}});
plot.on('plotly_unhover', () => {{
    if (!locked) {{
        previewImg.removeAttribute('src');
        previewLabel.textContent = '';
    }}
}});
plot.on('plotly_click', (event) => {{
    const data = event.points[0].customdata;
    locked = data;
    updatePreview(data);
}});
plot.on('plotly_doubleclick', () => {{
    locked = null;
    previewImg.removeAttribute('src');
    previewLabel.textContent = '';
}});
</script>
</body>
</html>
"""
    output_path.write_text(html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JEPA + Isomap viewer pipeline")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory of RGB images")
    parser.add_argument(
        "--action-file",
        type=Path,
        default=None,
        help="Optional path to actions.npz (defaults to image_dir/../actions.npz if present)",
    )
    parser.add_argument("--max-images", type=int, default=128, help="Limit number of frames")
    parser.add_argument("--image-size", type=int, default=64, help="Square resize used by the encoder")
    parser.add_argument("--n-neighbors", type=int, default=12, help="k for Isomap")
    parser.add_argument("--rollout-steps", type=int, default=4, help="Number of future rollouts per sample")
    parser.add_argument("--grid-rows", type=int, default=4, help="Rows to show in the visualization grid")
    parser.add_argument("--output-html", type=Path, default=Path("isomap_viewer.html"))
    parser.add_argument("--output-grid", type=Path, default=Path("isomap_grid.png"))
    parser.add_argument("--coords-path", type=Path, default=Path("isomap_coords.npy"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = sorted([p for p in args.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not image_paths:
        raise ValueError(f"No images found in {args.image_dir}")
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    action_array: np.ndarray | None = None
    action_source: Path | None = args.action_file
    if action_source is None:
        default_source = args.image_dir.parent / "actions.npz"
        if default_source.is_file():
            action_source = default_source
    if action_source is not None:
        if not action_source.is_file():
            raise FileNotFoundError(f"Unable to locate action file: {action_source}")
        action_array = load_actions_npz(action_source)
        usable = min(len(image_paths), action_array.shape[0])
        if usable == 0:
            raise ValueError("No overlapping timesteps between images and actions")
        if usable < len(image_paths):
            print(f"Truncating images from {len(image_paths)} to {usable} to match actions")
            image_paths = image_paths[:usable]
        if usable < action_array.shape[0]:
            action_array = action_array[:usable]
    cfg = WorldModelConfig(action_dim=action_array.shape[1]) if action_array is not None else WorldModelConfig()
    tensors, previews, encoded_images = load_images(image_paths, args.image_size)
    world_model = JEPAWorldModel(cfg)
    predictor = PredictorNetwork(cfg.latent_dim, cfg.hidden_dim, cfg.action_dim, cfg.embedding_dim)
    decoder = VisualizationDecoder(cfg.proj_vis_dim, args.image_size)
    if action_array is not None:
        actions = torch.from_numpy(action_array)
    else:
        print("Using synthetic sinusoidal actions (no action file detected).")
        actions = synthetic_actions(len(tensors), cfg.action_dim)
    embeddings, reconstructions, rollouts, z_states, h_states = process_sequence(
        tensors, world_model, predictor, decoder, actions, args.rollout_steps
    )
    coords = run_isomap(embeddings, args.n_neighbors)
    coords = normalize_coords(coords)
    args.coords_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.coords_path, coords)
    metadata = []
    total = len(image_paths)
    for idx, path in enumerate(image_paths):
        metadata.append(
            {
                "index": idx,
                "label": path.name,
                "color": float(idx) / max(1, total - 1),
            }
        )
    args.output_grid.parent.mkdir(parents=True, exist_ok=True)
    build_grid(previews, reconstructions, rollouts, args.grid_rows, args.output_grid)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    make_html(coords, metadata, encoded_images, args.output_html)
    print(f"Saved HTML viewer to {args.output_html}")
    print(f"Saved Isomap coordinates to {args.coords_path}")
    print(f"Saved visualization grid to {args.output_grid}")


if __name__ == "__main__":
    main()
