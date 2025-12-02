#!/usr/bin/env python3

# Standard library imports
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Third-party imports (import lines first, then from lines)
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import tyro
from PIL import Image
from sklearn.manifold import TSNE
from torchvision.models import ResNet18_Weights, resnet18
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm

from trajectory_utils import list_state_frames, list_traj_dirs

# Determine compute device (MPS for Mac, else CPU)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Type aliases
State = torch.Tensor  # Tensor shape: (channels, H, W)
Action = np.ndarray   # e.g., scalar or one-hot array

class MarioTrajectoryDataset(Dataset):
    """
    Dataset for multiple trajectories. Expects root_dir/:
      traj_*/
        states/
        actions.npz
    """
    def __init__(self,
                 root_dir: str,
                 transform: Optional[T.Compose] = None,
                 max_trajs: int | None = None,
    ) -> None:
        self.transform = transform or default_transform()
        self.index: List[Tuple[Path, List[Path], np.ndarray, int]] = []
        root = Path(root_dir)
        traj_dirs = set()
        for t, traj_path in enumerate(tqdm(list_traj_dirs(root), desc="Scanning trajectories")):
            if max_trajs and t > max_trajs:
                break
            if not traj_path.is_dir():
                continue
            states = traj_path / 'states'
            actions_file = traj_path / 'actions.npz'
            if not states.is_dir() or not actions_file.is_file():
                continue
            files = list_state_frames(states)
            # require at least 8 frames for 4 inputs + 4 outputs
            if len(files) < 8:
                continue
            with np.load(actions_file) as data:
                if 'actions' in data.files:
                    actions = data['actions']
                elif len(data.files) == 1:
                    actions = data[data.files[0]]
                else:
                    raise ValueError(f"Missing 'actions' in {actions_file}: {data.files}")
            traj_dirs.add(str(traj_path))
            # slide window for input 4 frames + next 4 frames
            for i in range(len(files) - 7):
                self.index.append((states, files, actions, i))
        self.num_trajectories = len(traj_dirs)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[State, Action, State]:
        states_dir, files, actions, offset = self.index[idx]
        # input sequence: 4 frames
        input_frames = []
        for i in range(4):
            with Image.open(files[offset + i]) as img:
                input_frames.append(self.transform(img))
        state = torch.cat(input_frames, dim=0)
        # ground truth next sequence: 4 frames
        gt_frames = []
        for i in range(4, 8):
            with Image.open(files[offset + i]) as img:
                gt_frames.append(self.transform(img))
        next_frames = torch.cat(gt_frames, dim=0)
        # action corresponding to last input frame
        action = actions[offset + 3]
        return state, action, next_frames


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = (nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        identity = self.skip(x)
        x = self.up(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity)


def default_transform() -> T.Compose:
    # use the ResNet18 IMAGENET1K_V1 preprocessing (resize→crop→normalize for 224×224)
    return ResNet18_Weights.DEFAULT.transforms()

class PredictionModel(nn.Module):
    def __init__(self, embedding_dim: int = 512, num_future_frames: int = 4) -> None:
        super().__init__()
        # load ResNet18 backbone pretrained for 224x224
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        # adjust first conv to accept 12 channels (4 stacked RGB frames)
        backbone.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # encoder stages
        self.enc1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # →112×112
        self.enc2 = nn.Sequential(backbone.maxpool, backbone.layer1)            # →56×56
        self.enc3 = backbone.layer2                                             # →28×28
        self.enc4 = backbone.layer3                                             # →14×14
        self.enc5 = backbone.layer4                                             # →7×7

        # embedding head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # pool to 1x1
        self.fc_emb = nn.Linear(512, embedding_dim)

        # decoder: U-Net with ResBlocks
        self.dec5 = ResBlock(512, 256)                    # 7→14
        self.dec4 = ResBlock(256 + 256, 128)              # cat(enc4) →28
        self.dec3 = ResBlock(128 + 128, 64)               # cat(enc3) →56
        self.dec2 = ResBlock(64 + 64, 64)                 # cat(enc2) →112
        self.dec1_up = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # →224

        # predict future frames
        self.final_conv = nn.Conv2d(64, 3 * num_future_frames, kernel_size=1)

    def forward(self, x: State) -> Tuple[State, torch.Tensor]:
        # encode
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)

        # embedding
        pooled = self.pool(f5)
        flat = torch.flatten(pooled, 1)
        emb = self.fc_emb(flat)

        # decode with U-Net skips
        d5 = self.dec5(f5)
        d4 = self.dec4(torch.cat([d5, f4], dim=1))
        d3 = self.dec3(torch.cat([d4, f3], dim=1))
        d2 = self.dec2(torch.cat([d3, f2], dim=1))
        d1 = F.relu(self.dec1_up(d2))
        decoded = self.final_conv(d1)
        return decoded, emb

class DistanceModel(nn.Module):
    def __init__(self, embedding_dim: int = 512) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * embedding_dim, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([e1, e2], dim=-1))

INV_MEAN = torch.tensor([0.485, 0.456, 0.406])  # same order as your Normalize
INV_STD  = torch.tensor([0.229, 0.224, 0.225])

def unnormalize(tensor):
    """
    tensor: 3×H×W, float, normalized.
    Returns a 3×H×W tensor in [0,1].
    """
    mean = INV_MEAN[:, None, None].to(tensor.device)
    std  = INV_STD[:, None, None].to(tensor.device)
    return (tensor * std + mean).clamp(0, 1)

def save_sample_images(samples, output_dir: Path, step: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    to_pil = T.ToPILImage()
    for idx, (state_stack, pred, gt) in enumerate(samples):
        # state_stack: 12×H×W — split into 4×(3×H×W)
        in_frames = state_stack.view(4, 3, state_stack.shape[1], state_stack.shape[2])
        gt_frames = gt.view(4, 3, gt.shape[1], gt.shape[2])
        pred_frames = pred.view(4, 3, pred.shape[1], pred.shape[2])

        # collect first row: 4 inputs + 4 ground-truths
        row1_imgs = []
        for i in range(4):
            row1_imgs.append(to_pil(unnormalize(in_frames[i])))
        for i in range(4):
            row1_imgs.append(to_pil(unnormalize(gt_frames[i])))

        # collect second row: 4 blank + 4 predictions
        W, H = row1_imgs[0].size
        row2_imgs = []
        blank = Image.new('RGB', (W, H), color=(0, 0, 0))
        for _ in range(4):
            row2_imgs.append(blank)
        for i in range(4):
            row2_imgs.append(to_pil(unnormalize(pred_frames[i])))

        # compose rows and final image
        row1 = Image.new('RGB', (W * 8, H))
        for i, im in enumerate(row1_imgs):
            row1.paste(im, (i * W, 0))
        row2 = Image.new('RGB', (W * 8, H))
        for i, im in enumerate(row2_imgs):
            row2.paste(im, (i * W, 0))
        final = Image.new('RGB', (W * 8, H * 2))
        final.paste(row1, (0, 0))
        final.paste(row2, (0, H))

        final.save(output_dir / f"step_{step:05d}_sample_{idx}.png")

# Write loss history to CSV
def write_loss_history(loss_history: List[Tuple[int, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'loss_history.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'loss'])
        writer.writerows(loss_history)

@dataclass
class Args:
    traj_dir: str = "/Users/dave/rl/nes-ai/runs/smb-search-v0__search_mario__1__2025-07-12_10-25-46/traj_dumps"
    batch_size: int = 64
    batches_per_epoch: int = 8  # number of batches per epoch
    num_epochs: int = 1024 * 1024
    lr: float = 1e-4
    output_dir: str = "out.predict_mario"  # where to save outputs
    pipeflush: bool = False


def main() -> None:
    args = tyro.cli(Args)

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output = Path(args.output_dir) / f"train__{date_str}"

    if args.pipeflush:
        args.batch_size = 8
        args.batches_per_epoch = 4
        args.num_epochs = 16

    max_trajs = 10 if args.pipeflush else None

    dataset = MarioTrajectoryDataset(args.traj_dir, max_trajs=max_trajs)
    print(f"Dataset: {len(dataset)} samples across {dataset.num_trajectories} trajectories.")

    sampler = RandomSampler(dataset, replacement=False, num_samples=args.batches_per_epoch * args.batch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    pred_model = PredictionModel().to(device)
    dist_model = DistanceModel().to(device)
    crit_pred = nn.MSELoss()
    crit_dist = nn.MSELoss()
    opt_pred = optim.Adam(pred_model.parameters(), lr=args.lr)
    opt_dist = optim.Adam(dist_model.parameters(), lr=args.lr)

    global_step = 0
    loss_history: List[Tuple[int, float]] = []
    for epoch in tqdm(range(args.num_epochs), desc="Epochs"):
        samples: List[Tuple[State, State, State]] = []
        embeddings_batch: List[np.ndarray] = []
        batch_pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)
        for state, action, next_frames in batch_pbar:
            state, next_frames = state.to(device), next_frames.to(device)
            pred_frames, emb = pred_model(state)
            pred_frames = F.interpolate(pred_frames, size=next_frames.shape[-2:], mode='bilinear', align_corners=False)

            loss_p = crit_pred(pred_frames, next_frames)
            e1, e2 = emb[:-1], emb[1:]
            loss_d = crit_dist(dist_model(e1, e2), torch.ones_like(e1[:, :1], device=device)*0.1)
            total = loss_p + loss_d

            opt_pred.zero_grad(); opt_dist.zero_grad()
            total.backward(); opt_pred.step(); opt_dist.step()

            embeddings_batch.append(emb.detach().cpu().numpy())

            global_step += 1
            loss_history.append((global_step, total.item()))
            batch_pbar.set_postfix({'loss': total.item()})

            for i in range(state.size(0)):
                if len(samples) < 3:
                    samples.append((state[i].cpu(), pred_frames[i].cpu(), next_frames[i].cpu()))

        save_sample_images(samples, output, global_step)

        if loss_history:
            steps, losses = zip(*loss_history)
            fig2, ax2 = plt.subplots()
            ax2.semilogy(steps, losses)
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Loss (log scale)')
            ax2.set_title(f'Loss up to step {global_step}')
            fig2.savefig(output / f'step_{global_step:05d}_loss.png')
            plt.close(fig2)

        batch_emb_array = np.concatenate(embeddings_batch, axis=0)
        n = batch_emb_array.shape[0]
        perp = min(5, max(1, n - 1))
        reduced = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(batch_emb_array)
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        ax3.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
        ax3.set_title(f"TSNE Embeddings Epoch {epoch+1}")
        ax3.axis('off')
        fig3.savefig(output / f"step_{global_step:05d}_tsne.png")
        plt.close(fig3)

    write_loss_history(loss_history, output)

if __name__ == '__main__':
    main()
