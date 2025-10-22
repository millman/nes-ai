#!/usr/bin/env python3
# %%

import glob
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.data import DataLoader, TensorDataset

#
# import ot  # requirements: pip install pot
# from geomloss import SamplesLoss

# ----- Model -----
class WorldModelCFM(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4 * 3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.action_embed = nn.Linear(action_dim, 256)
        self.flow_net = nn.Sequential(
            nn.Conv2d(256 + 1, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 3, kernel_size=1),
        )

    def forward(self, x_stack, action, t):
        B = x_stack.size(0)

        # x_stack: (B, 4, H, W, 3) -> (B, 4, 3, H, W)
        x_stack = x_stack.permute(0, 1, 4, 2, 3)
        # (B, 4, 3, H, W) -> (B, 12, H, W)
        x_stack = x_stack.reshape(B, 12, 224, 224)

        x_encoded = self.encoder(x_stack)

        a_embed = self.action_embed(action).view(B, 256, 1, 1)
        a_embed = a_embed.expand_as(x_encoded)

        t = t.view(B, 1, 1, 1).expand(B, 1, x_encoded.shape[2], x_encoded.shape[3])
        h = torch.cat([x_encoded + a_embed, t], dim=1)
        flow = self.flow_net(h)
        return flow


# ----- Flow Matching Loss (Linear Interpolation Direction) -----
def cfm_loss_basic(model, x_stack, action, x_next):
    B = x_stack.shape[0]
    t = torch.rand(B, device=x_stack.device).unsqueeze(1)

    # Get current (last) frame and next frame
    x0 = x_stack[:, -1].permute(0, 3, 1, 2)  # (B, 3, 224, 224)
    x1 = x_next.permute(0, 3, 1, 2)          # (B, 3, 224, 224)

    dx_dt = x1 - x0

    flow_pred = model(x_stack, action, t)
    return F.mse_loss(flow_pred, dx_dt)


def cfm_loss(model, x_stack, action, x_next):
    B = x_stack.shape[0]
    t = torch.rand(B, device=x_stack.device).unsqueeze(1)

    # Last image in stack (current)
    x0 = x_stack[:, -1].permute(0, 3, 1, 2)  # (B, 3, H, W)
    x1 = x_next.permute(0, 3, 1, 2)          # (B, 3, H, W)
    x_t = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1

    dx_dt = x1 - x0  # Same everywhere along the line (linear case)

    # Replace x_stack with x_t-only version for input
    # We fake a "stack" where all 4 frames are x_t to reuse the model structure
    x_stack_xt = x_stack.clone()
    x_stack_xt[:, :] = x_t.permute(0, 2, 3, 1).unsqueeze(1).repeat(1, 4, 1, 1, 1)

    flow_pred = model(x_stack_xt, action, t)
    return F.mse_loss(flow_pred, dx_dt)


# ----- Loss Function with Sinkhorn OT -----
# sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.05)
sinkhorn_loss = None

def cfm_ot_loss(model, x_stack, action, x_next):
    B = x_stack.shape[0]
    t = torch.rand(B, device=x_stack.device).unsqueeze(1)

    x0 = x_stack[:, -1]  # Last frame
    x1 = x_next

    x0 = x0.permute(0, 3, 1, 2)
    x1 = x1.permute(0, 3, 1, 2)
    x_t = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1

    flow_pred = model(x_stack, action, t)

    loss_total = 0.0
    for b in range(B):
        x0_flat = x0[b].reshape(3, -1).T  # (N, 3)
        x1_flat = x1[b].reshape(3, -1).T
        loss_total += sinkhorn_loss(x0_flat, x1_flat)

    return loss_total / B

# ----- Dummy Dataset -----
def create_dummy_dataset(num_samples=64, action_dim=6):
    x_stack = torch.rand(num_samples, 4, 224, 224, 3)
    actions = torch.randint(0, 2, (num_samples, action_dim)).float()
    x_next = torch.rand(num_samples, 224, 224, 3)
    return TensorDataset(x_stack, actions, x_next)


# ----- Training Loop -----
def train_example():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_dim = 6
    model = WorldModelCFM(action_dim=action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    dataset = create_dummy_dataset(num_samples=32, action_dim=action_dim)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for epoch in range(3):
        for batch in dataloader:
            x_stack, action, x_next = [b.to(device) for b in batch]
            loss = cfm_ot_loss(model, x_stack, action, x_next)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


def cfm_ot_direction_loss(model, x_stack, action, x_next):
    B = x_stack.shape[0]
    t = torch.rand(B, device=x_stack.device).unsqueeze(1)

    x0 = x_stack[:, -1]  # (B, 224, 224, 3)
    x1 = x_next

    # Convert to (B, 3, H, W)
    x0 = x0.permute(0, 3, 1, 2)
    x1 = x1.permute(0, 3, 1, 2)
    x_t = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1

    flow_pred = model(x_stack, action, t)  # (B, 3, 224, 224)

    loss_total = 0.0
    for b in range(B):
        # Reshape to N x 3
        x0_flat = x0[b].reshape(3, -1).T.cpu().detach().numpy()
        x1_flat = x1[b].reshape(3, -1).T.cpu().detach().numpy()

        n = x0_flat.shape[0]
        mu = np.ones(n) / n
        nu = np.ones(n) / n

        # Cost matrix (squared Euclidean distance)
        C = ot.dist(x0_flat, x1_flat, metric='euclidean') ** 2

        # Sinkhorn transport plan
        gamma = ot.sinkhorn(mu, nu, C, reg=0.1)

        # Compute expected transport direction for each source point
        transport_vectors = x1_flat[None, :, :] - x0_flat[:, None, :]  # (n, n, 3)
        expected_displacement = (gamma[:, :, None] * transport_vectors).sum(axis=1)  # (n, 3)

        # Compare to predicted flow
        pred = flow_pred[b].permute(1, 2, 0).reshape(-1, 3).cpu()
        target = torch.tensor(expected_displacement, dtype=torch.float32)

        loss_total += F.mse_loss(pred, target)

    return loss_total / B


def visualize_training_step(x0, x1, flow_pred):
    x0_np = x0[0].permute(1, 2, 0).cpu().numpy()
    x1_np = x1[0].permute(1, 2, 0).cpu().numpy()
    dx_gt_np = (x1 - x0)[0].permute(1, 2, 0).cpu().numpy()
    dx_pred_np = flow_pred[0].permute(1, 2, 0).detach().cpu().numpy()

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(x0_np)
    axs[0].set_title("x0 (current)")
    axs[1].imshow(x1_np)
    axs[1].set_title("x1 (next)")
    axs[2].imshow((dx_gt_np + 1) / 2)
    axs[2].set_title("GT Δx")
    axs[3].imshow((dx_pred_np + 1) / 2)
    axs[3].set_title("Predicted Δx")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


@dataclass
class Args:
    input_dir: str = "runs/SuperMarioBros-mame-v0__ppo_nes__1__2025-05-07_16-24-52/traj"
    """input dir containing trajectory.npz files"""

    work_dir: str = 'world_models'
    """used for intermediate and output files"""

    train: bool = False
    """run training loop"""
    visualize: bool = False
    """visualize samples of world model"""


# Visualize
def evaluate_model(model, x_stack, action, x1_true):
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        t = torch.ones(x_stack.shape[0], 1, device=device)
        x0 = x_stack[:, -1].permute(0, 3, 1, 2)
        x_stack_xt = x_stack.clone()
        x_stack_xt[:, :] = x0.permute(0, 2, 3, 1).unsqueeze(1).repeat(1, 4, 1, 1, 1)
        flow_pred = model(x_stack_xt.to(device), action.to(device), t)
        visualize_training_step(x0.to(device), x1_true.permute(0, 3, 1, 2).to(device), flow_pred)

# %%

def main():
    args = tyro.cli(Args)

    # %%

    args = Args(train=True, visualize=True)

    USE_DUMMY_DATASET = True

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        print("No GPU available, using CPU.")

    # %% Load datasets.

    if not USE_DUMMY_DATASET:
        obs_list = []
        actions_list = []
        logprobs_list = []
        rewards_list = []
        dones_list = []
        values_list = []

        # Load all .npz files into a list of dictionaries.
        npz_filenames = sorted(glob.glob(Path(args.input_dir) / "*.npz"))
        for path in npz_filenames:
            with np.load(path) as data:
                obs_list.append(data['obs'])
                actions_list.append(data['actions'])
                logprobs_list.append(data['logprobs'])
                rewards_list.append(data['rewards'])
                dones_list.append(data['dones'])
                values_list.append(data['values'])

        # Combine all objects together.
        obs = np.concatenate(obs_list, axis=0)
        actions = np.concatenate(actions_list, axis=0)
        logprobs = np.concatenate(logprobs_list, axis=0)
        rewards = np.concatenate(rewards_list, axis=0)
        dones = np.concatenate(dones_list, axis=0)
        values = np.concatenate(values_list, axis=0)

    # %tb

    # %% Run training.
    if args.train:
        print("HELLO")
        if USE_DUMMY_DATASET:
            action_dim = 6
            model = WorldModelCFM(action_dim=action_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            dataset = create_dummy_dataset(num_samples=32, action_dim=action_dim)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

            for epoch in range(3):
                for batch in dataloader:
                    x_stack, action, x_next = [b.to(device) for b in batch]
                    # loss = cfm_ot_loss(model, x_stack, action, x_next)
                    loss = cfm_loss(model, x_stack, action, x_next)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            for batch in dataloader:
                x_stack, action, x_next = batch
                loss = cfm_loss(model, x_stack, action, x_next)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            assert len(actions[0].shape) == 1, f"Unexpected action shape: {actions[0]}"
            action_dim = actions[0].size

    # %% Visualize
    if args.visualize:
        # After training
        batch = next(iter(dataloader))
        x_stack, action, x_next = [b.to(device) for b in batch]
        evaluate_model(model, x_stack, action, x_next)


# %%

if __name__ == "__main__":
    main()