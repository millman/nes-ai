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

    flow_pred = F.interpolate(flow_pred, size=(224, 224), mode='bilinear', align_corners=False)

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

    flow_pred = F.interpolate(flow_pred, size=(224, 224), mode='bilinear', align_corners=False)

    return F.mse_loss(flow_pred, dx_dt)


# ----- Dummy Dataset -----
def create_dummy_dataset(num_samples=64, action_dim=6):
    x_stack = torch.rand(num_samples, 4, 224, 224, 3)
    actions = torch.randint(0, 2, (num_samples, action_dim)).float()
    x_next = torch.rand(num_samples, 224, 224, 3)
    return TensorDataset(x_stack, actions, x_next)


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


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
    print("No GPU available, using CPU.")



action_dim = 6
model = WorldModelCFM(action_dim=action_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

dataset = create_dummy_dataset(num_samples=32, action_dim=action_dim)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for epoch in range(3):
    for batch in dataloader:
        x_stack, action, x_next = [b.to(device) for b in batch]
        # loss = cfm_ot_loss(model, x_stack, action, x_next)
        # loss = cfm_loss_basic(model, x_stack, action, x_next)
        loss = cfm_loss(model, x_stack, action, x_next)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# After training
batch = next(iter(dataloader))
x_stack, action, x_next = [b.to(device) for b in batch]
evaluate_model(model, x_stack, action, x_next)

# %%
