# %%

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP()
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

model.fc1.register_forward_hook(get_activation('fc1'))

x = torch.randn(16, 512)
_ = model(x)

act = activations['fc1'].cpu().numpy()

# %%

# Heatmap
plt.figure(figsize=(10, 4))
plt.imshow(act, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Activation Heatmap (fc1)')
plt.xlabel('Neurons')
plt.ylabel('Batch Samples')
plt.tight_layout()
#plt.savefig('activation_heatmap.png')
#plt.close()
plt.show()

# Histogram
plt.figure(figsize=(8, 4))
plt.hist(act.flatten(), bins=50, color='blue', alpha=0.7)
plt.title('Activation Histogram (fc1)')
plt.xlabel('Activation Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
#plt.savefig('activation_histogram.png')
#plt.close()
plt.show()

# %%

print("HMM")

# %%
