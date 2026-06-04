import os
import torch
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# config is defined; these are local mappings of your parameters
from config import delta_t, N_VAE, g, n, eta, T, k, b_size, lambda1, N_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def square_mean_torch(tensor):
    return torch.sum(torch.square(tensor)) / tensor.numel()


# Derivative of the activation function (tanh)
phi_prime = lambda x: 1.0 - torch.tanh(x) ** 2

# 1. Load Latent Space Matrix
# Expected matrix shape: (N_VAE, N_image)
matrix = np.load("compressed_matrix.npy")

# 2. Initialize Parameters
initial_J = torch.randn(N_VAE, N_VAE, device=device) * torch.sqrt(torch.tensor((g ** 2) / N_VAE, device=device))
initial_J.fill_diagonal_(0)  # Ensure diagonal is zero at initialization
J = torch.nn.Parameter(initial_J)

initial_B = torch.zeros(N_VAE, device=device)
B_bias = torch.nn.Parameter(initial_B)

# Define optimizer with weight decay only for the coupling matrix J
optimizer = torch.optim.Adam([
    {'params': [J], 'weight_decay': lambda1},
    {'params': [B_bias], 'weight_decay': 0.0}
], lr=eta)


# 3. Dataset and DataLoader
class ColumnDataset(Dataset):
    def __init__(self, X, device):
        super().__init__()
        self.X = torch.from_numpy(X).float().to(device)

    def __len__(self):
        return self.X.size(1)

    def __getitem__(self, idx):
        return self.X[:, idx]


dataset = ColumnDataset(matrix, device)
dataloader = DataLoader(dataset, batch_size=b_size, shuffle=True)

# 4. Prepare saving strategy and dynamic variables
os.makedirs('checkpoints_latent', exist_ok=True)
global_step = 0
# Define specific steps for model checkpoints using an exponential scale
target_save_steps = [round(201 * (1.7 ** i)) for i in range(13)]
save_step_dict = {step: idx for idx, step in enumerate(target_save_steps)}
print(f"Scheduled checkpoint global steps: {target_save_steps}")

noise_size = torch.sqrt(torch.tensor(2 * T * delta_t, device=device))
X = torch.randn(N_VAE, N_model, device=device)
kl_norm = []

# 5. Training Loop
for epoch in tqdm(range(n)):
    epoch_klnorm1 = 0.0

    for batch in dataloader:
        with torch.no_grad():
            # Data Phase: Calculate statistics from the dataset
            batch = batch.T
            phi_data = torch.tanh(batch)
            h_data = J @ phi_data
            x_h_data = batch - h_data - B_bias.unsqueeze(1)
            data = (1 / (T * (batch.shape[1]))) * x_h_data @ (phi_data.T)

            # Model Phase: Langevin dynamics simulation
            for j in range(k):
                tanh_X = torch.tanh(X)
                h = J @ tanh_X
                delta_h = h - X + B_bias.unsqueeze(1)
                delta_J = J.T @ delta_h
                X = ((1 - delta_t) * X + delta_t * (h + B_bias.unsqueeze(1)) -
                     delta_t * phi_prime(X) * delta_J +
                     noise_size * torch.randn(N_VAE, N_model, device=device))

            # Model Phase: Calculate statistics from the simulated state
            x_model = X
            phi_model = torch.tanh(x_model)
            h_model = J @ phi_model
            x_h_model = x_model - h_model - B_bias.unsqueeze(1)
            model = (1 / (T * (x_model.shape[1]))) * x_h_model @ (phi_model.T)

        # Gradient Updates
        optimizer.zero_grad()

        # Gradient for J: Difference between model and data, strictly zeroing the diagonal
        grad1 = model - data
        grad1.fill_diagonal_(0)
        J.grad = grad1

        # Gradient for B_bias
        grad2 = (1 / T) * (x_h_model.mean(dim=1) - x_h_data.mean(dim=1))
        B_bias.grad = grad2

        optimizer.step()

        # Logging and Checkpointing
        global_step += 1
        epoch_klnorm1 += square_mean_torch(grad1).item()  # Accumulate the mean squared gradient of current batch

        if global_step in save_step_dict:
            idx = save_step_dict[global_step]
            j_path = os.path.join('checkpoints_latent', f'J_idx_{idx}_step_{global_step}.npy')
            b_path = os.path.join('checkpoints_latent', f'B_idx_{idx}_step_{global_step}.npy')
            np.save(j_path, J.detach().cpu().numpy())
            np.save(b_path, B_bias.detach().cpu().numpy())

    # Record the average mean squared gradient for the entire Epoch
    avg_klnorm = epoch_klnorm1 / len(dataloader)
    kl_norm.append(avg_klnorm)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1} (Step {global_step}): J_grad_norm={avg_klnorm:.6f}")


# Extract last 30 epochs for a zoomed-in visualization
kl_norm_last_30 = kl_norm[-30:] if len(kl_norm) >= 30 else kl_norm

plt.figure(figsize=(12, 5))

# Plot full training process
plt.subplot(1, 2, 1)
plt.plot(kl_norm, color='b')
plt.title('Total Training Process (Latent Space)')
plt.xlabel('Epoch')
plt.ylabel('KL Norm Value')
plt.grid(True)

# Plot zoomed-in view of the last 30 epochs
plt.subplot(1, 2, 2)
plt.plot(kl_norm_last_30, marker='o', linestyle='-', color='r')
plt.title('Last 30 Epochs')
plt.xlabel('Epoch')
plt.ylabel('KL Norm Value')
plt.grid(True)

plt.tight_layout()
plt.show()