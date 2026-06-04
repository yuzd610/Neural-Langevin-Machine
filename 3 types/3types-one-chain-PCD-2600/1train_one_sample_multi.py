import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

# config is already defined; mapping local variables here
from config import delta_t, N, g, n, N_data, k, T, eta, b_size, lambda1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


def square_mean_torch(tensor):
    return torch.sum(torch.square(tensor)) / tensor.numel()


# Derivative of the activation function
phi_prime = lambda x: 1.0 - torch.tanh(x) ** 2

# --- 1. Data Preparation ---
transform = transforms.ToTensor()
# MNIST labels are 0-9. If you want to train on all digits, include 0-9 in the list.
desired_nums = [1, 3, 6]

mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Filter dataset for specific digits
train_mask = torch.isin(mnist_train_full.targets, torch.tensor(desired_nums))
mnist_train_selected = Subset(mnist_train_full, train_mask.nonzero().squeeze())

test_mask = torch.isin(mnist_test_full.targets, torch.tensor(desired_nums))
mnist_test_selected = Subset(mnist_test_full, test_mask.nonzero().squeeze())

# Combine and load data
all_data = ConcatDataset([mnist_train_selected, mnist_test_selected])
data_loader = DataLoader(all_data, batch_size=N_data, shuffle=False)

images, _ = next(iter(data_loader))
images = images.to(device)

# Data Preprocessing
scaled_matrix = images.squeeze(1) * 2 - 1
epsilon = 0.03
clipped_matrix = torch.clamp(scaled_matrix, min=-1.0 + epsilon, max=1.0 - epsilon)
transformed_matrix = torch.atanh(clipped_matrix)

# Dynamically get image count to prevent mismatch between N_data and actual filtered count
matrix = transformed_matrix.reshape(transformed_matrix.shape[0], -1).T  # Shape: (N, Actual_N_data)

# --- 2. Model Initialization ---
initial_J = torch.randn(N, N, device=device) * torch.sqrt(torch.tensor((g ** 2) / N, device=device))
# Crucial: Zero out the diagonal at initialization
initial_J.fill_diagonal_(0)
J = torch.nn.Parameter(initial_J)

initial_B = torch.zeros(N, device=device)
B_bias = torch.nn.Parameter(initial_B)

# Optimizer configuration
optimizer = torch.optim.Adam([
    {'params': [J], 'weight_decay': lambda1},     # Apply lambda1 weight decay to J
    {'params': [B_bias], 'weight_decay': 0.0}    # No weight decay for B_bias
], lr=eta)


class ColumnDataset(Dataset):
    def __init__(self, X): self.X = X

    def __len__(self): return self.X.size(1)

    def __getitem__(self, idx): return self.X[:, idx]


dataset = ColumnDataset(matrix)
dataloader = DataLoader(dataset, batch_size=b_size, shuffle=True)

noise_size = torch.sqrt(torch.tensor(2 * T * delta_t, device=device))

# Modification 1: Initialize a single persistent chain, no longer using a 3D matrix
X_persistent = torch.randn(N, 1, device=device)

kl_norm = []

# --- Prepare Saving Strategy ---
os.makedirs('checkpoints', exist_ok=True)  # Directory for intermediate models
global_step = 0
# Pre-calculate steps to save, using an exponential scale rounded to integers
target_save_steps = [round(201 * (1.7 ** i)) for i in range(13)]
print(f"Scheduled global steps for saving: {target_save_steps}")
# Convert to dictionary for easy tracking of indices
save_step_dict = {step: idx for idx, step in enumerate(target_save_steps)}


# --- 3. Training Loop ---
for epoch in tqdm(range(n)):
    for batch in dataloader:
        with torch.no_grad():
            # Data Phase: Calculate statistics from training data
            batch = batch.T  # Shape: (N, batch_size)
            phi_data = torch.tanh(batch)
            h_data = J @ phi_data
            # Adding b_i bias term as per the paper
            x_h_data = batch - h_data - B_bias.unsqueeze(1)
            data_grad_J = (1 / (T * batch.shape[1])) * x_h_data @ phi_data.T

            # Model Phase: Single Markov chain for Time Averaging
            X_curr = X_persistent
            samples = []
            
            # Run k = 2600 steps
            for step in range(1, k + 1):
                tanh_X = torch.tanh(X_curr)
                h = J @ tanh_X
                delta_h = h - X_curr + B_bias.unsqueeze(1)
                delta_J = J.T @ delta_h
                
                # State update
                X_curr = (
                        (1 - delta_t) * X_curr +
                        delta_t * (h + B_bias.unsqueeze(1)) -
                        delta_t * phi_prime(X_curr) * delta_J +
                        noise_size * torch.randn(N, 1, device=device)
                )

                # Modification 2: After 600 iterations, sample over time every 20 steps
                if step > 600 and (step - 600) % 20 == 0:
                    samples.append(X_curr.clone())

            # Update the persistent chain for the starting point of the next batch
            X_persistent = X_curr.detach()
            
            # Modification 3: Concatenate the time-sampled points to use as the batch for x_model
            # len(samples) will be 100, so the shape of x_model is (N, 100)
            x_model = torch.cat(samples, dim=1) 
            
            # The calculation for the remaining model state gradients remains unchanged
            phi_model = torch.tanh(x_model)
            h_model = J @ phi_model
            x_h_model = x_model - h_model - B_bias.unsqueeze(1)
            model_grad_J = (1 / (T * x_model.shape[1])) * x_h_model @ phi_model.T

        # Gradient Updates
        optimizer.zero_grad()

        # Gradient calculation for J
        J.grad = model_grad_J - data_grad_J

        # --- Core Modification: Zero out the diagonal gradient ---
        # This ensures Adam never updates the diagonal elements
        J.grad.fill_diagonal_(0)

        # Gradient calculation for B_bias
        B_bias.grad = (1 / T) * (x_h_model.mean(dim=1) - x_h_data.mean(dim=1))

        optimizer.step()

        # --- Global Step Counter and Saving Logic ---
        global_step += 1

        if global_step in save_step_dict:
            idx = save_step_dict[global_step]
            # Construct save paths with index and current step count
            j_path = os.path.join('checkpoints', f'J_idx_{idx}_step_{global_step}.npy')
            b_path = os.path.join('checkpoints', f'B_idx_{idx}_step_{global_step}.npy')

            np.save(j_path, J.detach().cpu().numpy())
            np.save(b_path, B_bias.detach().cpu().numpy())

    # Monitor KL Divergence metrics
    # Diagonal is already zero; this computes the mean squared variance of off-diagonal gradients
    klnorm1 = square_mean_torch(J.grad)
    klnorm2 = square_mean_torch(B_bias.grad)
    kl_norm.append(klnorm1.item())

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1} (Global Step {global_step}): J_grad_norm={klnorm1.item():.6f}, B_grad_norm={klnorm2.item():.6f}")

# --- 4. Final Save and Plotting ---
np.save('J_final.npy', J.detach().cpu().numpy())
np.save('B_bias.npy', B_bias.detach().cpu().numpy())

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(kl_norm)
plt.title('Total Training Process')
plt.xlabel('Epoch')
plt.ylabel('KL Norm')

plt.subplot(1, 2, 2)
plt.plot(kl_norm[-30:])
plt.title('Last 30 Epochs')
plt.xlabel('Epoch')
plt.show()