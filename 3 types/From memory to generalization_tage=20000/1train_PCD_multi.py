import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

# config is already defined, containing:
from config import delta_t, N, g, eta, T, k, b_size, lambda1, N_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def square_mean_torch(tensor):
    return torch.sum(torch.square(tensor)) / tensor.numel()


# Derivative of the activation function
phi_prime = lambda x: 1.0 - torch.tanh(x) ** 2

# --- 1. Data Preparation ---
transform = transforms.ToTensor()
desired_nums = [1, 3, 6]

mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_mask = torch.isin(mnist_train_full.targets, torch.tensor(desired_nums))
mnist_train_selected = Subset(mnist_train_full, train_mask.nonzero().squeeze())

test_mask = torch.isin(mnist_test_full.targets, torch.tensor(desired_nums))
mnist_test_selected = Subset(mnist_test_full, test_mask.nonzero().squeeze())

# Combine datasets and perform preprocessing
all_data = ConcatDataset([mnist_train_selected, mnist_test_selected])
# Load all images at once for preprocessing; data slicing happens in the subsequent loop
data_loader_full = DataLoader(all_data, batch_size=len(all_data), shuffle=False)

images, _ = next(iter(data_loader_full))
images = images.to(device)

scaled_matrix = images.squeeze(1) * 2 - 1
epsilon = 0.03
clipped_matrix = torch.clamp(scaled_matrix, min=-1.0 + epsilon, max=1.0 - epsilon)
transformed_matrix = torch.atanh(clipped_matrix)

# Final flattened data matrix. Shape: (N, Total_Available_Images)
matrix = transformed_matrix.reshape(transformed_matrix.shape[0], -1).T
total_available_data = matrix.shape[1]


class ColumnDataset(Dataset):
    def __init__(self, X): self.X = X

    def __len__(self): return self.X.size(1)

    def __getitem__(self, idx): return self.X[:, idx]


# --- 2. Basic Initialization Setup ---
save_dir = 'checkpoints_by_size'
os.makedirs(save_dir, exist_ok=True)

num_runs = 5 # Number of models to train for each dataset size
target_steps = 20000 # Fixed number of weight updates

# Pre-generate 5 sets of base initial weights to ensure identical starting points for different dataset sizes
base_initial_Js = []
base_initial_Bs = []
for _ in range(num_runs):
    J_init = torch.randn(N, N, device=device) * torch.sqrt(torch.tensor((g ** 2) / N, device=device))
    J_init.fill_diagonal_(0)
    B_init = torch.zeros(N, device=device)
    base_initial_Js.append(J_init)
    base_initial_Bs.append(B_init)

# Generate dataset size sequence: 5, 5*1.5, ..., 5*1.5^19 (cast to int)
data_sizes = [int(5 * (1.5 ** i)) for i in range(20)]
print(f"Scheduled dataset sizes: {data_sizes}")


# --- 3. Training Loop for Different Dataset Sizes ---
for current_size in data_sizes:
    # Prevent requested size from exceeding the total filtered data available
    if current_size > total_available_data:
        print(f"Warning: Requested size {current_size} exceeds available data. Capping at {total_available_data}.")
        current_size = total_available_data

    print(f"\n--- Starting training for dataset size: {current_size} ---")

    # Slice the corresponding subset of data
    current_matrix = matrix[:, :current_size]

    # Adjust batch_size according to current size
    current_b_size = min(current_size, b_size)

    dataset = ColumnDataset(current_matrix)
    # DataLoader defaults to drop_last=False, keeping the final incomplete batch
    dataloader = DataLoader(dataset, batch_size=current_b_size, shuffle=True)

    # Perform 5 independent training runs for the current dataset size
    for run_idx in range(num_runs):
        print(f"  -> Run {run_idx + 1}/{num_runs}")
        
        # Initialize parameters using the corresponding set of base initial weights
        J = torch.nn.Parameter(base_initial_Js[run_idx].clone())
        B_bias = torch.nn.Parameter(base_initial_Bs[run_idx].clone())

        optimizer = torch.optim.Adam([
            {'params': [J], 'weight_decay': lambda1},
            {'params': [B_bias], 'weight_decay': 0.0}
        ], lr=eta)

        # Re-initialize persistent chains and noise scale
        X = torch.randn(N, N_model, k + 1, device=device)
        noise_size = torch.sqrt(torch.tensor(2 * T * delta_t, device=device))

        global_step = 0
        pbar = tqdm(total=target_steps, desc=f"Size {current_size} | Run {run_idx+1}")

        # Start training until target update steps are reached
        while global_step < target_steps:
            for batch in dataloader:
                if global_step >= target_steps:
                    break  # Exit loop once target updates are completed

                with torch.no_grad():
                    # --- Data Phase ---
                    batch = batch.T
                    phi_data = torch.tanh(batch)
                    h_data = J @ phi_data
                    x_h_data = batch - h_data - B_bias.unsqueeze(1)
                    data_grad_J = (1 / (T * batch.shape[1])) * x_h_data @ phi_data.T

                    # --- Model Phase (PCD) ---
                    for j in range(k):
                        tanh_X = torch.tanh(X[:, :, j])
                        h = J @ tanh_X
                        delta_h = h - X[:, :, j] + B_bias.unsqueeze(1)
                        delta_J = J.T @ delta_h
                        X[:, :, j + 1] = (
                                (1 - delta_t) * X[:, :, j] +
                                delta_t * (h + B_bias.unsqueeze(1)) -
                                delta_t * phi_prime(X[:, :, j]) * delta_J +
                                noise_size * torch.randn(N, N_model, device=device)
                        )

                    X[:, :, 0] = X[:, :, -1]
                    x_model = X[:, :, -1]
                    phi_model = torch.tanh(x_model)
                    h_model = J @ phi_model
                    x_h_model = x_model - h_model - B_bias.unsqueeze(1)
                    model_grad_J = (1 / (T * x_model.shape[1])) * x_h_model @ phi_model.T

                # --- Gradient Update ---
                optimizer.zero_grad()
                J.grad = model_grad_J - data_grad_J
                J.grad.fill_diagonal_(0)
                B_bias.grad = (1 / T) * (x_h_model.mean(dim=1) - x_h_data.mean(dim=1))

                optimizer.step()

                global_step += 1
                pbar.update(1)

        pbar.close()

        # Training complete for current run, save model weights with run_idx in filename
        j_path = os.path.join(save_dir, f'J_size_{current_size}_run_{run_idx}.npy')
        b_path = os.path.join(save_dir, f'B_size_{current_size}_run_{run_idx}.npy')

        np.save(j_path, J.detach().cpu().numpy())
        np.save(b_path, B_bias.detach().cpu().numpy())

print("\nAll models trained and saved successfully.")