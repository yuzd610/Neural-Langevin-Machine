import os
import glob
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# 1. Exact Reconstruction and Extraction of Real Training Data
# (Ensuring strict alignment with the training subset)
# ==============================================================================
transform = transforms.ToTensor()
desired_nums = [1, 3, 6]

# Load MNIST datasets
mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Filter samples based on selected digits
train_mask = torch.isin(mnist_train_full.targets, torch.tensor(desired_nums))
mnist_train_selected = Subset(mnist_train_full, train_mask.nonzero().squeeze())

test_mask = torch.isin(mnist_test_full.targets, torch.tensor(desired_nums))
mnist_test_selected = Subset(mnist_test_full, test_mask.nonzero().squeeze())

# Merge datasets (Consistency must match the pre-processing logic used during training)
all_data = ConcatDataset([mnist_train_selected, mnist_test_selected])
# Note: shuffle=False is critical to ensure data_size slicing extracts the exact training subset
data_loader_full = DataLoader(all_data, batch_size=len(all_data), shuffle=False)

images, _ = next(iter(data_loader_full))
# Normalize real images to [-1, 1] and flatten (Shape: [Total_images, 784])
S_tensor_full = (images.view(-1, 784) * 2 - 1).to(device)
print(f"Total available real dataset size: {S_tensor_full.shape[0]}")


# ==============================================================================
# 2. Core Evaluation Metric (AAI Metric)
# ==============================================================================
def compute_AAI_full(real_data, gen_data):
    """
    Computes Adversarial Accuracy Improvement (AAI) using Nearest Neighbor classification.
    Returns AAI error, Acc_S (real identification accuracy), and Acc_T (generated accuracy).
    """
    N_s = real_data.shape[0]
    combined = torch.cat((real_data, gen_data), dim=0)
    labels = torch.cat((torch.zeros(N_s, device=device), torch.ones(N_s, device=device)))

    # Compute pairwise Euclidean distance matrix
    dist_matrix = torch.cdist(combined, combined, p=2)
    # Exclude self-matches by setting diagonal to infinity
    dist_matrix.fill_diagonal_(1e9)

    # Identify Nearest Neighbors
    nn_indices = torch.argmin(dist_matrix, dim=1)
    nn_labels = labels[nn_indices]

    # Evaluate classification accuracy
    matches = (nn_labels == labels).float()
    acc_S = matches[:N_s].mean().item()
    acc_T = matches[N_s:].mean().item()

    # Calculate AAI formulation
    aai_error = 0.5 * ((acc_S - 0.5) ** 2 + (acc_T - 0.5) ** 2)
    return aai_error, acc_S, acc_T


# ==============================================================================
# 3. Parse Generated Trajectories and Compute Metrics for Final Timestep
# ==============================================================================
traj_dir = 'generated_trajectories'
# Regex match for filenames containing run_idx
traj_files = sorted(glob.glob(os.path.join(traj_dir, 'traj_size_*_run_*.npy')))

# Dictionary to store results grouped by data_size: {size: {'aai': [], 'acc_s': [], 'acc_t': []}}
metrics_by_size = {}

print(f"\nStarting final-step AAI metrics calculation...")

for f_path in tqdm(traj_files, desc="Processing Models"):
    match = re.search(r'traj_size_(\d+)_run_(\d+)\.npy', os.path.basename(f_path))
    if match:
        data_size = int(match.group(1))
        
        data = np.load(f_path, allow_pickle=True).item()
        trajectories = data['trajectories']

        # Extract generated samples at the final timestep and transpose to (Batch_size, 784)
        gen_last_step = trajectories[-1]
        T_tensor = torch.from_numpy(gen_last_step).float().to(device).T

        # Strictly match real data: extract the first N samples (corresponding training subset)
        real_subset = S_tensor_full[:data_size]

        # Compute AAI metrics
        err, as_val, at_val = compute_AAI_full(real_subset, T_tensor)

        # Initialize grouping if data_size is seen for the first time
        if data_size not in metrics_by_size:
            metrics_by_size[data_size] = {'aai': [], 'acc_s': [], 'acc_t': []}
        
        metrics_by_size[data_size]['aai'].append(err)
        metrics_by_size[data_size]['acc_s'].append(as_val)
        metrics_by_size[data_size]['acc_t'].append(at_val)


# Sort data sizes and compute mean/std statistics
data_sizes = sorted(list(metrics_by_size.keys()))

mean_aais, std_aais = [], []
mean_acc_S, std_acc_S = [], []
mean_acc_T, std_acc_T = [], []

for ds in data_sizes:
    # AAI statistics
    mean_aais.append(np.mean(metrics_by_size[ds]['aai']))
    std_aais.append(np.std(metrics_by_size[ds]['aai']))
    
    # Acc_S statistics
    mean_acc_S.append(np.mean(metrics_by_size[ds]['acc_s']))
    std_acc_S.append(np.std(metrics_by_size[ds]['acc_s']))
    
    # Acc_T statistics
    mean_acc_T.append(np.mean(metrics_by_size[ds]['acc_t']))
    std_acc_T.append(np.std(metrics_by_size[ds]['acc_t']))

# Convert lists to NumPy arrays for element-wise operations (shading error bands)
mean_aais = np.array(mean_aais)
std_aais = np.array(std_aais)
mean_acc_S = np.array(mean_acc_S)
std_acc_S = np.array(std_acc_S)
mean_acc_T = np.array(mean_acc_T)
std_acc_T = np.array(std_acc_T)


# ==============================================================================
# 4. Figure 1: Final Steady-state AAI vs. Dataset Size (with Shaded Error Bands)
# ==============================================================================
plt.figure(figsize=(12, 8))

# Plot mean line
plt.plot(data_sizes, mean_aais, '-o', linewidth=5, markersize=10, color='purple')
# Plot transparent error band
plt.fill_between(data_sizes, mean_aais - std_aais, mean_aais + std_aais, color='purple', alpha=0.2)

# Add a vertical dashed line for the 5th data size point (index 8)
if len(data_sizes) > 7:
    plt.axvline(x=data_sizes[7], color='black', linestyle='--', linewidth=5)

plt.xscale('log')  # Log scale for improved visualization across magnitude orders
plt.xlabel('Training Data Size (Log Scale)', fontsize=39)
plt.ylabel(r'${\cal E}^{{\rm{AAI}}}$', fontsize=39)
plt.xticks(fontsize=31)
plt.yticks(fontsize=31)
plt.grid(True, which="both", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig('aai_vs_datasize.pdf', dpi=300)

# ==============================================================================
# 5. Figure 2: Acc_S and Acc_T vs. Dataset Size (with Shaded Error Bands)
# ==============================================================================
plt.figure(figsize=(12, 8))

# Acc_S: Red dashed line with square markers
plt.plot(data_sizes, mean_acc_S, 'r--s', linewidth=5, markersize=9, label='$Acc_S$ ')
plt.fill_between(data_sizes, mean_acc_S - std_acc_S, mean_acc_S + std_acc_S, color='red', alpha=0.2)

# Acc_T: Blue dashed line with triangle markers
plt.plot(data_sizes, mean_acc_T, 'b--^', linewidth=5, markersize=9, label='$Acc_T$ ')
plt.fill_between(data_sizes, mean_acc_T - std_acc_T, mean_acc_T + std_acc_T, color='blue', alpha=0.2)

# Random chance baseline
plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=8, alpha=0.8, label='Random Chance (0.5)')

# Add vertical dashed line for the 5th data size point
if len(data_sizes) > 7:
    plt.axvline(x=data_sizes[7], color='black', linestyle='--', linewidth=5)

plt.xscale('log')
plt.xlabel('Training Data Size (Log Scale)', fontsize=39)
plt.ylabel('Accuracy', fontsize=39)
plt.xticks(fontsize=31)
plt.yticks(fontsize=31)
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend(loc='best', fontsize=31)
plt.tight_layout()
plt.savefig('accuracy_vs_datasize.pdf', dpi=300)

plt.show()