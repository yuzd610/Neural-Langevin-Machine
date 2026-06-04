import os
import glob
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm import tqdm
from config import N_gen
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==========================================
# 1. Core Metrics Function Definitions
# ==========================================
def compute_AAI_full(real_data, gen_data):
    """
    Computes the AAI (Adversarial Accuracy Improvement) error based on nearest neighbor classification.
    """
    N_s = real_data.shape[0]
    combined = torch.cat((real_data, gen_data), dim=0)
    labels = torch.cat((torch.zeros(N_s, device=device), torch.ones(N_s, device=device)))

    dist_matrix = torch.cdist(combined, combined, p=2)
    dist_matrix.fill_diagonal_(1e9)

    nn_indices = torch.argmin(dist_matrix, dim=1)
    nn_labels = labels[nn_indices]

    matches = (nn_labels == labels).float()
    acc_S = matches[:N_s].mean().item()
    acc_T = matches[N_s:].mean().item()

    aai_error = 0.5 * ((acc_S - 0.5) ** 2 + (acc_T - 0.5) ** 2)
    return aai_error


def compute_cov_mse(cov_real, gen_data):
    """
    Computes the second-moment error (MSE of Covariance).
    """
    cov_gen = torch.cov(gen_data.T)
    mse = torch.mean((cov_real - cov_gen) ** 2).item()
    return mse


# ==========================================
# 2. Base Data Preparation (Preload full MNIST Train + Test)
# ==========================================
transform = transforms.ToTensor()
mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

N_image = N_gen

# ==========================================
# 3. Parse Trajectories and Calculate Metrics
# ==========================================
traj_dir = 'generated_trajectories'
traj_files = sorted(glob.glob(os.path.join(traj_dir, 'traj_digits_*.npy')))


# Sort files by the number of digit classes they contain
def extract_num_classes(filepath):
    match = re.search(r'digits_(\d+)', os.path.basename(filepath))
    return int(match.group(1)) if match else 0


traj_files.sort(key=extract_num_classes)

num_classes_list = []
aai_results = []
mse_results = []

print(f"\nStarting AAI and MSE metrics calculation for different digit sets...")

for f_path in tqdm(traj_files, desc="Processing Models"):
    num_classes = extract_num_classes(f_path)
    num_classes_list.append(num_classes)

    # ---------------- Strictly align data loading logic with training phase ----------------
    desired_nums = list(range(num_classes))

    train_mask = torch.isin(mnist_train_full.targets, torch.tensor(desired_nums))
    mnist_train_selected = Subset(mnist_train_full, train_mask.nonzero().squeeze())

    test_mask = torch.isin(mnist_test_full.targets, torch.tensor(desired_nums))
    mnist_test_selected = Subset(mnist_test_full, test_mask.nonzero().squeeze())

    # Merge Train and Test sets
    all_data = ConcatDataset([mnist_train_selected, mnist_test_selected])

    # Note: shuffle=False, intercept first N_image images to ensure data
    # matches exactly what the model saw during training.
    real_data_loader = DataLoader(all_data, batch_size=N_image, shuffle=False)
    images_real, _ = next(iter(real_data_loader))

    # Map real data to [-1, 1], flatten, and send to device
    S_tensor = (images_real.view(-1, 784) * 2 - 1).to(device)
    actual_sample_size = S_tensor.shape[0]  # Get actual number of images retrieved
    cov_S = torch.cov(S_tensor.T)
    # --------------------------------------------------------------------------------------

    # Read generated trajectory data
    data = np.load(f_path, allow_pickle=True).item()
    trajectories = data['trajectories']

    # Focus on the final steady-state result (the last timestep)
    final_traj_np = trajectories[-1]
    raw_T = torch.from_numpy(final_traj_np).float().to(device)

    # Transpose and truncate to match the sample size of real data
    # (AAI requires equal sample sizes on both sides)
    if raw_T.shape[0] == 784:
        T_tensor = raw_T.T[:actual_sample_size]
    else:
        T_tensor = raw_T[:actual_sample_size]

    # Calculate metrics
    aai_val = compute_AAI_full(S_tensor, T_tensor)
    mse_val = compute_cov_mse(cov_S, T_tensor)

    aai_results.append(aai_val)
    mse_results.append(mse_val)

# ==========================================
# 4. Chart 1: AAI Evolution vs. Number of Digit Classes
# ==========================================
plt.figure(figsize=(14, 10))
plt.plot(num_classes_list, aai_results, marker='o', color='darkred', markersize=12, linewidth=4)

plt.xlabel('Number of Digit Classes in Dataset', fontsize=39)
plt.ylabel(r'${\cal E}^{{\rm{AAI}}}$', fontsize=39)
plt.xticks(num_classes_list, fontsize=35)
plt.yticks(fontsize=35)
plt.grid(True, which="both", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig('aai_vs_classes.pdf', dpi=300)
print("Saved aai_vs_classes.pdf")

# ==========================================
# 5. Chart 2: MSE Evolution vs. Number of Digit Classes
# ==========================================
plt.figure(figsize=(14, 10))
plt.plot(num_classes_list, mse_results, marker='s', color='midnightblue', markersize=12, linewidth=4)

plt.xlabel('Number of Digit Classes in Dataset', fontsize=39)
plt.ylabel(r'${\cal E}^{(2)}$ ', fontsize=39)
plt.xticks(num_classes_list, fontsize=35)
plt.yticks(fontsize=35)
plt.grid(True, which="both", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig('mse_vs_classes.pdf', dpi=300)
print("Saved mse_vs_classes.pdf")

# Display both plots
plt.show()