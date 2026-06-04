import os
import glob
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from config import N_gen
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. Data Preparation
# ==========================================
transform = transforms.ToTensor()
mnist_train_full = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

# Digit filtering logic
desired_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
mask = torch.isin(mnist_train_full.targets, torch.tensor(desired_nums))
selected_indices = mask.nonzero().squeeze()

mnist_train_selected = Subset(mnist_train_full, selected_indices)
print(f"Filtered dataset size: {len(mnist_train_selected)}")

SAMPLE_SIZE = N_gen
real_data_loader = DataLoader(mnist_train_selected, batch_size=SAMPLE_SIZE, shuffle=False)
images_real, _ = next(iter(real_data_loader))

# Map to [-1, 1] and flatten
S_tensor = (images_real.view(-1, 784) * 2 - 1).to(device)

# Pre-calculate covariance matrix of real data to avoid redundant computation in loops
# torch.cov expects (num_features, num_samples), so we transpose to (784, 10000)
cov_S = torch.cov(S_tensor.T)


# ==========================================
# 2. Core Metrics (AAI Metric & MSE Metric)
# ==========================================
def compute_AAI_full(real_data, gen_data):
    """
    Computes the Adversarial Accuracy Improvement (AAI) error based on
    nearest neighbor classification accuracy between real and generated data.
    """
    N_s = real_data.shape[0]
    combined = torch.cat((real_data, gen_data), dim=0)
    labels = torch.cat((torch.zeros(N_s, device=device), torch.ones(N_s, device=device)))

    # Compute Euclidean distance matrix
    dist_matrix = torch.cdist(combined, combined, p=2)
    dist_matrix.fill_diagonal_(1e9)  # Ignore self-distance

    nn_indices = torch.argmin(dist_matrix, dim=1)
    nn_labels = labels[nn_indices]

    matches = (nn_labels == labels).float()
    acc_S = matches[:N_s].mean().item()  # Accuracy for real samples
    acc_T = matches[N_s:].mean().item()  # Accuracy for generated samples

    aai_error = 0.5 * ((acc_S - 0.5) ** 2 + (acc_T - 0.5) ** 2)
    return aai_error, acc_S, acc_T


# Compute Second Moment Error (MSE of Covariance)
def compute_cov_mse(cov_real, gen_data):
    """
    Computes the Mean Squared Error between the covariance matrices
    of real data and generated data.
    """
    # Calculate covariance matrix of generated data (784, 784)
    cov_gen = torch.cov(gen_data.T)
    # Compute MSE between the two covariance matrices
    mse = torch.mean((cov_real - cov_gen) ** 2).item()
    return mse


# ==========================================
# 3. Parsing and Batch Calculation
# ==========================================
# Ensure the path matches the 'output_dir' used during trajectory generation
traj_dir = 'generated_trajectories_latent'
traj_files = sorted(glob.glob(os.path.join(traj_dir, 'traj_*.npy')))
parsed_files = []

for f in traj_files:
    # Extract digits from filename, e.g., extract 201 from 'traj_201.npy'
    match = re.search(r'traj_.*?(\d+)\.npy', os.path.basename(f))
    if match:
        parsed_files.append((int(match.group(1)), f))
    else:
        print(f"Warning: Could not parse model update count from filename: {f}")

# Sort by model training steps (t_age) in ascending order
parsed_files.sort(key=lambda x: x[0])

all_results = []
print(f"\nCalculating AAI and MSE metrics for {len(parsed_files)} model files...")

for model_update_step, f_path in tqdm(parsed_files, desc="Processing Models"):
    data = np.load(f_path, allow_pickle=True).item()
    time_steps = data['steps']
    trajectories = data['trajectories']

    # Added 'mse_cov' list to result dictionary
    res_per_model = {'aai': [], 'acc_S': [], 'acc_T': [], 'mse_cov': []}
    for i in range(len(time_steps)):
        raw_T = torch.from_numpy(trajectories[i]).float().to(device)

        # Handle shape: trajectories[i] is (784, batch_size), needs to be (batch_size, 784)
        T_tensor = raw_T.T[:SAMPLE_SIZE] if raw_T.shape[0] == 784 else raw_T[:SAMPLE_SIZE]

        err, as_val, at_val = compute_AAI_full(S_tensor, T_tensor)

        # Calculate Covariance MSE for current step
        mse_val = compute_cov_mse(cov_S, T_tensor)

        res_per_model['aai'].append(err)
        res_per_model['acc_S'].append(as_val)
        res_per_model['acc_T'].append(at_val)
        res_per_model['mse_cov'].append(mse_val)

    all_results.append({
        'step_val': model_update_step,  # Represents the number of model parameter updates (t_age)
        'time_steps': time_steps,      # Logarithmic timesteps from Langevin dynamics
        'data': res_per_model
    })

# ==========================================
# 4. Figure 1: AAI Evolution over Generation Steps
# ==========================================
plt.figure(figsize=(19, 10))
num_models = len(all_results)
colors = cm.coolwarm_r(np.linspace(0.0, 1.0, num_models))

for idx, res in enumerate(all_results):
    plt.plot(res['time_steps'], res['data']['aai'], marker='o', markersize=6,
             linewidth=2.5, color=colors[idx], label=f"{res['step_val']}")

plt.xscale('log')
plt.xlabel('Generation Time Steps (Log Scale)', fontsize=39)
plt.ylabel(r'${\cal E}^{{\rm{AAI}}}$', fontsize=39)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.grid(True, which="both", linestyle="--", alpha=0.5)

# Legend placement outside the plotting area
plt.legend(title=r'$t_{age} \, [\mathrm{parameter \ updates}]$', bbox_to_anchor=(1.04, 1),
           loc="upper left", fontsize=26, title_fontsize=23)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('aai_evolution.pdf', dpi=300, bbox_inches='tight')


# ==========================================
# 5. Figure 2: Steady-state Metrics vs Training Updates (AAI Focus)
# ==========================================
final_steps = []
final_aai = []
final_acc_S = []
final_acc_T = []

for res in all_results:
    final_steps.append(res['step_val'])
    final_aai.append(res['data']['aai'][-1])
    final_acc_S.append(res['data']['acc_S'][-1])
    final_acc_T.append(res['data']['acc_T'][-1])

plt.figure(figsize=(14, 10))
plt.plot(final_steps, final_aai, 'k-o', linewidth=3, markersize=10, label=r'Steady-state ${\cal E}^{{\rm{AAI}}}$')
plt.plot(final_steps, final_acc_S, 'r--s', linewidth=2, markersize=8, label='Final $Acc_S$ (Real)')
plt.plot(final_steps, final_acc_T, 'b--^', linewidth=2, markersize=8, label='Final $Acc_T$ (Gen)')

plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=2, alpha=0.6, label='Random Chance (0.5)')

plt.xscale('log')
plt.xlabel(r'$t_{age} \, [\mathrm{parameter \ updates}]$', fontsize=39)
plt.ylabel('Metric Values', fontsize=39)
plt.xticks(fontsize=31)
plt.yticks(fontsize=31)
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend(loc='best', fontsize=29, frameon=True)
plt.tight_layout()
plt.savefig('steady_state.pdf', dpi=300)


# ==========================================
# 6. Figure 3: MSE of Second Moment Evolution (No Broken Axis)
# ==========================================
plt.figure(figsize=(19, 10))

for idx, res in enumerate(all_results):
    plt.plot(res['time_steps'], res['data']['mse_cov'], marker='o', markersize=6,
             linewidth=2.5, color=colors[idx], label=f"{res['step_val']}")

plt.xscale('log')
plt.xlabel('Generation Time Steps (Log Scale)', fontsize=39)
plt.ylabel(r'${\cal E}^{(2)}$', fontsize=39)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.grid(True, which="both", linestyle="--", alpha=0.5)

# Legend placement outside the plotting area matching Figure 1
plt.legend(title=r'$t_{age} \, [\mathrm{parameter \ updates}]$', bbox_to_anchor=(1.04, 1),
           loc="upper left", fontsize=26, title_fontsize=23)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('mse_cov_evolution.pdf', dpi=300, bbox_inches='tight')

# Display all plots
plt.show()