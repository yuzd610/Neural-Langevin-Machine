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
mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Digit filtering logic
desired_nums = [ 1, 3,  6]
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
cov_S = torch.cov(S_tensor.T)


# ==========================================
# 2. Core Metrics (AAI Metric & MSE Metric)
# ==========================================
def compute_AAI_full(real_data, gen_data):
    """
    Computes Adversarial Accuracy Improvement (AAI) error based on
    nearest neighbor classification between real and generated samples.
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
    return aai_error, acc_S, acc_T


def compute_cov_mse(cov_real, gen_data):
    """
    Compute Second Moment Error (MSE of Covariance)
    """
    cov_gen = torch.cov(gen_data.T)
    mse = torch.mean((cov_real - cov_gen) ** 2).item()
    return mse


# ==========================================
# 3. Parsing and Batch Calculation
# ==========================================
traj_dir = 'generated_trajectories'
traj_files = sorted(glob.glob(os.path.join(traj_dir, 'traj_*.npy')))
parsed_files = []

for f in traj_files:
    match = re.search(r'step_(\d+)', os.path.basename(f))
    if match:
        parsed_files.append((int(match.group(1)), f))
parsed_files.sort(key=lambda x: x[0])

all_results = []
print(f"\nStarting AAI and MSE metrics calculation...")

for step_val, f_path in tqdm(parsed_files, desc="Processing Models"):
    data = np.load(f_path, allow_pickle=True).item()
    time_steps = data['steps']
    trajectories = data['trajectories']

    res_per_model = {'aai': [], 'acc_S': [], 'acc_T': [], 'mse_cov': []}

    for i in range(len(time_steps)):
        raw_T = torch.from_numpy(trajectories[i]).float().to(device)
        T_tensor = raw_T.T[:SAMPLE_SIZE] if raw_T.shape[0] == 784 else raw_T[:SAMPLE_SIZE]

        err, as_val, at_val = compute_AAI_full(S_tensor, T_tensor)
        mse_val = compute_cov_mse(cov_S, T_tensor)

        res_per_model['aai'].append(err)
        res_per_model['acc_S'].append(as_val)
        res_per_model['acc_T'].append(at_val)
        res_per_model['mse_cov'].append(mse_val)

    all_results.append({
        'step_val': step_val,
        'time_steps': time_steps,
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

# 添加 500 时间步的竖直虚线
plt.axvline(x=500, color='gray', linestyle='--', linewidth=5, alpha=0.8, label='Step 500')

plt.xscale('log')
plt.xlabel('Generation Time Steps (Log Scale)', fontsize=39)
plt.ylabel(r'${\cal E}^{{\rm{AAI}}}$', fontsize=39)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.grid(True, which="both", linestyle="--", alpha=0.5)

# Restored legend and placed it outside the plotting area
plt.legend(title=r'$t_{age} \, [\mathrm{parameter \ updates}]$', bbox_to_anchor=(1.04, 1),
           loc="upper left", fontsize=26, title_fontsize=23)

# Add right margin to prevent legend from being cropped in interactive window
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('aai_evolution.pdf', dpi=300, bbox_inches='tight')


# ==========================================
# 5. Figure 2: Steady-state Metrics vs Training Updates
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

# Random chance baseline
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
# 6. Figure 3: MSE of Second Moment Evolution
# ==========================================

fig = plt.figure(figsize=(14, 10))

# Create broken axis layout
gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.08)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

max_mse = max([max(res['data']['mse_cov']) for res in all_results])
top_limit = max(0.005, max_mse * 1.05)

for idx, res in enumerate(all_results):
    ax1.plot(res['time_steps'], res['data']['mse_cov'], marker='o', markersize=6,
             linewidth=2.5, color=colors[idx], label=f"{res['step_val']}")
    ax2.plot(res['time_steps'], res['data']['mse_cov'], marker='o', markersize=6,
             linewidth=2.5, color=colors[idx])

# 为截断图的上下两个部分都添加 500 时间步的竖直虚线
ax1.axvline(x=500, color='gray', linestyle='--', linewidth=5, alpha=0.8, label='Step 500')
ax2.axvline(x=500, color='gray', linestyle='--', linewidth=5, alpha=0.8)

ax1.set_ylim(0.005, top_limit)
ax2.set_ylim(-0.0002, 0.0042)
ax1.set_xscale('log')
ax2.set_xscale('log')

# Visual handling of the broken axis
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.tick_params(labelbottom=False, bottom=False)
ax2.xaxis.tick_bottom()

# Draw broken axis diagonal markers
d = 0.015
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=1.5)
ax1.plot((-d, +d), (-d * 4, +d * 4), **kwargs)
ax1.plot((1 - d, 1 + d), (-d * 4, +d * 4), **kwargs)

kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

ax2.axhline(y=0.004, color='gray', linestyle='--', linewidth=2, alpha=0.8)
ax2.set_yticks([0, 0.001, 0.002, 0.003, 0.004])

ax1.grid(True, which="both", linestyle="--", alpha=0.5)
ax2.grid(True, which="both", linestyle="--", alpha=0.5)

ax1.tick_params(axis='y', labelsize=31)
ax2.tick_params(axis='both', labelsize=35)

# Kept the x-axis negative offset (x=-0.02) so the label doesn't overlap with large fonts
fig.supylabel(r'${\cal E}^{(2)}$ ', fontsize=39, x=-0.02)
ax2.set_xlabel('Generation Time Steps (Log Scale)', fontsize=39)

lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, title=r'$t_{age} \, [\mathrm{parameter \ updates}]$',
           bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=26, title_fontsize=23)

# Kept the left margin padding (0.08) for the offset ylabel
plt.tight_layout(rect=[0.08, 0, 1, 1])
plt.savefig('mse_cov_evolution.pdf', dpi=300, bbox_inches='tight')

# Display all plots
plt.show()