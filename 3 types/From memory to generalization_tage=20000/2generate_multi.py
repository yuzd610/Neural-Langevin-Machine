import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

# config is already defined
from config import delta_t, N, T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(29)


def phi_prime(x):
    return 1.0 - torch.tanh(x) ** 2


# ==========================================
# 1. Core Parameters & Timestep Settings
# ==========================================
noise_size = torch.sqrt(torch.tensor(2 * T * delta_t, device=device))

# Generate logarithmic timesteps: 5 * 1.5^i, 22 points in total (max ~24939)

log_steps = [50000]
max_step = max(log_steps)

print(f"Total logarithmic timesteps to sample: {len(log_steps)}\n{log_steps}")
print(f"Maximum iteration steps per model: {max_step}")

# ==========================================
# 2. Batch Load Models and Generate Trajectories
# ==========================================
checkpoint_dir = 'checkpoints_by_size'
output_dir = 'generated_trajectories'
os.makedirs(output_dir, exist_ok=True)

j_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'J_size_*_run_*.npy')))

if not j_files:
    print(f"No model files found in {checkpoint_dir}!")

for j_path in j_files:
    # Match the corresponding B_bias path
    b_path = j_path.replace('J_', 'B_')
    if not os.path.exists(b_path):
        print(f"Warning: Corresponding bias file {b_path} not found. Skipping this model.")
        continue

    # Update regex to extract both dataset size (size) and run index (run_idx)
    basename = os.path.basename(j_path)
    match = re.search(r'J_size_(\d+)_run_(\d+)\.npy', basename)
    if match:
        current_data_size = int(match.group(1))
        run_idx = int(match.group(2))
    else:
        print(f"Warning: Could not parse size and run_idx from {basename}. Skipping.")
        continue

    current_batch_size = current_data_size  # Number of generated images = Training dataset size
    model_name = f"size_{current_data_size}_run_{run_idx}"

    print(f"\n---> Evaluating model {model_name}. Generating {current_batch_size} images.")

    # Load current model parameters
    J = torch.from_numpy(np.load(j_path)).float().to(device)
    B_bias = torch.from_numpy(np.load(b_path)).float().to(device)

    # Initialize the starting point for Langevin dynamics: Y_current (Shape: N, current_batch_size)
    y_current = torch.randn(size=(N, current_batch_size), device=device)

    # Initialize Tensor to store results at the 22 key logarithmic timesteps
    saved_trajectories = torch.zeros((len(log_steps), N, current_batch_size), device=device)

    # Dynamic iteration process
    for step in tqdm(range(max_step + 1), desc=f"Simulating {model_name}"):

        # If current step is in the predefined log_steps, record the state
        # (Store the images after tanh activation)
        if step in log_steps:
            idx = log_steps.index(step)
            saved_trajectories[idx] = torch.tanh(y_current)

        # Langevin dynamics physical update
        with torch.no_grad():
            tanh_Y = torch.tanh(y_current)
            h = J @ tanh_Y
            delta_h = h - y_current + B_bias.unsqueeze(1)
            delta_J = J.T @ delta_h

            # Compute next state for y_current; noise dimension must match current_batch_size
            y_current = (
                    (1 - delta_t) * y_current +
                    delta_t * (h + B_bias.unsqueeze(1)) -
                    delta_t * phi_prime(y_current) * delta_J +
                    noise_size * torch.randn(N, current_batch_size, device=device)
            )

    # 3. Save generated multi-step trajectories to disk
    save_data = {
        'steps': log_steps,
        'data_size': current_data_size,       # Record training data size
        'run_idx': run_idx,                  # Record the run index for this trajectory
        'generated_count': current_batch_size, # Record total generated image count
        'trajectories': saved_trajectories.cpu().numpy()
    }

    # Include run_idx in the filename to ensure the 5 trajectory files do not overwrite each other
    out_path = os.path.join(output_dir, f'traj_size_{current_data_size}_run_{run_idx}.npy')
    np.save(out_path, save_data)

print("\nAll model trajectories generated successfully!")