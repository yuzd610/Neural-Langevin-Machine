import os
import glob
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import delta_t, N, T,N_gen

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
# 1. Core Parameters & Timesteps
# ==========================================
batch_size = N_gen
noise_size = torch.sqrt(torch.tensor(2 * T * delta_t, device=device))

# Generate logarithmic timesteps: 5 * 1.5^i, total 22 recording points (max ~24939)
num_log_steps = 22
log_steps = [int(round(5 * (1.5 ** i))) for i in range(num_log_steps)]
max_step = max(log_steps)

print(f"Total logarithmic timesteps to sample: {len(log_steps)}\n{log_steps}")
print(f"Maximum iteration steps per model: {max_step}")

# ==========================================
# 2. Batch Loading Models and Generating Trajectories
# ==========================================
checkpoint_dir = 'checkpoints'  # Directory where weights are stored
output_dir = 'generated_trajectories'
os.makedirs(output_dir, exist_ok=True)


# Get all J matrix files and sort them based on the number of digits they contain (2, 3, ... 10)
def extract_num_classes(filepath):
    match = re.search(r'digits_(\d+)', filepath)
    return int(match.group(1)) if match else 0


j_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'J_digits_*.npy')), key=extract_num_classes)

if not j_files:
    print(f"No model files found in {checkpoint_dir}! Please check the path and filenames.")

for j_path in j_files:
    # Match corresponding B_bias filename
    b_path = j_path.replace('J_digits_', 'B_bias_digits_')

    if not os.path.exists(b_path):
        print(f"Warning: Corresponding bias file {b_path} not found. Skipping this model.")
        continue

    # Extract model identifier (e.g., digits_2, digits_10, etc.)
    model_name = os.path.basename(j_path).replace('J_', '').replace('.npy', '')
    print(f"\n---> Evaluating model: {model_name} (Number of digit classes: {extract_num_classes(j_path)})")

    # Load weights for the current model
    J = torch.from_numpy(np.load(j_path)).float().to(device)
    B_bias = torch.from_numpy(np.load(b_path)).float().to(device)

    # Initialize starting point Y_current for Langevin Dynamics (Shape: N, batch_size)
    y_current = torch.randn(size=(N, batch_size), device=device)

    # Pre-allocate Tensor to store results at the 22 key timesteps (Shape: 22, N, batch_size)
    saved_trajectories = torch.zeros((len(log_steps), N, batch_size), device=device)

    # Iteratively generate trajectories
    for step in tqdm(range(max_step + 1), desc=f"Simulating {model_name}"):

        # If current step is in the predefined logarithmic steps, record the state
        # (Store the image data after tanh activation)
        if step in log_steps:
            idx = log_steps.index(step)
            saved_trajectories[idx] = torch.tanh(y_current)

        # Langevin Dynamics physical update
        tanh_Y = torch.tanh(y_current)
        h = J @ tanh_Y
        delta_h = h - y_current + B_bias.unsqueeze(1)
        delta_J = J.T @ delta_h

        # Compute next y_current
        y_current = (
                (1 - delta_t) * y_current +
                delta_t * (h + B_bias.unsqueeze(1)) -
                delta_t * phi_prime(y_current) * delta_J +
                noise_size * torch.randn(N, batch_size, device=device)
        )

    # ==========================================
    # 3. Save multi-step trajectories to local disk
    # ==========================================
    # Save as a dictionary for easy mapping of steps to trajectories during loading
    save_data = {
        'steps': log_steps,
        'trajectories': saved_trajectories.cpu().numpy()  # Convert back to numpy for saving
    }

    out_path = os.path.join(output_dir, f'traj_{model_name}.npy')
    np.save(out_path, save_data)
    print(f"Saved trajectories to: {out_path}")

print("\nAll model trajectories generated successfully!")