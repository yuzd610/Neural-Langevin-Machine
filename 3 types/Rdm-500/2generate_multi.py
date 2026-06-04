import os
import glob
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


set_seed(30)


def phi_prime(x):
    return 1.0 - torch.tanh(x) ** 2



# ==========================================
# 1. Core Parameters and Time-step Settings
# ==========================================
batch_size = N_gen
noise_size = torch.sqrt(torch.tensor(2 * T * delta_t, device=device))


num_log_steps = 18
# Generate logarithmic time steps for sampling
log_steps = [int(round(13 * (1.5 ** i))) for i in range(num_log_steps)]
max_step = max(log_steps)

print(f"Total log steps to be sampled ({len(log_steps)}):\n{log_steps}")
print(f"Maximum iteration steps per model: {max_step}")

# ==========================================
# 2. Batch Read Models and Generate Trajectories
# ==========================================
checkpoint_dir = 'checkpoints'  # Directory where models are stored
output_dir = 'generated_trajectories'
os.makedirs(output_dir, exist_ok=True)

# Get all saved J matrix files and sort them by name
j_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'J_*.npy')))

if not j_files:
    print(f"No model files found in {checkpoint_dir}!")

for j_path in j_files:
    # Resolve the corresponding B_bias path
    b_path = j_path.replace('J_', 'B_')
    if not os.path.exists(b_path):
        print(f"Warning: Corresponding bias file {b_path} not found, skipping this model.")
        continue

    # Extract model identifier (e.g., idx_0_step_201)
    model_name = os.path.basename(j_path).replace('J_', '').replace('.npy', '')
    print(f"\n---> Evaluating model: {model_name}")

    # Load current model parameters
    J = torch.from_numpy(np.load(j_path)).float().to(device)
    B_bias = torch.from_numpy(np.load(b_path)).float().to(device)

    # Initialize starting point Y_current for Langevin dynamics (Shape: N, batch_size)
    y_current = torch.normal(mean=0.0, std=1.0, size=(N, batch_size), device=device)

    # Tensor to store results for the specific log steps (Shape: num_log_steps, N, batch_size)
    saved_trajectories = torch.zeros((len(log_steps), N, batch_size), device=device)

    # Dynamic iteration process
    for step in tqdm(range(max_step + 1), desc=f"Simulating {model_name}"):

        # If current step is in our preset log steps, record the state (store tanh-activated images)
        if step in log_steps:
            idx = log_steps.index(step)
            saved_trajectories[idx] = torch.tanh(y_current)

        # Langevin dynamics physical update
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

    # 3. Save the multi-step trajectories of the current model to local storage
    # Saved as a dictionary to associate data with specific time steps
    save_data = {
        'steps': log_steps,
        'trajectories': saved_trajectories.cpu().numpy()  # Convert back to numpy for storage
    }

    out_path = os.path.join(output_dir, f'traj_{model_name}.npy')
    np.save(out_path, save_data)

    # Optional: Uncomment below to visualize the result of the last time step during processing
    # plot_mnist_grid(saved_trajectories[-1].T, grid_size=(6, 6), title=f"Model: {model_name} | Step: {max_step}")

print("\nAll model trajectories generated!")