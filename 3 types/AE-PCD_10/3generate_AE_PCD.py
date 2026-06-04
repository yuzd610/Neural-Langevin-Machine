import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import necessary parameters from config
from config import delta_t, N, T, N_VAE,N_gen

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed):
    """Sets seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(29)

def phi_prime(x):
    """Derivative of the activation function (tanh)."""
    return 1.0 - torch.tanh(x) ** 2

# ==========================================
# 1. Define and Load VAE Model
# ==========================================
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=N_VAE):
        super(VAE, self).__init__()
        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # Use tanh to map output to [-1, 1] range
        return torch.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Initialize and load the pre-trained VAE
loaded_vae = VAE(latent_dim=N_VAE).to(device)

try:
    loaded_vae.load_state_dict(torch.load("vae_mnist_30dim.pth"))
    print("VAE model weights loaded successfully!")
except FileNotFoundError:
    print("Error: 'vae_mnist_30dim.pth' not found. Please check the file path.")
    exit()

loaded_vae.eval()

# ==========================================
# 2. Core Parameters and Timestep Setup
# ==========================================
batch_size = N_gen  # Number of latent vectors to simulate
noise_size = torch.sqrt(torch.tensor(2 * T * delta_t, device=device))

# Generate logarithmic sampling steps for trajectory tracking
num_log_steps = 22
log_steps = [int(round(5 * (1.5 ** i))) for i in range(num_log_steps)]
max_step = max(log_steps)

print(f"Logarithmic sampling steps ({len(log_steps)} total):\n{log_steps}")
print(f"Maximum iteration steps per model: {max_step}")

# ==========================================
# 3. Batch Process Models and Generate Trajectories
# ==========================================
checkpoint_dir = 'checkpoints_latent'
output_dir = 'generated_trajectories_latent'
os.makedirs(output_dir, exist_ok=True)

# Find all interaction matrix files (J_*.npy)
j_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'J_*.npy')))

if not j_files:
    print(f"No model files found in {checkpoint_dir}!")

for j_path in j_files:
    # Identify corresponding bias file
    b_path = j_path.replace('J_', 'B_')
    if not os.path.exists(b_path):
        print(f"Warning: Corresponding bias file {b_path} not found. Skipping model.")
        continue

    model_name = os.path.basename(j_path).replace('J_', '').replace('.npy', '')
    print(f"\n---> Evaluating latent space model: {model_name}")

    # Load interaction matrix J and bias B
    J = torch.from_numpy(np.load(j_path)).float().to(device)
    B_bias = torch.from_numpy(np.load(b_path)).float().to(device)

    # Initialize Langevin dynamics starting point (mu_current) in latent space
    # Shape: (N_VAE, batch_size)
    mu_current = torch.randn(size=(N_VAE, batch_size), device=device)

    # Tensor to store decoded images in pixel space (784 dimensions)
    # Shape: (Num_Steps, Pixels, Batch_Size)
    saved_trajectories = torch.zeros((len(log_steps), N, batch_size), device=device)

    # Main Simulation Loop
    for step in tqdm(range(max_step + 1), desc=f"Simulating {model_name}"):

        # If current step is in log_steps, decode and save the state
        if step in log_steps:
            idx = log_steps.index(step)
            with torch.no_grad():
                # Transpose mu_current from (N_VAE, batch_size) to (batch_size, N_VAE) for VAE
                decoded_images = loaded_vae.decode(mu_current.T)
                # Store as (Pixels, Batch_Size)
                saved_trajectories[idx] = decoded_images.T

        # --- Langevin Dynamics Update in Latent Space ---
        
        # Calculate field h and gradient terms
        tanh_mu = torch.tanh(mu_current)
        h = J @ tanh_mu
        delta_h = h - mu_current + B_bias.unsqueeze(1)
        delta_J = J.T @ delta_h

        # Update mu_current based on the Langevin equation
        mu_current = (
                (1 - delta_t) * mu_current +
                delta_t * (h + B_bias.unsqueeze(1)) -
                delta_t * phi_prime(mu_current) * delta_J +
                noise_size * torch.randn(N_VAE, batch_size, device=device)
        )

    # 4. Save simulation results
    save_data = {
        'steps': log_steps,
        'trajectories': saved_trajectories.cpu().numpy()
    }

    out_path = os.path.join(output_dir, f'traj_{model_name}.npy')
    np.save(out_path, save_data)

print("\nAll latent space trajectories generated and decoded successfully!")