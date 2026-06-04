import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

# config is already defined; importing hyperparameters here
from config import delta_t, N, g, n, N_data, eta, T, k, b_size, lambda1, N_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def square_mean_torch(tensor):
    return torch.sum(torch.square(tensor)) / tensor.numel()


# Derivative of the activation function
phi_prime = lambda x: 1.0 - torch.tanh(x) ** 2


class ColumnDataset(Dataset):
    def __init__(self, X): self.X = X

    def __len__(self): return self.X.size(1)

    def __getitem__(self, idx): return self.X[:, idx]


# --- 1. Download and preload the complete dataset ---
transform = transforms.ToTensor()
mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create folder for storing weights
os.makedirs('checkpoints', exist_ok=True)

# Used to store metrics for all training scenarios for final plotting
all_kl_norms = {}

# --- 2. Core outer loop: Iterate through different subsets of digit combinations ---
# Range 1 to 9, corresponding digit lists are [0,1], [0,1,2] ... [0..9]
for max_digit in range(1, 10):
    desired_nums = list(range(max_digit + 1))
    num_classes = len(desired_nums)

    print(f"\n" + "=" * 50)
    print(f"Starting training for dataset: {desired_nums} (Fixed {n} Epochs)")
    print("=" * 50)

    # ---------------- Filter current required data ----------------
    train_mask = torch.isin(mnist_train_full.targets, torch.tensor(desired_nums))
    mnist_train_selected = Subset(mnist_train_full, train_mask.nonzero().squeeze())

    test_mask = torch.isin(mnist_test_full.targets, torch.tensor(desired_nums))
    mnist_test_selected = Subset(mnist_test_full, test_mask.nonzero().squeeze())

    # Merge data
    all_data = ConcatDataset([mnist_train_selected, mnist_test_selected])

    # Select the first N_data samples
    data_loader = DataLoader(all_data, batch_size=N_data, shuffle=False)
    images, _ = next(iter(data_loader))
    images = images.to(device)

    # Data preprocessing
    scaled_matrix = images.squeeze(1) * 2 - 1
    epsilon = 0.03
    clipped_matrix = torch.clamp(scaled_matrix, min=-1.0 + epsilon, max=1.0 - epsilon)
    transformed_matrix = torch.atanh(clipped_matrix)

    matrix = transformed_matrix.reshape(transformed_matrix.shape[0], -1).T
    dataset = ColumnDataset(matrix)
    dataloader = DataLoader(dataset, batch_size=b_size, shuffle=True)

    # ---------------- Re-initialize model and optimizer ----------------
    initial_J = torch.randn(N, N, device=device) * torch.sqrt(torch.tensor((g ** 2) / N, device=device))
    initial_J.fill_diagonal_(0)  # Zero out the diagonal
    J = torch.nn.Parameter(initial_J)

    initial_B = torch.zeros(N, device=device)
    B_bias = torch.nn.Parameter(initial_B)

    optimizer = torch.optim.Adam([
        {'params': [J], 'weight_decay': lambda1},
        {'params': [B_bias], 'weight_decay': 0.0}
    ], lr=eta)

    noise_size = torch.sqrt(torch.tensor(2 * T * delta_t, device=device))
    # Re-initialize the sampling Markov Chain for the current experiment
    X = torch.randn(N, N_model, k + 1, device=device)

    kl_norm = []

    # ---------------- Start training for the current dataset ----------------
    for epoch in tqdm(range(n), desc=f"Training subset {num_classes}"):
        for batch in dataloader:
            with torch.no_grad():
                # --- Data Phase ---
                batch = batch.T
                phi_data = torch.tanh(batch)
                h_data = J @ phi_data
                x_h_data = batch - h_data - B_bias.unsqueeze(1)
                data_grad_J = (1 / (T * batch.shape[1])) * x_h_data @ phi_data.T

                # --- Model Phase (PCD Langevin Dynamics) ---
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
            # Core modification: Ensure diagonal elements are not updated
            J.grad.fill_diagonal_(0)

            B_bias.grad = (1 / T) * (x_h_model.mean(dim=1) - x_h_data.mean(dim=1))
            optimizer.step()

        # Record KL Divergence norm metrics
        klnorm1 = square_mean_torch(J.grad)
        klnorm2 = square_mean_torch(B_bias.grad)
        kl_norm.append(klnorm1.item())

        # Dynamic print to avoid no log output if n is too small
        if (epoch + 1) % max(1, n // 5) == 0:
            print(f"  [Epoch {epoch + 1}/{n}] J_grad_norm={klnorm1.item():.6f}")

    # ---------------- Training complete for current dataset, save results ----------------
    all_kl_norms[f"{num_classes} Digits"] = kl_norm

    j_path = os.path.join('checkpoints', f'J_digits_{num_classes}.npy')
    b_path = os.path.join('checkpoints', f'B_bias_digits_{num_classes}.npy')
    np.save(j_path, J.detach().cpu().numpy())
    np.save(b_path, B_bias.detach().cpu().numpy())
    print(f">> Saved current subset weights to: {j_path}")

# --- 3. Plotting results after all experiments ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for label, norm_history in all_kl_norms.items():
    plt.plot(norm_history, label=label)
plt.title('Total Training Process (KL Norm)')
plt.xlabel('Epoch')
plt.ylabel('KL Norm')
plt.legend()

plt.subplot(1, 2, 2)
for label, norm_history in all_kl_norms.items():
    # Plot the last min(30, n) steps
    plot_len = min(30, len(norm_history))
    plt.plot(range(len(norm_history) - plot_len, len(norm_history)), norm_history[-plot_len:], label=label)
plt.title(f'Convergence Comparison (Last {min(30, n)} Epochs)')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()