import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import matplotlib.pyplot as plt

# Hyperparameters
beta = 0
from config import N_data, N_VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Data Preparation ---
transform = transforms.ToTensor()
desired_nums = [1, 3, 6]

# Loading datasets
mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Filter samples for specific digits
train_mask = torch.isin(mnist_train_full.targets, torch.tensor(desired_nums))
train_indices = train_mask.nonzero().squeeze()
mnist_train_selected = Subset(mnist_train_full, train_indices)

test_mask = torch.isin(mnist_test_full.targets, torch.tensor(desired_nums))
test_indices = test_mask.nonzero().squeeze()
mnist_test_selected = Subset(mnist_test_full, test_indices)

# Combine train and test sets
all_data = ConcatDataset([mnist_train_selected, mnist_test_selected])

# Create DataLoader for training (Batch Size 128 for stability)
train_loader = DataLoader(all_data, batch_size=128, shuffle=True)

# Create DataLoader for the final N_data data (used for final compression)
# shuffle=False ensures consistent ordering
final_data_loader = DataLoader(all_data, batch_size=N_data, shuffle=False)
images_full, _ = next(iter(final_data_loader))
images_full = images_full.to(device)

# Preprocessing: Scale from [0, 1] to [-1, 1]
transformed_matrix = images_full.squeeze(1) * 2 - 1
# Shape: (N_data, 784). PyTorch networks usually expect (N, Dim)
input_data_tensor = transformed_matrix.reshape( N_data, -1)


# --- 2. Define VAE Model ---
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
        # Use Tanh because the input data is scaled to [-1, 1]
        return torch.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Initialize model and optimizer
vae = VAE(latent_dim=N_VAE).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)


# --- 3. Define Loss Function ---
# Reconstruction (MSE) + KL divergence losses
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x.view(-1, 784), reduction='sum')

    # KL Divergence formula: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + beta * KLD


# --- 4. Training Loop ---
epochs = 50
print("Starting VAE training...")

vae.train()
for epoch in range(epochs):
    train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

    for batch_idx, (data, _) in enumerate(progress_bar):
        data = data.to(device)

        # Preprocessing: map [0, 1] to [-1, 1] to match model design
        data = data * 2 - 1
        data = data.view(-1, 784)

        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        progress_bar.set_postfix({'loss': loss.item() / len(data)})

    print(f'====> Epoch: {epoch + 1} Average loss: {train_loss / len(train_loader.dataset):.4f}')

# --- 5. Compress Data and Save ---
print("Training complete. Generating compressed matrix...")

vae.eval()
with torch.no_grad():
    # Use input_data_tensor (N_image, 784) prepared earlier
    # We use mu (mean) as the deterministic latent representation
    mu, logvar = vae.encode(input_data_tensor)
    z_compressed = mu  # shape: (N_image, N_VAE)

    # Transpose to (N_VAE, N_image) to match matrix multiplication conventions
    compressed_matrix = z_compressed.T.cpu().numpy()

# Save VAE model parameters
torch.save(vae.state_dict(), "vae_mnist_30dim.pth")
print("Model saved as: vae_mnist_30dim.pth")

# Save the compressed matrix J
np.save("compressed_matrix.npy", compressed_matrix)
print(f"Compressed matrix saved as: compressed_matrix.npy, shape: {compressed_matrix.shape}")

# --- 6. (Optional) Decoding Test ---
print("\n--- Decoding Test ---")
# Load model
loaded_vae = VAE(latent_dim=N_VAE).to(device)
loaded_vae.load_state_dict(torch.load("vae_mnist_30dim.pth"))
loaded_vae.eval()

# Load matrix
loaded_compressed_matrix = np.load("compressed_matrix.npy")  # (20, N)
# Take the 0-th sample for decoding
z_sample = torch.tensor(loaded_compressed_matrix[:, 0]).float().to(device).unsqueeze(0)  # (1, 20)

with torch.no_grad():
    recon_img = loaded_vae.decode(z_sample)
    # Rescale back to [0, 1] for visualization
    recon_img = (recon_img + 1) / 2
    recon_img = recon_img.view(28, 28).cpu()

plt.imshow(recon_img, cmap='gray')
plt.title(f"Reconstructed Image from {N_VAE}-dim Latent")
plt.show()