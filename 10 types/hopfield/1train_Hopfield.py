import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Set random seed for reproducibility
torch.manual_seed(42)

# 1. Data Preparation: Select 10 different digits (0-9)
transform = transforms.ToTensor()
mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

patterns = []
labels_found = set()

# Iterate through the dataset to extract one sample for each digit from 0 to 9
for img, label in mnist_train_full:
    if label not in labels_found:
        # Flatten the 28x28 image to 784 and map values from [0, 1] to [-1, 1]
        img_scaled = img.view(-1) * 2.0 - 1.0
        
        # Apply clipping and atanh transformation to map to unbounded space
        epsilon = 0.03
        img_clipped = torch.clamp(img_scaled, min=-1.0 + epsilon, max=1.0 - epsilon)
        img_transformed = torch.atanh(img_clipped)
        
        patterns.append(img_transformed)
        labels_found.add(label)
    if len(labels_found) == 10:
        break

# Stack the list into a tensor with shape (10, 784)
X = torch.stack(patterns)

# 2. Define model parameters J and b
N = 784
# Randomly initialize J and initialize b to zero
J = nn.Parameter(torch.randn(N, N) * torch.sqrt(torch.tensor(4.0 / (N))))
b = nn.Parameter(torch.zeros(N))

# Define a diagonal mask to enforce J_ii = 0
mask = 1.0 - torch.eye(N)

# 3. Set up the optimizer
# Use Adam optimizer for gradient descent
optimizer = optim.Adam([J, b], lr=0.001)

# 4. Training Loop (Minimize the target energy function)
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Enforce zero diagonal elements in J
    J_masked = J * mask
    
    # Calculate the local field h and reconstruction result
    local_field = torch.matmul(torch.tanh(X), J_masked.t())
    
    # Calculate the internal error term of the energy model
    error_term = -X + local_field + b
    
    # Calculate total energy E(x)
    energy = 0.5 * torch.sum(error_term ** 2)
    
    # Backpropagation
    energy.backward()
    optimizer.step()
    
    # Print training progress
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Total Energy: {energy.item():.4f}")

# 5. Save final weights
# Apply the mask one last time to J before saving
J_final = (J * mask).detach()
b_final = b.detach()

torch.save({
    'J': J_final,
    'b': b_final
}, 'hopfield_energy_weights.pth')

print("Training complete! Weights successfully saved to 'hopfield_energy_weights.pth'")