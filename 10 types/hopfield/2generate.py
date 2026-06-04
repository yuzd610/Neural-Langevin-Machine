import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Increase global font sizes
plt.rcParams.update({'font.size': 24})

# 1. Data Preparation: Extract only the first digit
torch.manual_seed(42)
transform = transforms.ToTensor()
mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

patterns = []
labels_found = set()

for img, label in mnist_train_full:
    if label not in labels_found:
        img_scaled = img.view(-1) * 2.0 - 1.0
        
        # Consistent mapping with training: clamp and atanh
        epsilon = 0.03
        img_clipped = torch.clamp(img_scaled, min=-1.0 + epsilon, max=1.0 - epsilon)
        img_transformed = torch.atanh(img_clipped)
        
        patterns.append(img_transformed)
        labels_found.add(label)
    if len(labels_found) == 10:
        break

# Take the first digit, maintaining shape (1, 784)
X_single = torch.stack(patterns)[6:7]

# 2. Load energy model weights
try:
    weights = torch.load('hopfield_energy_weights.pth')
    J = weights['J']
    b = weights['b']
except FileNotFoundError:
    print("Error: 'hopfield_energy_weights.pth' not found. Ensure the training code has run successfully.")
    exit()

# 3. Construct 50% noise
def add_noise(x, noise_ratio):
    x_noisy = x.clone()
    mask = torch.rand_like(x) < noise_ratio
    # 这里的噪声依然可以在 [-1, 1] 甚至更大的范围内添加
    random_noise = torch.rand_like(x) * 2.0 - 1.0 
    x_noisy[mask] = random_noise[mask]
    return x_noisy

X_noisy = add_noise(X_single, noise_ratio=0.25)

# 4. Dynamics Evolution and Energy Logging
total_steps = 451
dt = 0.01
# Select 4 specific timesteps to save images
record_steps = [0, 150, 300, 450] 

energies = []
saved_points = []

X = X_noisy.clone()

for step in range(total_steps):
    # Calculate local field and error term
    h = torch.matmul(torch.tanh(X), J.t())
    e = -X + h + b
    
    # Record current system energy
    energy = 0.5 * torch.sum(e ** 2).item()
    energies.append(energy)
    
    # Extract and save image data at specified steps
    if step in record_steps:
        # Apply tanh to restore the values from unbounded space back to [-1, 1]
        img_tensor = torch.tanh(X.clone()).cpu().view(28, 28)
        img_disp = (img_tensor + 1.0) / 2.0  # Map back to [0, 1] for display
        saved_points.append((step, img_disp.numpy()))
        
    # Calculate driving force F and update X (Euler method)
    phi_prime = 1.0 - torch.tanh(X)**2
    J_T_e = torch.matmul(e, J)
    F = e - phi_prime * J_T_e
    X = X + F * dt


# 5. Plot and save the 4 digit images as independent PDF files
for (step, img) in saved_points:
    fig_img, ax_img = plt.subplots(figsize=(2, 2))
    ax_img.imshow(img, cmap='gray')
    ax_img.axis('off') # Remove axes
    img_save_path = f'digit_step_{step}.pdf'
    plt.savefig(img_save_path, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig_img)
    print(f"Saved digit image: {img_save_path}")

# 6. Plot the main chart (Relationship between Kinetic Energy and Dynamics Step)
fig, ax = plt.subplots(figsize=(12, 7))

# Plot the energy descent curve
ax.plot(range(total_steps), energies, color='#2c3e50', linewidth=3.5)

# Set labels with enlarged fonts
ax.set_xlabel("Dynamics Step", fontsize=28)
ax.set_ylabel("Kinetic Energy", fontsize=28)

# Set tick parameters
ax.tick_params(axis='both', which='major', labelsize=22)
ax.grid(True, linestyle='--', alpha=0.7)

# Set reasonable y-axis margins
ax.set_ylim(min(energies) * 0.9, max(energies) * 1.1)

# Save main plot as a PDF file
main_plot_path = 'energy_descent_main.pdf'
plt.savefig(main_plot_path, format='pdf', bbox_inches='tight')
print(f"Saved main plot: {main_plot_path}")

# Display the plot
plt.show()