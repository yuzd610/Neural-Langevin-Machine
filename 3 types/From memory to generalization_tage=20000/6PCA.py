import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import os
from config import delta_t, N, T  # Note: Removed N_image1 which is not defined in config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. Target Parameters Setting & Weight Loading
# ==========================================
target_size = int(5 * (1.5 ** 10))  # Calculated to be 288
run_idx = 0  # Default to use weights from the 0-th run, can be modified as needed
save_dir = 'checkpoints_by_size'

# Define the folder to store PDF plots and create it automatically
output_pdf_dir = f'pca_plots_size_{target_size}'
os.makedirs(output_pdf_dir, exist_ok=True)

print(f"Targeting Dataset Size: {target_size}, Run Index: {run_idx}")
print(f"All PDF plots will be saved to: {output_pdf_dir}/")

# Construct file paths and load J and B
j_path = os.path.join(save_dir, f'J_size_{target_size}_run_{run_idx}.npy')
b_path = os.path.join(save_dir, f'B_size_{target_size}_run_{run_idx}.npy')

J = torch.from_numpy(np.load(j_path)).float().to(device)
B = torch.from_numpy(np.load(b_path)).float().to(device)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(20)

def phi_prime(x):
    return 1.0 - torch.tanh(x) ** 2

# Modify the grid plotting function to support saving as PDF via a save path
def plot_mnist_grid(columns, grid_size=(10, 10), image_size=(28, 28), save_path=None):
    batch_size = columns.shape[0]
    rows, cols = grid_size
    assert rows * cols == batch_size
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()
    for idx in range(batch_size):
        image = (columns[idx].cpu().numpy() + 1) / 2.0
        image = image.reshape(image_size)
        axes[idx].imshow(image, cmap='gray')
        axes[idx].axis('off')
    plt.tight_layout()
    # Save as PDF format if a save path is specified
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig) # Explicitly close the figure to release memory

# ==========================================
# 2. Dynamics Iteration Process (VRAM Optimized Version)
# ==========================================
a = 120000
b = 150000
c = 10

batch_size = 100
noise_size = torch.sqrt(torch.tensor(2 * T * delta_t, device=device))

# Only need to record the "current step" state, discarding the huge time dimension of b+1
Y_curr = torch.normal(mean=0.0, std=1.0, size=(N, batch_size), device=device)

# Used to store single trajectory data from step a to b on CPU to prevent GPU OOM
trajectory_list = []

# Dynamic evolution including the bias term B
for i in tqdm(range(b), desc="Simulations Progress"):
    tanh_Y = torch.tanh(Y_curr)
    h = J @ tanh_Y
    delta_h = h - Y_curr + B.unsqueeze(1)
    delta_J = J.T @ delta_h
    
    Y_next = (
        (1 - delta_t) * Y_curr +
        delta_t * (h + B.unsqueeze(1)) - 
        delta_t * phi_prime(Y_curr) * delta_J + 
        noise_size * torch.randn(N, batch_size, device=device)
    )
    
    # Reach the recording interval, plot in real-time
    if i >= a and (i - a) % int((b - a) / c) == 0:
        print(f"Step: {i}")
        grid_save_path = os.path.join(output_pdf_dir, f'mnist_grid_step_{i}.pdf')
        # Transfer the current tensor to CPU for plotting
        plot_mnist_grid(torch.tanh(Y_curr.T), grid_size=(10, 10), image_size=(28, 28), save_path=grid_save_path)

    # Collect trajectory after step a (only collect the last sample of the batch and move to CPU memory)
    if i >= a:
        trajectory_list.append(Y_curr[:, 20].detach().cpu())
        
    Y_curr = Y_next

# Concatenate the collected list into a single tensor with shape (b-a, N)
trajectory_tensor = torch.stack(trajectory_list, dim=0)

# ==========================================
# 3. Data Filtering & Alignment (Using exactly target_size amount of data)
# ==========================================
transform = transforms.ToTensor()
desired_nums = [1, 3, 6]
mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_mask = torch.isin(mnist_train_full.targets, torch.tensor(desired_nums))
mnist_train_selected = Subset(mnist_train_full, train_mask.nonzero().squeeze())

test_mask = torch.isin(mnist_test_full.targets, torch.tensor(desired_nums))
mnist_test_selected = Subset(mnist_test_full, test_mask.nonzero().squeeze())

all_data_selected = ConcatDataset([mnist_train_selected, mnist_test_selected])

# Intercept the first target_size samples exactly consistent with training for PCA
data_loader = DataLoader(all_data_selected, batch_size=target_size, shuffle=False)
images_for_pca, labels_for_pca = next(iter(data_loader))

actual_num_images = images_for_pca.shape[0]
flattened_images = (images_for_pca.squeeze(1).reshape(actual_num_images, -1) * 2 - 1)
images_np = flattened_images.cpu().numpy()
labels_np = labels_for_pca.cpu().numpy()

# ==========================================
# 4. Perform PCA & 3D Trajectory Plotting
# ==========================================
pca = PCA(n_components=3)
pca_result = pca.fit_transform(images_np)

colors_map = {1: 'red', 3: 'green', 6: 'blue'}

# Direct use of trajectory_tensor concatenated on CPU
processed_new_vectors = torch.tanh(trajectory_tensor).numpy()
print(f"Trajectory shape: {processed_new_vectors.shape}")

# Plot individual slices of images at the late stage of evolution
for i in range(a, b, int((b - a) / c)):
    image = processed_new_vectors[i - a, :].reshape(28, 28)
    fig = plt.figure(figsize=(3, 3))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    # Define the PDF save path for a single image slice and call savefig
    single_img_save_path = os.path.join(output_pdf_dir, f'single_image_step_{i}.pdf')
    plt.savefig(single_img_save_path, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig) # Removed the original plt.show() to prevent the program from freezing due to loop pop-ups

# Transform trajectory and plot 3D scatter plot
new_pca_result = pca.transform(processed_new_vectors)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.tick_params(axis='both', which='major', labelsize=25)

# Plot reference manifold (based only on data points of target_size)
for digit in desired_nums:
    indices = (labels_np == digit)
    ax.scatter(pca_result[indices, 0],
               pca_result[indices, 1],
               pca_result[indices, 2],
               color=colors_map[digit],
               label=f'Digit {digit}',
               alpha=0.3,
               s=50)

# Plot generated trajectory
trajectory = ax.scatter(new_pca_result[:, 0], new_pca_result[:, 1], new_pca_result[:, 2],
                        color='black', marker='o', s=1, alpha=0.7, label='Trajectory')

# Mark key time nodes
key_steps = [120000, 132000, 138000, 147000]
colors = ['yellow', 'blue', 'yellow', 'blue']
for step, color in zip(key_steps, colors):
    if step - a >= 0 and step - a < new_pca_result.shape[0]:
        ax.scatter(new_pca_result[step - a, 0], new_pca_result[step - a, 1], new_pca_result[step - a, 2],
                   color=color, marker='*', s=300)

ax.set_xlabel('X', fontsize=25, labelpad=25)
ax.set_ylabel('Y', fontsize=25, labelpad=25)
ax.set_zlabel('Z', fontsize=25, labelpad=25)
ax.legend(fontsize=16, loc='upper right')

plt.grid(True)
plt.tight_layout()

# Define the save path for the 3D trajectory plot and execute save before show()
pca_3d_save_path = os.path.join(output_pdf_dir, 'pca_3d_trajectory.pdf')
plt.savefig(pca_3d_save_path, format='pdf', bbox_inches='tight')

plt.show()
plt.close(fig)