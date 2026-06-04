import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset

# ==========================================
# 1. Configuration & Data Preparation
# ==========================================
input_dir = 'generated_trajectories'
output_combined_dir = 'combined_visualizations'
os.makedirs(output_combined_dir, exist_ok=True)

# Calculate target data sizes: int(5 * 1.5^i)
target_powers = [0, 3, 9, 15, 19]
target_sizes = [int(5 * (1.5 ** i)) for i in target_powers]
print(f"Target data sizes to visualize: {target_sizes}")

img_h, img_w = 28, 28

# --- Extract real dataset (strictly identical to training) ---
print("Loading real dataset for comparison...")
transform = transforms.ToTensor()
desired_nums = [1, 3, 6]

mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_mask = torch.isin(mnist_train_full.targets, torch.tensor(desired_nums))
mnist_train_selected = Subset(mnist_train_full, train_mask.nonzero().squeeze())
test_mask = torch.isin(mnist_test_full.targets, torch.tensor(desired_nums))
mnist_test_selected = Subset(mnist_test_full, test_mask.nonzero().squeeze())

# Combine without shuffling to ensure the first N samples are the exact subset
# used for training the model of size N.
all_data = ConcatDataset([mnist_train_selected, mnist_test_selected])
data_loader_full = DataLoader(all_data, batch_size=len(all_data), shuffle=False)

images, _ = next(iter(data_loader_full))
# Map real images to [-1, 1] range to match generative data color scale, and convert to numpy
real_images_full = (images.view(-1, 784) * 2 - 1).numpy()


# ==========================================
# 2. Visualization Logic
# ==========================================
def create_2x10_grid(images_flat, max_n=20):
    """Extract first max_n images, pad with black (-1.0) if insufficient, and stitch into a 2x10 grid"""
    num_samples = min(max_n, images_flat.shape[0])

    # Initialize with black background (-1.0 represents pure black in grayscale with vmin=-1, vmax=1)
    padded_images = np.full((max_n, img_h * img_w), -1.0)
    if num_samples > 0:
        padded_images[:num_samples] = images_flat[:num_samples]

    samples = padded_images.reshape(-1, img_h, img_w)
    row1 = np.concatenate(samples[0:10], axis=1)
    row2 = np.concatenate(samples[10:20], axis=1)
    return np.concatenate([row1, row2], axis=0)


def generate_comparative_plot():
    all_files = os.listdir(input_dir)
    selected_files = []

    for size in target_sizes:
        # 修改提取方式：使用正则匹配所有包含当前 size 和 run 编号的文件
        match = [f for f in all_files if re.match(rf'traj_size_{size}_run_\d+\.npy', f)]
        if match:
            # 排序后默认选取第一个（例如 run_0）用于可视化展示
            match.sort()
            selected_files.append((size, match[0]))
        else:
            print(f"Warning: File for size {size} not found. Skipping.")

    if not selected_files:
        print("No matching .npy files found.")
        return

    num_models = len(selected_files)

    # --- Layout Calculation (Two-column comparison) ---
    fig_width = 22  # Increased width to accommodate left and right columns
    model_block_height = 2
    gap_height = 0.1
    fig_height = (model_block_height + gap_height) * num_models

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create num_models rows, 2 columns grid
    outer_grid = gridspec.GridSpec(
        num_models, 2,
        hspace=gap_height / model_block_height,
        wspace=0.05,  # Spacing between the two columns
        left=0.08, right=0.98, top=0.92, bottom=0.05
    )

    for idx, (size, file_name) in enumerate(selected_files):
        try:
            # 1. Prepare Generated Data
            file_path = os.path.join(input_dir, file_name)
            data = np.load(file_path, allow_pickle=True).item()
            # Original trajectories shape is (num_steps, N, batch_size)
            gen_images_flat = data['trajectories'][-1].T
            gen_grid = create_2x10_grid(gen_images_flat)

            # 2. Prepare Real Training Data (Extract the first 'size' samples used during training)
            real_subset_flat = real_images_full[:size]
            real_grid = create_2x10_grid(real_subset_flat)

            # --- Plot Real Data (Left Column) ---
            ax_real = fig.add_subplot(outer_grid[idx, 0])
            ax_real.imshow(real_grid, cmap='gray', aspect='equal', interpolation='nearest', vmin=-1, vmax=1)
            ax_real.axis('off')

            # --- Plot Generated Data (Right Column) ---
            ax_gen = fig.add_subplot(outer_grid[idx, 1])
            ax_gen.imshow(gen_grid, cmap='gray', aspect='equal', interpolation='nearest', vmin=-1, vmax=1)
            ax_gen.axis('off')

            # --- Labels and Titles ---
            # Add data size label on the left side
            y_center = real_grid.shape[0] / 2.0
            x_offset = -10
            ax_real.text(x_offset, y_center, f"Size:\n{size}",
                         fontsize=26,
                         fontweight='bold',
                         color='black',
                         ha='right',
                         va='center',
                         clip_on=False)

            # Add column titles only for the first row
            if idx == 0:
                ax_real.set_title("Real Training Data ", fontsize=30, pad=20)
                ax_gen.set_title("Generated Data ", fontsize=30, pad=20)

        except Exception as e:
            print(f"Error processing size {size}: {e}")

    # --- Save Result ---
    save_filename = "real_vs_generated_comparison.pdf"
    # 修复了原代码这里的拼写错误 (save_filePname -> save_filename)
    save_path = os.path.join(output_combined_dir, save_filename)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.close()
    print(f"\nVisualization complete! Saved to: {save_path}")


if __name__ == "__main__":
    generate_comparative_plot()