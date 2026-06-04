import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm

# ==========================================
# 1. Configuration Parameters
# ==========================================
input_dir = 'generated_trajectories_latent'
output_combined_dir = 'combined_visualizations'
os.makedirs(output_combined_dir, exist_ok=True)

# Target model numbers (iteration counts)
target_models = [201, 988, 4852, 23836, 117107]
img_h, img_w = 28, 28


def generate_compact_seamless_plot():
    # --- 1. Search for matching files ---
    selected_files = []

    for m_num in target_models:
        # Use glob to search for files containing the number, e.g., traj_*201.npy
        search_pattern = os.path.join(input_dir, f'traj_*{m_num}.npy')
        matches = glob.glob(search_pattern)

        if matches:
            # If multiple matches exist, take the first one
            selected_files.append((m_num, matches[0]))
        else:
            print(f"Warning: No trajectory file found containing number {m_num} (Search pattern: {search_pattern})")

    if not selected_files:
        print("Error: No matching model files found. Please check the generated .npy filenames.")
        return

    num_rows = len(selected_files)

    # --- 2. Generate color gradient ---
    colors = cm.coolwarm_r(np.linspace(0.0, 1.0, num_rows))

    # --- 3. Layout calculations ---
    fig_width = 12
    model_block_height = 2
    gap_height = 0.05
    fig_height = (model_block_height + gap_height) * num_rows

    fig = plt.figure(figsize=(fig_width, fig_height))
    outer_grid = gridspec.GridSpec(
        num_rows, 1,
        hspace=gap_height / model_block_height,
        left=0.15, right=0.98, top=0.95, bottom=0.05
    )

    # --- 4. Iterate and plot each row ---
    for idx, (m_num, file_path) in enumerate(selected_files):
        try:
            # Load data
            data = np.load(file_path, allow_pickle=True).item()

            # Extract trajectories from the last step [last_step, 784, batch_size]
            # Transpose to [batch_size, 784]
            images = data['trajectories'][-1].T

            # Select samples
            samples = images[80:100].reshape(-1, img_h, img_w)

            # Concatenate into a 2x10 grid
            row1 = np.concatenate(samples[0:10], axis=1)
            row2 = np.concatenate(samples[10:20], axis=1)
            combined_img = np.concatenate([row1, row2], axis=0)

            # Plotting
            ax_img = fig.add_subplot(outer_grid[idx, 0])
            ax_img.imshow(combined_img, cmap='gray', aspect='equal', interpolation='nearest')
            ax_img.axis('off')

            # Place labels
            y_center = combined_img.shape[0] / 2.0
            x_offset = -8

            # Display the model iteration number directly
            ax_img.text(x_offset, y_center, str(m_num),
                        fontsize=34,
                        fontweight='bold',
                        color=colors[idx],
                        ha='right',
                        va='center',
                        clip_on=False)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # --- 5. Save the results ---
    save_filename = "compact_model_comparison_latent.pdf"
    save_path = os.path.join(output_combined_dir, save_filename)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.close()
    print(f"\nVisualization complete! Comparison plot saved to: {save_path}")


if __name__ == "__main__":
    generate_compact_seamless_plot()