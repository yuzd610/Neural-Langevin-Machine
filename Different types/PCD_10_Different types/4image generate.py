import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==========================================
# 1. Configuration Parameters
# ==========================================
input_dir = 'generated_trajectories'
output_combined_dir = 'combined_visualizations'
os.makedirs(output_combined_dir, exist_ok=True)

# Target number of digit classes
target_classes = [2, 4, 6, 8, 10]

# Mapping for the text labels on the left side
label_map = {
    2: "0-1",
    4: "0-3",
    6: "0-5",
    8: "0-7",
    10: "0-9"
}

img_h, img_w = 28, 28

def generate_compact_seamless_plot():
    # Look for corresponding trajectory files based on target classes
    selected_files = []
    for cls in target_classes:
        file_name = f'traj_digits_{cls}.npy'
        file_path = os.path.join(input_dir, file_name)
        if os.path.exists(file_path):
            selected_files.append((cls, file_name))
        else:
            print(f"Warning: File not found - {file_path}")

    if not selected_files:
        print("No matching .npy files found. Please check input_dir and filename formats.")
        return

    num_models = len(selected_files)

    # --- 1. Layout Calculation ---
    # fig_width reserves margin on the left for text labels
    fig_width = 12
    model_block_height = 2
    gap_height = 0.05
    fig_height = (model_block_height + gap_height) * num_models

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Single column layout; increase 'left' value to ensure space for long labels
    outer_grid = gridspec.GridSpec(
        num_models, 1,
        hspace=gap_height / model_block_height,
        left=0.15, right=0.98, top=0.95, bottom=0.05
    )

    # --- 2. Iterate through each row for plotting ---
    for idx, (cls, file_name) in enumerate(selected_files):
        try:
            file_path = os.path.join(input_dir, file_name)
            data = np.load(file_path, allow_pickle=True).item()

            # Data processing: Arrange samples into a 2x10 grid
            # trajectories shape is usually [steps, Dim, Batch].
            # Take the last timestep and transpose.
            images = data['trajectories'][-1].T
            samples = images[0:20].reshape(-1, img_h, img_w)
            row1 = np.concatenate(samples[0:10], axis=1)
            row2 = np.concatenate(samples[10:20], axis=1)
            combined_img = np.concatenate([row1, row2], axis=0)

            # Create subplot
            ax_img = fig.add_subplot(outer_grid[idx, 0])
            ax_img.imshow(combined_img, cmap='gray', aspect='equal', interpolation='nearest')
            ax_img.axis('off')

            # --- Label Placement ---
            # y-coordinate: half of the image height for vertical centering
            y_center = combined_img.shape[0] / 2.0

            # x-coordinate fine-tuning to place it close to the left of the image
            x_offset = -12

            # Get the string label for the current class (e.g., "0-3")
            label_text = label_map[cls]

            ax_img.text(x_offset, y_center, label_text,
                        fontsize=36,
                        fontweight='bold',
                        color='black',    # Unified black color
                        ha='right',       # Right-align text to the x_offset anchor
                        va='center',      # Vertically center text to the y_center anchor
                        clip_on=False)    # Allow text to be drawn outside the subplot area

        except Exception as e:
            print(f"Error processing class {cls}: {e}")

    # --- 3. Save Results ---
    save_filename = "digits_generation_comparison.pdf"
    save_path = os.path.join(output_combined_dir, save_filename)

    # bbox_inches='tight' automatically crops excess white margins
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.close()
    print(f"\nVisualization complete! Image saved to: {save_path}")


if __name__ == "__main__":
    generate_compact_seamless_plot()