import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm

# ==========================================
# 1. Configuration Parameters
# ==========================================
input_dir = 'generated_trajectories'
output_combined_dir = 'combined_visualizations'
os.makedirs(output_combined_dir, exist_ok=True)

# Target iteration steps (parameter updates)
target_steps =  [201, 988, 4852, 23836, 117107]



img_h, img_w = 28, 28


def generate_compact_seamless_plot():
    # Filter files based on target steps
    all_files = os.listdir(input_dir)
    selected_files = []
    for step in target_steps:
        # Search for files ending with specific step count
        match = [f for f in all_files if f.endswith(f'_step_{step}.npy')]
        if match:
            selected_files.append((step, match[0]))

    if not selected_files:
        print("No matching .npy files found. Please check input_dir and filename formats.")
        return

    num_models = len(selected_files)

    # --- 1. Generate color gradient ---
    colors = cm.coolwarm_r(np.linspace(0.0, 1.0, num_models))

    # --- 2. Layout calculation ---
    # Fig_width provides enough margin on the left for text labels
    fig_width = 12
    model_block_height = 2
    gap_height = 0.05
    fig_height = (model_block_height + gap_height) * num_models

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Single column layout; 'left' margin is set to ~0.12 to accommodate text
    outer_grid = gridspec.GridSpec(
        num_models, 1,
        hspace=gap_height / model_block_height,
        left=0.12, right=0.98, top=0.95, bottom=0.05
    )

    # --- 3. Iterate through each row to plot ---
    for idx, (step, file_name) in enumerate(selected_files):
        try:
            file_path = os.path.join(input_dir, file_name)
            data = np.load(file_path, allow_pickle=True).item()

            # Data Processing: Arrange samples into a 2x10 grid
            # trajectories shape is usually [steps, Dim, Batch], we take last step and transpose
            images = data['trajectories'][-1].T
            samples = images[80:100].reshape(-1, img_h, img_w)
            row1 = np.concatenate(samples[0:10], axis=1)
            row2 = np.concatenate(samples[10:20], axis=1)
            combined_img = np.concatenate([row1, row2], axis=0)

            # Create subplot
            ax_img = fig.add_subplot(outer_grid[idx, 0])
            ax_img.imshow(combined_img, cmap='gray', aspect='equal', interpolation='nearest')
            ax_img.axis('off')

            # --- Label Placement: Place text within the image coordinate system ---
            # combined_img.shape[0] is height (56), shape[1] is width (280)
            # y coordinate: half of the image height to center vertically
            y_center = combined_img.shape[0] / 2.0

            # [Adjustment]: Values closer to 0 move the text closer to the image
            x_offset = -8

            ax_img.text(x_offset, y_center, str(step),
                        fontsize=34,
                        fontweight='bold',
                        color=colors[idx],
                        ha='right',   # Align right side of text to the x_offset point
                        va='center',  # Align vertical center of text to the y_center point
                        clip_on=False) # Important: Allow text to be drawn outside the subplot area

        except Exception as e:
            print(f"Error processing step {step}: {e}")

    # --- 4. Save results ---
    save_filename = "compact_comparison_fixed_gap.pdf"
    save_path = os.path.join(output_combined_dir, save_filename)

    # Small pad_inches helps eliminate extra white margins in the PDF
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.close()
    print(f"\nVisualization complete! Margin between numbers and images reduced. Saved to: {save_path}")


if __name__ == "__main__":
    generate_compact_seamless_plot()