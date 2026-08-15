import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator

def plot_eigenvalue_spectra_by_size(input_dir='checkpoints_by_size', output_dir='eigenvalue_plots_by_size'):
    """
    Load weight matrices (J), compute their eigenvalues, and plot them in the complex plane.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"Directory '{input_dir}' not found. Please check the path.")
        return

    # Filter for J matrix weight files (.npy)
    files = os.listdir(input_dir)
    j_files = [f for f in files if f.startswith('J_size_') and f.endswith('.npy')]

    if not j_files:
        print(f"No J matrix weight files found in '{input_dir}'.")
        return

    # Helper function: extract size and run index from filename for sorting
    def extract_info(filename):
        match = re.search(r'J_size_(\d+)_run_(\d+)', filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        return -1, -1

    # Sort files by dataset size and run_idx
    j_files.sort(key=extract_info)

    print(f"Found {len(j_files)} J-matrix files. Starting eigenvalue calculation and plotting...")

    for filename in tqdm(j_files, desc="Processing"):
        filepath = os.path.join(input_dir, filename)
        
        # Extract metadata (size and run) for figure naming and setting title
        match = re.search(r'J_size_(\d+)_run_(\d+)', filename)
        if not match:
            continue
            
        size = match.group(1)
        run = match.group(2)

        # 1. Load weight matrix J
        J = np.load(filepath)

        # 2. Compute eigenvalues
        eigenvalues = np.linalg.eigvals(J)

        # 3. Extract real and imaginary parts
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)

        # 4. Visualize and plot
        plt.figure(figsize=(8, 8))
        
        # Scatter plot of eigenvalues (using marker size and alpha to observe density distribution)
        plt.scatter(real_parts, imag_parts, s=45, alpha=0.6, color='blue', edgecolors='none')

        # Add axis reference lines
        plt.axhline(0, color='black', linewidth=3, linestyle='--')
        plt.axvline(0, color='black', linewidth=3, linestyle='--')
        
        # Plot style settings
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.xlabel('Re', fontsize=32)
        plt.ylabel('Im', fontsize=32)
        
        # Update title format to N_data=xxx
        # Note: Matplotlib MathText syntax is used here; double braces {{}} escape in Python f-strings
        plt.title(f'$N_{{data}}={size}$', fontsize=32)
        
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))  
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
        
        # Key setting: 1:1 aspect ratio to ensure geometric shape (e.g., elliptic law) is not distorted
        plt.axis('equal') 

        # 5. Save figure
        # Include size and run in filename to prevent overwriting
        save_filename = f'eigen_spectrum_size_{size}_run_{run}.pdf'
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory

    print(f"\nProcessing complete! Plots are saved in the '{output_dir}' folder.")

if __name__ == '__main__':
    plot_eigenvalue_spectra_by_size()
