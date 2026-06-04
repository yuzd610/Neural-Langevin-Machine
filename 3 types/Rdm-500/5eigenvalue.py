import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
def plot_eigenvalue_spectra(input_dir='checkpoints', output_dir='eigenvalue_plots'):
    """
    Load weight matrices, compute their eigenvalues, and plot them in the complex plane.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"Directory '{input_dir}' not found. Please check the path.")
        return

    # Filter for J matrix weight files (.npy)
    files = os.listdir(input_dir)
    j_files = [f for f in files if f.startswith('J_idx_') and f.endswith('.npy')]

    if not j_files:
        print(f"No J matrix weight files found in '{input_dir}'.")
        return

    # Helper function to extract training step for sorting
    def extract_step(filename):
        match = re.search(r'step_(\d+)', filename)
        return int(match.group(1)) if match else -1

    # Sort files chronologically by training step
    j_files.sort(key=extract_step)

    print(f"Found {len(j_files)} J-matrix files. Starting eigenvalue calculation and plotting...")

    for filename in tqdm(j_files, desc="Processing"):
        filepath = os.path.join(input_dir, filename)
        
        # Extract metadata (idx and step) for naming the plots
        match = re.search(r'idx_(\d+)_step_(\d+)', filename)
        idx = match.group(1) if match else "unknown"
        step = match.group(2) if match else "unknown"

        # 1. Load the weight matrix J
        J = np.load(filepath)

        # 2. Compute eigenvalues 
        # (J is typically non-symmetric, so eigenvalues are distributed in the complex plane)
        eigenvalues = np.linalg.eigvals(J)

        # 3. Extract real and imaginary components
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)

        # 4. Visualization
        plt.figure(figsize=(8, 8))
        
        # Plot eigenvalues as scatter points; alpha helps visualize density distribution
        plt.scatter(real_parts, imag_parts, s=45, alpha=0.6, color='blue', edgecolors='none')

        # Add reference lines for the axes
        plt.axhline(0, color='black', linewidth=3, linestyle='--')
        plt.axvline(0, color='black', linewidth=3, linestyle='--')
        
        # Styling the plot
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.xlabel('Re', fontsize=32)
        plt.ylabel('Im', fontsize=32)
        plt.title(f'$t_{{age}}$: {step}', fontsize=32)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))  
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
        
        

        # Crucial: Set 1:1 axis aspect ratio to accurately represent the 
        # geometric shape of the Elliptic Law and prevent distortion.
        plt.axis('equal') 

        # 5. Save the plot
        save_filename = f'eigen_spectrum_idx_{idx}_step_{step}.pdf'
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() # Close plot to release memory

    print(f"\nProcessing complete! Plots are saved in the '{output_dir}' folder.")

if __name__ == '__main__':
    plot_eigenvalue_spectra()