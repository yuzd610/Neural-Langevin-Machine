EBM Scaling and Generalization on Multi-Digit MNIST

Overview
This project investigates the impact of training dataset size on the generalization capabilities of Energy-Based Models (EBMs). By training models on exponentially growing subsets of MNIST digits (1, 3, and 6) using Persistent Contrastive Divergence (PCD), the project analyzes how data volume affects model fidelity, stability, and memorization. To ensure statistical robustness, multiple independent runs are executed for each dataset size, accompanied by deep structural analyses including Eigenvalue spectrums and 3D PCA manifold tracking.

Key Features
- Exponential Data Scaling: Trains models on dataset sizes growing exponentially (from 5 up to thousands of samples) to plot generalization curves.
- Statistical Robustness: Executes 5 independent training runs for every dataset size to compute means and standard deviations (error bands) for evaluation metrics.
- Spectral Analysis: Computes and visualizes the complex eigenvalue spectrum of the learned interaction matrix (J) to study system stability and dynamics.
- Manifold Tracking: Uses 3D PCA to project and visualize how the Langevin dynamics generation trajectory navigates the learned energy landscape towards data attractors.
- Comparative Evaluation: Strictly compares the generated data against the exact real data subset used during training to quantify memorization vs. generalization using the AAI metric.

Project Structure

[Scripts]
- 1train_PCD_multi.py : Slices the dataset into exponential sizes, trains 5 runs per size, and saves weights.
- 2generate_multi.py : Simulates Langevin dynamics for the trained models to generate image trajectories.
- 3AAI_multi_MSE.py : Evaluates generation fidelity (AAI, Acc_S, Acc_T) with error bands across dataset sizes.
- 4image_generate.py : Visualizes side-by-side grids of exact real training data vs. generated data.
- 5eigenvalue.py : Analyzes the structural stability of the J matrix by plotting its complex eigenvalues.
- 6PCA.py : Performs 3D PCA dimensionality reduction to visualize generation trajectories against the true data manifold.
- config.py : Centralized hyperparameters.

[Directories]
- data/ : MNIST dataset storage.
- checkpoints_by_size/ : Stores J and B_bias matrices labeled by dataset size and run index.
- generated_trajectories/ : Stores generated trajectories for each specific model.
- eigenvalue_plots_by_size/ : Stores PDF plots of complex eigenvalue spectrums.
- pca_plots_size_*/ : Stores PCA 3D trajectory plots and related image slices.
- combined_visualizations/ : Stores output image grids.

[Outputs & Checkpoints]
- aai_vs_datasize.pdf : Plot showing AAI error trends with standard deviation bands across dataset sizes.
- accuracy_vs_datasize.pdf : Plot showing Real (Acc_S) vs Generated (Acc_T) classification accuracies.
- real_vs_generated_comparison.pdf : Grid visualization directly comparing real inputs to generated outputs.
- eigen_spectrum_size_*_run_*.pdf : Scatter plots of Eigenvalues in the complex plane.
- pca_3d_trajectory.pdf : 3D visualization of the generated data moving toward the real data manifold.

Configuration
Key parameters defined in config.py:
- N: 784 (Flattened image dimension)
- g: 2 (Initialization scaling factor for J matrix)
- delta_t: 0.01 (Time step for Langevin dynamics)
- k: 10 (PCD Langevin steps per training update)
- T: 1 (Temperature/Noise scale)
- eta: 0.00005 (Learning rate for Adam optimizer)
- lambda1: 0.0000005 (L2 weight decay applied to J matrix)
- b_size: 1000 (Maximum batch size)

Usage

1. Train Models Across Dataset Sizes
Run: python 1train_PCD_multi.py
Splits MNIST (digits 1, 3, 6) into exponentially growing sizes. Trains 5 separate models per size for 20,000 steps each. (Note: This step trains up to 100 models and will take time!)

2. Generate Trajectories
Run: python 2generate_multi.py
Loads the trained models and simulates Langevin dynamics (50,000 steps), saving the final converged trajectories.

3. Evaluate Generalization Metrics
Run: python 3AAI_multi_MSE.py
Computes AAI, Acc_S, and Acc_T. Groups the 5 runs per dataset size to calculate means and standard deviations, outputting smooth log-scale plots with error bands.

4. Generate Visual Comparisons
Run: python 4image_generate.py
Creates side-by-side visual grids comparing the actual training subsets with the models' generated outputs.

5. Analyze Eigenvalue Spectrum
Run: python 5eigenvalue.py
Loads the saved J matrices and plots their eigenvalues in the complex plane to visually analyze the structural properties and stability of the learned energy landscapes.

6. Visualize 3D PCA Trajectories
Run: python 6PCA.py
Projects the real data and the continuous Langevin generation process into a 3D PCA space, showing how noise settles into the targeted digit attractors over time.


