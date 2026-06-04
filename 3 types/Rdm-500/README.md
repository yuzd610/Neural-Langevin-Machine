Short-Run MCMC Energy-Based Model (Rdm-k) on MNIST

Overview
This project implements an Energy-Based Model (EBM) trained on MNIST digits (1, 3, and 6) using Short-Run MCMC (Contrastive Divergence with k steps). Unlike Persistent Contrastive Divergence (PCD), this approach resets the Markov chains to random noise at every training iteration and runs Langevin dynamics for a fixed k=500 steps. The project heavily focuses on analyzing the temporal evolution of the generation process, specifically investigating how the model behaves when generating samples at, before, and after the exact k-steps it was trained on.

Key Features
- Short-Run MCMC Training: Resets the Langevin dynamics chain to pure noise at every parameter update, strictly training the model to map noise to data in exactly k=500 steps.
- Generation Trajectory Analysis: Evaluates Adversarial Accuracy Improvement (AAI) and Covariance MSE over logarithmic generation steps, identifying the optimal stopping point for generation.
- Critical Step Highlighting: Visually and statistically marks Step 500 in the evaluation plots to demonstrate the correlation between the training constraint and generation fidelity.
- Eigenvalue Stability Analysis: Computes and plots the complex eigenvalue spectrum of the learned J-matrix to ensure the structural stability of the learned energy landscape over training time.
- Dynamics Visualization: Generates a visual timeline of a single model's sampling process, showing how random noise morphs into recognizable digits over specific Langevin steps.

Project Structure

[Scripts]
- 1train_RDM_multi.py : Trains the EBM using Short-Run MCMC (CD-500) and saves model checkpoints.
- 2generate_multi.py : Simulates Langevin dynamics from noise, recording states at logarithmic timesteps.
- 3AAI_multi_MSE.py : Evaluates AAI and Covariance MSE over generation steps, highlighting Step 500.
- 4image generate.py : Creates a visual grid showing the evolution of images during the Langevin sampling process.
- 5eigenvalue.py : Analyzes and plots the complex eigenvalue spectrum of the weight matrices.
- config.py : Centralized hyperparameters.

[Directories]
- checkpoints/ : Directory storing saved J and B_bias matrices over parameter updates.
- generated_trajectories/ : Directory storing the multi-step trajectories (.npy).
- eigenvalue_plots/ : Directory storing PDF plots of the complex eigenvalue spectrums.
- combined_visualizations/ : Directory storing output image grids.
- data/ : MNIST dataset storage.

[Outputs]
- aai_evolution.pdf : Plot of AAI error over generation steps (with Step 500 marked).
- steady_state.pdf : Plot of steady-state metrics vs training updates.
- mse_cov_evolution.pdf : Broken-axis plot of Covariance MSE evolution (with Step 500 marked).
- evolution_t_age_*.pdf : Visual grid tracking image evolution from noise to digits.
- eigen_spectrum_idx_*_step_*.pdf : Scatter plots of Eigenvalues in the complex plane over time.

Configuration
Key hyperparameters defined in config.py:
- N: 784 (Flattened image dimension)
- g: 2 (Initialization scaling factor for J matrix)
- delta_t: 0.01 (Time step for Langevin dynamics)
- n: 2400 (Total training epochs)
- k: 500 (Strict Langevin steps per training update from random noise)
- T: 1 (Temperature/Noise scale)
- eta: 0.00005 (Learning rate)
- lambda1: 0.0000005 (L2 weight decay applied to J matrix)

Usage

1. Train the EBM with Short-Run MCMC
Run: python 1train_RDM_multi.py
Filters the dataset for digits 1, 3, and 6. Trains the model by running 500 Langevin steps from pure noise at every update. Saves checkpoints logarithmically.

2. Generate Trajectories
Run: python 2generate_multi.py
Loads the saved models and runs an extended Langevin dynamics simulation (up to thousands of steps), recording the image states at specific logarithmic timesteps to capture the full evolution.

3. Evaluate Generation Dynamics
Run: python 3AAI_multi_MSE.py
Computes AAI and Covariance MSE across the generation timesteps. Generates plots with a distinct vertical line at Step 500 to evaluate if generation quality peaks exactly where the model was trained to stop.

4. Visualize Sampling Evolution
Run: python 4image generate.py
Focuses on a specific trained model (e.g., at update 23836) and visually plots the progression of digits forming from noise at sampling steps like 20, 99, 500, 2530, and 12808.

5. Analyze Spectral Stability
Run: python 5eigenvalue.py
Calculates and plots the complex eigenvalues of the interaction matrix J for each saved checkpoint, verifying the mathematical stability (e.g., adherence to the Elliptic Law) of the learned network.

Workflow
1. Training Constraint: Forces the network to learn a fast-mixing trajectory from noise to data within exactly 500 continuous steps.
2. Unbounded Simulation: Runs generation far beyond the 500-step training limit to observe if the distribution holds steady or degrades.
3. Metric Validation: Uses AAI and MSE curves to objectively identify the optimal generation time step.
4. Visual & Structural Proof: Correlates the statistical metrics with visual digit formation and deep spectral analysis of the weight matrix.

Tip: Execute the scripts sequentially (1 -> 5). Pay special attention to the output plots from Step 3 and 4, as they perfectly illustrate how Short-Run MCMC models are highly optimized for their specific training horizon (k=500)!
