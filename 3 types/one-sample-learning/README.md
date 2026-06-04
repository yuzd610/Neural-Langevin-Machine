Scheduled Dynamics Energy-Based Model (PCD) on MNIST

Overview
This project implements a continuous Energy-Based Model (EBM) trained on a subset of MNIST digits (1, 3, and 6) using one chain Persistent Contrastive Divergence (PCD) and one samples. The defining and unique characteristic of this specific implementation is the use of a Scheduled Langevin Dynamics mechanism during the training phase. Specifically, the complex non-linear reaction/coupling term in the dynamics equation is artificially suppressed for the first 150 steps and only activated for the remaining 450 steps. This two-stage "warm-up" allows the persistent chain to undergo initial thermalization before introducing full asymmetric interactions.

Key Features
- Scheduled Langevin Dynamics: Employs a time-dependent binary mask (transform_steps) that turns off the non-linear driving force (phi_prime * delta_J) for the first 150 steps of the MCMC chain, activating it only for the final 450 steps.
- Persistent Contrastive Divergence (PCD): Retains the final state of the Markov chain across training updates to serve as the starting point for the next batch, drastically reducing the required mixing time.
- Continuous Unbounded State Space: Maps discrete, bounded pixel values [-1, 1] to an unbounded continuous space using an atanh transformation for smooth gradient flow.
- Rigorous Evaluation: Generates samples over logarithmic timesteps and tracks convergence using statistical metrics like Adversarial Accuracy Improvement (AAI) and Covariance MSE, avoiding reliance on purely visual inspections.

Project Structure

[Scripts]
- 1train_one_sample_multi.py : Trains the EBM via PCD with the unique 150/450 scheduled dynamics split, saving intermediate checkpoints.
- 2generate_multi.py : Simulates full, unmasked Langevin dynamics starting from pure noise to generate sample trajectories.
- 3AAI_multi_MSE.py : Evaluates generation fidelity (AAI, Acc_S, Acc_T, and Covariance MSE) across generation timesteps.
- 4image generate.py : Compiles a seamless, compact visual grid tracking the formation of digits across specific parameter updates.
- config.py : Centralized hyperparameters.

[Directories]
- checkpoints/ : Directory storing saved J and B_bias matrices over parameter updates.
- generated_trajectories/ : Directory storing the generated trajectories (.npy).
- combined_visualizations/ : Directory storing output image grids.
- data/ : MNIST dataset storage (downloaded automatically).

[Outputs]
- aai_evolution.pdf : Plot of AAI error over generation steps.
- steady_state.pdf : Plot of steady-state metrics vs training updates.
- mse_cov_evolution.pdf : Broken-axis plot of Covariance MSE evolution.
- compact_comparison_fixed_gap.pdf : High-quality grid visualization of generated images.

Configuration
Key hyperparameters defined in config.py:
- N: 784 (Flattened image dimension)
- g: 2 (Initialization scaling factor for J matrix)
- delta_t: 0.01 (Time step for Langevin dynamics)
- n: 120 (Total training epochs)
- k: 600 (Langevin steps per PCD update; split into 150 masked + 450 unmasked steps)
- T: 1 (Temperature/Noise scale)
- N_data: 1000 (Number of training samples used)
- eta: 0.0005 (Learning rate for Adam optimizer)
- lambda1: 0.000005 (L2 weight decay applied to J matrix)

Usage

1. Train the EBM with Scheduled Dynamics
Run: python 1train_one_sample_multi.py
Loads 1000 samples of digits 1, 3, and 6. Runs PCD where the internal Langevin dynamics use the `transform_steps` array to schedule the interaction term. Saves model matrices logarithmically.

2. Generate Trajectories
Run: python 2generate_multi.py
Loads the saved model checkpoints and simulates standard Langevin dynamics (without the training mask) starting from random noise. Saves the states at 22 specific logarithmic timesteps.

3. Evaluate Metrics
Run: python 3AAI_multi_MSE.py
Computes the AAI metric (a nearest-neighbor 2-sample test) and Second Moment (Covariance) MSE comparing the generated samples against the true MNIST subset. Generates evaluation plots.

4. Visualize Generated Images
Run: python 4image generate.py
Parses the generated trajectory files and creates a visually appealing grid showing the generated digits at specific training checkpoints (e.g., 201, 988, 4852, 23836, 117107).


