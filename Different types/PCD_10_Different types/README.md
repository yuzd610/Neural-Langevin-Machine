EBM Capacity and Scaling Across MNIST Digit Classes

Overview
This project investigates the learning capacity and generalizability of continuous Energy-Based Models (EBMs) as the complexity of the data distribution increases. By training separate models on progressively larger subsets of MNIST digit classes (e.g., from just digits "0-1" up to the full "0-9" dataset), the project quantifies how data diversity impacts the model's ability to learn a stable energy landscape using Persistent Contrastive Divergence (PCD).

Key Features
- Progressive Complexity Scaling: Automatically subsets the MNIST dataset to train independent models on 2 classes, 3 classes, and so on, up to 10 classes.
- Continuous PCD Training: Maps discrete image pixels to an unbounded continuous space via atanh and trains models using Persistent Contrastive Divergence with k=10 Langevin steps.
- Capacity Evaluation Metrics: Tracks the degradation or stability of generation fidelity using Adversarial Accuracy Improvement (AAI) and Covariance MSE as the number of target digit classes increases.
- Side-by-Side Visualizations: Generates a seamless visual grid mapping the specific digit class subsets (e.g., "0-1", "0-3", "0-9") directly to their generated image counterparts.

Project Structure

[Scripts]
- 1train_PCD_multi.py : Iteratively filters MNIST by class count, trains separate EBMs, and saves weights.
- 2generate_multi.py : Loads the trained models and simulates Langevin dynamics to generate sample trajectories.
- 3AAI_multi_MSE.py : Evaluates generation fidelity (AAI and Covariance MSE) against the number of digit classes.
- 4image generate.py : Creates a visual grid showing generated digits corresponding to increasing class counts.
- config.py : Centralized hyperparameters.

[Directories]
- checkpoints/ : Directory storing saved J and B_bias matrices for each class count subset.
- generated_trajectories/ : Directory storing generated multi-step trajectories (.npy).
- combined_visualizations/ : Directory storing the final visual grid PDF.
- data/ : MNIST dataset storage (downloaded automatically).

[Outputs]
- aai_vs_classes.pdf : Line plot showing AAI error evolution as the dataset complexity (class count) increases.
- mse_vs_classes.pdf : Line plot showing Covariance MSE evolution against the number of digit classes.
- digits_generation_comparison.pdf : Clean visual grid comparing generated digits across subsets like "0-1", "0-3", "0-5", "0-7", and "0-9".

Configuration
Key hyperparameters defined in config.py:
- N: 784 (Flattened image dimension)
- g: 2 (Initialization scaling factor for J matrix)
- delta_t: 0.01 (Time step for Langevin dynamics)
- n: 500 (Total training epochs per model)
- k: 10 (Langevin steps per PCD update)
- T: 1 (Temperature/Noise scale)
- N_data: 10000 (Maximum number of training samples used per subset)
- eta: 0.00005 (Learning rate for Adam optimizer)
- lambda1: 0.0000005 (L2 weight decay applied to J matrix)

Usage

1. Train Models on Increasing Digit Classes
Run: python 1train_PCD_multi.py
Iterates through target class counts (2 through 10). Filters the dataset, trains an independent PCD model for 500 epochs on each subset, and saves the resulting J and B matrices to the checkpoints directory.

2. Generate Trajectories
Run: python 2generate_multi.py
Loads the checkpoint for each class-count model and runs Langevin dynamics from random noise, saving the trajectory states across 22 logarithmic timesteps.

3. Evaluate Capacity Metrics
Run: python 3AAI_multi_MSE.py
Reads the generated samples from the final timestep and strictly aligns them with the exact real training subsets. Computes and plots AAI and Covariance MSE to statistically demonstrate how model performance scales with data diversity.

4. Visualize Generated Digits
Run: python 4image generate.py
Parses the trajectory files specifically for models trained on 2, 4, 6, 8, and 10 digit classes. Generates a perfectly formatted, label-aligned PDF grid showing what the model produces at each complexity level.

Workflow
1. Subsetting Data: The script slices MNIST not just by sample count, but by the number of unique categorical modes (digits) the distribution contains.
2. Independent Training: A dedicated energy landscape is learned for each subset level using PCD.
3. Metric Validation: Evaluates if a fixed-size network (784x784 parameters) suffers from capacity bottlenecks when asked to memorize/generalize 10 digits versus just 2 digits.
4. Visual Confirmation: Outputs a grid that visually corroborates the statistical findings from the AAI and MSE plots.

Tip: Step 1 trains 9 separate models back-to-back for 500 epochs each. Ensure you are running this on a GPU, or adjust the "n" (epochs) in config.py for a quicker exploratory run!
