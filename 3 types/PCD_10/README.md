# PCD-10: Core Persistent Contrastive Divergence

## Overview

This is the **core implementation** of Persistent Contrastive Divergence (PCD) trained on 10 MNIST digits. This project serves as the foundational experiment for understanding the generative learning framework and is recommended as the starting point for exploring the entire experimental suite.

PCD is a training algorithm for energy-based models that approximates the maximum likelihood gradient by simulating model dynamics. This implementation demonstrates how networks with local, asymmetric learning rules can learn complex distributions.

## Key Features

- **Core Algorithm**: Reference implementation of Persistent Contrastive Divergence
- **MNIST 10 Digits**: Trained on full diversity of digit types (0-9)
- **Scalable Architecture**: 784-dimensional input space
- **Comprehensive Analysis**: Includes eigenvalue and PCA analysis
- **Generative Learning**: Creates novel digit samples after training

## Project Structure

```
PCD_10/
├── 1train_PCD_multi.py            # Train the model
├── 2generate_multi.py             # Generate new samples
├── 3AAI_multi_MSE.py              # MSE analysis
├── 4image generate.py             # Visualize results
├── 5eigenvalue.py                 # Eigenvalue spectrum analysis
├── config.py                      # Hyperparameters
├── checkpoints/                   # Training checkpoints
├── combined_visualizations/       # Output visualizations
├── eigenvalue_plots/              # Eigenvalue analysis plots
├── generated_trajectories/        # Generation dynamics data
└── data/                          # Dataset storage
```

## Configuration

Key parameters in `config.py`:

- `N`: System dimensionality (784 for 28×28 images)
- `g`: Coupling strength of synaptic weights
- `delta_t`: Time step for discrete simulation (typically 0.01)
- `n`: Total training epochs (typically 12000)
- `k`: Learning rate for weight updates (typically 10)
- `T`: Effective temperature controlling noise (typically 1.0)
- `N_data`: Number of training samples (10000)
- `N_gen`: Number of generation samples (10000)
- `eta`: Learning rate decay parameter
- `lambda1`: L2 regularization strength
- `b_size`: Batch size (typically 1000)
- `N_model`: Number of parallel model simulations

## Usage

### 1. Train the Model

```bash
python 1train_PCD_multi.py
```

Trains the PCD model on MNIST 10-digit dataset. Saves weight matrix and training logs.

### 2. Generate New Samples

```bash
python 2generate_multi.py
```

Loads trained weights and generates novel digit images by simulating network dynamics from random initialization.

### 3. Compute MSE Analysis

```bash
python 3AAI_multi_MSE.py
```

Analyzes the statistical difference between generated and real samples using Mean Squared Error metrics.

### 4. Visualize Generated Images

```bash
python 4image generate.py
```

Creates comprehensive visualizations comparing generated samples with training data.

### 5. Analyze Eigenvalue Spectrum

```bash
python 5eigenvalue.py
```

Computes and visualizes the eigenvalue spectrum of the learned weight matrix, revealing information about the network's capacity and learned structure.

## Output

- **checkpoints/**: Weight matrices saved at different training stages
- **generated_trajectories/**: Time-series of network states during generation
- **combined_visualizations/**: Side-by-side comparisons of real vs. generated samples
- **eigenvalue_plots/**: Eigenvalue distribution and spectral analysis
- **data/**: MNIST dataset cache

