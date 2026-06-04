# AE-PCD-10: Autoencoder-Based Latent Space Learning

## Overview

This project extends the PCD framework by combining **Autoencoders with PCD** trained on 10 MNIST digits. By learning in a compressed latent space, this approach enables faster training while exploring how dimensionality reduction interacts with generative learning.

The autoencoder learns a compressed representation of MNIST images, then PCD-based learning happens in this lower-dimensional latent space, combining representation learning with generative dynamics.

## Key Features

- **Latent Space Representation**: VAE compresses 784D images to ~30D latent vectors
- **PCD in Compressed Space**: Generative learning in efficient latent representation
- **Faster Convergence**: Reduced dimensionality enables quicker training
- **Quality Reconstruction**: Maintains visual quality through VAE decoder
- **Analysis-Ready**: Latent space is amenable to visualization and interpretation

## Project Structure

```
AE-PCD_10/
├── 1AE.py                         # Train Variational Autoencoder
├── 2train_AE_PCD.py               # Train PCD in latent space
├── 3generate_AE_PCD.py            # Generate samples from latent space
├── 4AAI_multi_MSE.py              # Multi-MSE analysis
├── 5image generate.py             # Visualize results
├── config.py                      # Hyperparameters
├── vae_mnist_30dim.pth            # Pre-trained VAE
├── compressed_matrix.npy          # Latent space data matrix
├── checkpoints_latent/            # Training checkpoints
├── combined_visualizations/       # Output visualizations
├── generated_trajectories_latent/ # Generation trajectory data
└── data/                          # Dataset storage
```

## Configuration

Key parameters in `config.py`:

- `N`: Original dimensionality (784 for 28×28 images)
- `N_VAE`: Latent space dimensionality (typically 20-30)
- `g`: Coupling strength in latent space
- `delta_t`: Time step for dynamics
- `n`: Training epochs for PCD phase
- `k`: Learning rate
- `T`: Temperature (noise)
- `lambda1`: L2 regularization
- `N_data`: Number of training samples (10000)
- `b_size`: Batch size

## Usage

### 1. Train Autoencoder

```bash
python 1AE.py
```

Trains VAE on MNIST digits, learning compression to latent space. Saves `vae_mnist_30dim.pth` and creates `compressed_matrix.npy` with encoded training data.

### 2. Train PCD in Latent Space

```bash
python 2train_AE_PCD.py
```

Trains the generative model using PCD on latent representations. Much faster than training on full image space.

### 3. Generate Samples

```bash
python 3generate_AE_PCD.py
```

Generates novel digit samples by:
- Sampling from learned latent distribution
- Decoding through VAE to image space
- Visualizing generation process

### 4.AAI- Multi-MSE Analysis

```bash
python 4AAI_multi_MSE.py
```

Computes quality metrics for generated vs. real samples in both latent and image space.

### 5. Visualize Results

```bash
python 5image generate.py
```

Creates comprehensive visualizations of generated digits alongside training data.

## Output

- **checkpoints_latent/**: PCD training checkpoints in latent space
- **generated_trajectories_latent/**: Generation dynamics in latent space
- **combined_visualizations/**: Generated images and comparisons
- **vae_mnist_30dim.pth**: Trained VAE weights
- **compressed_matrix.npy**: Encoded training data

## Workflow

1. **Autoencoding**: Learn efficient 30D representation of 784D images
2. **Latent Learning**: Train generative model in compact space
3. **Generation**: Sample latent codes, decode to images
4. **Analysis**: Evaluate quality in both latent and image domains

---

**Tip**: Compare results with PCD-10 to understand the trade-off between speed and working in full image space vs. compressed latent representations.
