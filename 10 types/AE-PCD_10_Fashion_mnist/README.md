# Autoencoder-PCD on Fashion MNIST

## Overview

This project combines **Variational Autoencoders (VAE)** with **Persistent Contrastive Divergence (PCD)** for learning generative models on **Fashion MNIST** dataset. By learning in a compressed latent space, this approach enables efficient training on image data while maintaining high-quality generation.

The key insight is to use a VAE for dimensionality reduction, then apply PCD-based learning in the resulting latent space, combining the benefits of representation learning with generative modeling.

## Key Features

- **Latent Space Learning**: VAE compression reduces 784-dimensional images to compact representation
- **PCD in Latent Space**: Generative learning happens in low-dimensional latent space
- **Fashion MNIST Domain**: Extended to clothing item category prediction
- **Efficient Training**: Faster convergence through dimensionality reduction
- **High-Quality Generation**: Reconstructed images maintain visual quality

## Project Structure

```
AE-PCD_10_Fashion_mnist/
├── 1AE.py                         # Train the Variational Autoencoder
├── 2train_AE_PCD.py               # Train PCD in latent space
├── 3generate_AE_PCD.py            # Generate samples
├── 4AAI_multi_MSE.py              # Multi-MSE analysis
├── 5image generate.py             # Visualize generated images
├── config.py                      # Hyperparameters
├── vae_mnist_30dim.pth            # Pre-trained VAE weights
├── compressed_matrix.npy          # Compressed data representation
├── checkpoints_latent/            # Training checkpoints in latent space
├── combined_visualizations/       # Output visualizations
├── generated_trajectories_latent/ # Generation trajectory data
└── data/                          # Dataset storage
```

## Configuration

Key parameters in `config.py`:

- `N`: Dimensionality of original images (784 for 28×28)
- `N_VAE`: Dimensionality of VAE latent space (e.g., 20-30 dimensions)
- `g`: Coupling strength in latent space
- `delta_t`: Time step for dynamics
- `n`: Training epochs
- `k`: Learning rate
- `T`: Temperature (noise)
- `lambda1`: L2 regularization
- `N_data`: Number of training samples
- `N_model`: Batch size

## Usage

### 1. Train Autoencoder

```bash
python 1AE.py
```

Trains VAE on Fashion MNIST and saves to `vae_mnist_30dim.pth`. Creates compressed data matrix.

### 2. Train PCD in Latent Space

```bash
python 2train_AE_PCD.py
```

Trains the generative model using PCD dynamics in the VAE's latent space.

### 3. Generate Samples

```bash
python 3generate_AE_PCD.py
```

Generates new Fashion MNIST images from random noise, starting in latent space then reconstructing to image space.

### 4. Multi-MSE Analysis

```bash
python 4AAI_multi_MSE.py
```

Computes Mean Squared Error metrics comparing generated vs. real samples.

### 5. Visualize Results

```bash
python 5image generate.py
```

Creates comprehensive visualizations of generated samples alongside real data.

## Output

- **checkpoints_latent/**: Training checkpoints
- **generated_trajectories_latent/**: Time-series of generation process
- **combined_visualizations/**: Generated images and analysis plots
- **vae_mnist_30dim.pth**: Trained VAE weights
- **compressed_matrix.npy**: Encoded training data

## Workflow

1. **Preprocessing**: VAE learns to compress Fashion MNIST images to latent vectors
2. **Latent Learning**: PCD learns the distribution of latent codes
3. **Generation**: Sample from learned latent distribution, then decode to image space
4. **Analysis**: Compare statistical properties and quality of generated items

## Key Advantages

- **Scalability**: Significantly faster than operating on 784-dimensional space
- **Interpretability**: Latent space can be analyzed and visualized
- **Quality**: Combines representation learning with generative modeling
- **Flexibility**: Can extend to other image datasets by retraining VAE

---

**Tip**: Start with step 1 (VAE training) to learn the compression, then proceed with PCD training for generative modeling in the latent space.
