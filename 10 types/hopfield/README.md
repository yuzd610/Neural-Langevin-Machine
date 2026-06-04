# Hopfield Network - Content-Addressable Memory

## Overview

This project implements a classical **Hopfield network** for content-addressable memory and generative dynamics. The Hopfield network is an early neural network model capable of storing and retrieving patterns through energy-based learning, making it a foundational architecture for understanding modern generative models.

The implementation demonstrates how local, asymmetric learning rules can enable a network to learn complex patterns and generate novel variants by simulating the network's natural dynamics.

## Key Features

- **Classical Architecture**: Pure Hopfield network implementation without modern extensions
- **Content-Addressable Memory**: Store and retrieve patterns with partial or noisy cues
- **Energy-Based Learning**: Convergence to learned patterns through energy minimization


## Project Structure

```
hopfield/
├── 1train_Hopfield.py              # Train the Hopfield network
├── 2generate.py                    # Generate new samples
├── 3origin.py                      # Utility for origin/reference patterns
├── hopfield_energy_weights.pth     # Saved network weights
├── config.py                       # Hyperparameters configuration
└── data/                           # Training data directory
```

## Configuration

Edit `config.py` to customize the network:

- `N`: Network size (dimensionality)
- `g`: Coupling strength of synaptic weights
- `delta_t`: Time step for dynamics simulation
- `n`: Number of training epochs
- `k`: Learning rate for weight updates
- `T`: Temperature (noise level)
- `N_data`: Number of training patterns
- `lambda1`: L2 regularization coefficient

## Usage

### 1. Training

```bash
python 1train_Hopfield.py
```

Trains the Hopfield network on MNIST patterns and saves weights to `hopfield_energy_weights.pth`.

### 2. Generation

```bash
python 2generate.py
```

Loads trained weights and generates new samples by simulating network dynamics from random initialization.

### 3. Origin Reference

```bash
python 3origin.py
```

Shows original training patterns for comparison with generated samples.

## Output

- **checkpoints/**: Training checkpoints
- **data/**: Dataset storage
- **hopfield_energy_weights.pth**: Final trained network weights

## How It Works

1. **Learning**: Patterns are stored in the weight matrix using Hebbian-like rules
2. **Retrieval**: Given a partial or noisy pattern, the network converges to the nearest stored pattern
3. **Generation**: Starting from noise, the network dynamics naturally evolve toward learned patterns


