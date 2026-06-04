# 3types-One-Chain-PCD-2600: Extended Scale Experiment

## Overview

This project explores PCD learning at **significantly extended scale** (2600-dimensional chain-based configuration) across **multiple digit types**. By increasing both the system dimensionality and using chain-based or structured sampling patterns, this experiment investigates scalability of the core PCD algorithm and how it handles larger representational capacity.

The "3types" nomenclature suggests this variant explores patterns across different digit categories or learning regimes simultaneously.

## Key Features

- **Extended Dimensionality**: 2600-D system vs. standard 784-D
- **Chain-Based Sampling**: Potentially structured or sequential sampling
- **Multi-Type Learning**: Learning across multiple digit categories
- **Scalability Analysis**: Tests algorithm performance at scale
- **Larger Capacity**: More parameters for capturing complex distributions

## Project Structure

```
3types-one-chain-PCD-2600/
├── 1train_one_sample_multi.py     # Train with one-sample regime
├── 2generate_multi.py             # Generate samples
├── 3AAI_multi_MSE.py              # MSE analysis
├── 4image generate.py             # Visualize results
├── config.py                      # Hyperparameters
├── B_bias.npy                     # Learned biases
├── J_final.npy                    # Final weights
├── checkpoints/                   # Training checkpoints
├── combined_visualizations/       # Visualizations
├── generated_trajectories/        # Generation dynamics
└── data/                          # Dataset storage
```

## Configuration

Key parameters in `config.py`:

- `N`: System dimensionality (2600 - extended size)
- `g`: Coupling strength
- `delta_t`: Time step
- `n`: Training epochs
- `k`: Learning rate
- `T`: Temperature
- `N_data`: Training samples
- `lambda1`: L2 regularization
- `b_size`: Batch size

## Usage

### 1. Train Extended Scale Model

```bash
python 1train_one_sample_multi.py
```

Trains the 2600-D model, potentially from one representative sample per type to test generalization at scale.

### 2. Generate Samples

```bash
python 2generate_multi.py
```

Generates samples from the extended model.

### 3. Compute MSE Analysis

```bash
python 3AAI_multi_MSE.py
```

Analyzes quality metrics for scaled experiment.

### 4. Visualize Results

```bash
python 4image generate.py
```

Visualizes generated content (with dimensionality considerations).

## Output

- **checkpoints/**: Training checkpoints
- **generated_trajectories/**: Generation dynamics
- **combined_visualizations/**: Results and analysis
- **B_bias.npy**, **J_final.npy**: Learned large-scale parameters

## Workflow

1. **Training**: PCD at extended (2600-D) scale
2. **Generation**: Large-capacity network dynamics
3. **Analysis**: Understanding scalability
4. **Comparison**: Performance vs. standard 784-D

## Research Questions

- **Scalability**: How does PCD perform with 3x larger dimensionality?
- **Chain Structure**: Does structured sampling improve learning?
- **Multi-Type Learning**: Can network learn multiple categories simultaneously?
- **Convergence**: Does learning become more or less efficient at scale?
- **Capacity**: Does extra capacity improve or hurt generative performance?

## Performance Considerations

- **Training Time**: Significantly longer than 784-D (higher dimensional optimization)
- **Memory Usage**: ~10x larger weight matrix than standard
- **Convergence**: May require modified learning rates/regularization
- **Generation**: Slower simulation due to larger system

## Comparison

| Metric | Standard 784-D | 3types-2600-D |
|--------|----------------|---------------|
| Dimensionality | 784 | 2600 |
| Weight Matrix Size | 614,656 | ~6.76M parameters |
| Capacity | Low | High |
| Training Speed | Fast | Slower |
| Generation Quality | Known | Comparative |

---

**Research Focus**: This experiment is key for understanding scalability limits of local, asymmetric learning rules. Does performance improve with capacity or does it saturate/degrade?
