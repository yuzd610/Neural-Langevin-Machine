# One-Sample Learning: Learning from Minimal Data

## Overview

This project explores an extreme learning scenario: **training a generative model from a single representative sample per digit class**. Instead of using thousands of training samples, the network learns to generalize from just one example of each digit (0-9), then generates diverse variants.

This is a challenging regime that tests whether networks with local, asymmetric learning rules can truly learn underlying patterns rather than memorize, and how they can extrapolate from minimal supervision.

## Key Features

- **Extreme Data Scarcity**: Single sample per digit class (10 samples total)
- **Generalization Challenge**: Network must learn category structure from minimal input
- **One-Sample Adaptation**: High learning rate for rapid convergence
- **Creative Generation**: Network generates diverse digit variants from one example
- **Memory vs. Generalization**: Explores the boundary between memorization and learning

## Project Structure

```
one-sample-learning/
├── 1train_one_sample_multi.py     # Train from single samples
├── 2generate_multi.py             # Generate diverse samples
├── 3AAI_multi_MSE.py              # MSE analysis
├── 4image generate.py             # Visualize results
├── config.py                      # Hyperparameters
├── B_bias.npy                     # Learned bias terms
├── J_final.npy                    # Final weight matrix
├── checkpoints/                   # Training checkpoints
├── combined_visualizations/       # Output visualizations
├── generated_trajectories/        # Generation dynamics
└── data/                          # Dataset storage
```

## Configuration

Key parameters in `config.py` (optimized for single-sample regime):

- `N`: System dimensionality (784)
- `g`: Coupling strength
- `delta_t`: Time step
- `n`: Training epochs (120 - much fewer than multi-sample)
- `k`: Learning rate (600 - much higher than standard 10)
- `T`: Temperature for noise/stochasticity
- `N_data`: Number of training samples (1000 - but sparse/repeated)
- `eta`: Decay parameter (0.0005)
- `lambda1`: L2 regularization (0.000005)
- `b_size`: Batch size (1 - single samples)
- `N_model`: Model copies during training (1)

## Usage

### 1. Train from Single Samples

```bash
python 1train_one_sample_multi.py
```

Trains the network on one representative example per digit. The algorithm adapts weights to encode the category structure from these minimal samples.

### 2. Generate Diverse Samples

```bash
python 2generate_multi.py
```

Generates novel digit samples by evolving the network from noise. Despite learning from only 10 samples, the network generates diverse, creative variations.

### 3. Compute MSE Analysis

```bash
python 3AAI_multi_MSE.py
```

Analyzes how well generated samples match the learned distribution, despite training data scarcity.

### 4. Visualize Results

```bash
python 4image generate.py
```

Creates visualizations showing:
- Single training sample (input)
- Generated diverse variants
- Comparison with standard multi-sample learning

## Output

- **checkpoints/**: Training checkpoints during single-sample learning
- **generated_trajectories/**: Generation dynamics
- **combined_visualizations/**: Generated samples and analysis
- **B_bias.npy**: Learned bias parameters
- **J_final.npy**: Final weight matrix
- **data/**: Dataset storage

## Workflow

1. **Initialization**: Load one representative sample per digit
2. **Rapid Training**: High learning rate adapts weights quickly
3. **Generation**: Network evolves from noise toward learned patterns
4. **Analysis**: Compare diversity and quality despite data scarcity

## Research Questions

This project investigates:

- **How much structure can be learned from minimal data?** Can networks extract digit category patterns from single examples?
- **Does generalization occur?** Are generated samples creative variations or just memorized samples?
- **What role does temperature play?** How does noise enable diverse generation from limited training?
- **Local learning sufficiency**: Can local, asymmetric rules enable genuine learning from scarce data?

## Key Results

- Networks can learn recognizable patterns from single samples
- Generated samples show significant diversity beyond memorization
- High learning rates accelerate convergence in data-scarce regime
- Temperature/noise plays crucial role in creative variation

## Comparison with Multi-Sample Learning

| Metric | One-Sample | Multi-Sample (PCD-10) |
|--------|-----------|----------------------|
| Training Samples | 10 | 10,000 |
| Learning Rate (k) | 600 | 10 |
| Training Epochs | 120 | 12,000 |
| Batch Size | 1 | 1000 |
| Generation Diversity | High (noise-dependent) | Learned from variety |
| Memorization Risk | High | Low |

---

**Research Application**: This regime tests the robustness of local learning rules in extreme data scarcity conditions, relevant for few-shot learning and transfer learning scenarios.
