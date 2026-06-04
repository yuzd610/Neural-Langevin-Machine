# From Memory to Generalization: Extended Training Study

## Overview

This project investigates the **transition from memorization to true generalization** through **extended training (20,000+ epochs)**. By training for dramatically longer periods than standard approaches, this experiment explores:

- How networks distinguish between memorizing training samples and learning underlying structure
- Whether extended training improves or degrades generative quality  
- The dynamics of learning curves over very long timescales
- Generalization performance as a function of training stage

This is a deep investigation into the learning dynamics of generative models using local, asymmetric learning rules.

## Key Features

- **Extended Training**: 20,000+ epochs (vs. typical 1,000-12,000)
- **Generalization Analysis**: Systematic study of memorization vs. learning
- **Multiple Checkpoints**: Captures learning at different stages
- **Long-Term Dynamics**: Understands convergence behavior over long training
- **Memory-Generalization Tradeoff**: Empirically studies theoretical concepts
- **PCA Analysis**: Principal component projection reveals learned structure

## Project Structure

```
From memory to generalization_tage=20000/
├── 1train_PCD_multi.py            # Extended training
├── 2generate_multi.py             # Generate from checkpoints
├── 3AAI_multi_MSE.py              # MSE analysis
├── 4image_generate.py             # Visualize results
├── 5eigenvalue.py                 # Eigenvalue spectrum
├── 6PCA.py                        # PCA analysis at different stages
├── config.py                      # Hyperparameters
├── checkpoints_by_size/           # Checkpoints at training stages
├── pca_plots_size_288/            # PCA visualizations
├── combined_visualizations/       # Output visualizations
├── generated_trajectories/        # Generation dynamics
└── data/                          # Dataset storage
```

## Configuration

Key parameters in `config.py`:

- `N`: System dimensionality (784)
- `g`: Coupling strength
- `delta_t`: Time step
- `n`: Training epochs (20000+)
- `k`: Learning rate
- `T`: Temperature
- `N_data`: Training samples (10000)
- `lambda1`: L2 regularization (important for controlling overfitting)
- `b_size`: Batch size

## Usage

### 1. Extended Training

```bash
python 1train_PCD_multi.py
```

Trains for 20,000+ epochs, saving checkpoints at regular intervals to track learning progression.

### 2. Generate from Checkpoints

```bash
python 2generate_multi.py
```

Generates samples at different training stages to visualize how generation quality evolves.

### 3. Compute MSE Analysis

```bash
python 3AAI_multi_MSE.py
```

Analyzes MSE metrics across different training stages, showing memorization vs. generalization.

### 4. Visualize Generated Images

```bash
python 4image_generate.py
```

Creates visualizations showing:
- How generated images evolve during training
- Transition from memorization to generalization
- Quality changes over 20,000 epochs

### 5. Analyze Eigenvalue Spectrum

```bash
python 5eigenvalue.py
```

Computes eigenvalues at different training stages to understand spectral evolution.

### 6. PCA Projection Analysis

```bash
python 6PCA.py
```

Projects generative trajectories onto PCA basis of training data, revealing:
- How learned manifold develops over training
- Convergence to data manifold
- Dimensionality of learned structure

## Output

- **checkpoints_by_size/**: Weights at different training epochs (early, mid, late)
- **pca_plots_size_288/**: PCA projections at different stages
- **combined_visualizations/**: Generated samples across training stages
- **generated_trajectories/**: Full generation dynamics
- **Eigenvalue evolution**: How spectrum changes during training

## Workflow

1. **Early Training** (epochs 1-3000): Rapid learning, possible memorization
2. **Mid Training** (epochs 3000-10000): Transition period
3. **Late Training** (epochs 10000-20000): Convergence and generalization
4. **Analysis**: Compare samples, eigenvalues, and PCA projections across stages

## Key Research Questions

- **When does memorization give way to generalization?** At what epoch does quality stabilize?
- **Does performance improve monotonically?** Can extended training hurt generalization?
- **Eigenvalue evolution**: How do spectral properties change?
- **Data manifold convergence**: Does network better explore data manifold over time?
- **Optimal stopping point**: Is there an epoch where generalization peaks before overfitting?

## Expected Results

- **Early epochs**: Generated images recognizable but low quality, high variation
- **Mid epochs**: Quality improves, some structure learned
- **Late epochs**: High-quality generation, stable patterns, true generalization
- **Very late epochs**: Possible quality saturation or slight degradation

## Comparison: Learning Curves

The project captures evolution across training:

| Training Stage | Samples | Quality | Diversity | Generalization |
|----------------|---------|---------|-----------|-----------------|
| Early (1K) | Memorized patterns | Low | High (noise) | Poor |
| Mid (10K) | Mixed | Medium | Medium | Improving |
| Late (20K) | Learned patterns | High | Moderate | Excellent |

## Applications

This experiment is crucial for:
- **Understanding convergence**: How to monitor generalization during training
- **Stopping criteria**: When should training stop?
- **Regularization tuning**: How does regularization affect memory-generalization tradeoff?
- **Theoretical insights**: Validating theories about local learning and generalization

---

**Key Insight**: This extended study provides empirical evidence for how local, asymmetric learning rules balance memorization and generalization over realistic training timescales.
