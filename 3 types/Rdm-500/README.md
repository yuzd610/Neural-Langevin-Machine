# Rdm-500: Random Model Variant

## Overview

This project explores a **variant PCD configuration with randomized initialization or augmented dimensionality**. By using a random model baseline or higher-dimensional system (500-D), this experiment investigates how network size and initialization strategies affect generative learning capacity.

This serves as an important control and variant study to understand the role of network architecture in the success of generative learning.

## Key Features

- **Extended Dimensionality**: 500-dimensional random feature space
- **Variant Initialization**: Explores alternative weight initialization schemes
- **Baseline Comparison**: Provides reference for understanding core 784-D experiments
- **Scalability Analysis**: Tests how algorithm performs at different scales
- **Control Experiment**: Isolates effects of network size from learning algorithm

## Project Structure

```
Rdm-500/
├── 1train_RDM_multi.py            # Train random/variant model
├── 2generate_multi.py             # Generate samples
├── 3AAI_multi_MSE.py              # MSE analysis
├── 4image generate.py             # Visualize results
├── 5eigenvalue.py                 # Eigenvalue analysis
├── config.py                      # Hyperparameters
├── B_bias.npy                     # Learned biases
├── J_final.npy                    # Final weights
├── checkpoints/                   # Training checkpoints
├── combined_visualizations/       # Visualizations
├── eigenvalue_plots/              # Spectral analysis
├── generated_trajectories/        # Generation dynamics
└── data/                          # Dataset storage
```

## Configuration

Key parameters in `config.py`:

- `N`: System dimensionality (500 - variant size)
- `g`: Coupling strength
- `delta_t`: Time step
- `n`: Training epochs
- `k`: Learning rate
- `T`: Temperature
- `N_data`: Training samples (10000)
- `N_gen`: Generation samples (10000)
- `lambda1`: L2 regularization
- `b_size`: Batch size (1000)

## Usage

### 1. Train Random/Variant Model

```bash
python 1train_RDM_multi.py
```

Trains the model with randomized or augmented architecture, learning from MNIST dataset.

### 2. Generate Samples

```bash
python 2generate_multi.py
```

Generates samples in the variant configuration space.

### 3. Compute MSE Analysis

```bash
python 3AAI_multi_MSE.py
```

Analyzes generation quality with variant architecture.

### 4. Visualize Results

```bash
python 4image generate.py
```

Creates visualizations (may require dimensionality projection to 28×28 for display).

### 5. Analyze Eigenvalue Spectrum

```bash
python 5eigenvalue.py
```

Studies the spectral properties of learned 500-D weight matrix.

## Output

- **checkpoints/**: Training checkpoints
- **generated_trajectories/**: Generation time-series
- **combined_visualizations/**: Generated samples and analysis
- **eigenvalue_plots/**: Spectral properties
- **B_bias.npy**, **J_final.npy**: Learned parameters

## Workflow

1. **Training**: PCD learning in 500-D random/variant space
2. **Generation**: Network dynamics in extended space
3. **Analysis**: Eigenvalue spectrum reveals structure
4. **Comparison**: Results vs. standard 784-D full-image learning

## Research Questions

- **Does 500-D augmentation improve learning?** Can additional random dimensions help?
- **Network size effects**: How does system size affect convergence and quality?
- **Spectral properties**: How do eigenvalues differ from full 784-D learning?
- **Generalization**: Does variant architecture hurt or help?

## Comparison

| Aspect | Standard PCD | Rdm-500 |
|--------|--------------|---------|
| Dimensionality | 784 | 500 |
| Nature | Full image space | Random/variant space |
| Role | Primary method | Baseline/variant |
| Eigenvalue spectrum | 784 eigenvalues | 500 eigenvalues |
| Generation quality | Baseline | Comparative |

## Applications

This variant is useful for:
- **Control experiments**: Isolating effects of dimensionality
- **Ablation studies**: Understanding importance of architecture choices
- **Scalability**: Testing algorithm performance at different scales
- **Random feature analysis**: Understanding role of feature space

---

**Tip**: Compare eigenvalue spectra and MSE metrics between this variant and standard PCD-10 to understand how network architecture affects learning dynamics.
