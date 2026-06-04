# PCD-10 Different Types: Cross-Domain Generalization

## Overview

This project applies the core PCD framework to **diverse data types and modalities beyond standard MNIST**. By training on different digit styles, non-digit image categories, or alternative data representations, this experiment investigates the generalizability of the local learning rule across different domains.

This serves as a critical test of whether the learned dynamics principles are truly general or specific to handwritten digits.

## Key Features

- **Multi-Domain Learning**: PCD applied beyond standard MNIST
- **Cross-Type Generalization**: Tests algorithm robustness
- **Alternative Data Modalities**: Explores different input types
- **Generalization Principles**: Validates core learning rule universality
- **Comparative Analysis**: Benchmark against standard MNIST performance
- **Domain Adaptation**: Understands how hyperparameters adapt across types

## Project Structure

```
PCD_10_Different types/
├── 1train_PCD_multi.py            # Train on different data types
├── 2generate_multi.py             # Generate across types
├── 3AAI_multi_MSE.py              # MSE analysis
├── 4image generate.py             # Visualize results
├── config.py                      # Hyperparameters
├── checkpoints/                   # Training checkpoints
├── combined_visualizations/       # Output visualizations
├── generated_trajectories/        # Generation dynamics
└── data/                          # Diverse dataset storage
```

## Configuration

Key parameters in `config.py`:

- `N`: System dimensionality (varies by data type)
- `g`: Coupling strength
- `delta_t`: Time step
- `n`: Training epochs
- `k`: Learning rate
- `T`: Temperature
- `N_data`: Training samples (varies by domain)
- `lambda1`: L2 regularization
- `b_size`: Batch size

## Usage

### 1. Train on Different Data Types

```bash
python 1train_PCD_multi.py
```

Trains PCD on non-standard MNIST data. May involve:
- Different digit fonts or writing styles
- Non-digit image categories (Fashion MNIST, SVHN, etc.)
- Synthetic or augmented data
- Mixed domain data

### 2. Generate from Different Domains

```bash
python 2generate_multi.py
```

Generates samples for each data type to assess domain-specific performance.

### 3. Compute MSE Analysis

```bash
python 3AAI_multi_MSE.py
```

Analyzes generation quality across different data types.

### 4. Visualize Cross-Domain Results

```bash
python 4image generate.py
```

Creates side-by-side visualizations comparing:
- Real samples from different types
- Generated samples per type
- Quality assessment across domains

## Output

- **checkpoints/**: Per-type training checkpoints
- **generated_trajectories/**: Generation dynamics for each type
- **combined_visualizations/**: Cross-type comparisons
- **data/**: Multi-type dataset storage

## Workflow

1. **Multi-Type Loading**: Load diverse data types
2. **Per-Type Training**: Train PCD on each data type
3. **Generation**: Generate samples per domain
4. **Cross-Domain Analysis**: Compare performance across types

## Research Questions

- **Algorithm Generality**: Does PCD work equally well across different data types?
- **Hyperparameter Transfer**: Do optimal hyperparameters transfer across domains?
- **Learning Curves**: How do convergence rates differ per type?
- **Generation Quality**: Which domains are easier/harder to learn?
- **Domain Adaptation**: What minimal adjustments enable cross-domain success?

## Expected Results by Domain

For different data types, you might observe:

| Data Type | Learning Difficulty | Generation Quality | Training Time |
|-----------|--------------------|--------------------|---------------|
| Standard MNIST | Baseline | Baseline | Baseline |
| Different Fonts | Easy | High | Fast |
| Fashion Items | Medium | Medium | Moderate |
| Synthetic Data | Easy/Hard | Varies | Variable |

## Comparative Analysis

Metrics to compare across types:
- **MSE**: Mean squared error between real and generated
- **Convergence**: Epochs to stable generation
- **Diversity**: Sample-to-sample variation
- **Recognizability**: Visual quality of generated samples

## Applications

This experiment validates:
- **Universality**: Do local learning rules work for diverse problems?
- **Robustness**: How sensitive is PCD to data type changes?
- **Scalability**: Can the approach scale to complex domains?
- **Practical utility**: How broadly applicable is the method?

## Domain Recommendations

Suggested data types to test:
1. **Different MNIST variants**: Rotated, distorted, noisy digits
2. **Fashion items**: Shoes, shirts, bags (Fashion MNIST)
3. **Other categories**: Letters, symbols, shapes
4. **Real-world data**: Street view house numbers (SVHN)
5. **Synthetic data**: Computer-generated patterns

---

**Key Insight**: If PCD successfully learns across diverse domains with minimal hyperparameter adjustment, it validates the generality of local, asymmetric learning rules as a fundamental principle for generative learning.
