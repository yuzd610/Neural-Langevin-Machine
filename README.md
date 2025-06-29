# Neural Langevin Machine: A Local Asymmetric Learning Rule Can Be Creative

## Overview

This repository provides the code for the framework presented in the paper "Neural Langevin Machine: a local asymmetric learning rule can be creative". This work introduces a novel generative model capable of learning complex data distributions and creating new, high-quality samples.

- The core of our work is the **Neural Langevin Machine**, a recurrent network model that learns from data using a local and asymmetric synaptic update rule. This rule is derived from the principle of matching the statistics of a free-running model to the statistics of the data, akin to a generative learning process based on Langevin dynamics.
- We demonstrate that this learning rule enables the network to capture the underlying structure of a complex dataset, such as MNIST handwritten digits. After training, the network can autonomously generate novel samples that are stylistically consistent with the training data, starting from a state of pure random noise. This showcases the "creative" potential of the learning mechanism.
- The repository includes scripts to train the network on a subset of MNIST digits, generate new images, and analyze the learned dynamics. The analysis includes projecting the generative trajectories onto the principal component space of the real data, visualizing how the system navigates the learned data manifold.

## Requirements

- Python 3.11
- PyTorch 2.3.0 (with CUDA support recommended)
- NumPy 1.24.3
- Matplotlib 3.8.4
- Tqdm 4.66.2
- Scikit-learn (for PCA analysis)

## Installation

You can create a conda environment and install the required dependencies with the following commands:

```bash
conda create -n nlm python=3.11
conda activate nlm
conda install pytorch==2.3.0 torchvision cudatoolkit=11.8 -c pytorch
conda install numpy matplotlib tqdm scikit-learn
```

## Usage

The experiments are organized into scripts that should be run in numerical order. The primary parameters for all experiments are specified in `config.py`.

- **`config.py`**: Specifies all hyperparameters for training and generation.
- **`1training.py`**: Loads the MNIST dataset, trains the Neural Langevin Machine by updating the weight matrix `J`, and saves the final weights to `J_cupy1.npy`.
- **`2generate.py`**: Loads the trained weight matrix `J` and simulates the network's dynamics from a random initial state to generate new images. It visualizes the evolution of the generated samples over time.
- **`3pca.py`**: Performs Principal Component Analysis (PCA) on the real training data to define a low-dimensional latent space. It then projects the trajectory of a generated sample into this space to visualize how the system converges to the learned data manifold.
- **`4trajectory.py`**: Analyzes and plots the activity of a single neuron over the course of the generative process, providing insight into the system's fine-grained temporal dynamics.

#### **The settings file (`config.py`) contains the following key arguments:**

-   `N`: The dimension of the system (e.g., 784 for 28x28 MNIST images).
-   `g`: The coupling strength of the synaptic weights.
-   `delta_t`: The time step for the discrete-time simulation.
-   `n`: The total number of training epochs.
-   `n_t`: The number of inner loop time steps for simulating the model dynamics during training.
-   `T`: The effective temperature, controlling the level of noise in the dynamics.
-   `N_image`: The number of images used from the dataset for training.
-   `N_image1`: The number of images used from the dataset for pca base vector.
-   `k`: The learning rate for the weight matrix updates.
-   `lambda1`: The L2 regularization (weight decay) parameter.
-   `b_size`: The batch size for training data samples.
-   `N_model`: The number of model states simulated in parallel during the training phase.



## Contact

If you have any questions or issues with the code, please feel free to open an issue on this repository or contact us at `yuzd610@163.com`.
