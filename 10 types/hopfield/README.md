Continuous Hopfield Network on MNIST

Overview
This project implements a Continuous Hopfield Network (Energy-based Model) trained on the MNIST dataset. It demonstrates how to store specific digit patterns using gradient descent to minimize an energy function in an unbounded continuous space, and how to retrieve them from highly noisy states using dynamic continuous evolution (Euler method).

The key insight is mapping bounded image pixels into an unbounded continuous space using an atanh transformation, learning the weight matrix via gradient descent, and strictly tracking the kinetic energy descent during pattern retrieval.

Key Features
- Continuous State Space: Uses atanh transformation to map bounded pixel values [-1, 1] to unbounded continuous space.
- Energy-Based Learning: Optimizes weight matrix J (with zero-diagonal) and bias b via Gradient Descent (Adam optimizer).
- Robust Pattern Retrieval: Reconstructs digits from 25% random noise using continuous Euler dynamics.
- Energy Tracking: Visualizes the strict monotonic descent of the system's kinetic energy during the retrieval process.

Project Structure
Hopfield_Continuous_MNIST/
├── 1train_Hopfield.py             # Trains the Hopfield network on 10 distinct digits
├── 2generate.py                   # Adds noise and runs dynamic evolution to recover patterns
├── 3origin.py                     # Extracts and saves the clean original image for comparison
├── hopfield_energy_weights.pth    # Pre-trained model weights (Generated after step 1)
├── digit_step_*.pdf               # Snapshots of the image at different dynamic steps
├── energy_descent_main.pdf        # Visualization of the energy descent curve
├── original_digit_6.pdf           # Ground truth clean image
└── data/                          # Dataset storage (downloaded automatically)

Configuration
Key parameters used in the scripts:

- N: Dimensionality of original images (784 for 28x28 flattened MNIST)
- epsilon: 0.03 (clipping boundary before atanh transformation)
- epochs: 1000 (training iterations for the energy model)
- lr: 0.001 (Learning rate for Adam optimizer)
- noise_ratio: 0.25 (25% random noise added during retrieval)
- total_steps: 451 (Number of dynamics steps for the Euler method)
- dt: 0.01 (Time step size for Euler integration)

Usage

1. Train the Hopfield Network
Run: python 1train_Hopfield.py
Extracts 10 unique digits, transforms them into continuous space, trains the energy model to minimize the target energy function, and saves to hopfield_energy_weights.pth.

2. Generate and Retrieve Patterns
Run: python 2generate.py
Loads the trained weights, takes a stored digit, injects 25% noise, and simulates continuous dynamics evolution to restore the image. Saves intermediate images as PDFs and plots the energy descent curve.

3. Extract Original Ground Truth
Run: python 3origin.py
Extracts the clean, un-noised version of the targeted digit and saves it as a PDF for direct comparison with the reconstructed output.

Output
- hopfield_energy_weights.pth: Trained network parameters (J and b).
- digit_step_{0, 150, 300, 450}.pdf: Visualization of the noisy image recovering step-by-step.
- energy_descent_main.pdf: Plot showing the continuous decrease in kinetic energy over dynamics steps.
- original_digit_6.pdf: Clean baseline image.

Workflow
1. Preprocessing: Scale, clamp, and apply atanh transformation to map MNIST digits into unbounded continuous space.
2. Training: Learn matrix J and bias b using backpropagation to minimize the internal error/energy for the 10 stored patterns.
3. Dynamics Evolution: Inject random noise and simulate continuous system evolution using Euler integration.
4. Evaluation: Compare the final stabilized state with the original clean digit and analyze the kinetic energy descent trajectory.

Tip: Ensure you start with step 1 (1train_Hopfield.py) to generate the required weight files before attempting to run the generation or visualization scripts!
