# Optimizer Module

## Overview

This module, `optimizer.rs`, implements parameter update algorithms such as Stochastic Gradient Descent (SGD) and Adam. Optimizers are critical components in training machine learning models, as they determine how the model's parameters are adjusted to minimize the loss function.

## Purpose

The optimizer updates model parameters based on the computed gradients from the loss function. This process ensures the model improves its predictions over time by reducing the loss. Both SGD and Adam are widely used optimization algorithms, offering distinct trade-offs in convergence speed and stability.

## Algorithms

### Stochastic Gradient Descent (SGD)

SGD updates parameters by applying the gradients scaled by a learning rate:

Where:

- : Model parameter (e.g., weight or bias).
- : Learning rate, a scalar determining the step size.
- : Gradient of the loss function with respect to .

### Adam

Adam combines the advantages of SGD with momentum and adaptive learning rates, making it more robust for complex optimization landscapes. The update equations are as follows:

1. Compute biased moment estimates:

2. Apply bias corrections:

3. Update parameters:

Where:

- : Exponential moving average of gradients (first moment).
- : Exponential moving average of squared gradients (second moment).
- : Hyperparameters controlling the decay rates for the moments.
- : A small constant to prevent division by zero.
- : Timestep.

## Key Properties

**SGD**:

- **Simplicity**: Straightforward implementation and fast computation.
- **Limitations**: Fixed learning rate may lead to slow convergence or overshooting.

**Adam**:

- **Adaptive Learning Rates**: Adjusts learning rates based on gradient magnitudes.
- **Momentum**: Accelerates convergence by smoothing updates.
- **Bias Correction**: Ensures unbiased moment estimates during early iterations.

## Implementation Details

### API

- **`OptimizerType`**: Enum to select between SGD and Adam.
- **`Optimizer`**: Struct encapsulating the optimizer's state and hyperparameters.
  - Fields:
    - `learning_rate`: Step size for updates.
    - `beta1`, `beta2`: Momentum decay factors (Adam-specific).
    - `epsilon`: Small constant for numerical stability (Adam-specific).
    - `moment1`, `moment2`: Optional moment estimates (Adam-specific).
    - `timestep`: Tracks iterations (Adam-specific).

### Workflow

1. Initialize the optimizer with:

   ```rust
   let optimizer = Optimizer::new(OptimizerType::Adam);
   ```

2. Perform an optimization step:

   ```rust
   optimizer.step(&mut params.view_mut(), &grads.view());
   ```

3. Update is applied element-wise to match parameter and gradient shapes.

## Mathematical Foundation

### SGD:

- Updates are proportional to the gradient's magnitude, scaled by the learning rate.
- Suitable for tasks with well-scaled gradients and simple optimization surfaces.

### Adam:

- Introduces two adaptive mechanisms:
  - Momentum: Smooths gradient updates over time.
  - Adaptive scaling: Adjusts learning rates based on the gradient's variance.
- Effective for sparse gradients and complex loss landscapes.
