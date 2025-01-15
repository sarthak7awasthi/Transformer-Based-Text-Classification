# Layer Normalization Module

A Rust implementation of layer normalization for deep neural networks, particularly optimized for Transformer architectures.

## Overview

The `layer_norm.rs` module provides an efficient implementation of layer normalization, a critical technique for stabilizing deep neural network training. This implementation is especially useful in Transformer models where consistent input distributions across layers are essential for optimal performance.

## Features

- Instance-specific normalization independent of batch size
- Numerically stable computations with epsilon parameter
- Learnable scale (gamma) and shift (beta) parameters
- Optimized for Transformer architectures
- Comprehensive test coverage

## Mathematical Foundation

Layer normalization operates by normalizing feature vectors independently along the feature axis using the following steps:

1. Calculate mean (μ):

   ```
   μ = (1/d) * Σ(x_i)
   ```

2. Calculate variance (σ²):

   ```
   σ² = (1/d) * Σ(x_i - μ)²
   ```

3. Normalize values:

   ```
   x̂_i = (x_i - μ) / sqrt(σ² + ε)
   ```

4. Scale and shift:
   ```
   y_i = γ * x̂_i + β
   ```

Where:

- d: number of features
- ε: small constant for numerical stability (typically 1e-5)
- γ: learnable scaling parameter
- β: learnable shifting parameter

## Usage

```rust
use layer_norm::apply_layer_norm;

// Example input vector
let input = vec![1.0, 2.0, 3.0, 4.0];
let gamma = 1.0;
let beta = 0.0;
let epsilon = 1e-5;

// Apply layer normalization
let normalized = apply_layer_norm(&input, gamma, beta, epsilon);
```

## Key Functions

### `apply_layer_norm`

```rust
pub fn apply_layer_norm(
    input: &[f32],
    gamma: f32,
    beta: f32,
    epsilon: f32
) -> Vec<f32>
```

Normalizes an input vector using layer normalization.

Parameters:

- `input`: Input feature vector
- `gamma`: Scaling parameter
- `beta`: Shifting parameter
- `epsilon`: Small constant for numerical stability

Returns:

- Normalized vector with the same dimensions as the input


## Role in Transformer Architecture

This implementation is designed to work seamlessly in Transformer architectures where layer normalization is typically applied:

- Before/after multi-head attention mechanisms
- Before/after feed-forward networks
- In pre-norm or post-norm configurations

