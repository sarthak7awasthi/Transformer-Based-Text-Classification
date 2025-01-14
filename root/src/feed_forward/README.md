# Feed-Forward Network Module

A Rust implementation of the position-wise feed-forward network used in Transformer models. This module provides a two-layer neural network with ReLU activation that processes each position in a sequence independently.

## Overview

The feed-forward network (FFN) is a crucial component of Transformer layers, providing non-linear transformations of token representations. It processes each position independently and identically, enhancing the model's ability to learn complex patterns.

## Features

- Two-layer neural network architecture
- ReLU activation function
- Position-wise processing
- Configurable input and hidden dimensions
- Efficient parallel computation
- Random weight initialization

## Mathematical Foundation

The feed-forward network applies two sequential transformations:

1. First Layer (with ReLU activation):

   ```
   h = ReLU(xW₁ + b₁)
   ```

2. Second Layer:
   ```
   y = hW₂ + b₂
   ```

Where:

- x: Input vector
- W₁, W₂: Weight matrices
- b₁, b₂: Bias vectors
- ReLU(z) = max(0, z)

## Usage

```rust
use feed_forward::FeedForwardNetwork;


let ffn = FeedForwardNetwork::new(
    input_dim: 512,
    hidden_dim: 2048
);


let output = ffn.forward(&input);
```

## API Reference

### `FeedForwardNetwork` Struct

```rust
pub struct FeedForwardNetwork {
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
    hidden_dim: usize,
    input_dim: usize,
}
```

#### Fields

- `w1`: First layer weight matrix
- `b1`: First layer bias vector
- `w2`: Second layer weight matrix
- `b2`: Second layer bias vector
- `hidden_dim`: Hidden layer dimension
- `input_dim`: Input/output dimension

## Role in Transformer Architecture

The feed-forward network is applied:

- After self-attention mechanisms
- Independently to each position in the sequence
- Before residual connections and layer normalization

## Key Properties

### Position-Wise Processing

- Each position processed independently
- Same transformation applied to all positions
- Maintains sequence length and dimensionality

### Dimensionality Changes

- Input dimension → Hidden dimension (expansion)
- Hidden dimension → Input dimension (projection)
- Typically hidden_dim = 4 × input_dim

### Non-Linearity

- ReLU activation enables learning complex patterns
- Prevents linear collapse of multiple layers
