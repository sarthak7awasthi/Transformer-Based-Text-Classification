# Attention Mechanism Module

A Rust implementation of attention mechanisms for Transformer models, including Scaled Dot-Product Attention and Multi-Head Attention.

## Overview

The module provides efficient implementations of attention mechanisms that enable Transformer models to dynamically focus on relevant parts of input sequences. It replaces traditional sequential processing with parallelizable attention-based computations.

## Key Components

### Scaled Dot-Product Attention

Computes attention weights through query-key interactions and applies them to values:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

where:

- Q: Query matrix
- K: Key matrix
- V: Value matrix
- d_k: Key dimensionality

### Multi-Head Attention

Extends scaled dot-product attention by:

- Splitting input into multiple attention heads
- Processing each head independently
- Concatenating and projecting outputs

## Features

- Parallelizable attention computation
- Scaled dot-product for training stability
- Multi-head processing for enhanced feature extraction
- Softmax normalization for interpretable weights
- Support for self-attention and cross-attention

## Usage

### Scaled Dot-Product Attention

```rust
use attention::scaled_dot_product_attention;

let output = scaled_dot_product_attention(
    &query,    // Query matrix
    &key,      // Key matrix
    &value     // Value matrix
);
```

### Multi-Head Attention

```rust
use attention::multi_head_attention;

let output = multi_head_attention(
    &query,     // Query matrix
    &key,       // Key matrix
    &value,     // Value matrix
    num_heads   // Number of attention heads
);
```

## Mathematical Foundation

### Scaled Dot-Product Attention

1. Compute attention scores:

   ```
   Scores = QK^T / √d_k
   ```

2. Apply softmax:

   ```
   Weights = softmax(Scores)
   ```

3. Generate output:
   ```
   Output = Weights · V
   ```

### Multi-Head Attention

1. Split input into h heads
2. Apply scaled dot-product attention to each head
3. Concatenate outputs:
   ```
   MultiHead(Q,K,V) = Concat(head_1,...,head_h)W_O
   ```

## Key Functions

### `scaled_dot_product_attention`

```rust
pub fn scaled_dot_product_attention(
    query: &[Vec<f32>],
    key: &[Vec<f32>],
    value: &[Vec<f32>]
) -> Vec<Vec<f32>>
```

Parameters:

- `query`: Query matrix
- `key`: Key matrix
- `value`: Value matrix

Returns:

- Attention-weighted output matrix

### `multi_head_attention`

```rust
pub fn multi_head_attention(
    query: &[Vec<f32>],
    key: &[Vec<f32>],
    value: &[Vec<f32>],
    num_heads: usize
) -> Vec<Vec<f32>>
```

Parameters:

- `query`: Query matrix
- `key`: Key matrix
- `value`: Value matrix
- `num_heads`: Number of attention heads

Returns:

- Concatenated and projected attention outputs

## Role in Transformer Architecture

- **Encoder Self-Attention**: Relates tokens within input sequence
- **Decoder Self-Attention**: Processes target sequence
- **Cross-Attention**: Models input-output dependencies
