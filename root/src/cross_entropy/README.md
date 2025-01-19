# Cross-Entropy Loss Module

A Rust implementation of cross-entropy loss computation for multi-class classification tasks in neural networks.

## Overview

This module (`loss.rs`) implements cross-entropy loss calculation, combining softmax transformation with negative log-likelihood to evaluate model predictions against ground truth labels in classification tasks.

## Mathematical Foundation

The implementation is based on two key components:

### Softmax Function

Converts raw logits into probabilities for class i:

```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

### Cross-Entropy Loss

Computes the negative log-likelihood of true classes across a batch:

```
loss = -1/N * Σ_i log(p_i,y_i)
```

where:

- N is the batch size
- p_i,y_i is the predicted probability for the true class y_i of the i-th sample

## Features

- Numerically stable softmax implementation
- Batch-averaged loss computation
- Proper probability normalization
- High sensitivity to confident incorrect predictions
- Batch size invariance

## Usage

### Input Format

- **Logits**: Raw model outputs (shape: [batch_size, num_classes])
- **Labels**: Ground truth class indices (shape: [batch_size])

### Output

Returns a scalar loss value representing the average cross-entropy across the batch.

## Implementation Details

### Numerical Stability

The implementation includes safeguards for numerical stability:

1. Logit shifting by subtracting the maximum value
2. Stable probability normalization
3. Safe logarithm computation

### Processing Steps

1. Softmax computation with numerical stability measures
2. True class probability selection
3. Negative log-likelihood calculation
4. Batch averaging

## Integration

The module is designed to work seamlessly within the Transformer pipeline:

- Accepts logits from `classification_head.rs`
- Takes ground truth labels from `data_loader.rs`
- Outputs loss values for optimization

## Key Benefits

- Provides reliable training feedback
- Penalizes overconfident incorrect predictions
- Generates well-calibrated probability outputs
