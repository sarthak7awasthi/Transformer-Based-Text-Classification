# Trainer Module

## Overview

The `trainer.rs` module is responsible for training a Transformer model using a dataset. It manages the forward and backward passes, computes the loss, updates model parameters, and tracks metrics like loss and accuracy over multiple epochs. It also saves the model's state after each epoch and upon completion.

---

## Purpose

The Trainer module provides the following functionalities:

1. **Model Training**: Executes the training process over multiple epochs with batch updates.
2. **Performance Tracking**: Calculates loss and accuracy metrics for each epoch.
3. **Model Persistence**: Saves the model at the end of each epoch and after training for reproducibility.

---

## Workflow

### 1. **Dataset Preparation**

The training dataset is loaded using the `DataLoader` module. Inputs and labels are batched for efficient processing.

### 2. **Training Loop**

For each epoch:

- **Forward Pass**: The input batch is passed through the Transformer model to generate logits.
- **Loss Calculation**: Cross-entropy loss is computed between the logits and ground truth labels.
- **Backward Pass**: Gradients are calculated and applied to update model parameters.
- **Metrics Computation**: Accuracy and loss are tracked for the current epoch.

### 3. **Model Saving**

The model's state is saved at the end of each epoch and upon completion of the training process.

---

## Key Functions

### `new(model: Transformer, optimizer: Optimizer, data_loader: &'a DataLoader, epochs: usize) -> Self`

Initializes a new `Trainer` instance with:

- A `Transformer` model.
- An `Optimizer` for parameter updates.
- A reference to a `DataLoader` instance for managing dataset loading and batching.
- The number of training epochs.

### `train(&mut self, dataset_path: &str, save_path: &str)`

Trains the model over the specified number of epochs.

#### Steps:

1. Load the dataset from `dataset_path`.
2. Create batches of inputs and labels.
3. For each epoch:
   - Perform forward and backward passes for each batch.
   - Compute epoch-level loss and accuracy.
   - Save the model's state.
4. Save the final model to `save_path`.

### `compute_correct_predictions(&self, logits: &Array2<f64>, labels: &[usize]) -> usize`

Calculates the number of correct predictions in a batch.

#### Steps:

1. For each logit vector, identify the index with the highest probability (predicted label).
2. Compare the predicted label with the ground truth label.
3. Count the number of matches.

---

## Key Properties

1. **Batch Processing**: Supports mini-batch training for efficient model updates.
2. **Performance Metrics**: Tracks epoch-wise accuracy and loss to monitor training progress.
3. **Model Persistence**: Ensures the model's state is saved after each epoch for reproducibility.

---
