# Evaluator Module

## Overview

The `evaluator.rs` module evaluates the performance of a trained Transformer model on a given dataset. By computing key metrics such as **accuracy**, **precision**, **recall**, and **F1-score**, the module provides insights into how well the model generalizes to unseen data, enabling validation before deployment.

---

## Purpose

The `Evaluator` module serves the following key purposes:

1. Validating the trained model's performance on a test dataset.
2. Generating quantitative metrics to assess the model’s strengths and weaknesses.
3. Ensuring the reliability of predictions before production use.

---

## Methodology

### Workflow

1. **Dataset Loading**:

   - The `DataLoader` module loads and tokenizes the dataset, splitting it into inputs and labels.
   - Converts tokenized inputs into a 2D matrix suitable for Transformer processing.

2. **Model Predictions**:

   - Runs a forward pass through the Transformer model to generate logits.

3. **Metric Computation**:

   - **Accuracy**: Measures the proportion of correct predictions.
   - **Precision, Recall, F1-Score**: Provides deeper insights into per-class performance.

4. **Output**:
   - Metrics are logged in a user-friendly format for interpretability.

---

## Mathematical Foundation

### Accuracy

\[
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
\]
Measures the overall correctness of the model’s predictions.

---

### Precision

For a given class \( c \):
\[
\text{Precision}\_c = \frac{\text{True Positives}\_c}{\text{True Positives}\_c + \text{False Positives}\_c}
\]
Indicates how many predicted positives were actual positives.

---

### Recall

For a given class \( c \):
\[
\text{Recall}\_c = \frac{\text{True Positives}\_c}{\text{True Positives}\_c + \text{False Negatives}\_c}
\]
Measures the model's ability to correctly identify actual positives.

---

### F1-Score

\[
\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\]
Harmonic mean of precision and recall, balancing both metrics.

---

## Key Functions

### `new(model_path: &str, data_loader: &DataLoader) -> Result<Self, std::io::Error>`

Creates a new `Evaluator` instance, initializing it with:

- The trained Transformer model loaded from `model_path`.
- A `DataLoader` instance for loading the evaluation dataset.

---

### `evaluate(&self, dataset_path: &str) -> Result<(), Box<dyn std::error::Error>>`

Performs evaluation on a dataset by:

1. Loading the inputs and labels from `dataset_path`.
2. Running predictions using the Transformer model.
3. Computing metrics such as accuracy, precision, recall, and F1-score.
4. Printing metrics in a user-readable format.

---

### `compute_accuracy(&self, logits: &Array2<f64>, labels: &[usize]) -> f64`

Computes the accuracy of predictions:

- **Inputs**: Logits (model outputs) and ground truth labels.
- **Output**: A floating-point accuracy value.

---

### `compute_metrics(&self, logits: &Array2<f64>, labels: &[usize]) -> (f64, f64, f64)`

Computes precision, recall, and F1-score:

- **Inputs**: Logits and ground truth labels.
- **Output**: A tuple containing precision, recall, and F1-score.

---

## Key Properties

1. **Scalability**: Can handle large datasets efficiently due to batch processing.
2. **Flexibility**: Computes a wide range of metrics, catering to various performance assessment needs.
3. **Integration**: Seamlessly integrates with the Transformer model and DataLoader module.

---
