// src/evaluator.rs

use std::collections::HashMap;
use crate::transformer::Transformer;
use crate::cross_entropy::loss::Loss;

/// Struct to evaluate the performance of the trained model.
pub struct Evaluator;

impl Evaluator {
    /// Creates a new Evaluator instance.
    ///
    /// # Returns
    /// A new instance of the Evaluator struct.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the model's performance on test data.
    ///
    /// # Arguments
    /// * `model` - A reference to the trained Transformer model.
    /// * `inputs` - A reference to the test inputs (tokenized and padded sequences).
    /// * `labels` - A reference to the ground truth labels corresponding to the inputs.
    ///
    /// # Returns
    /// * A tuple containing:
    ///   - The loss value (f64).
    ///   - A HashMap of metrics (accuracy, precision, recall, F1 score, etc.).
    pub fn evaluate(
        &self,
        model: &Transformer,
        inputs: &[Vec<usize>],
        labels: &[usize],
    ) -> (f64, HashMap<String, f64>) {
        // Forward pass: Generate logits from the model
        let logits = model.forward(inputs);

        // Compute loss
        let loss = Loss::cross_entropy_loss(&logits, labels);

        // Compute accuracy and other metrics
        let metrics = self.calculate_metrics(&logits, labels);

        (loss, metrics)
    }

    /// Calculates evaluation metrics (accuracy, precision, recall, F1 score).
    ///
    /// # Arguments
    /// * `logits` - A reference to the model's output logits.
    /// * `labels` - A reference to the ground truth labels.
    ///
    /// # Returns
    /// A HashMap containing calculated metrics.
    fn calculate_metrics(
        &self,
        logits: &[Vec<f64>],
        labels: &[usize],
    ) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        // Compute predictions from logits
        let predictions: Vec<usize> = logits
            .iter()
            .map(|logit| {
                logit
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap()
            })
            .collect();

        // Calculate accuracy
        let correct = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(pred, label)| pred == label)
            .count();
        let accuracy = correct as f64 / labels.len() as f64 * 100.0;

        // Calculate precision, recall, and F1 score
        let (precision, recall, f1_score) = self.calculate_prf(&predictions, labels);

        // Insert metrics into the HashMap
        metrics.insert("accuracy".to_string(), accuracy);
        metrics.insert("precision".to_string(), precision);
        metrics.insert("recall".to_string(), recall);
        metrics.insert("f1_score".to_string(), f1_score);

        metrics
    }

    /// Computes precision, recall, and F1 score.
    ///
    /// # Arguments
    /// * `predictions` - A reference to the predicted labels.
    /// * `labels` - A reference to the ground truth labels.
    ///
    /// # Returns
    /// A tuple containing precision, recall, and F1 score.
    fn calculate_prf(
        &self,
        predictions: &[usize],
        labels: &[usize],
    ) -> (f64, f64, f64) {
        let mut true_positive = 0;
        let mut false_positive = 0;
        let mut false_negative = 0;

        for (pred, label) in predictions.iter().zip(labels.iter()) {
            if *pred == *label {
                true_positive += 1;
            } else {
                if *pred == 1 {
                    false_positive += 1;
                }
                if *label == 1 {
                    false_negative += 1;
                }
            }
        }

        let precision = if true_positive + false_positive > 0 {
            true_positive as f64 / (true_positive + false_positive) as f64
        } else {
            0.0
        };

        let recall = if true_positive + false_negative > 0 {
            true_positive as f64 / (true_positive + false_negative) as f64
        } else {
            0.0
        };

        let f1_score = if precision + recall > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        };

        (precision, recall, f1_score)
    }
}
