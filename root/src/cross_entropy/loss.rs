use ndarray::{Array2, Axis};
use ndarray::prelude::*;
use std::f64;

/// Module for calculating loss functions, specifically Cross-Entropy Loss.
///
/// Purpose:
/// - Implements cross-entropy loss for multi-class classification problems.
/// - Converts logits (raw model outputs) to probabilities using the softmax function.
/// - Computes the negative log-likelihood of the target class.
///
/// Input:
/// - `logits`: A 2D array containing the model's logits. Shape: [batch_size, num_classes].
/// - `labels`: A vector of ground truth labels. Shape: [batch_size].
///
/// Output:
/// - Scalar loss value (f64), averaged over the batch.

pub struct Loss;

impl Loss {
    /// Computes the softmax function for a batch of logits.
    ///
    /// # Arguments
    /// * `logits` - A 2D array of logits. Shape: [batch_size, num_classes].
    ///
    /// # Returns
    /// * A 2D array of probabilities. Shape: [batch_size, num_classes].
    pub fn softmax(logits: &Array2<f64>) -> Array2<f64> {
        let mut probabilities = logits.clone();

        for mut row in probabilities.outer_iter_mut() {
            let max_logit = row.iter().cloned().fold(f64::MIN, f64::max); // For numerical stability

            let exp_sum: f64 = row.iter().map(|logit| (logit - max_logit).exp()).sum();

            row.mapv_inplace(|logit| (logit - max_logit).exp() / exp_sum);
        }

        probabilities
    }

    /// Computes the cross-entropy loss for a batch of logits and labels.
    ///
    /// # Arguments
    /// * `logits` - A 2D array of logits. Shape: [batch_size, num_classes].
    /// * `labels` - A vector of ground truth labels. Shape: [batch_size].
    ///
    /// # Returns
    /// * A scalar loss value averaged over the batch.
    pub fn cross_entropy_loss(logits: &Array2<f64>, labels: &[usize]) -> f64 {
        assert_eq!(logits.nrows(), labels.len(), "Logits and labels batch sizes must match.");

        let probabilities = Self::softmax(logits);

        let mut total_loss = 0.0;
        for (i, &label) in labels.iter().enumerate() {
            assert!(
                label < probabilities.ncols(),
                "Label index out of bounds for logits."
            );
         
            total_loss -= probabilities[(i, label)].ln();
        }

        total_loss / labels.len() as f64 // Return average loss
    }

    /// Computes gradients of the cross-entropy loss with respect to logits.
    ///
    /// # Arguments
    /// * `logits` - A 2D array of logits. Shape: [batch_size, num_classes].
    /// * `labels` - A vector of ground truth labels. Shape: [batch_size].
    ///
    /// # Returns
    /// * A 2D array of gradients. Shape: [batch_size, num_classes].
    pub fn gradients(logits: &Array2<f64>, labels: &[usize]) -> Array2<f64> {
        let probabilities = Self::softmax(logits);

        let mut gradients = probabilities;

        for (i, &label) in labels.iter().enumerate() {
            gradients[(i, label)] -= 1.0; 
        }

        gradients / labels.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_softmax() {
        let logits = array![
            [1.0, 2.0, 3.0],
            [1.0, 1.0, 1.0],
        ];

        let probabilities = Loss::softmax(&logits);

        assert_eq!(probabilities.nrows(), 2);
        assert_eq!(probabilities.ncols(), 3);
        assert!((probabilities[(0, 0)] - 0.09003).abs() < 1e-5);
        assert!((probabilities[(0, 1)] - 0.24473).abs() < 1e-5);
        assert!((probabilities[(0, 2)] - 0.66524).abs() < 1e-5);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let logits = array![
            [1.0, 2.0, 3.0],
            [1.0, 1.0, 1.0],
        ];
        let labels = vec![2, 1];

        let loss = Loss::cross_entropy_loss(&logits, &labels);
        assert!((loss - 0.71356).abs() < 1e-5);
    }

    #[test]
    fn test_gradients() {
        let logits = array![
            [1.0, 2.0, 3.0],
            [1.0, 1.0, 1.0],
        ];
        let labels = vec![2, 1];

        let gradients = Loss::gradients(&logits, &labels);

        assert_eq!(gradients.nrows(), 2);
        assert_eq!(gradients.ncols(), 3);
        assert!((gradients[(0, 2)] - (-0.33476)).abs() < 1e-5);
    }
}
