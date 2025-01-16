/////////////////////////////////////////////////////////////
// src/classification_head/classification_head.rs
//
// Purpose:
// Maps encoder output to the final output classes (e.g., spam vs. not spam).
//
// Input: Final encoder output. Shape: [batch_size, d_model].
//
// Output: Class probabilities (logits). Shape: [batch_size, num_classes].
//
// How it fits later:
// Provides the output for classification tasks in the Transformer pipeline.
/////////////////////////////////////////////////////////////

use ndarray::{Array2, Axis};

pub struct ClassificationHead {
    weights: Array2<f64>, // Weight matrix for the output layer
    biases: Vec<f64>,     // Bias vector for the output layer
}

impl ClassificationHead {
    /// Creates a new `ClassificationHead`.
    ///
    /// # Arguments
    /// * `d_model` - Dimension of the transformer encoder output.
    /// * `num_classes` - Number of output classes.
    ///
    /// # Returns
    /// A new instance of `ClassificationHead`.
    pub fn new(d_model: usize, num_classes: usize) -> Self {
        let weights = Array2::zeros((d_model, num_classes));
        let biases = vec![0.0; num_classes];
        ClassificationHead { weights, biases }
    }

    /// Performs a forward pass through the classification head.
    ///
    /// # Arguments
    /// * `pooled_output` - Pooled encoder output. Shape: [batch_size, d_model].
    ///
    /// # Returns
    /// * Logits. Shape: [batch_size, num_classes].
    pub fn forward(&self, pooled_output: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let batch_size = pooled_output.len();
        let num_classes = self.biases.len();

        // Compute logits for each example in the batch
        let mut logits = Vec::with_capacity(batch_size);
        for example in pooled_output {
            let mut example_logits = vec![0.0; num_classes];
            for (class_idx, bias) in self.biases.iter().enumerate() {
                let mut logit = *bias;
                for (feature_idx, &value) in example.iter().enumerate() {
                    logit += value * self.weights[(feature_idx, class_idx)];
                }
                example_logits[class_idx] = logit;
            }
            logits.push(example_logits);
        }

        logits // Shape: [batch_size, num_classes]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_classification_head_forward() {
        let d_model = 4;
        let num_classes = 2;

        let mut head = ClassificationHead::new(d_model, num_classes);

        // Example weights and biases for testing
        head.weights = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]];
        head.biases = vec![0.1, -0.1];

        let pooled_output = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4.0, 3.0, 2.0, 1.0],
        ];

        let logits = head.forward(&pooled_output);

        assert_eq!(logits.len(), 2);
        assert_eq!(logits[0].len(), 2);

        // Check values manually
        assert!((logits[0][0] - 4.1).abs() < 1e-6); // Example 1, Class 0
        assert!((logits[0][1] - 5.5).abs() < 1e-6); // Example 1, Class 1
        assert!((logits[1][0] - 3.8).abs() < 1e-6); // Example 2, Class 0
        assert!((logits[1][1] - 4.6).abs() < 1e-6); // Example 2, Class 1
    }
}
