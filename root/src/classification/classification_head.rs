use ndarray::{Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct ClassificationHead {
    weights: Array2<f64>,
    biases: Array2<f64>,
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
        let weights = Array2::random((d_model, num_classes), Uniform::new(-0.1, 0.1));
        let biases = Array2::zeros((1, num_classes));
        ClassificationHead { weights, biases }
    }

    /// Performs a forward pass through the classification head.
    ///
    /// # Arguments
    /// * `pooled_output` - Pooled encoder output. Shape: [batch_size, d_model].
    ///
    /// # Returns
    /// * Logits. Shape: [batch_size, num_classes].
    pub fn forward(&self, pooled_output: &Array2<f64>) -> Array2<f64> {
        pooled_output.dot(&self.weights) + &self.biases
    }

    /// Collects mutable references to all trainable parameters in the classification head.
    pub fn parameters_mut(&mut self) -> Vec<&mut f64> {
        let mut params = vec![];

        // Add mutable references to weight matrix values
        for value in self.weights.iter_mut() {
            params.push(value);
        }

        // Add mutable references to bias values
        for value in self.biases.iter_mut() {
            params.push(value);
        }

        params
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
        head.biases = array![[0.1, -0.1]];

        let pooled_output = array![
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ];

        let logits = head.forward(&pooled_output);

        assert_eq!(logits.shape(), &[2, 2]);

        // Check values manually
        assert!((logits[[0, 0]] - 4.1).abs() < 1e-6); // Example 1, Class 0
        assert!((logits[[0, 1]] - 5.5).abs() < 1e-6); // Example 1, Class 1
        assert!((logits[[1, 0]] - 3.8).abs() < 1e-6); // Example 2, Class 0
        assert!((logits[[1, 1]] - 4.6).abs() < 1e-6); // Example 2, Class 1
    }

    #[test]
    fn test_parameters_mut() {
        let d_model = 4;
        let num_classes = 2;
        let mut head = ClassificationHead::new(d_model, num_classes);

        let params = head.parameters_mut();
        assert_eq!(params.len(), d_model * num_classes + num_classes);
    }

    #[test]
    fn test_serialization() {
        let d_model = 4;
        let num_classes = 2;
        let head = ClassificationHead::new(d_model, num_classes);

        // Test serialization
        let serialized = serde_json::to_string(&head).expect("Failed to serialize");
        let deserialized: ClassificationHead = serde_json::from_str(&serialized).expect("Failed to deserialize");

        // Verify dimensions are preserved
        assert_eq!(head.weights.shape(), deserialized.weights.shape());
        assert_eq!(head.biases.shape(), deserialized.biases.shape());
    }
}