use crate::attention::scaled_dot_product_attention;
use crate::feed_forward::FeedForwardNetwork;
use crate::layer_norm::apply_layer_norm;
use ndarray::{Array2, Axis};

pub struct EncoderLayer {
    pub feed_forward: FeedForwardNetwork,
    pub epsilon: f64,
}

impl EncoderLayer {
    /// Creates a new encoder layer with the specified dimensions
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, epsilon: f64) -> Self {
        Self {
            feed_forward: FeedForwardNetwork::new(d_model, d_ff),
            epsilon,
        }
    }

    /// Forward pass for the encoder layer
    ///
    /// # Arguments
    /// - `x`: Input embeddings with positional encodings (shape: [batch_size, seq_len, d_model]).
    ///
    /// # Returns
    /// - Processed embeddings (shape: [batch_size, seq_len, d_model]).
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        // Attention computation
        let attention_output = scaled_dot_product_attention(x, x, x);

        // Add & normalize (Residual Connection 1)
        let residual1 = x + &attention_output;
        let norm1 = apply_layer_norm(&residual1, self.epsilon);

        // Feed-forward computation
        let ffn_output = self.feed_forward.forward(&norm1);

        // Add & normalize (Residual Connection 2)
        let residual2 = &norm1 + &ffn_output;
        apply_layer_norm(&residual2, self.epsilon)
    }

    /// Collect mutable parameters for optimization
    pub fn parameters_mut(&mut self) -> Vec<&mut f64> {
        self.feed_forward.parameters_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_encoder_layer() {
        let d_model = 4;
        let num_heads = 2;
        let d_ff = 8;
        let epsilon = 1e-6;

        let encoder_layer = EncoderLayer::new(d_model, num_heads, d_ff, epsilon);

        let input = array![
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ];

        let output = encoder_layer.forward(&input);

        assert_eq!(output.shape(), input.shape());
    }
}
