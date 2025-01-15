use crate::attention::scaled_dot_product_attention;
use crate::feed_forward::FeedForwardNetwork;
use crate::layer_norm::apply_layer_norm;

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
    pub fn forward(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {

        let attention_output = scaled_dot_product_attention(x, x, x);

       
        let residual1: Vec<Vec<f64>> = x.iter()
            .zip(attention_output.iter())
            .map(|(xi, ai)| xi.iter().zip(ai.iter()).map(|(xij, aij)| xij + aij).collect())
            .collect();
        let gamma = vec![1.0; residual1[0].len()];
        let beta = vec![0.0; residual1[0].len()];
        let norm1: Vec<Vec<f64>> = residual1.iter()
            .map(|row| apply_layer_norm(row, self.epsilon, &gamma, &beta))
            .collect();

 
        let ffn_output = self.feed_forward.forward(&ndarray::Array2::from_shape_vec(
            (norm1.len(), norm1[0].len()),
            norm1.iter().flat_map(|v| v.clone()).collect(),
        ).unwrap());


        let residual2: Vec<Vec<f64>> = norm1.iter()
            .zip(ffn_output.outer_iter())
            .map(|(ni, fi)| ni.iter().zip(fi.iter()).map(|(nij, fij)| nij + fij).collect())
            .collect();
        residual2.iter()
            .map(|row| apply_layer_norm(row, self.epsilon, &gamma, &beta))
            .collect()
    }
}
