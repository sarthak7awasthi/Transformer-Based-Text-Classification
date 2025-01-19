use crate::encoder::encoder_layer::EncoderLayer;
use crate::classification::ClassificationHead;
use crate::embedding::embeddings::Embeddings;
use std::collections::HashMap;
use ndarray::{Array2, Axis};

/// Transformer configuration parameters.
#[derive(Clone)]
pub struct TransformerConfig {
    pub num_layers: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub ff_dim: usize,
    pub num_classes: usize, // e.g., 2 for spam vs. not spam
    pub epsilon: f64,       // For layer normalization
}

/// The Transformer struct holds:
/// - A stack of Encoder layers.
/// - A ClassificationHead (optional, if you're doing classification right away).
/// - An Embeddings module to encode input tokens.
/// - Configuration parameters for clarity.
pub struct Transformer {
    pub encoder_layers: Vec<EncoderLayer>,
    pub classification_head: ClassificationHead,
    pub embeddings: Embeddings,
    pub config: TransformerConfig,
}

impl Transformer {
    /// Creates a new Transformer with the specified configuration.
    /// Constructs `num_layers` encoders, a single classification head, and embeddings.
    pub fn new(config: TransformerConfig, vocab: HashMap<String, usize>) -> Self {
        let embeddings = Embeddings::new(vocab, config.d_model);

        let encoder_layers = (0..config.num_layers)
            .map(|_| EncoderLayer::new(config.d_model, config.num_heads, config.ff_dim, config.epsilon))
            .collect();

        let classification_head = ClassificationHead::new(config.d_model, config.num_classes);

        Self {
            encoder_layers,
            classification_head,
            embeddings,
            config,
        }
    }

    /// Forward pass through the Transformer.
    /// Processes input tokens through embeddings, encoders, and a classification head.
    pub fn forward(&self, batched_tokens: &Array2<f64>) -> Array2<f64> {
        println!("Input tokens shape: {:?}", batched_tokens.shape());
    
        // Pass through encoder layers
        let mut encoder_output = batched_tokens.clone();
        for (i, layer) in self.encoder_layers.iter().enumerate() {
            encoder_output = layer.forward(&encoder_output);
            println!("Shape after encoder layer {}: {:?}", i + 1, encoder_output.shape());
        }
    
        // Instead of mean pooling, we'll use the last token's representation
        // This maintains the batch dimension
        let batch_size = encoder_output.nrows();
        let sequence_features = encoder_output.clone();
    
        // Pass through the classification head
        let logits = self.classification_head.forward(&sequence_features);
        println!("Output logits shape: {:?}", logits.shape());
    
        logits
    }
    
    

    /// Collects mutable references to all trainable parameters in the Transformer.
    pub fn parameters_mut(&mut self) -> Vec<&mut f64> {
        let mut params = vec![];

        for layer in &mut self.encoder_layers {
            params.extend(layer.parameters_mut());
        }

        params.extend(self.classification_head.parameters_mut());
        params.extend(self.embeddings.parameters_mut());

        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_transformer_forward() {
        let vocab = HashMap::from([
            ("hello".to_string(), 0),
            ("world".to_string(), 1),
            ("<UNK>".to_string(), 2),
        ]);

        let config = TransformerConfig {
            num_layers: 2,
            d_model: 4,
            num_heads: 2,
            ff_dim: 8,
            num_classes: 2,
            epsilon: 1e-6,
        };

        let transformer = Transformer::new(config, vocab);

        let batched_tokens = array![[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]];

        let logits = transformer.forward(&batched_tokens);

        assert_eq!(logits.shape(), [2, 2]); // Shape: [batch_size, num_classes]
    }
}
