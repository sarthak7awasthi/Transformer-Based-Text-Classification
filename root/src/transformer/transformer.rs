/////////////////////////////////////////////////////////////
// src/transformer.rs
//
// Purpose: Integrates multiple Encoder layers (stacked)
// and passes the final output to a ClassificationHead.
//
// Input:  Batched token embeddings with positional encodings
//         Shape: [batch_size][sequence_length][embedding_dim]
//
// Output: Encoded representation (after ClassificationHead,
//         or the final context if you only need encoder output)
//
// How it fits later: Represents the core transformer logic
// that can be extended for tasks like classification.
/////////////////////////////////////////////////////////////

use crate::encoder::encoder_layer::EncoderLayer;
use crate::classification::ClassificationHead;

/// Transformer configuration parameters.
/// You can also store these in a central config module (`config.rs`)
pub struct TransformerConfig {
    pub num_layers: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub ff_dim: usize,
    pub num_classes: usize, // e.g., 2 for spam vs. not spam
    pub epsilon: f64,       // For layer normalization
}

/// The Transformer struct holds:
/// - A stack of Encoder layers
/// - A ClassificationHead (optional, if you're doing classification right away)
/// - Configuration parameters for clarity
pub struct Transformer {
    pub encoder_layers: Vec<EncoderLayer>,
    pub classification_head: ClassificationHead,
    pub config: TransformerConfig,
}

impl Transformer {
    /// Creates a new Transformer with the specified configuration.
    /// Constructs `num_layers` encoders and a single classification head.
    pub fn new(config: TransformerConfig) -> Self {
        // Build the encoder layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let encoder = EncoderLayer::new(config.d_model, config.num_heads, config.ff_dim, config.epsilon);
            layers.push(encoder);
        }

        // Build the classification head
        let classification_head = ClassificationHead::new(config.d_model, config.num_classes);

        Self {
            encoder_layers: layers,
            classification_head,
            config,
        }
    }

    /// Forward pass through the Transformer.
    /// 1. Pass embeddings through each encoder layer in sequence.
    /// 2. (Optional) Aggregate the final encoder outputs.
    /// 3. Pass aggregated encoding to the classification head (if classification is desired here).
    ///
    /// Returns the final classification logits (or a representation).
    pub fn forward(
        &self,
        batched_embeddings: &Vec<Vec<Vec<f64>>>,
    ) -> Vec<Vec<f64>> {
        // 1) Process each sequence in the batch through the encoder layers
        let mut encoder_output: Vec<Vec<Vec<f64>>> = batched_embeddings.clone();

        for layer in &self.encoder_layers {
            encoder_output = encoder_output
                .iter()
                .map(|sequence| layer.forward(sequence))
                .collect();
        }

        // 2) Aggregate the outputs using average pooling
        let pooled_output = self.average_pooling(&encoder_output);

        // 3) Pass the pooled representation into the classification head
        let logits = self.classification_head.forward(&pooled_output);

        logits // Shape: [batch_size][num_classes]
    }

    /// A simple average pooling function to get a single vector per sequence.
    /// If your output shape is [batch_size][sequence_length][d_model],
    /// the pooled output shape will be [batch_size][d_model].
    fn average_pooling(
        &self,
        encoder_output: &Vec<Vec<Vec<f64>>>,
    ) -> Vec<Vec<f64>> {
        let batch_size = encoder_output.len();
        let mut pooled = Vec::with_capacity(batch_size);

        for sequence in encoder_output {
            // sequence shape: [sequence_length][d_model]
            let seq_length = sequence.len();
            if seq_length == 0 {
                // Handle edge case: empty sequence
                pooled.push(vec![0.0; self.config.d_model]);
                continue;
            }

            let mut sum_vec = vec![0.0; self.config.d_model];

            // Accumulate embeddings across the sequence length
            for token_vec in sequence {
                for (i, val) in token_vec.iter().enumerate() {
                    sum_vec[i] += val;
                }
            }

            // Divide by sequence length to compute the mean
            for i in 0..sum_vec.len() {
                sum_vec[i] /= seq_length as f64;
            }

            pooled.push(sum_vec);
        }

        pooled // Shape: [batch_size][d_model]
    }
}
