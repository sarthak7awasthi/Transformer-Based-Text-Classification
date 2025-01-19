// Module: embeddings.rs

/// Purpose:
/// Converts tokens into dense vectors and adds positional encodings.
/// Input: Tokenized input.
/// Output: Token embeddings with positional encodings.

use std::collections::HashMap;
use ndarray::{Array2, Array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub struct Embeddings {
    token_embedding_matrix: Array2<f64>, // Matrix to store token embeddings
    vocab: HashMap<String, usize>,      // Vocabulary mapping tokens to indices
    model_dim: usize,                   // Embedding dimension
}

impl Embeddings {
    /// Creates a new `Embeddings` instance.
    ///
    /// # Arguments
    /// * `vocab` - Vocabulary mapping tokens to indices.
    /// * `model_dim` - Dimension of the embeddings.
    pub fn new(vocab: HashMap<String, usize>, model_dim: usize) -> Self {
        let vocab_size = vocab.len();
        let token_embedding_matrix = Array2::random((vocab_size, model_dim), Uniform::new(-0.1, 0.1));
        Embeddings {
            token_embedding_matrix,
            vocab,
            model_dim,
        }
    }

    /// Generates positional encodings for a given sequence length.
    ///
    /// # Arguments
    /// * `seq_len` - Length of the input sequence.
    ///
    /// # Returns
    /// * A matrix of shape (seq_len, model_dim) containing positional encodings.
    pub fn generate_positional_encodings(&self, seq_len: usize) -> Array2<f64> {
        let mut positional_encodings = Array2::zeros((seq_len, self.model_dim));

        for pos in 0..seq_len {
            for i in 0..self.model_dim {
                let angle = pos as f64 / 10000f64.powf((2 * (i / 2)) as f64 / self.model_dim as f64);
                positional_encodings[[pos, i]] = if i % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                };
            }
        }

        positional_encodings
    }

    /// Converts tokenized input into dense vectors and adds positional encodings.
    ///
    /// # Arguments
    /// * `tokenized_input` - A slice of tokenized input indices.
    ///
    /// # Returns
    /// * A matrix of shape (seq_len, model_dim) with embeddings and positional encodings.
    pub fn encode(&self, tokenized_input: &[usize]) -> Array2<f64> {
        let seq_len = tokenized_input.len();
        let mut embeddings = Array2::zeros((seq_len, self.model_dim));

        for (idx, &token_idx) in tokenized_input.iter().enumerate() {
            if token_idx < self.token_embedding_matrix.nrows() {
                embeddings.row_mut(idx).assign(&self.token_embedding_matrix.row(token_idx));
            } else {
                // Handle unknown token index
                let unknown_idx = self.vocab.get("<UNK>").copied().unwrap_or(0);
                embeddings.row_mut(idx).assign(&self.token_embedding_matrix.row(unknown_idx));
            }
        }

        let positional_encodings = self.generate_positional_encodings(seq_len);
        embeddings + positional_encodings
    }

    /// Collects mutable references to all trainable parameters in the embeddings.
    pub fn parameters_mut(&mut self) -> Vec<&mut f64> {
        let mut params = vec![];

        // Add mutable references to embedding matrix values
        for value in self.token_embedding_matrix.iter_mut() {
            params.push(value);
        }

        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embeddings() {
        let vocab = HashMap::from([
            ("hello".to_string(), 0),
            ("world".to_string(), 1),
            ("<UNK>".to_string(), 2),
        ]);
        let model_dim = 4;

        let embeddings = Embeddings::new(vocab.clone(), model_dim);

        let input = vec![0, 1, 3]; // Using indices directly for testing
        let encoded = embeddings.encode(&input);

        assert_eq!(encoded.shape(), &[3, model_dim]);
    }
}
