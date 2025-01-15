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
    /// * `tokenized_input` - A vector of tokenized input strings.
    ///
    /// # Returns
    /// * A matrix of shape (seq_len, model_dim) with embeddings and positional encodings.
    pub fn encode(&self, tokenized_input: &[String]) -> Array2<f64> {
        let seq_len = tokenized_input.len();
        let mut embeddings = Array2::zeros((seq_len, self.model_dim));

        for (idx, token) in tokenized_input.iter().enumerate() {
            if let Some(&token_idx) = self.vocab.get(token) {
                embeddings.row_mut(idx).assign(&self.token_embedding_matrix.row(token_idx));
            } else {
                // Handle unknown token
                let unknown_idx = self.vocab.get("<UNK>").unwrap_or(&0);
                embeddings.row_mut(idx).assign(&self.token_embedding_matrix.row(*unknown_idx));
            }
        }

        let positional_encodings = self.generate_positional_encodings(seq_len);
        embeddings + positional_encodings
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

        let input = vec!["hello".to_string(), "world".to_string(), "unknown".to_string()];
        let encoded = embeddings.encode(&input);

        assert_eq!(encoded.shape(), &[3, model_dim]);
    }
}
