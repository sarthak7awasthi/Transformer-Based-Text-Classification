use std::collections::HashMap;

use crate::configurration::config::{PAD_TOKEN, UNK_TOKEN, MAX_SEQ_LENGTH};

/// Tokenizer structure for managing tokenization and padding
pub struct Tokenizer {
    pub vocab: HashMap<String, usize>, // Vocabulary mapping tokens to indices
    pub max_seq_length: usize,         // Maximum sequence length for padding
}

impl Tokenizer {
    /// new Tokenizer instance
    pub fn new(vocab: HashMap<String, usize>, max_seq_length: usize) -> Self {
        // Ensure special tokens are in the vocabulary
        Self::verify_vocab(&vocab);
        Tokenizer { vocab, max_seq_length }
    }

  
    fn verify_vocab(vocab: &HashMap<String, usize>) {
        let required_tokens = [PAD_TOKEN, UNK_TOKEN];
        for &token in &required_tokens {
            if !vocab.contains_key(token) {
                panic!("Missing required special token in vocabulary: {}", token);
            }
        }
    }

 // function to build vocab dynamically utilizing the dataset
    pub fn build_vocab(
        dataset: &[String],
        special_tokens: &[&str],
        max_vocab_size: Option<usize>,
    ) -> HashMap<String, usize> {
        let mut token_counts: HashMap<String, usize> = HashMap::new();

     
        for text in dataset {
            let tokens = Self::preprocess_text(text);
            for token in tokens {
                *token_counts.entry(token).or_insert(0) += 1;
            }
        }

      
        let mut vocab: HashMap<String, usize> = HashMap::new();
        for (i, &token) in special_tokens.iter().enumerate() {
            vocab.insert(token.to_string(), i);
        }


        let mut sorted_tokens: Vec<_> = token_counts.into_iter().collect();
        sorted_tokens.sort_by(|a, b| b.1.cmp(&a.1));
        let max_vocab_size = max_vocab_size.unwrap_or(sorted_tokens.len());

        let mut index = special_tokens.len();
        for (token, _) in sorted_tokens.into_iter().take(max_vocab_size - index) {
            vocab.insert(token, index);
            index += 1;
        }

        vocab
    }

    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        let tokens = Self::preprocess_text(text);
        tokens
            .into_iter()
            .map(|token| *self.vocab.get(&token).unwrap_or(&self.vocab[UNK_TOKEN]))
            .collect()
    }

    
    pub fn pad_sequence(&self, sequence: Vec<usize>) -> Vec<usize> {
        let mut padded_sequence = sequence;
        padded_sequence.resize(self.max_seq_length, self.vocab[PAD_TOKEN]);
        padded_sequence
    }


    pub fn tokenize_and_pad_batch(&self, texts: &[String]) -> Vec<Vec<usize>> {
        texts
            .iter()
            .map(|text| {
                let tokenized = self.tokenize(text);
                self.pad_sequence(tokenized)
            })
            .collect()
    }


    fn preprocess_text(text: &str) -> Vec<String> {
        text.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_verification() {
        let vocab = HashMap::from([(PAD_TOKEN.to_string(), 0), (UNK_TOKEN.to_string(), 1)]);
        Tokenizer::verify_vocab(&vocab);
    }

    #[test]
    #[should_panic(expected = "Missing required special token in vocabulary: [UNK]")]
    fn test_vocab_missing_special_token() {
        let vocab = HashMap::from([(PAD_TOKEN.to_string(), 0)]);
        Tokenizer::verify_vocab(&vocab);
    }

    #[test]
    fn test_tokenization_and_padding() {
        let vocab = HashMap::from([
            (PAD_TOKEN.to_string(), 0),
            (UNK_TOKEN.to_string(), 1),
            ("hello".to_string(), 2),
            ("world".to_string(), 3),
        ]);
        let tokenizer = Tokenizer::new(vocab.clone(), 5);

        let tokenized = tokenizer.tokenize("hello world unknown");
        assert_eq!(tokenized, vec![2, 3, 1]);

        let padded = tokenizer.pad_sequence(tokenized);
        assert_eq!(padded, vec![2, 3, 1, 0, 0]);
    }

    #[test]
    fn test_build_vocab() {
        let dataset = vec![
            "hello world".to_string(),
            "hello Rust".to_string(),
            "hello hello".to_string(),
        ];
        let special_tokens = &[PAD_TOKEN, UNK_TOKEN];
        let vocab = Tokenizer::build_vocab(&dataset, special_tokens, Some(5));

        assert!(vocab.contains_key(PAD_TOKEN));
        assert!(vocab.contains_key(UNK_TOKEN));
        assert!(vocab.contains_key("hello"));
        assert!(vocab.contains_key("world"));
    }
}
