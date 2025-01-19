use crate::transformer::Transformer;
use crate::tokenization::tokenizer::Tokenizer;
use ndarray::Array2;
use std::error::Error;

pub struct Inference<'a> {
    pub model: Transformer,
    pub tokenizer: &'a Tokenizer,
}

impl<'a> Inference<'a> {
    /// Creates a new `Inference` instance with the loaded model and tokenizer.
    pub fn new(model_path: &str, tokenizer: &'a Tokenizer) -> Result<Self, Box<dyn Error>> {
        let model = Transformer::load(model_path)?;
        Ok(Inference { model, tokenizer })
    }

    /// Perform inference on a single input text.
    pub fn predict(&self, input_text: &str) -> Result<(usize, Vec<f64>), Box<dyn Error>> {
   
        let tokenized_input = self.tokenizer.tokenize(input_text);

      
        let padded_input = self.tokenizer.pad_sequence(tokenized_input);

      
        let input_array = Array2::from_shape_vec(
            (1, padded_input.len()), 
            padded_input.into_iter().map(|x| x as f64).collect(),
        )?;

        let logits = self.model.forward(&input_array);

        
        let logits_slice = logits.row(0).to_vec(); 
        let exp_values: Vec<f64> = logits_slice.iter().map(|x| x.exp()).collect();
        let sum_exp: f64 = exp_values.iter().sum();
        let probabilities: Vec<f64> = exp_values.iter().map(|p| p / sum_exp).collect();

  
        let predicted_class = probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap();

        Ok((predicted_class, probabilities))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenization::tokenizer::Tokenizer;
    use crate::transformer::{Transformer, TransformerConfig};
    use std::collections::HashMap;

    #[test]
    fn test_inference_predict() {
    
        let vocab = HashMap::from([
            ("hello".to_string(), 0),
            ("world".to_string(), 1),
            ("[PAD]".to_string(), 2),
            ("[UNK]".to_string(), 3),
        ]);

        let config = TransformerConfig {
            num_layers: 2,
            d_model: 4,
            num_heads: 2,
            ff_dim: 8,
            num_classes: 2,
            epsilon: 1e-6,
        };

        let transformer = Transformer::new(config, vocab.clone());
        let tokenizer = Tokenizer::new(vocab, 128);

        let model_path = "mock_model.json";
        transformer.save(model_path).unwrap();

   
        let inference = Inference::new(model_path, &tokenizer).unwrap();

        let (predicted_class, probabilities) = inference.predict("hello world").unwrap();
        println!("Predicted Class: {}", predicted_class);
        println!("Probabilities: {:?}", probabilities);

     
        std::fs::remove_file(model_path).unwrap();
    }
}
