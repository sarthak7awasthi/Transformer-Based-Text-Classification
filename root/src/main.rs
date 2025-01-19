mod positional_encoding;
mod attention;
mod feed_forward;
mod layer_norm;
mod encoder;
mod embedding;
mod transformer;
mod classification;
mod tokenization;
mod configurration;
mod data_handler;
mod cross_entropy;
mod model_optimizer;
mod training;
mod model_evaluator;
mod model_inference;

use std::collections::HashMap;
use std::fs;
use serde_json::Value;
use transformer::{Transformer, TransformerConfig};
use tokenization::tokenizer::Tokenizer;
use data_handler::data_loader::DataLoader;
use model_optimizer::optimizer::{Optimizer, OptimizerType};
use training::trainer::Trainer;
use model_evaluator::evaluator::Evaluator;
use model_inference::inference::Inference;
use crate::configurration::config::{PAD_TOKEN, UNK_TOKEN, MAX_SEQ_LENGTH};

fn main() {
    println!("Starting Transformer NLP Pipeline...\n");

    let vocab = build_vocab("src/train_dataset.json");

 
    let tokenizer = Tokenizer::new(vocab.clone(), MAX_SEQ_LENGTH);
    let data_loader = DataLoader::new(&tokenizer);


    let transformer_config = TransformerConfig {
        num_layers: 2,
        d_model: 128,
        num_heads: 8,
        ff_dim: 256,
        num_classes: 2,
        epsilon: 1e-6,
    };

  
    train_model(&transformer_config, &vocab, &data_loader);

  
    evaluate_model(&data_loader);

  
    perform_inference(&tokenizer);

    println!("\nPipeline Execution Completed Successfully!");
}


fn build_vocab(training_dataset_path: &str) -> HashMap<String, usize> {

    let file_content = fs::read_to_string(training_dataset_path)
        .expect("Failed to read the training dataset file");
    let data: Value = serde_json::from_str(&file_content)
        .expect("Failed to parse the training dataset as JSON");

    let mut dataset = Vec::new();
    if let Some(array) = data.as_array() {
        for item in array {
            if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                dataset.push(text.to_string());
            }
        }
    }


    let special_tokens = &[PAD_TOKEN, UNK_TOKEN];


    Tokenizer::build_vocab(&dataset, special_tokens, Some(100))
}


fn train_model(config: &TransformerConfig, vocab: &HashMap<String, usize>, data_loader: &DataLoader) {
    println!("\nTraining the Transformer Model...");

 
    let transformer = Transformer::new(config.clone(), vocab.clone());
    let optimizer = Optimizer::new(OptimizerType::SGD);
    let mut trainer = Trainer::new(transformer, optimizer, data_loader, 10);

   
    trainer.train("src/train_dataset.json", "src/trained_model.json");
    println!("Model Training Completed.\n");
}


fn evaluate_model(data_loader: &DataLoader) {
    println!("\nEvaluating the Transformer Model...");

 
    let model_path = "src/trained_model.json";
    match Evaluator::new(model_path, data_loader) {
        Ok(evaluator) => {
       
            let test_dataset_path = "src/test_dataset.json";
            if let Err(e) = evaluator.evaluate(test_dataset_path) {
                eprintln!("Evaluation error: {}", e);
            } else {
                println!("Model Evaluation Completed.\n");
            }
        }
        Err(e) => eprintln!("Failed to load model for evaluation: {}", e),
    }
}


fn perform_inference(tokenizer: &Tokenizer) {
    println!("\nPerforming Inference...");


    let model_path = "src/trained_model.json";
    match Inference::new(model_path, tokenizer) {
        Ok(inference) => {
            let input_text = "Exclusive deal: Buy 1 Get 1 Free!";
            match inference.predict(input_text) {
                Ok((predicted_class, probabilities)) => {
                    println!("Input: {}", input_text);
                    println!("Predicted Class: {}", predicted_class);
                    println!("Probabilities: {:?}", probabilities);
                }
                Err(e) => eprintln!("Error during inference: {}", e),
            }
        }
        Err(e) => eprintln!("Failed to load inference model: {}", e),
    }
}
