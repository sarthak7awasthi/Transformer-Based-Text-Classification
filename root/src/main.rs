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

use std::collections::HashMap;
use positional_encoding::position_encoding_calculator;
use attention::{scaled_dot_product_attention, multi_head_attention};
use feed_forward::FeedForwardNetwork;
use layer_norm::{apply_layer_norm, test_apply_layer_norm};
use encoder::EncoderLayer;
use embedding::embeddings::Embeddings;
use transformer::{Transformer, TransformerConfig};
use tokenization::tokenizer::Tokenizer;
use data_handler::data_loader::DataLoader;
use cross_entropy::loss::Loss;
use model_optimizer::optimizer::{Optimizer, OptimizerType};
use training::trainer::Trainer;
use crate::configurration::config::{PAD_TOKEN, UNK_TOKEN, MAX_SEQ_LENGTH};

fn main() {
    println!("Running all module tests...\n");

    test_positional_encoding();
    test_attention();
    test_feed_forward();
    test_layer_norm();
    test_encoder_layer();
    test_embeddings();
    test_transformer();
    test_tokenizer();
    test_data_loader();
    test_cross_entropy_loss();
    test_optimizer();
    test_trainer();

    println!("\nAll module tests completed successfully!");
}

fn test_positional_encoding() {
    let seq_len = 4;
    let embed_dim = 6;
    let encodings = position_encoding_calculator(seq_len, embed_dim);
    println!("Positional Encodings:\n{:?}\n", encodings);
}

fn test_attention() {
    let query = ndarray::array![[1.0, 0.0], [0.0, 1.0]];
    let key = ndarray::array![[1.0, 0.0], [0.0, 1.0]];
    let value = ndarray::array![[1.0, 0.0], [0.0, 1.0]];
    let attention_result = scaled_dot_product_attention(&query, &key, &value);
    println!("Attention Result:\n{:?}\n", attention_result);
}

fn test_feed_forward() {
    let ff = FeedForwardNetwork::new(4, 8);
    let input = ndarray::array![[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]];
    let output = ff.forward(&input);
    println!("Feed-Forward Output:\n{:?}\n", output);
}

fn test_layer_norm() {
    test_apply_layer_norm();
}

fn test_encoder_layer() {
    let d_model = 8;
    let num_heads = 2;
    let d_ff = 16;
    let epsilon = 1e-6;
    let seq_len = 4;

    let encoder_layer = EncoderLayer::new(d_model, num_heads, d_ff, epsilon);
    let input = ndarray::Array2::<f64>::ones((seq_len, d_model));
    let output = encoder_layer.forward(&input);

    println!("Encoder Layer Output:\n{:?}\n", output);
}

fn test_embeddings() {
    let vocab = std::collections::HashMap::from([
        ("hello".to_string(), 0),
        ("world".to_string(), 1),
        ("<UNK>".to_string(), 2),
    ]);
    let model_dim = 4;

    let embeddings = Embeddings::new(vocab.clone(), model_dim);
    let input = vec![0, 1, 2];
    let encoded = embeddings.encode(&input);

    println!("Embeddings with Positional Encodings:\n{:?}\n", encoded);
}

fn test_transformer() {
    let vocab = std::collections::HashMap::from([
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

    let transformer = Transformer::new(config, vocab.clone());

    let input_tokens = ndarray::array![[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]];
    println!("Input tokens:\n{:?}", input_tokens);

    let logits = transformer.forward(&input_tokens);
    println!("Logits:\n{:?}", logits);

    assert_eq!(logits.shape(), [2, 2]); // Ensure logits have the correct shape
}



fn test_tokenizer() {
    let dataset = vec![
        "Hello world!".to_string(),
        "Rust is amazing.".to_string(),
    ];

    let special_tokens = vec!["[PAD]", "[UNK]"];
    let vocab = Tokenizer::build_vocab(&dataset, &special_tokens, Some(100));
    let tokenizer = Tokenizer::new(vocab.clone(), MAX_SEQ_LENGTH);

    let tokenized = tokenizer.tokenize_and_pad_batch(&dataset);
    println!("Tokenized and Padded Sequences:\n{:?}\n", tokenized);
}

fn test_data_loader() {
    let mut vocab = HashMap::from([
        (PAD_TOKEN.to_string(), 0),
        (UNK_TOKEN.to_string(), 1),
        ("hello".to_string(), 2),
        ("world".to_string(), 3),
    ]);

    let tokenizer = Tokenizer::new(vocab.clone(), MAX_SEQ_LENGTH);
    let data_loader = DataLoader::new(&tokenizer);

    let mock_dataset_path = "src/test_dataset.json";
    match data_loader.load_dataset(mock_dataset_path) {
        Ok((inputs, labels)) => {
            println!("DataLoader Inputs (first few):\n{:?}", &inputs[..2]);
            println!("DataLoader Labels (first few):\n{:?}", &labels[..2]);
        }
        Err(e) => println!("DataLoader Error: {}", e),
    }
}

fn test_cross_entropy_loss() {
    let logits = ndarray::array![[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]];
    let labels = vec![2, 1];
    let loss = Loss::cross_entropy_loss(&logits, &labels);
    println!("Cross-Entropy Loss: {:.5}", loss);
}

fn test_optimizer() {
    let mut params = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
    let grads = ndarray::array![[0.1, 0.2], [0.3, 0.4]];

    let mut sgd_optimizer = Optimizer::new(OptimizerType::SGD);
    sgd_optimizer.step(&mut params.view_mut(), &grads.view());
    println!("Parameters after SGD:\n{:?}\n", params);

    params = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
    let mut adam_optimizer = Optimizer::new(OptimizerType::Adam);
    adam_optimizer.step(&mut params.view_mut(), &grads.view());
    println!("Parameters after Adam:\n{:?}\n", params);
}


fn test_trainer() {
    // Create vocabulary with special tokens
    let mut vocab = std::collections::HashMap::from([
        (PAD_TOKEN.to_string(), 0),
        (UNK_TOKEN.to_string(), 1),
        ("hello".to_string(), 2),
        ("world".to_string(), 3),
    ]);

    let config = TransformerConfig {
        num_layers: 2,
        d_model: 128, // Changed from 4 to match sequence length
        num_heads: 8, // Should be a factor of d_model
        ff_dim: 256, // Typically 2-4x d_model
        num_classes: 2,
        epsilon: 1e-6,
    };

    let transformer = Transformer::new(config, vocab.clone());
    let optimizer = Optimizer::new(OptimizerType::SGD);
    let tokenizer = Tokenizer::new(vocab, MAX_SEQ_LENGTH);
    let data_loader = DataLoader::new(&tokenizer);

    let mut trainer = Trainer::new(transformer, optimizer, &data_loader, 3);
    trainer.train("src/test_dataset.json");
}