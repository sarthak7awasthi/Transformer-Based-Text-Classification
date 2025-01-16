mod positional_encoding;
mod attention;
mod feed_forward;
mod layer_norm;
mod encoder;
mod embedding;
mod transformer;
mod classification;

use positional_encoding::position_encoding_calculator;
use attention::{scaled_dot_product_attention, multi_head_attention};
use feed_forward::FeedForwardNetwork;
use layer_norm::{apply_layer_norm, test_apply_layer_norm};
use encoder::EncoderLayer;
use embedding::embeddings::Embeddings;
use transformer::{Transformer, TransformerConfig};

fn main() {
    println!("Running all module tests...\n");

    test_positional_encoding();
    test_attention();
    test_feed_forward();
    test_layer_norm();
    test_encoder_layer();
    test_embeddings();
    test_transformer();

    println!("\nAll module tests completed successfully!");
}

fn test_positional_encoding() {
    let seq_len = 4;
    let embed_dim = 6;
    let encodings = position_encoding_calculator(seq_len, embed_dim);
    println!("Positional Encodings:\n{:?}\n", encodings);
}

fn test_attention() {
    let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let key = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let value = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
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

    let input = vec![
        vec![0.1; d_model];
        seq_len
    ];

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

    let input = vec!["hello".to_string(), "world".to_string(), "unknown".to_string()];
    let encoded = embeddings.encode(&input);

    println!("Embeddings with Positional Encodings:\n{:?}\n", encoded);
}

fn test_transformer() {
    // Create a mock vocabulary and embeddings
    let vocab = std::collections::HashMap::from([
        ("hello".to_string(), 0),
        ("world".to_string(), 1),
        ("<UNK>".to_string(), 2),
    ]);
    let d_model = 4;

    // Instantiate embeddings
    let embeddings = Embeddings::new(vocab.clone(), d_model);

    // Transformer configuration
    let config = TransformerConfig {
        num_layers: 2,
        d_model,
        num_heads: 2,
        ff_dim: 8,
        num_classes: 2,
        epsilon: 1e-6,
    };

    // Create the transformer
    let transformer = Transformer::new(config);

    // Input tokens
    let input_tokens = vec!["hello".to_string(), "world".to_string(), "unknown".to_string()];
    let batched_embeddings: Vec<Vec<Vec<f64>>> = vec![
        embeddings
            .encode(&input_tokens)
            .outer_iter()
            .map(|row| row.to_vec())
            .collect(),
    ];

    // Forward pass through the transformer
    let logits = transformer.forward(&batched_embeddings);

    println!("Transformer Output (Logits):\n{:?}\n", logits);
}
