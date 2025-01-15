mod positional_encoding;
mod attention;
mod feed_forward;
mod layer_norm;
mod encoder;

use positional_encoding::position_encoding_calculator;
use attention::{scaled_dot_product_attention, multi_head_attention};
use feed_forward::FeedForwardNetwork;
use layer_norm::{apply_layer_norm, test_apply_layer_norm};
use encoder::EncoderLayer;

fn main() {
    println!("Running all module tests...\n");

    test_positional_encoding();


    test_attention();

 
    test_feed_forward();

  
    test_layer_norm();


    test_encoder_layer();

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
