mod positional_encoding;
mod attention;
mod feed_forward;
mod layer_norm;

use positional_encoding::position_encoding_calculator;
use attention::{scaled_dot_product_attention, multi_head_attention};
use feed_forward::FeedForwardNetwork;
use layer_norm::{apply_layer_norm, test_apply_layer_norm};

fn main() {
    println!("Running all module tests...");

    // Positional Encoding Test
    let seq_len = 4;
    let embed_dim = 6;
    let encodings = position_encoding_calculator(seq_len, embed_dim);
    println!("Positional Encodings: {:?}", encodings);

    // Attention Test
    let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let key = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let value = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let scale = 1.0;
    let attention_result = scaled_dot_product_attention(&query, &key, &value);
    println!("Attention Result: {:?}", attention_result);

    // Feed-Forward Test
    let ff = FeedForwardNetwork::new(4, 8);
    let input = ndarray::array![[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]];
    let output = ff.forward(&input);
    println!("Feed-Forward Output: {:?}", output);

    // LayerNorm Test
    test_apply_layer_norm();

    println!("All module tests completed successfully!");
}
