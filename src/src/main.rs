mod PositionalEncoder;
use PositionalEncoder::positional_encoder::position_encoding_calculator;

fn main() {
    let sequence_length = 4;
    let embedding_dimensions = 6;
    let encoding = position_encoding_calculator(sequence_length, embedding_dimensions);
    println!("{:?}", encoding);
}
