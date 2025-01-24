

// I tweaked the math a bit in this module. the original formula is slightly different. refer the readme i made for this module.
use std::f64::consts::PI;

/// module for positional encoding
/// Functional: position_encoding_calculator: Generates positional encodings for a sequence of tokens.
/// Parameters: 
///           "sequence_length": The number of tokens in the sequence (usize)
///           "embedding_dimensions": The dimensionality  of each token's embedding

/// Return:    A 2D vector (Vec<Vec<f64>>). Each row represents positional encoding for a specific token in the sequence.
///


pub fn position_encoding_calculator(sequence_length: usize, embedding_dimensions: usize) -> Vec<Vec<f64>>{
	let mut positional_encoding = vec![vec![0.0; embedding_dimensions]; sequence_length];



	for pos in 0..sequence_length{
		for dim in 0..embedding_dimensions{
			let angle = (pos as f64) / (10000f64.powf( dim as f64 / embedding_dimensions as f64));

			if dim % 2 == 0{
				positional_encoding[pos][dim] = angle.sin()
			}
			else{
				positional_encoding[pos][dim] = angle.cos()
			}
		}
	}


	positional_encoding
}





/// Test Section of the module(as it's recommended to put tests in same file for Rust)


#[cfg(test)]
mod tests {
    use super::*; 

    #[test]
    fn test_positional_encoding_dimensions() {
        let sequence_length = 4;
        let embedding_dimensions = 6;
        let encodings = position_encoding_calculator(sequence_length, embedding_dimensions);

        
        assert_eq!(encodings.len(), sequence_length, "Sequence length mismatch");
        assert_eq!(encodings[0].len(), embedding_dimensions, "Embedding dimension mismatch");
    }

    #[test]
    fn test_positional_encoding_values() {
        let sequence_length = 2;
        let embedding_dimensions = 4;
        let encodings = position_encoding_calculator(sequence_length, embedding_dimensions);

      
        let expected_sin_value = (0.0 / 10000f64.powf(0.0)).sin();
        assert!((encodings[0][0] - expected_sin_value).abs() < 1e-6);

        let expected_cos_value = (1.0 / 10000f64.powf(0.25)).cos();
        assert!((encodings[1][1] - expected_cos_value).abs() < 1e-6);
    }
}

