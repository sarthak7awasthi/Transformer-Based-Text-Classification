use ndarray::{Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;


/// Functional: FeedForwardNetwork
/// Implements a two-layer feed-forward network with ReLU activation, used in transformer encoder layers.
/// 
/// Fields:
///   - `w1`: Weight matrix for the first dense layer. Shape: (input_dim, hidden_dim)
///   - `b1`: Bias vector for the first dense layer. Shape: (1, hidden_dim)
///   - `w2`: Weight matrix for the second dense layer. Shape: (hidden_dim, input_dim)
///   - `b2`: Bias vector for the second dense layer. Shape: (1, input_dim)
///   - `hidden_dim`: Size of the hidden layer (usize).
///   - `input_dim`: Size of the input (and output) vectors (usize).
///







/// Functional: new
/// Creates a new instance of the feed-forward network with randomly initialized weights and biases.
/// 
/// Parameters:
///   - `input_dim`: The dimensionality of the input vectors (usize).
///   - `hidden_dim`: The dimensionality of the hidden layer (usize).
/// 
/// Return: 
///   A `FeedForwardNetwork` instance.
///
/// Functional: forward
/// Applies the feed-forward network to a batch of input data.
///
/// Parameters:
///   - `x`: A 2D array of shape (batch_size, input_dim), representing the input batch.
/// 
/// Return: 
///   A 2D array of shape (batch_size, input_dim), representing the processed output.
///













pub struct FeedForwardNetwork {
    w1: Array2<f64>, // Weight matrix for the first dense layer
    b1: Array2<f64>, // Bias vector for the first dense layer
    w2: Array2<f64>, // Weight matrix for the second dense layer
    b2: Array2<f64>, // Bias vector for the second dense layer
    hidden_dim: usize, // Hidden layer dimension
    input_dim: usize,  // Input dimension
}

impl FeedForwardNetwork {
  
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
  
        let w1 = Array2::random((input_dim, hidden_dim), Uniform::new(-0.1, 0.1));
        let b1 = Array2::zeros((1, hidden_dim));
        let w2 = Array2::random((hidden_dim, input_dim), Uniform::new(-0.1, 0.1));
        let b2 = Array2::zeros((1, input_dim));

        Self {
            w1,
            b1,
            w2,
            b2,
            hidden_dim,
            input_dim,
        }
    }

  
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
  
        assert_eq!(x.shape()[1], self.input_dim, "Input dimensions do not match!");

     
        let mut h = x.dot(&self.w1) + &self.b1;
        h.mapv_inplace(|v| v.max(0.0));

        // Second dense layer
        let y = h.dot(&self.w2) + &self.b2; 

        y 
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;


    #[test]
    fn test_feed_forward() {
        let input_dim = 4;
        let hidden_dim = 8;
        let batch_size = 2;


        let ff = FeedForwardNetwork::new(input_dim, hidden_dim);

     
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0]
        ];


        let y = ff.forward(&x);

   
        assert_eq!(y.shape(), &[batch_size, input_dim]);

        
        println!("Input: {:?}", x);
        println!("Output: {:?}", y);
    }
}
