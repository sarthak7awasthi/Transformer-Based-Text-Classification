use ndarray::{Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct FeedForwardNetwork {
    w1: Array2<f64>,
    b1: Array2<f64>,
    w2: Array2<f64>,
    b2: Array2<f64>,
    hidden_dim: usize,
    input_dim: usize,
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

  
        let y = h.dot(&self.w2) + &self.b2;

        y
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut f64> {
        let mut params = vec![];

        for value in self.w1.iter_mut() {
            params.push(value);
        }
        for value in self.b1.iter_mut() {
            params.push(value);
        }
        for value in self.w2.iter_mut() {
            params.push(value);
        }
        for value in self.b2.iter_mut() {
            params.push(value);
        }

        params
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

        let ff = FeedForwardNetwork::new(input_dim, hidden_dim);

        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0]
        ];

        let y = ff.forward(&x);

        assert_eq!(y.shape(), &[2, input_dim]);
    }

    #[test]
    fn test_serialization() {
        let input_dim = 4;
        let hidden_dim = 8;

        let ff = FeedForwardNetwork::new(input_dim, hidden_dim);

        let serialized = serde_json::to_string(&ff).expect("Serialization failed");
        let deserialized: FeedForwardNetwork = serde_json::from_str(&serialized).expect("Deserialization failed");

        assert_eq!(ff.input_dim, deserialized.input_dim);
        assert_eq!(ff.hidden_dim, deserialized.hidden_dim);
    }
}
