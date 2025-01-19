/// Module for implementing optimizers such as SGD and Adam.
///
/// Purpose:
/// - Updates model parameters using computed gradients.
/// - Supports both SGD and Adam optimization methods.
///
/// Input:
/// - Gradients of all layers.
/// Output:
/// - Updated parameters.

use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use crate::configurration::config::{LEARNING_RATE, BETA1, BETA2, EPSILON};

/// Optimizer enum to choose between different optimization algorithms.
pub enum OptimizerType {
    SGD,
    Adam,
}

pub struct Optimizer {
    optimizer_type: OptimizerType,
    learning_rate: f64,
    beta1: f64,       
    beta2: f64,       
    epsilon: f64,     
    moment1: Option<Array2<f64>>, 
    moment2: Option<Array2<f64>>,
    timestep: usize, 
}

impl Optimizer {
    /// new optimizer.
    pub fn new(optimizer_type: OptimizerType) -> Self {
        Self {
            optimizer_type,
            learning_rate: LEARNING_RATE,
            beta1: BETA1,
            beta2: BETA2,
            epsilon: EPSILON,
            moment1: None,
            moment2: None,
            timestep: 0,
        }
    }

    /// Applies gradients to update parameters using the specified optimizer type.
    ///
    /// # Arguments
    /// * `params` - A mutable reference to the parameters to be updated.
    /// * `grads` - A reference to the gradients corresponding to the parameters.
    pub fn step(&mut self, params: &mut ArrayViewMut2<f64>, grads: &ArrayView2<f64>) {
        match self.optimizer_type {
            OptimizerType::SGD => self.sgd_step(params, grads),
            OptimizerType::Adam => self.adam_step(params, grads),
        }
    }

 
    fn sgd_step(&self, params: &mut ArrayViewMut2<f64>, grads: &ArrayView2<f64>) {
        assert_eq!(params.shape(), grads.shape(), "Parameter and gradient shapes must match.");

        params.zip_mut_with(grads, |param, &grad| {
            *param -= self.learning_rate * grad;
        });
    }

  
    fn adam_step(&mut self, params: &mut ArrayViewMut2<f64>, grads: &ArrayView2<f64>) {
        assert_eq!(params.shape(), grads.shape(), "Parameter and gradient shapes must match.");

  
        if self.moment1.is_none() {
            self.moment1 = Some(Array2::zeros(params.raw_dim()));
            self.moment2 = Some(Array2::zeros(params.raw_dim()));
        }

        let moment1 = self.moment1.as_mut().unwrap();
        let moment2 = self.moment2.as_mut().unwrap();

        self.timestep += 1;
        let t = self.timestep as f64;

        moment1.zip_mut_with(grads, |m1, &grad| {
            *m1 = self.beta1 * *m1 + (1.0 - self.beta1) * grad;
        });

        moment2.zip_mut_with(grads, |m2, &grad| {
            *m2 = self.beta2 * *m2 + (1.0 - self.beta2) * grad.powi(2);
        });

     
        let bias_corrected_m1 = moment1.mapv(|m1| m1 / (1.0 - self.beta1.powf(t)));
        let bias_corrected_m2 = moment2.mapv(|m2| m2 / (1.0 - self.beta2.powf(t)));

        params.zip_mut_with(&bias_corrected_m1, |param, &m1| {
					*param -= self.learning_rate * m1 / (bias_corrected_m2.mapv(f64::sqrt) + self.epsilon)
							.iter()
							.fold(0.0, |acc, &val| acc + val); 
				});
			
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sgd() {
        let mut params = array![[1.0, 2.0], [3.0, 4.0]];
        let grads = array![[0.1, 0.2], [0.3, 0.4]];
        let mut optimizer = Optimizer::new(OptimizerType::SGD);

        optimizer.step(&mut params.view_mut(), &grads.view());

        assert_eq!(params, array![[0.9999, 1.9998], [2.9997, 3.9996]]);
    }

    #[test]
    fn test_adam() {
        let mut params = array![[1.0, 2.0], [3.0, 4.0]];
        let grads = array![[0.1, 0.2], [0.3, 0.4]];
        let mut optimizer = Optimizer::new(OptimizerType::Adam);

        optimizer.step(&mut params.view_mut(), &grads.view());

      
        assert_eq!(params.shape(), [2, 2]); 
    }
}
