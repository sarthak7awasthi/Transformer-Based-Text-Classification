/// Module for layer normalization
/// Functional: apply_layer_norm: Applies layer normalization to stabilize training.
///
/// Parameters:
///   - `inputs`: A reference to a vector of feature values for a single instance (`Vec<f64>`).
///   - `epsilon`: A small constant to prevent division by zero (`f64`).
///   - `gamma`: A scaling parameter for normalized outputs (`Vec<f64>`).
///   - `beta`: A shifting parameter for normalized outputs (`Vec<f64>`).
///
/// Return:
///   A vector (`Vec<f64>`) representing the normalized output for the given instance.
pub fn apply_layer_norm(
	inputs: &Vec<f64>,
	epsilon: f64,
	gamma: &Vec<f64>,
	beta: &Vec<f64>,
) -> Vec<f64> {
	assert_eq!(inputs.len(), gamma.len(), "Gamma size mismatch");
	assert_eq!(inputs.len(), beta.len(), "Beta size mismatch");


	let mean: f64 = inputs.iter().sum::<f64>() / inputs.len() as f64;

	let variance: f64 = inputs
			.iter()
			.map(|&x| (x - mean).powi(2))
			.sum::<f64>()
			/ inputs.len() as f64;


	inputs
			.iter()
			.enumerate()
			.map(|(i, &x)| {
					let normalized = (x - mean) / ((variance + epsilon).sqrt());
					gamma[i] * normalized + beta[i]
			})
			.collect()
}

/// Functional: test_apply_layer_norm: Tests the `apply_layer_norm` function.
///
/// Parameters: None.
/// Return: Prints test results to confirm functionality.
pub fn test_apply_layer_norm() {

	let inputs = vec![1.0, 2.0, 3.0, 4.0];
	let gamma = vec![1.0, 1.0, 1.0, 1.0];
	let beta = vec![0.0, 0.0, 0.0, 0.0];
	let epsilon = 1e-5;


	let normalized = apply_layer_norm(&inputs, epsilon, &gamma, &beta);


	let mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
	let variance: f64 = normalized
			.iter()
			.map(|&x| (x - mean).powi(2))
			.sum::<f64>()
			/ normalized.len() as f64;

	println!("Normalized Values: {:?}", normalized);
	println!("Mean: {:.6}", mean);
	println!("Variance: {:.6}", variance);


	assert!((mean - 0.0).abs() < 1e-6, "Mean is not zero!");
	assert!((variance - 1.0).abs() < 1e-5, "Variance is not close to one!"); 

	println!("LayerNorm test passed!");
}
