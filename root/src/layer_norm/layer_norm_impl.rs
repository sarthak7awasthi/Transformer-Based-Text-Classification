use ndarray::{Array2, ArrayView2, Axis};

/// Applies layer normalization to stabilize training.
///
/// # Arguments
/// - `inputs`: A 2D array of feature values for a batch. Shape: [batch_size, feature_dim].
/// - `epsilon`: A small constant to prevent division by zero.
/// 
/// # Returns
/// - A 2D array of normalized outputs. Shape: [batch_size, feature_dim].
pub fn apply_layer_norm(inputs: &Array2<f64>, epsilon: f64) -> Array2<f64> {
	let mean = inputs.mean_axis(Axis(1)).unwrap();
	let variance = inputs.var_axis(Axis(1), 0.0);

	let mut normed = inputs.clone();
	for ((mut row, &m), &v) in normed.outer_iter_mut().zip(mean.iter()).zip(variance.iter()) {
			let std = (v + epsilon).sqrt();
			row.mapv_inplace(|x| (x - m) / std);
	}
	normed
}



pub fn test_apply_layer_norm() {
    let inputs = Array2::from_shape_vec(
        (2, 4),
        vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
    )
    .unwrap();
    let epsilon = 1e-5;

    let normalized = apply_layer_norm(&inputs, epsilon);

   
    for row in normalized.outer_iter() {
        let mean: f64 = row.mean().unwrap();
        let variance: f64 = row.var(1e-5);
        assert!((mean - 0.0).abs() < 1e-6, "Mean is not zero!");
        assert!((variance - 1.0).abs() < 1e-5, "Variance is not one!");
    }

    println!("LayerNorm test passed!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_basic() {
        test_apply_layer_norm();
    }
}
