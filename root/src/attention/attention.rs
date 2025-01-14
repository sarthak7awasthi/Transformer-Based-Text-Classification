

// todo(for self): try to make this more optimal. taking too long to execute for bigger inputs.




/// module for attention mechanism
/// Functional: scaled_dot_product_attention: Computes the scaled dot-product attention for a set of queries, keys, and values.
/// Parameters:
///           "query": The Q matrix (Vec<Vec<f32>>) representing the query vectors.
///           "key": The K matrix (Vec<Vec<f32>>) representing the key vectors.
///           "value": The V matrix (Vec<Vec<f32>>) representing the value vectors.
/// Return:    A 2D vector (Vec<Vec<f32>>) representing the attention-weighted output for each query.


pub fn scaled_dot_product_attention(
	query: &[Vec<f32>],
	key: &[Vec<f32>],



	value: &[Vec<f32>],
) -> Vec<Vec<f32>> {
	assert!(!query.is_empty() && !key.is_empty() && !value.is_empty(), "Inputs must be non-empty.");
	assert_eq!(query[0].len(), key[0].len(), "Query and Key dimensions must match.");
	assert_eq!(key.len(), value.len(), "Key and Value must have the same number of tokens.");

	let d_k = key[0].len() as f32;


	let mut qk_transpose = vec![vec![0.0; key.len()]; query.len()];
	for i in 0..query.len() {
			for j in 0..key.len() {
					qk_transpose[i][j] = query[i]
							.iter()
							.zip(&key[j])
							.map(|(q, k)| q * k)
							.sum();
			}
	}


	for row in &mut qk_transpose {
			for val in row.iter_mut() {
					*val /= d_k.sqrt();
			}
	}

	// Applying softmax for probability distribution
	for row in &mut qk_transpose {
			let max_val = row.iter().cloned().fold(f32::MIN, f32::max); 

			let sum_exp: f32 = row.iter().map(|val| (val - max_val).exp()).sum();
			for val in row.iter_mut() {
					*val = (*val - max_val).exp() / sum_exp;
			}
	}


	let mut attention_output = vec![vec![0.0; value[0].len()]; query.len()];
	for i in 0..qk_transpose.len() {
			for j in 0..value[0].len() {
					attention_output[i][j] = qk_transpose[i]
							.iter()
							.zip(value.iter())
							.map(|(weight, v)| weight * v[j])
							.sum();
			}
	}

	attention_output
}

/// module for multi-head attention
/// Functional: multi_head_attention: Implements multi-head attention by splitting inputs into multiple heads, computing scaled dot-product attention for each, and concatenating the results.
/// Parameters:
///           "query": The Q matrix (Vec<Vec<f32>>) representing the query vectors.
///           "key": The K matrix (Vec<Vec<f32>>) representing the key vectors.
///           "value": The V matrix (Vec<Vec<f32>>) representing the value vectors.
///           "num_heads": The number of attention heads (usize) for the computation.
/// Return:    A 2D vector (Vec<Vec<f32>>) representing the concatenated and projected multi-head attention output.
pub fn multi_head_attention(
	query: &[Vec<f32>],
	key: &[Vec<f32>],
	value: &[Vec<f32>],
	num_heads: usize,
) -> Vec<Vec<f32>> {
	assert!(!query.is_empty() && !key.is_empty() && !value.is_empty(), "Inputs must be non-empty.");
	assert_eq!(query[0].len(), key[0].len(), "Query and Key dimensions must match.");
	assert_eq!(key.len(), value.len(), "Key and Value must have the same number of tokens.");

	let d_model = query[0].len(); 
	assert_eq!(d_model % num_heads, 0, "d_model must be divisible by num_heads");

	let head_dim = d_model / num_heads;


	let mut query_heads = vec![vec![vec![0.0; head_dim]; query.len()]; num_heads];
	let mut key_heads = vec![vec![vec![0.0; head_dim]; key.len()]; num_heads];
	let mut value_heads = vec![vec![vec![0.0; head_dim]; value.len()]; num_heads];

	for h in 0..num_heads {
			for i in 0..query.len() {
					for j in 0..head_dim {
							query_heads[h][i][j] = query[i][h * head_dim + j];
					}
			}
			for i in 0..key.len() {
					for j in 0..head_dim {
							key_heads[h][i][j] = key[i][h * head_dim + j];
					}
			}
			for i in 0..value.len() {
					for j in 0..head_dim {
							value_heads[h][i][j] = value[i][h * head_dim + j];
					}
			}
	}


	let mut head_outputs = vec![vec![vec![0.0; head_dim]; query.len()]; num_heads];
	for h in 0..num_heads {
			head_outputs[h] = scaled_dot_product_attention(&query_heads[h], &key_heads[h], &value_heads[h]);
	}

	
	let mut concatenated = vec![vec![0.0; d_model]; query.len()];
	for i in 0..query.len() {
			for h in 0..num_heads {
					for j in 0..head_dim {
							concatenated[i][h * head_dim + j] = head_outputs[h][i][j];
					}
			}
	}

	concatenated
}
