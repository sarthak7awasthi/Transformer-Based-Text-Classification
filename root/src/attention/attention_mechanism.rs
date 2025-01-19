use ndarray::{Array2, Axis, s};
use ndarray::Zip;

/// Functional: `scaled_dot_product_attention`
/// Computes the scaled dot-product attention for a set of queries, keys, and values.
///
/// Parameters:
///   - `query`: The Q matrix (`Array2<f64>`) representing the query vectors.
///   - `key`: The K matrix (`Array2<f64>`) representing the key vectors.
///   - `value`: The V matrix (`Array2<f64>`) representing the value vectors.
///
/// Return:
///   A matrix (`Array2<f64>`) representing the attention-weighted output.
pub fn scaled_dot_product_attention(
	query: &Array2<f64>,
	key: &Array2<f64>,
	value: &Array2<f64>,
) -> Array2<f64> {
	assert_eq!(query.shape()[1], key.shape()[1], "Query and Key dimensions must match.");
	assert_eq!(key.shape()[0], value.shape()[0], "Key and Value must have the same number of tokens.");

	let d_k = key.shape()[1] as f64;

	let mut qk_transpose = query.dot(&key.t());
	qk_transpose.mapv_inplace(|x| x / d_k.sqrt());

	// Apply softmax
	for mut row in qk_transpose.outer_iter_mut() {
			let max = row.iter().cloned().fold(f64::MIN, f64::max);
			let exp_sum: f64 = row.iter().map(|&x| (x - max).exp()).sum();
			row.mapv_inplace(|x| (x - max).exp() / exp_sum);
	}

	qk_transpose.dot(value)
}

/// Functional: `multi_head_attention`
/// Implements multi-head attention by splitting inputs into multiple heads, computing scaled dot-product attention for each, and concatenating the results.
///
/// Parameters:
///   - `query`: The Q matrix (`Array2<f64>`) representing the query vectors.
///   - `key`: The K matrix (`Array2<f64>`) representing the key vectors.
///   - `value`: The V matrix (`Array2<f64>`) representing the value vectors.
///   - `num_heads`: The number of attention heads (`usize`) for the computation.
///
/// Return:
///   A matrix (`Array2<f64>`) representing the concatenated and projected multi-head attention output.
pub fn multi_head_attention(
    query: &Array2<f64>,
    key: &Array2<f64>,
    value: &Array2<f64>,
    num_heads: usize,
) -> Array2<f64> {
    assert_eq!(query.ncols() % num_heads, 0, "d_model must be divisible by num_heads");
    let head_dim = query.ncols() / num_heads;

    let split_heads = |matrix: &Array2<f64>| -> Vec<Array2<f64>> {
        (0..num_heads)
            .map(|h| matrix.slice(s![.., h * head_dim..(h + 1) * head_dim]).to_owned())
            .collect()
    };

    let query_heads = split_heads(query);
    let key_heads = split_heads(key);
    let value_heads = split_heads(value);

    let head_outputs: Vec<Array2<f64>> = query_heads
        .iter()
        .zip(&key_heads)
        .zip(&value_heads)
        .map(|((q, k), v)| scaled_dot_product_attention(q, k, v))
        .collect();

    // Concatenate head outputs
    let mut concatenated = Array2::zeros((query.nrows(), query.ncols()));
    for (h, head_output) in head_outputs.iter().enumerate() {
        concatenated
            .slice_mut(s![.., h * head_dim..(h + 1) * head_dim])
            .assign(&head_output);
    }

    concatenated
}
