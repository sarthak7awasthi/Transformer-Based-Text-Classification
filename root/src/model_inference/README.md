# Inference Module

## Overview

The `inference.rs` module facilitates the usage of a trained Transformer model to generate predictions for new, unseen input data. It supports tokenizing input text, running it through the model, and interpreting the results by providing class predictions along with their probabilities.

---

## Purpose

The `Inference` module enables:

1. **Utilizing Trained Models**: Loads a pre-trained Transformer model from a saved file.
2. **Text Classification**: Processes raw text inputs and predicts their corresponding class labels.
3. **Probability Distribution**: Outputs a probability distribution across all possible classes, aiding interpretability.

---

## Methodology

### Workflow

1. **Model Loading**:

   - Loads the Transformer model from a specified path using `Transformer::load`.

2. **Text Preprocessing**:

   - Tokenizes raw text inputs using the `Tokenizer` module.
   - Pads or truncates the tokens to ensure a uniform sequence length.

3. **Forward Pass**:

   - Converts tokenized inputs into a numerical format suitable for the model.
   - Performs a forward pass through the Transformer model to obtain logits.

4. **Softmax Computation**:

   - Applies the softmax function to the logits to generate a probability distribution.

5. **Class Prediction**:
   - Identifies the class with the highest probability as the predicted label.

---

## Key Functions

### `new(model_path: &str, tokenizer: &'a Tokenizer) -> Result<Self, Box<dyn Error>>`

Creates a new `Inference` instance with:

- A Transformer model loaded from the specified `model_path`.
- A reference to the `Tokenizer` instance used for input processing.

### `predict(&self, input_text: &str) -> Result<(usize, Vec<f64>), Box<dyn Error>>`

Performs inference on a single input string.

#### Steps:

1. Tokenizes and pads the input text.
2. Runs the tokenized input through the Transformer model.
3. Computes the softmax probabilities of all classes.
4. Returns:
   - The predicted class index.
   - A vector of probabilities for all classes.

---

## Mathematical Foundation

### Softmax Function

Converts logits into probabilities:
\[
P(y*i) = \frac{e^{x_i}}{\sum*{j} e^{x_j}}
\]
Where:

- \( x_i \): Logit for class \( i \).
- \( P(y_i) \): Predicted probability for class \( i \).

### Predicted Class

The predicted class \( c \) is the index of the maximum probability:
\[
c = \text{argmax}(P(y))
\]

---

## Key Properties

1. **Simplicity**: Provides an intuitive API for generating predictions from raw text inputs.
2. **Flexibility**: Handles arbitrary input lengths by tokenizing and padding.
3. **Interpretability**: Outputs both predicted labels and class probabilities.

---

## Example Usage

### Performing Inference

```rust
use model_inference::inference::Inference;

let tokenizer = Tokenizer::new(vocab.clone(), 128);
let inference = Inference::new("trained_model.json", &tokenizer)?;

let input_text = "hello world";
let (predicted_class, probabilities) = inference.predict(input_text)?;
println!("Predicted Class: {}", predicted_class);
println!("Probabilities: {:?}", probabilities);
```

---
