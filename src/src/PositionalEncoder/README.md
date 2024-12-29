# Positional Encoding Module

## Overview

This module, positional_encoding.rs, provides functionality for generating positional encodings for a sequence of tokens. Positional encoding is a key component of Transformer models, as it provides information about the order of tokens in a sequence, which is otherwise lost due to the model's parallel nature.

## Purpose

In Transformers, tokens are processed in parallel, unlike Recurrent Neural Networks (RNNs), which inherently capture sequential information. Positional encoding addresses this limitation by injecting positional information into token embeddings, allowing the model to consider the order of tokens.

## Mathematical Foundation

The positional encoding values are generated using sine and cosine functions. For a given position  and embedding dimension :

1. For even dimensions (dim % 2 == 0):

  PE(pos, 2i)= $\sin\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$


2. For odd dimensions (dim % 2 == 1):

  PE(pos, 2i) = $\cos\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$

Where:

pos : Position of the token in the sequence (e.g., 0, 1, 2, ...).

i : Index of the embedding dimension (e.g., 0, 1, 2, ...).

d_model: Total dimensionality of the embedding vector.

10000: A fixed scaling factor that spreads the sine and cosine values across dimensions.

## Key Properties

**Smooth Variation**: The sine and cosine functions ensure smooth transitions between positions.

**Relative Positioning**: The difference between positional encodings at positions  and  reflects their relative positional relationships.

**Generalization**: The periodic nature of sine and cosine functions allows the model to generalize to unseen sequence lengths during inference.
