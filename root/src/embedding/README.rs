# Embeddings Module

A Rust implementation of token embeddings and positional encodings for Transformer models.

## Overview

The `embeddings.rs` module provides essential functionality for converting discrete tokens into continuous vector representations, enhanced with positional information. This forms the foundational input layer for Transformer-based models, enabling them to process sequential text data effectively.

## Features

- Token to dense vector mapping
- Sinusoidal positional encodings
- Combined embeddings with positional information
- Efficient matrix operations for embedding lookups
- Support for variable sequence lengths

## Mathematical Foundation

### Token Embeddings

Tokens are mapped to dense vectors using a learnable embedding matrix E of shape V × d_model:

```
embedding(t) = E[t]
```

where t is the token index.

### Positional Encodings

Position information is encoded using sinusoidal functions:

For even dimensions (i % 2 = 0):
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
```

For odd dimensions (i % 2 = 1):
```
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Combined Output

The final embedding combines both components:
```
FinalEmbedding = TokenEmbedding + PositionalEncoding
```

## Configuration

The module can be configured with the following parameters:

- `vocab_size`: Size of the vocabulary
- `d_model`: Dimension of the embedding vectors
- `max_sequence_length`: Maximum supported sequence length
- `dropout_rate`: Optional dropout rate for regularization

## Key Properties

1. **Semantic Representation**
   - Captures semantic relationships between tokens
   - Similar tokens have similar vector representations

2. **Position Awareness**
   - Maintains sequence order information
   - Handles variable-length sequences
   - Generalizes to unseen sequence lengths

3. **Performance**
   - Efficient matrix operations
   - Optimized memory usage
   - Support for batch processing

## Integration

This module serves as the first layer in a Transformer architecture, transforming raw token sequences into vector representations that can be processed by subsequent attention and feed-forward layers.

