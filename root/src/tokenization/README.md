# Tokenizer Module

A Rust module for efficient text tokenization, vocabulary management, and sequence padding in Natural Language Processing (NLP) pipelines.

## Overview

The `tokenizer.rs` module provides essential functionality for converting raw text into machine-readable numerical sequences. It serves as a crucial preprocessing step for NLP tasks, handling tokenization, vocabulary management, and sequence padding.

## Features

### Dynamic Vocabulary Management

- Runtime vocabulary creation from input datasets
- Configurable maximum vocabulary size
- Built-in special tokens (`[PAD]`, `[UNK]`)
- Efficient token-to-index mapping

### Text Processing

- Whitespace-based text tokenization
- Automatic handling of unknown tokens
- Sequence padding and truncation to uniform length
- Batch processing support

### Mathematical Foundation

The module implements these core operations:

1. **Tokenization**: `T = split(S)`

   - Converts input text `S` into token set `T`
   - Example: `"Hello world!" → {"Hello", "world!"}`

2. **Index Mapping**: `I = {V[t₁], V[t₂], ..., V[tₙ]}`

   - Maps tokens to indices using vocabulary `V`
   - Handles unknown tokens with `[UNK]` index

3. **Sequence Normalization**:
   - Padding: Extends sequences shorter than `MAX_SEQ_LENGTH`
   - Truncation: Cuts sequences longer than `MAX_SEQ_LENGTH`

## Special Tokens

- `[PAD]`: Used for padding sequences to uniform length
- `[UNK]`: Represents tokens not found in vocabulary
