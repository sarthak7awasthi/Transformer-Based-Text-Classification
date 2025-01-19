# Data Loader Module

## Overview

The `data_loader` module provides functionality for parsing datasets into tokenized and padded inputs, along with corresponding labels, for use in Transformer-based models. It supports multiple dataset formats (e.g., JSON and CSV), making it versatile for a variety of text classification tasks. The module integrates with the tokenizer to preprocess raw text into numerical sequences and handles batching to prepare data for efficient training and testing.

## Purpose

In machine learning pipelines, data preprocessing is critical to ensure raw datasets are converted into model-compatible formats. The `data_loader` module facilitates this by:

- Parsing raw datasets from supported file formats
- Tokenizing and padding text data using a tokenizer
- Converting labels into numerical representations for classification
- Preparing data batches for training and evaluation

This module bridges the gap between raw datasets and model training by automating preprocessing, reducing manual intervention.

## Supported File Formats

### CSV Format

- Each row contains a `text` field and a `label` field
- Example:

```
text,label
"Win a free iPhone now!",spam
"Meeting scheduled for 10 AM tomorrow.",not spam
```

### JSON Format

- A list of objects, each containing `text` and `label` keys
- Example:

```json
[
  { "text": "Win a free iPhone now!", "label": "spam" },
  { "text": "Meeting scheduled for 10 AM tomorrow.", "label": "not spam" }
]
```

## Key Functionalities

### File Parsing

- Detects file format based on extension (e.g., `.csv`, `.json`)
- Reads and extracts text and label fields from the dataset

### Tokenization and Padding

- Uses the tokenizer to convert text into tokenized and padded sequences of fixed length

### Label Conversion

- Maps string labels (e.g., "spam", "not spam") to numerical categories (e.g., `1`, `0`)

### Batch Preparation

- Divides tokenized data and labels into batches for efficient processing during training and testing

## Mathematical Foundation

### Tokenization and Padding

Given an input sentence, the tokenizer splits it into tokens, maps each token to an integer using a vocabulary, and applies padding to ensure all sequences are of equal length.

Let:

- `text` be the input sentence
- `vocab(token)` be the vocabulary index for a token
- `PAD_INDEX` be the index for the padding token
- `MAX_SEQ_LENGTH` be the fixed sequence length

**Tokenization**: For a sentence `text`:

1. Split into tokens: `tokens = preprocess(text)`
2. Map tokens to indices: `token_indices = [vocab(token) for token in tokens]`

**Padding**: To ensure uniform sequence length:

```python
if len(token_indices) < MAX_SEQ_LENGTH:
    token_indices.extend([PAD_INDEX] * (MAX_SEQ_LENGTH - len(token_indices)))
else:
    token_indices = token_indices[:MAX_SEQ_LENGTH]
```

### Label Conversion

Labels are converted into numerical categories for use in classification. For example:

- `"spam" -> 1`
- `"not spam" -> 0`

## Key Properties

### Versatility

- Supports multiple input formats (CSV, JSON)
- Handles diverse text classification tasks with configurable tokenization and padding

### Scalability

- Efficient batch preparation for large datasets
- Seamless integration with Transformer-based models

### Error Handling

- Provides meaningful errors for unsupported file formats or missing fields
