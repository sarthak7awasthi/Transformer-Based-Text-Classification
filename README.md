# Transformer-Based Text Classification

## Overview
In this project I implemented a **Transformer-based text classification pipeline from scratch in Rust without any external Machine learning libraries.**
It is designed to classify textual inputs into predefined categories using advanced Natural Language Processing (NLP) techniques. Currently I built it for spam detection use case but can be easily changed for applications of sentiment analysis, topic categorization, content moderation, etc.
Each module has its own detailed README file. Below is a high-level summary of the modules, followed by a comprehensive description of the configuration settings.

---

## Key Features
- **Custom Transformer Model**: A fully implemented Transformer architecture optimized for text classification tasks.
- **End-to-End Pipeline**: Includes modules for tokenization, training, evaluation, and inference.
- **Scalability**: Designed for extensibility, enabling support for more advanced features or larger datasets.

---

## Project Modules

### 1. **Positional Encoding Module**
This module generates positional encodings to inject sequential information into token embeddings. The encodings are calculated using sine and cosine functions.

- **Purpose**: Provides position-based context to the model.
- [Read Full Documentation](https://github.com/sarthak7awasthi/Transformer/tree/main/root/src/positional_encoding)

---

### 2. **Attention Module**
Implements the scaled dot-product attention and multi-head attention mechanisms, which are critical for enabling the Transformer to focus on relevant parts of the input sequence.

- **Purpose**: Allows the model to weigh the importance of different tokens dynamically.
- [Read Full Documentation](https://github.com/sarthak7awasthi/Transformer/tree/main/root/src/attention)

---

### 3. **Feed Forward Module**
Defines a feed-forward neural network used within the Transformer layers. This module processes attention outputs with non-linearity for feature transformation.

- **Purpose**: Applies dense layers with activation functions.
- [Read Full Documentation](https://github.com/sarthak7awasthi/Transformer/tree/main/root/src/feed_forward)

---

### 4. **Layer Normalization Module**
Applies layer normalization to stabilize the training process by normalizing inputs within each layer.

- **Purpose**: Prevents internal covariate shifts during training.
- [Read Full Documentation](https://github.com/sarthak7awasthi/Transformer/tree/main/root/src/layer_norm)

---

### 5. **Transformer Encoder Module**
Combines attention, feed-forward, and normalization mechanisms to form the building blocks of the Transformer model.

- **Purpose**: Represents a single Transformer encoder layer.
- [Read Full Documentation](https://github.com/sarthak7awasthi/Transformer/tree/main/root/src/encoder)

---

### 6. **Embedding Module**
Handles token embedding and integrates positional encodings into the input sequence representation.

- **Purpose**: Converts tokens into dense vectors with positional context.
- [Read Full Documentation](https://github.com/sarthak7awasthi/Transformer/tree/main/root/src/embedding)

---

### 7. **Transformer Module**
Integrates multiple encoder layers and provides a complete Transformer architecture for text classification.

- **Purpose**: Acts as the core model architecture.
- [Read Full Documentation](https://github.com/sarthak7awasthi/Transformer/tree/main/root/src/transformer)

---

### 8. **Cross-Entropy Loss Module**
Implements cross-entropy loss for multi-class classification tasks, providing both loss computation and gradient calculations.

- **Purpose**: Guides the optimization process by computing classification errors.
- [Read Full Documentation](https://github.com/sarthak7awasthi/Transformer/tree/main/root/src/cross_entropy)

---

### 9. **Optimizer Module**
Implements optimization algorithms like Stochastic Gradient Descent (SGD) and Adam for updating model parameters during training.

- **Purpose**: Efficiently minimizes the loss function.
- [Read Full Documentation](https://github.com/sarthak7awasthi/Transformer/tree/main/root/src/model_optimizer)

---

### 10. **Data Loader Module**
Facilitates the loading, batching, and preprocessing of datasets for training and evaluation.

- **Purpose**: Manages dataset handling for input to the model.
- [Read Full Documentation](https://github.com/sarthak7awasthi/Transformer/tree/main/root/src/data_handler)

---

### 11. **Trainer Module**
Handles the training loop, including forward and backward passes, loss computation, parameter updates, and metric tracking.

- **Purpose**: Automates the training process for multiple epochs.
- [Read Full Documentation](https://github.com/sarthak7awasthi/Transformer/tree/main/root/src/training)

---

### 12. **Evaluator Module**
Evaluates the model on a test dataset, computing metrics such as accuracy, precision, recall, and F1-score.

- **Purpose**: Validates model performance on unseen data.
- [Read Full Documentation](https://github.com/sarthak7awasthi/Transformer/tree/main/root/src/model_evaluator)

---

### 13. **Inference Module**
Runs predictions on new inputs, providing both class labels and probability distributions for each prediction.

- **Purpose**: Enables practical usage of the trained model.
- [Read Full Documentation](https://github.com/sarthak7awasthi/Transformer/tree/main/root/src/model_inference)

---

## Configuration

The configuration settings are defined in the `config.rs` file and are crucial for controlling model behavior, training dynamics, and tokenization. Below are the key parameters:

### **Tokenization Settings**
- **`MAX_SEQ_LENGTH`**: Maximum length of input sequences (default: 128).
- **`PAD_TOKEN`**: Padding token (`[PAD]`) used to ensure uniform sequence lengths.
- **`UNK_TOKEN`**: Unknown token (`[UNK]`) for handling out-of-vocabulary words.
- **`CLS_TOKEN`**: Classification token (`[CLS]`) added at the start of each input sequence.
- **`SEP_TOKEN`**: Separator token (`[SEP]`) added between sentence pairs.

### **Training Parameters**
- **`BATCH_SIZE`**: Number of samples processed simultaneously during training (default: 32).
- **`LEARNING_RATE`**: Learning rate for the optimizer (default: 0.001).
- **`BETA1`**: Beta1 parameter for the Adam optimizer (default: 0.9).
- **`BETA2`**: Beta2 parameter for the Adam optimizer (default: 0.999).
- **`EPSILON`**: Small constant for numerical stability in Adam updates (default: 1e-8).

---

## Project Workflow

1. **Tokenization**:
   - Preprocesses text into tokenized and padded sequences.

2. **Model Training**:
   - Uses the `Trainer` module to train the Transformer model on the training dataset.

3. **Evaluation**:
   - Validates the model’s performance using the `Evaluator` module.

4. **Inference**:
   - Deploys the trained model for predictions using the `Inference` module.

---

## Getting Started

### Prerequisites
- **Rust**: Ensure Rust is installed. Install it using:
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

### Setting Up the Project
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd transformer_project
   ```

2. Build the project:
   ```bash
   cargo build
   ```

3. Run the project:
   ```bash
   cargo run
   ```

---

## Future Improvements
1. **Multi-GPU Support**: Enable distributed training.
2. **Visualization Tools**: Add support for attention heatmaps and loss graphs.
3. **Pre-trained Models**: Incorporate pre-trained weights for fine-tuning.

---



