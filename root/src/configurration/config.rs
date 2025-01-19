pub const MAX_SEQ_LENGTH: usize = 128; // Maximum sequence length for tokenized inputs
pub const BATCH_SIZE: usize = 32;      // Batch size for model training

// Special Tokens
pub const PAD_TOKEN: &str = "[PAD]";
pub const UNK_TOKEN: &str = "[UNK]";
pub const CLS_TOKEN: &str = "[CLS]";
pub const SEP_TOKEN: &str = "[SEP]";

// Optimizer Hyperparameters
pub const LEARNING_RATE: f64 = 0.001; // Learning rate for the optimizer
pub const BETA1: f64 = 0.9;           // Momentum for Adam
pub const BETA2: f64 = 0.999;         // Second moment for Adam
pub const EPSILON: f64 = 1e-8;        // To prevent division by zero
