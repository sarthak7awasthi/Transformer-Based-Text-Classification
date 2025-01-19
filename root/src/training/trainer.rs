use crate::data_handler::data_loader::DataLoader;
use crate::cross_entropy::loss::Loss;
use crate::model_optimizer::optimizer::Optimizer;
use crate::transformer::Transformer;
use crate::configurration::config::{BATCH_SIZE, LEARNING_RATE};
use ndarray::Array2;

pub struct Trainer<'a> {
    pub model: Transformer,
    pub optimizer: Optimizer,
    pub data_loader: &'a DataLoader<'a>,
    pub epochs: usize,
}

impl<'a> Trainer<'a> {
    /// Create a new Trainer instance.
    pub fn new(
        model: Transformer,
        optimizer: Optimizer,
        data_loader: &'a DataLoader,
        epochs: usize,
    ) -> Self {
        Trainer {
            model,
            optimizer,
            data_loader,
            epochs,
        }
    }

    /// Train the model over the specified number of epochs.
    pub fn train(&mut self, dataset_path: &str) {
        // Load dataset
        let (inputs, labels) = self.data_loader.load_dataset(dataset_path).unwrap();
        let batches = self.data_loader.create_batches(inputs, labels);

        for epoch in 0..self.epochs {
            println!("Epoch {}/{}", epoch + 1, self.epochs);

            let mut epoch_loss = 0.0;
            let mut correct_predictions = 0;
            let mut total_samples = 0;

            for (batch_inputs, batch_labels) in &batches {
                // Convert inputs into a format compatible with the model
                let batch_array: Array2<f64> = Array2::from_shape_vec(
									(batch_inputs.len(), batch_inputs[0].len()),
									batch_inputs.iter().flatten().map(|&x| x as f64).collect(),
								).unwrap();
							

                // Forward pass
                let logits = self.model.forward(&batch_array);

                // Compute loss
                let loss = Loss::cross_entropy_loss(&logits, batch_labels);
                epoch_loss += loss;

                // Compute gradients
                let gradients = Loss::gradients(&logits, batch_labels);

                // Update parameters
                let mut params = self.model.parameters_mut();
                for (param, grad) in params.iter_mut().zip(gradients.iter()) {
									**param -= LEARNING_RATE * grad;
								}
							

                // Compute accuracy
                correct_predictions += self.compute_correct_predictions(&logits, batch_labels);
                total_samples += batch_labels.len();
            }

            let epoch_accuracy = correct_predictions as f64 / total_samples as f64;
            println!(
                "Epoch {}: Loss: {:.4}, Accuracy: {:.2}%",
                epoch + 1,
                epoch_loss / batches.len() as f64,
                epoch_accuracy * 100.0
            );
        }
    }

    /// Compute the number of correct predictions in a batch.
    fn compute_correct_predictions(&self, logits: &Array2<f64>, labels: &[usize]) -> usize {
        logits
            .outer_iter()
            .zip(labels.iter())
            .filter(|(logit, &label)| {
                let predicted_label = logit
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(index, _)| index)
                    .unwrap_or(0);
                predicted_label == label
            })
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::TransformerConfig;
    use crate::model_optimizer::optimizer::OptimizerType;

    #[test]
    fn test_trainer() {
        let vocab = std::collections::HashMap::from([
            ("hello".to_string(), 0),
            ("world".to_string(), 1),
            ("<UNK>".to_string(), 2),
        ]);

        let config = TransformerConfig {
            num_layers: 2,
            d_model: 4,
            num_heads: 2,
            ff_dim: 8,
            num_classes: 2,
            epsilon: 1e-6,
        };

        let transformer = Transformer::new(config, vocab);
        let optimizer = Optimizer::new(OptimizerType::SGD);
        let tokenizer = crate::tokenization::tokenizer::Tokenizer::new(vocab, 128);
        let data_loader = DataLoader::new(&tokenizer);

        let mut trainer = Trainer::new(transformer, optimizer, &data_loader, 3);

        trainer.train("mock_dataset.json");
    }
}
