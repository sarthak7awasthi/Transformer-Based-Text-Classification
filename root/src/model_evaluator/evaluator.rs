use crate::transformer::Transformer;
use crate::data_handler::data_loader::DataLoader;
use ndarray::Array2;

pub struct Evaluator<'a> {
    pub model: Transformer,
    pub data_loader: &'a DataLoader<'a>,
}

impl<'a> Evaluator<'a> {
    /// Creates a new Evaluator instance.
    pub fn new(model_path: &str, data_loader: &'a DataLoader) -> Result<Self, std::io::Error> {
        let model = Transformer::load(model_path)?;
        Ok(Evaluator { model, data_loader })
    }

   
    pub fn evaluate(&self, dataset_path: &str) -> Result<(), Box<dyn std::error::Error>> {
   
        let (inputs, labels) = self.data_loader.load_dataset(dataset_path)?;

      
        let batch_array = Array2::from_shape_vec(
            (inputs.len(), inputs[0].len()),
            inputs.iter().flatten().map(|&x| x as f64).collect(),
        )?;

   
        let logits = self.model.forward(&batch_array);

        let accuracy = self.compute_accuracy(&logits, &labels);
        println!("Accuracy: {:.2}%", accuracy * 100.0);

        let metrics = self.compute_metrics(&logits, &labels);
        println!(
            "Precision: {:.2}%, Recall: {:.2}%, F1-Score: {:.2}%",
            metrics.0 * 100.0,
            metrics.1 * 100.0,
            metrics.2 * 100.0
        );

        Ok(())
    }

  
    fn compute_accuracy(&self, logits: &Array2<f64>, labels: &[usize]) -> f64 {
        let correct_predictions = logits
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
            .count();

        correct_predictions as f64 / labels.len() as f64
    }

  
    fn compute_metrics(&self, logits: &Array2<f64>, labels: &[usize]) -> (f64, f64, f64) {
        let num_classes = logits.shape()[1];
        let mut true_positives = vec![0; num_classes];
        let mut false_positives = vec![0; num_classes];
        let mut false_negatives = vec![0; num_classes];

        for (logit, &label) in logits.outer_iter().zip(labels.iter()) {
            let predicted_label = logit
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(index, _)| index)
                .unwrap_or(0);

            if predicted_label == label {
                true_positives[label] += 1;
            } else {
                false_positives[predicted_label] += 1;
                false_negatives[label] += 1;
            }
        }

        let precision: f64 = true_positives
            .iter()
            .zip(false_positives.iter())
            .map(|(tp, fp)| *tp as f64 / (*tp + *fp).max(1) as f64)
            .sum::<f64>()
            / num_classes as f64;

        let recall: f64 = true_positives
            .iter()
            .zip(false_negatives.iter())
            .map(|(tp, fn_val)| *tp as f64 / (*tp + *fn_val).max(1) as f64)
            .sum::<f64>()
            / num_classes as f64;

        let f1_score = if precision + recall > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        };

        (precision, recall, f1_score)
    }
}
