use crate::configurration::config::BATCH_SIZE;
use crate::tokenization::tokenizer::Tokenizer; 
use std::fs;
use std::path::Path;
use std::error::Error;
use serde_json::Value;

pub struct DataLoader<'a> {
    pub tokenizer: &'a Tokenizer,
}

impl<'a> DataLoader<'a> {
    pub fn new(tokenizer: &'a Tokenizer) -> Self {
        DataLoader { tokenizer }
    }

    pub fn load_dataset(
        &self,
        file_path: &str,
    ) -> Result<(Vec<Vec<usize>>, Vec<usize>), Box<dyn Error>> {
        let path = Path::new(file_path);
        let extension = path.extension().and_then(|ext| ext.to_str());

        match extension {
            Some("csv") => self.load_csv(file_path),
            Some("json") => self.load_json(file_path),
            _ => Err(format!("Unsupported file format: {:?}", extension).into()),
        }
    }

    fn load_csv(
        &self,
        file_path: &str,
    ) -> Result<(Vec<Vec<usize>>, Vec<usize>), Box<dyn Error>> {
        let mut reader = csv::Reader::from_path(file_path)?;
        let mut inputs = Vec::new();
        let mut labels = Vec::new();

        for result in reader.records() {
            let record = result?;
            let text = record.get(0).ok_or("Missing text field")?;
            let label: usize = record.get(1)
                .ok_or("Missing label field")?
                .parse()?;

            inputs.push(self.tokenizer.tokenize_and_pad_batch(&[text.to_string()])[0].clone());
            labels.push(label);
        }

        Ok((inputs, labels))
    }

    fn load_json(
        &self,
        file_path: &str,
    ) -> Result<(Vec<Vec<usize>>, Vec<usize>), Box<dyn Error>> {
        let file_content = fs::read_to_string(file_path)?;
        let data: Value = serde_json::from_str(&file_content)?;

        let mut inputs = Vec::new();
        let mut labels = Vec::new();

        if let Some(array) = data.as_array() {
            for item in array {
                let text = item.get("text")
                    .and_then(|v| v.as_str())
                    .ok_or("Missing text field in JSON entry")?;
                let label = item.get("label")
                    .ok_or("Missing label field in JSON entry")?
                    .as_u64()
                    .ok_or("Label must be a number")?;

                inputs.push(self.tokenizer.tokenize_and_pad_batch(&[text.to_string()])[0].clone());
                labels.push(label as usize);
            }
        }

        Ok((inputs, labels))
    }

    pub fn create_batches(
        &self,
        inputs: Vec<Vec<usize>>,
        labels: Vec<usize>,
    ) -> Vec<(Vec<Vec<usize>>, Vec<usize>)> {
        inputs
            .chunks(BATCH_SIZE)
            .zip(labels.chunks(BATCH_SIZE))
            .map(|(input_chunk, label_chunk)| {
                (input_chunk.to_vec(), label_chunk.to_vec())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::configurration::config::{PAD_TOKEN, UNK_TOKEN};

    #[test]
    fn test_data_loader() {
        let vocab = HashMap::from([
            (PAD_TOKEN.to_string(), 0),
            (UNK_TOKEN.to_string(), 1),
            ("hello".to_string(), 2),
            ("world".to_string(), 3),
        ]);

        let tokenizer = Tokenizer::new(vocab, 128);
        let data_loader = DataLoader::new(&tokenizer);

        // Test with JSON file
        let result = data_loader.load_dataset("src/test_dataset.json");
        assert!(result.is_ok());
    }
}