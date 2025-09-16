use {
    crate::{error::GSVError, onnx_builder::create_onnx_cpu_session, text::utils::BERT_TOKENIZER},
    log::{debug, warn},
    ndarray::{Array1, Array2, Axis, concatenate},
    ort::{inputs, value::Tensor},
    std::{path::Path, str::FromStr, sync::Arc},
    tokenizers::Tokenizer,
};

#[derive(Debug)]
pub struct BertModel {
    model: Option<ort::session::Session>,
    tokenizers: Option<Arc<tokenizers::Tokenizer>>,
}

impl BertModel {
    pub fn new<P: AsRef<Path>>(path: Option<P>) -> Result<Self, GSVError> {
        let mut model = None;
        if let Some(path) = path {
            model = Some(create_onnx_cpu_session(path)?);
        }
        Ok(Self {
            model: model,
            tokenizers: Some(Arc::new(Tokenizer::from_str(BERT_TOKENIZER).unwrap())),
        })
    }

    pub fn get_bert(
        &mut self,
        text: &str,
        word2ph: &[i32],
        total_phones: usize,
    ) -> Result<Array2<f32>, GSVError> {
        if self.model.is_some() && self.tokenizers.is_some() {
            let tmp = self.get_real_bert(text, word2ph)?;
            debug!("use real bert, {}", text);
            if tmp.shape()[0] != total_phones {
                warn!(
                    "tmp.shape()[0]: {} != total_phones: {}, use empty",
                    tmp.shape()[0],
                    total_phones
                );
                return Ok(self.get_fake_bert(total_phones));
            }
            Ok(tmp)
        } else {
            debug!("use empty bert, {}", text);
            Ok(self.get_fake_bert(total_phones))
        }
    }

    fn get_real_bert(&mut self, text: &str, word2ph: &[i32]) -> Result<Array2<f32>, GSVError> {
        let tokenizer = self.tokenizers.as_ref().unwrap();
        let session = self.model.as_mut().unwrap();

        let encoding = tokenizer.encode(text, true).unwrap();
        let (input_ids, attention_mask, token_type_ids): (Vec<i64>, Vec<i64>, Vec<i64>) = (
            encoding.get_ids().iter().map(|&id| id as i64).collect(),
            encoding
                .get_attention_mask()
                .iter()
                .map(|&m| m as i64)
                .collect(),
            encoding.get_type_ids().iter().map(|&t| t as i64).collect(),
        );

        let inputs = inputs![
            "input_ids" => Tensor::from_array(Array2::from_shape_vec((1, input_ids.len()), input_ids).unwrap()).unwrap(),
            "attention_mask" => Tensor::from_array(Array2::from_shape_vec((1, attention_mask.len()), attention_mask).unwrap()).unwrap(),
            "token_type_ids" => Tensor::from_array(Array2::from_shape_vec((1, token_type_ids.len()), token_type_ids).unwrap()).unwrap()
        ];

        let bert_out = session.run(inputs)?;
        let bert_feature = bert_out["bert_feature"]
            .try_extract_array::<f32>()?
            .to_owned();

        let bert_feature_2d: Array2<f32> = bert_feature.into_dimensionality()?;

        Ok(build_phone_level_feature(
            bert_feature_2d,
            Array1::from_vec(word2ph.to_vec()),
        ))
    }

    fn get_fake_bert(&self, total_phones: usize) -> Array2<f32> {
        // The BERT model outputs features of size 1024
        Array2::<f32>::zeros((total_phones, 1024))
    }
}

// Helper function to expand word-level features to phone-level features.
// This function is required by get_real_bert.
fn build_phone_level_feature(res: Array2<f32>, word2ph: Array1<i32>) -> Array2<f32> {
    let phone_level_features = word2ph
        .into_iter()
        .enumerate()
        .map(|(i, count)| {
            if i < res.dim().0 {
                let row = res.row(i);
                Array2::from_shape_fn((count as usize, res.ncols()), |(_j, k)| row[k])
            } else {
                // If word2ph has more elements than res rows, duplicate the last feature.
                let last_row = res.row(res.dim().0 - 1);
                Array2::from_shape_fn((count as usize, res.ncols()), |(_j, k)| last_row[k])
            }
        })
        .collect::<Vec<_>>();

    concatenate(
        Axis(0),
        &phone_level_features
            .iter()
            .map(|x| x.view())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}
