use {
    crate::{error::GSVError, onnx_builder::create_onnx_cpu_session, text::dict},
    arpabet::Arpabet,
    log::debug,
    ndarray::{Array, s},
    ort::{inputs, session::Session, value::Tensor},
    std::{path::Path, str::FromStr},
    tokenizers::Tokenizer,
};

static MINI_BART_G2P_TOKENIZER: &str = include_str!("tokenizer.mini-bart-g2p.json");

static DECODER_START_TOKEN_ID: u32 = 2;

#[allow(unused)]
static BOS_TOKEN: &str = "<s>";
#[allow(unused)]
static EOS_TOKEN: &str = "</s>";

#[allow(unused)]
static BOS_TOKEN_ID: u32 = 0;
static EOS_TOKEN_ID: u32 = 2;

pub struct G2PEnModel {
    encoder_model: Session,
    decoder_model: Session,
    tokenizer: Tokenizer,
}

impl G2PEnModel {
    pub fn new<P: AsRef<Path>>(encoder_path: P, decoder_path: P) -> Result<Self, GSVError> {
        let encoder_model = create_onnx_cpu_session(encoder_path)?;
        let decoder_model = create_onnx_cpu_session(decoder_path)?;
        let tokenizer = Tokenizer::from_str(MINI_BART_G2P_TOKENIZER)?;

        Ok(Self {
            encoder_model,
            decoder_model,
            tokenizer,
        })
    }

    pub fn get_phoneme(&mut self, text: &str) -> Result<Vec<String>, GSVError> {
        debug!("processing {:?}", text);
        let encoding = self.tokenizer.encode(text, true)?;
        let input_ids = encoding
            .get_ids()
            .iter()
            .map(|x| *x as i64)
            .collect::<Vec<i64>>();
        let mut decoder_input_ids = vec![DECODER_START_TOKEN_ID as i64];

        let input_id_len = input_ids.len();
        let input_ids_tensor =
            Tensor::from_array(Array::from_shape_vec((1, input_id_len), input_ids.clone())?)?;
        let attention_mask_tensor =
            Tensor::from_array(Array::from_elem((1, input_id_len), 1 as i64))?;
        let encoder_outputs = self.encoder_model.run(inputs![
            "input_ids" => input_ids_tensor.clone(),
            "attention_mask" => attention_mask_tensor.clone()
        ])?;

        for _ in 0..50 {
            // Prepare input tensors
            // Run inference

            let encoder_output = encoder_outputs["last_hidden_state"].view();

            let decoder_input_ids_tensor = Tensor::from_array(Array::from_shape_vec(
                (1, decoder_input_ids.len()),
                decoder_input_ids.clone(),
            )?)?;

            let outputs = self.decoder_model.run(inputs![
                "input_ids" => decoder_input_ids_tensor,
                "encoder_attention_mask" => attention_mask_tensor.clone(),
                "encoder_hidden_states" => encoder_output,
            ])?;

            let output_array = outputs["logits"].try_extract_array::<f32>()?;

            // Get the last token's logits
            let last_token_logits = &output_array.slice(s![0, output_array.shape()[1] - 1, ..]);

            // Find the argmax
            let next_token_id = last_token_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as i64)
                .ok_or(GSVError::DecodeTokenFailed)?;

            decoder_input_ids.push(next_token_id);
            if next_token_id == EOS_TOKEN_ID as i64 {
                break;
            }
        }

        let decoder_input_ids = decoder_input_ids
            .iter()
            .map(|x| *x as u32)
            .collect::<Vec<u32>>();
        Ok(self
            .tokenizer
            .decode(&decoder_input_ids, true)?
            .split(" ")
            .map(|v| v.to_owned())
            .collect::<Vec<String>>())
    }
}

pub struct G2pEn {
    model: Option<G2PEnModel>,
    arpabet: Arpabet,
}

impl G2pEn {
    pub fn new<P: AsRef<Path>>(path: Option<P>) -> Result<Self, GSVError> {
        let arpabet = arpabet::load_cmudict().clone();
        if let Some(path) = path {
            let path = path.as_ref();
            Ok(G2pEn {
                model: Some(G2PEnModel::new(
                    path.join("encoder_model.onnx"),
                    path.join("decoder_model.onnx"),
                )?),
                arpabet: arpabet,
            })
        } else {
            Ok(G2pEn {
                model: None,
                arpabet: arpabet,
            })
        }
    }

    pub fn g2p(&mut self, text: &str) -> Result<Vec<String>, GSVError> {
        if let Some(v) = dict::en_word_dict(text) {
            return Ok(v.to_owned());
        }
        match &mut self.model {
            Some(model) => {
                let words = text.split_whitespace();
                let mut phonemes = Vec::new();
                for word in words {
                    let phones = model.get_phoneme(word)?;
                    phonemes.extend(phones.into_iter());
                }
                Ok(phonemes)
            }
            None => {
                // Split text into words and process each with Arpabet
                let words = text.split_whitespace();
                let mut phonemes = Vec::new();
                for word in words {
                    if let Some(phones) = self.arpabet.get_polyphone_str(word) {
                        phonemes.extend(phones.iter().map(|&p| p.to_string()));
                    } else {
                        // Fallback to character-level processing
                        for c in word.chars() {
                            let c_str = c.to_string();
                            if let Some(phones) = self.arpabet.get_polyphone_str(&c_str) {
                                phonemes.extend(phones.iter().map(|&p| p.to_string()));
                            } else {
                                phonemes.push(c_str);
                            }
                        }
                    }
                }
                Ok(phonemes)
            }
        }
    }
}
