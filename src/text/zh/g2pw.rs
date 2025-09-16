use {
    crate::{
        error::GSVError,
        onnx_builder::create_onnx_cpu_session,
        text::{BERT_TOKENIZER, DICT_MONO_CHARS, DICT_POLY_CHARS, argmax_2d},
    },
    ndarray::Array,
    ort::value::Tensor,
    std::{
        fmt::Debug,
        path::Path,
        str::FromStr,
        sync::{Arc, LazyLock},
    },
    tokenizers::Tokenizer,
};

pub static LABELS: &str = include_str!("dict_poly_index_list.json");

pub static POLY_LABLES: LazyLock<Vec<String>> =
    LazyLock::new(|| serde_json::from_str(LABELS).unwrap());

#[derive(Clone)]
pub enum G2PWOut {
    Pinyin(String),
    Yue(String),
    RawChar(char),
}

impl Debug for G2PWOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pinyin(s) => write!(f, "\"{}\"", s),
            Self::Yue(s) => write!(f, "\"{}\"", s),
            Self::RawChar(s) => write!(f, "\"{}\"", s),
        }
    }
}

#[derive(Debug)]
pub struct G2PW {
    model: Option<ort::session::Session>,
    tokenizers: Option<Arc<tokenizers::Tokenizer>>,
}

impl G2PW {
    pub fn new<P: AsRef<Path>>(g2pw_path: Option<P>) -> Result<Self, GSVError> {
        if let Some(g2pw_path) = g2pw_path {
            log::info!("G2PW model is loading...");
            let model = create_onnx_cpu_session(g2pw_path)?;
            log::info!("G2PW model is loaded.");
            Ok(Self {
                model: Some(model),
                tokenizers: Some(Arc::new(Tokenizer::from_str(BERT_TOKENIZER).unwrap())),
            })
        } else {
            Ok(Self {
                model: None,
                tokenizers: None,
            })
        }
    }

    pub fn g2p<'s>(&mut self, text: &'s str) -> Vec<G2PWOut> {
        if self.model.is_some() && self.tokenizers.is_some() {
            self.get_pinyin_ml(text)
                .unwrap_or(self.simple_get_pinyin(text))
        } else {
            self.simple_get_pinyin(text)
        }
    }

    pub fn simple_get_pinyin(&self, text: &str) -> Vec<G2PWOut> {
        let mut pre_data = vec![];
        for (_, c) in text.chars().enumerate() {
            if let Some(mono) = DICT_MONO_CHARS.get(&c) {
                pre_data.push(G2PWOut::Pinyin(mono.phone.clone()));
            } else if let Some(poly) = DICT_POLY_CHARS.get(&c) {
                pre_data.push(G2PWOut::Pinyin(poly.phones[0].0.clone()));
            } else {
                pre_data.push(G2PWOut::RawChar(c));
            }
        }
        pre_data
    }

    fn get_pinyin_ml<'s>(&mut self, text: &'s str) -> Result<Vec<G2PWOut>, GSVError> {
        let c = self.tokenizers.as_ref().unwrap().encode(text, true)?;
        let input_ids = c.get_ids().iter().map(|x| *x as i64).collect::<Vec<i64>>();
        let token_type_ids = vec![0i64; input_ids.len()];
        let attention_mask = vec![1i64; input_ids.len()];

        let mut phoneme_masks = vec![];
        let mut pre_data = vec![];
        let mut query_id = vec![];
        let mut chars_id = vec![];

        for (i, c) in text.chars().enumerate() {
            if let Some(mono) = DICT_MONO_CHARS.get(&c) {
                pre_data.push(G2PWOut::Pinyin(mono.phone.clone()));
            } else if let Some(poly) = DICT_POLY_CHARS.get(&c) {
                pre_data.push(G2PWOut::Pinyin("".to_owned()));
                // 这个位置是 tokens 的位置，它的前后添加了 '[CLS]' 和 '[SEP]' 两个特殊字符
                query_id.push(i + 1);
                chars_id.push(poly.index);
                let mut phoneme_mask = vec![0f32; POLY_LABLES.len()];
                for (_, i) in &poly.phones {
                    phoneme_mask[*i] = 1.0;
                }
                phoneme_masks.push(phoneme_mask);
            } else {
                pre_data.push(G2PWOut::RawChar(c));
            }
        }
        let input_ids =
            Tensor::from_array(Array::from_shape_vec((1, input_ids.len()), input_ids).unwrap())
                .unwrap();
        let token_type_ids = Tensor::from_array(
            Array::from_shape_vec((1, token_type_ids.len()), token_type_ids).unwrap(),
        )
        .unwrap();
        let attention_mask = Tensor::from_array(
            Array::from_shape_vec((1, attention_mask.len()), attention_mask).unwrap(),
        )
        .unwrap();

        for ((position_id, phoneme_mask), char_id) in query_id
            .iter()
            .zip(phoneme_masks.iter())
            .zip(chars_id.iter())
        {
            let phoneme_mask = Tensor::from_array(
                Array::from_shape_vec((1, phoneme_mask.len()), phoneme_mask.to_vec()).unwrap(),
            )
            .unwrap();
            let position_id_t =
                Tensor::from_array(Array::from_vec([*position_id as i64].to_vec())).unwrap();
            let char_id = Tensor::from_array(Array::from_vec([*char_id as i64].to_vec())).unwrap();

            let model_ouput = self.model.as_mut().unwrap().run(ort::inputs![
                "input_ids" => input_ids.clone(),
                "token_type_ids" => token_type_ids.clone(),
                "attention_mask" => attention_mask.clone(),
                "phoneme_mask"=> phoneme_mask,
                "char_ids" => char_id,
                "position_ids"=> position_id_t,
            ])?;

            let probs = model_ouput["probs"].try_extract_array::<f32>().unwrap();

            let probs_view = probs.view();

            let i = argmax_2d(&probs_view);

            pre_data[*position_id - 1] = G2PWOut::Pinyin(POLY_LABLES[i.1 as usize].clone());
        }

        Ok(pre_data)
    }
}
