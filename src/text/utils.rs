use std::{collections::HashMap, sync::LazyLock};

use ndarray::{ArrayView, IntoDimension, IxDyn};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PolyChar {
    pub index: usize,
    pub phones: Vec<(String, usize)>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MonoChar {
    pub phone: String,
}

pub static MONO_CHARS_DIST_STR: &str = include_str!("dict_mono_chars.json");
pub static POLY_CHARS_DIST_STR: &str = include_str!("dict_poly_chars.json");
pub static DEFAULT_ZH_WORD_DICT: &str = include_str!("zh_word_dict.json");
pub static BERT_TOKENIZER: &str = include_str!("g2pw_tokenizer.json");

pub fn load_mono_chars() -> HashMap<char, MonoChar> {
    if let Ok(dir) = std::env::var("G2PW_DIST_DIR") {
        let s = std::fs::read_to_string(format!("{}/dict_mono_chars.json", dir))
            .expect("dict_mono_chars.json not found");
        serde_json::from_str(&s).expect("dict_mono_chars.json parse error")
    } else {
        serde_json::from_str(MONO_CHARS_DIST_STR).unwrap()
    }
}

pub fn load_poly_chars() -> HashMap<char, PolyChar> {
    if let Ok(dir) = std::env::var("G2PW_DIST_DIR") {
        let s = std::fs::read_to_string(format!("{}/dict_poly_chars.json", dir))
            .expect("dict_poly_chars.json not found");
        serde_json::from_str(&s).expect("dict_poly_chars.json parse error")
    } else {
        serde_json::from_str(POLY_CHARS_DIST_STR).unwrap()
    }
}

pub static DICT_MONO_CHARS: LazyLock<HashMap<char, MonoChar>> = LazyLock::new(load_mono_chars);
pub static DICT_POLY_CHARS: LazyLock<HashMap<char, PolyChar>> = LazyLock::new(load_poly_chars);

pub fn str_is_chinese(s: &str) -> bool {
    let mut r = true;
    for c in s.chars() {
        if !DICT_MONO_CHARS.contains_key(&c) && !DICT_POLY_CHARS.contains_key(&c) {
            r &= false;
        }
    }
    r
}

// Finds the index of the maximum value in a 2D tensor
pub fn argmax_2d(tensor: &ArrayView<f32, IxDyn>) -> (usize, usize) {
    let mut max_index = (0, 0);
    let mut max_value = tensor
        .get(IxDyn::zeros(2))
        .copied()
        .unwrap_or(f32::NEG_INFINITY);

    for i in 0..tensor.shape()[0] {
        for j in 0..tensor.shape()[1] {
            if let Some(value) = tensor.get((i, j).into_dimension()) {
                if *value > max_value {
                    max_value = *value;
                    max_index = (i, j);
                }
            }
        }
    }
    max_index
}
