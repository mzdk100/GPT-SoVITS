use std::{collections::HashMap, sync::LazyLock};

use crate::text::utils::*;
use std::path::PathBuf;

static ZN_DICT: LazyLock<HashMap<String, Vec<String>>> = LazyLock::new(|| {
    let word_dict_path = std::env::var("GPT_SOVITS_DICT_PATH").unwrap_or(".".to_string());
    let path = PathBuf::from(word_dict_path.as_str()).join("zh_word_dict.json");
    if path.is_file() {
        let zh_word_dict = std::fs::read_to_string(path).unwrap();
        serde_json::from_str(&zh_word_dict).unwrap()
    } else {
        serde_json::from_str(DEFAULT_ZH_WORD_DICT).unwrap()
    }
});
static EN_DICT: LazyLock<HashMap<String, Vec<String>>> = LazyLock::new(|| {
    let word_dict_path = std::env::var("GPT_SOVITS_DICT_PATH").unwrap_or(".".to_string());
    let path = PathBuf::from(word_dict_path.as_str()).join("en_word_dict.json");
    if path.is_file() {
        let en_word_dict = std::fs::read_to_string(path).unwrap();
        serde_json::from_str(&en_word_dict).unwrap()
    } else {
        HashMap::default()
    }
});

pub fn zh_word_dict(word: &str) -> Option<&'static [String]> {
    ZN_DICT.get(word).map(|s| s.as_slice())
}

pub fn en_word_dict(word: &str) -> Option<&'static [String]> {
    EN_DICT.get(word).map(|s| s.as_slice())
}
