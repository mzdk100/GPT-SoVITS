mod bert;
mod dict;
mod en;
mod num;
mod phone_symbol;
mod utils;
mod zh;

use {
    crate::error::GSVError,
    jieba_rs::Jieba,
    log::{debug, warn},
    ndarray::Array2,
    regex::Regex,
    std::sync::LazyLock,
    unicode_segmentation::UnicodeSegmentation,
};
pub use {
    bert::BertModel,
    en::{EnSentence, EnWord, G2pEn},
    num::{NumSentence, is_numeric},
    phone_symbol::get_phone_symbol,
    utils::{BERT_TOKENIZER, DICT_MONO_CHARS, DICT_POLY_CHARS, argmax_2d, str_is_chinese},
    zh::{G2PW, G2PWOut, ZhMode, ZhSentence},
};

// Regex to handle emojis and symbols
static CLEANUP_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F900}-\u{1F9FF}\u{2600}-\u{27BF}\u{2000}-\u{206F}\u{2300}-\u{23FF}]+",
    )
    .unwrap()
});

// Simplified regex for tokenization
static TOKEN_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?x)
        \p{Han}+ |              # Chinese characters
        [a-zA-Z]+(?:['-][a-zA-Z]+)* | # English words with optional apostrophes/hyphens
        \d+(?:\.\d+)? |          # Numbers (including decimals)
        [.,!?;:()\[\]<>\-"$/\u{3001}\u{3002}\u{FF01}\u{FF1F}\u{FF1B}\u{FF1A}\u{FF0C}\u{2018}\u{2019}\u{201C}\u{201D}] | # Punctuation
        \s+                      # Whitespace
        "#,
    )
    .unwrap()
});

/// Filters out emojis and other non-essential symbols from the text.
fn cleanup_text(text: &str) -> String {
    CLEANUP_REGEX.replace_all(text, " ").into_owned()
}

pub fn split_text(text: &str) -> Vec<String> {
    let mut items = Vec::with_capacity(text.len() / 20);
    let mut current = String::with_capacity(64);
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        // Handle newlines separately - don't add them to current sentence
        if c == '\n' || c == '\r' {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                items.push(trimmed.to_string());
            }
            current.clear();
            continue;
        }

        current.push(c);

        // Check if current character is end punctuation
        let is_end_punctuation = matches!(c, '。' | '！' | '？' | '；' | '.' | '!' | '?' | ';');

        if is_end_punctuation {
            // Special handling for period (.)
            if c == '.' {
                if let Some(&next_char) = chars.peek() {
                    // Case 1: Abbreviation like "Dr. Smith" - next char is space followed by uppercase
                    if next_char == ' ' {
                        let mut peek_iter = chars.clone();
                        peek_iter.next(); // Skip the space
                        if let Some(after_space) = peek_iter.next() {
                            if after_space.is_uppercase() {
                                // This is likely an abbreviation, continue
                                continue;
                            }
                        }
                    }

                    // Case 2: Decimal number like "1.0版本" - next char is digit
                    if next_char.is_digit(10) {
                        continue;
                    }

                    // Case 3: Abbreviation with lowercase letter following
                    if next_char.is_lowercase() {
                        continue;
                    }
                }
            }
            // For other punctuation, check if next character is lowercase letter
            else if matches!(c, '!' | '?' | ';') {
                if let Some(&next_char) = chars.peek() {
                    if next_char.is_lowercase() {
                        continue;
                    }
                }
            }

            let trimmed = current.trim();
            if !trimmed.is_empty() {
                items.push(trimmed.to_string());
            }
            current.clear();
        }
    }

    // Handle any remaining text
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        items.push(trimmed.to_string());
    }

    items
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Lang {
    Zh,
    En,
}

#[derive(Debug, Clone, Copy)]
pub enum LangId {
    Auto,    // Mandarin
    AutoYue, // Cantonese
}

pub trait SentenceProcessor {
    fn get_text_for_bert(&self) -> String;
    fn get_word2ph(&self) -> &[i32];
    fn get_phone_ids(&self) -> &[i64];
}

impl SentenceProcessor for EnSentence {
    fn get_text_for_bert(&self) -> String {
        let mut result = String::with_capacity(self.text.len() * 10);
        for word in &self.text {
            match word {
                EnWord::Word(w) => {
                    if !result.is_empty() && !result.ends_with(' ') {
                        result.push(' ');
                    }
                    result.push_str(w);
                }
                EnWord::Punctuation(p) => {
                    result.push_str(p);
                }
            }
        }
        debug!("English BERT text: {}", result);
        result
    }

    fn get_word2ph(&self) -> &[i32] {
        &self.word2ph
    }

    fn get_phone_ids(&self) -> &[i64] {
        &self.phone_ids
    }
}

impl SentenceProcessor for ZhSentence {
    fn get_text_for_bert(&self) -> String {
        debug!("Chinese BERT text: {}", self.text);
        self.text.clone()
    }

    fn get_word2ph(&self) -> &[i32] {
        &self.word2ph
    }

    fn get_phone_ids(&self) -> &[i64] {
        &self.phone_ids
    }
}

pub struct TextProcessor {
    pub jieba: Jieba,
    pub g2pw: G2PW,
    pub g2p_en: G2pEn,
    pub bert_model: BertModel,
}

impl TextProcessor {
    pub fn new(g2pw: G2PW, g2p_en: G2pEn, bert_model: BertModel) -> Result<Self, GSVError> {
        Ok(Self {
            jieba: Jieba::new(),
            g2pw,
            g2p_en,
            bert_model,
        })
    }

    pub fn get_phone_and_bert(
        &mut self,
        text: &str,
        lang_id: LangId,
    ) -> Result<Vec<(String, Vec<i64>, Array2<f32>)>, GSVError> {
        if text.trim().is_empty() {
            return Err(GSVError::InputEmpty);
        }

        let cleaned_text = cleanup_text(text);
        let chunks = split_text(&cleaned_text);
        let mut result = Vec::with_capacity(chunks.len());

        for chunk in chunks.iter() {
            debug!("Processing chunk: {}", chunk);
            let mut phone_builder = PhoneBuilder::new(chunk);
            phone_builder.extend_text(&self.jieba, chunk);

            if !chunk
                .trim_end()
                .ends_with(['。', '.', '?', '？', '!', '！', '；', ';', '\n'])
            {
                phone_builder.push_punctuation(".");
            }

            // --- A. Collect data for all sub-sentences in the chunk ---
            #[derive(Debug)]
            struct SubSentenceData {
                bert_text: String,
                word2ph: Vec<i32>,
                phone_ids: Vec<i64>,
            }
            let mut sub_sentences_data: Vec<SubSentenceData> = Vec::new();

            for mut sentence in phone_builder.sentences {
                let g2p_result = match &mut sentence {
                    Sentence::Zh(zh) => {
                        let mode = if matches!(lang_id, LangId::AutoYue) {
                            ZhMode::Cantonese
                        } else {
                            ZhMode::Mandarin
                        };
                        zh.g2p(&mut self.g2pw, mode);
                        zh.build_phone()
                    }
                    Sentence::En(en) => en.g2p(&mut self.g2p_en).and_then(|_| en.build_phone()),
                };

                match g2p_result {
                    Ok(phone_seq) => {
                        if phone_seq.is_empty() {
                            continue; // Skip parts that produce no phonemes
                        }
                        sub_sentences_data.push(SubSentenceData {
                            bert_text: sentence.get_text_for_bert(),
                            word2ph: sentence.get_word2ph().to_vec(),
                            phone_ids: sentence.get_phone_ids().to_vec(),
                        });
                    }
                    Err(e) => {
                        warn!("G2P failed for a sentence part in chunk '{}': {}", chunk, e);

                        // Continue processing other parts of the chunk
                    }
                }
            }

            // --- B. Group sub-sentences into logically complete sentences ---
            #[derive(Default, Debug)]
            struct GroupedSentence {
                text: String,
                word2ph: Vec<i32>,
                phone_ids: Vec<i64>,
            }
            let mut grouped_sentences: Vec<GroupedSentence> = Vec::new();
            let mut current_group = GroupedSentence::default();

            for data in sub_sentences_data {
                let ends_sentence = data
                    .bert_text
                    .find(['。', '.', '?', '？', '!', '！', '；', ';']);

                current_group.text.push_str(&data.bert_text);
                current_group.word2ph.extend(data.word2ph);
                current_group.phone_ids.extend(data.phone_ids);
                if ends_sentence.is_some() {
                    grouped_sentences.push(current_group);
                    current_group = GroupedSentence::default()
                }
            }
            // Add any remaining part that didn't end with punctuation
            if !current_group.text.is_empty() {
                grouped_sentences.push(current_group);
            }

            // --- C. Process each complete sentence with BERT ---
            for group in grouped_sentences {
                debug!("Processing grouped sentence: '{}'", group.text);
                let total_expected_bert_len = group.phone_ids.len();

                match self
                    .bert_model
                    .get_bert(&group.text, &group.word2ph, total_expected_bert_len)
                {
                    Ok(bert_features) => {
                        if bert_features.shape()[0] != total_expected_bert_len {
                            let error_msg = format!(
                                "BERT output length mismatch for text '{}': expected {}, got {}",
                                group.text,
                                total_expected_bert_len,
                                bert_features.shape()[0]
                            );
                            warn!("{}", error_msg);

                            continue;
                        }
                        result.push((group.text, group.phone_ids, bert_features));
                    }
                    Err(e) => {
                        warn!(
                            "Failed to get BERT features for text '{}': {}",
                            group.text, e
                        );
                    }
                }
            }
        }

        debug!("RESULT (total sentences: {})", result.len());
        if result.is_empty() {
            return Err(GSVError::GeneratePhonemesOrBertFeaturesFailed(
                text.to_owned(),
            ));
        }
        Ok(result)
    }
}

fn parse_punctuation(p: &str) -> Option<&'static str> {
    match p {
        "，" | "," => Some(","),
        "。" | "." => Some("."),
        "！" | "!" => Some("!"),
        "？" | "?" => Some("?"),
        "；" | ";" => Some(";"),
        "：" | ":" => Some(":"),
        "‘" | "’" | "'" => Some("'"),
        "＇" => Some("'"),
        "“" | "”" | "\"" => Some("\""),
        "＂" => Some("\""),
        "（" | "(" => Some("("),
        "）" | ")" => Some(")"),
        "【" | "[" => Some("["),
        "】" | "]" => Some("]"),
        "《" | "<" => Some("<"),
        "》" | ">" => Some(">"),
        "—" | "–" => Some("-"),
        "～" | "~" => Some("~"),
        "…" | "..." => Some("..."),
        "·" => Some("·"),
        "、" => Some("、"),
        "$" => Some("$"),
        "/" => Some("/"),
        "\n" => Some("\n"), // Corrected escape sequence
        " " => Some(" "),
        _ => None,
    }
}

#[derive(Debug)]
enum Sentence {
    Zh(ZhSentence),
    En(EnSentence),
}

impl SentenceProcessor for Sentence {
    fn get_text_for_bert(&self) -> String {
        match self {
            Sentence::Zh(zh) => zh.get_text_for_bert(),
            Sentence::En(en) => en.get_text_for_bert(),
        }
    }

    fn get_word2ph(&self) -> &[i32] {
        match self {
            Sentence::Zh(zh) => zh.get_word2ph(),
            Sentence::En(en) => en.get_word2ph(),
        }
    }

    fn get_phone_ids(&self) -> &[i64] {
        match self {
            Sentence::Zh(s) => s.get_phone_ids(),
            Sentence::En(s) => s.get_phone_ids(),
        }
    }
}

struct PhoneBuilder {
    sentences: Vec<Sentence>,
    sentence_lang: Lang,
}

impl PhoneBuilder {
    fn new(text: &str) -> Self {
        let sentence_lang = detect_sentence_language(text);
        Self {
            sentences: Vec::with_capacity(16),
            sentence_lang,
        }
    }

    fn extend_text(&mut self, jieba: &Jieba, text: &str) {
        let tokens: Vec<&str> = if str_is_chinese(text) {
            jieba.cut(text, true).into_iter().collect()
        } else {
            TOKEN_REGEX.find_iter(text).map(|m| m.as_str()).collect()
        };

        for t in tokens {
            if let Some(p) = parse_punctuation(t) {
                self.push_punctuation(p);
                continue;
            }

            if is_numeric(t) {
                let ns = NumSentence {
                    text: t.to_owned(),
                    lang: self.sentence_lang,
                };
                let txt = match ns.to_lang_text() {
                    Ok(txt) => txt,
                    Err(e) => {
                        warn!("Failed to process numeric token '{}': {}", t, e);
                        t.to_string()
                    }
                };
                match self.sentence_lang {
                    Lang::Zh => self.push_zh_word(&txt),
                    Lang::En => self.push_en_word(&txt),
                }
            } else if str_is_chinese(t) {
                self.push_zh_word(t);
            } else if t
                .chars()
                .all(|c| c.is_ascii_alphabetic() || c == '\'' || c == '-')
            {
                self.push_en_word(t);
            } else {
                // Handle mixed-language tokens by re-tokenizing the mixed token
                for sub_token in TOKEN_REGEX.find_iter(t) {
                    let sub_token_str = sub_token.as_str();
                    if let Some(p) = parse_punctuation(sub_token_str) {
                        self.push_punctuation(p);
                    } else if is_numeric(sub_token_str) {
                        let ns = NumSentence {
                            text: sub_token_str.to_owned(),
                            lang: self.sentence_lang,
                        };
                        let txt = match ns.to_lang_text() {
                            Ok(txt) => txt,
                            Err(e) => {
                                warn!("Failed to process numeric token '{}': {}", sub_token_str, e);
                                sub_token_str.to_string()
                            }
                        };
                        match self.sentence_lang {
                            Lang::Zh => self.push_zh_word(&txt),
                            Lang::En => self.push_en_word(&txt),
                        }
                    } else if str_is_chinese(sub_token_str) {
                        self.push_zh_word(sub_token_str);
                    } else if sub_token_str
                        .chars()
                        .all(|c| c.is_ascii_alphabetic() || c == '\'' || c == '-')
                    {
                        self.push_en_word(sub_token_str);
                    }
                }
            }
        }
    }

    fn push_punctuation(&mut self, p: &'static str) {
        match self.sentences.last_mut() {
            Some(Sentence::Zh(zh)) => {
                zh.text.push_str(p);
                zh.phones.push(G2PWOut::RawChar(p.chars().next().unwrap()));
            }
            Some(Sentence::En(en)) => {
                // Simplified condition check
                if p == " " && matches!(en.text.last(), Some(EnWord::Word(w)) if w == "a") {
                    return;
                }
                en.text.push(EnWord::Punctuation(p));
            }
            None => {
                let en = EnSentence {
                    phone_ids: Vec::with_capacity(16),
                    phones: Vec::with_capacity(16),
                    text: vec![EnWord::Punctuation(p)],
                    word2ph: Vec::with_capacity(16),
                };
                self.sentences.push(Sentence::En(en));
            }
        }
    }

    fn push_en_word(&mut self, word: &str) {
        if word.ends_with(['。', '.', '?', '？', '!', '！', '；', ';', '\n']) {
            let en = EnSentence {
                phone_ids: Vec::with_capacity(16),
                phones: Vec::with_capacity(16),
                text: vec![EnWord::Word(word.to_string())],
                word2ph: Vec::with_capacity(16),
            };
            self.sentences.push(Sentence::En(en));
        }
        match self.sentences.last_mut() {
            Some(Sentence::En(en)) => {
                // Simplified condition check using matches! macro
                if matches!(en.text.last(), Some(EnWord::Punctuation(p)) if *p == "'" || *p == "-")
                {
                    let p = en.text.pop().unwrap();
                    if let Some(EnWord::Word(last_word)) = en.text.last_mut() {
                        if let EnWord::Punctuation(p_str) = p {
                            last_word.push_str(p_str);
                            last_word.push_str(word);
                            return;
                        }
                    }
                    en.text.push(p); // Push back if not applicable
                }
                en.text.push(EnWord::Word(word.to_string()));
            }
            _ => {
                let en = EnSentence {
                    phone_ids: Vec::with_capacity(16),
                    phones: Vec::with_capacity(16),
                    text: vec![EnWord::Word(word.to_string())],
                    word2ph: Vec::with_capacity(16),
                };
                self.sentences.push(Sentence::En(en));
            }
        }
    }

    fn push_zh_word(&mut self, word: &str) {
        fn add_zh_word(zh: &mut ZhSentence, word: &str) {
            zh.text.push_str(word);
            match dict::zh_word_dict(word) {
                Some(phones) => {
                    zh.phones.extend(
                        phones
                            .into_iter()
                            .map(|p: &String| G2PWOut::Pinyin(p.clone())),
                    );
                }
                None => {
                    zh.phones
                        .extend(word.chars().map(|_| G2PWOut::Pinyin(String::new())));
                }
            }
        }

        if word.ends_with(['。', '.', '?', '？', '!', '！', '；', ';', '\n']) {
            let zh = ZhSentence {
                phone_ids: Vec::with_capacity(16),
                phones: Vec::with_capacity(16),
                word2ph: Vec::with_capacity(16),
                text: String::with_capacity(32),
            };
            self.sentences.push(Sentence::Zh(zh));
        }

        match self.sentences.last_mut() {
            Some(Sentence::Zh(zh)) => add_zh_word(zh, word),
            _ => {
                let mut zh = ZhSentence {
                    phone_ids: Vec::with_capacity(16),
                    phones: Vec::with_capacity(16),
                    word2ph: Vec::with_capacity(16),
                    text: String::with_capacity(32),
                };
                add_zh_word(&mut zh, word);
                self.sentences.push(Sentence::Zh(zh));
            }
        }
    }
}

/// Detects the dominant language of a sentence based on character distribution.
fn detect_sentence_language(text: &str) -> Lang {
    let graphemes = text.graphemes(true).collect::<Vec<&str>>();
    let total_chars = graphemes.len();
    if total_chars == 0 {
        return Lang::Zh; // Default to Chinese for empty input
    }

    let zh_count = graphemes.iter().filter(|&&g| str_is_chinese(g)).count();
    let zh_percent = zh_count as f32 / total_chars as f32;

    debug!("chinese percent {}", zh_percent);
    if zh_percent > 0.3 { Lang::Zh } else { Lang::En }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_text() {
        assert_eq!(split_text("Dr. Smith"), ["Dr. Smith"]);
        assert_eq!(split_text("1.0版本"), ["1.0版本"]);
    }
}
