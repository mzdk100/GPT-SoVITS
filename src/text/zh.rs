mod g2pw;
mod jyutping_list;
mod split;
mod yue;

use {
    crate::{error::GSVError, text::get_phone_symbol},
    log::{debug, warn},
};
pub use {
    g2pw::{G2PW, G2PWOut},
    split::split_zh_ph,
};

#[derive(Debug)]
pub enum ZhMode {
    Mandarin,
    Cantonese,
}

#[derive(Debug, Default)]
pub struct ZhSentence {
    pub phone_ids: Vec<i64>,
    pub phones: Vec<G2PWOut>,
    pub word2ph: Vec<i32>,
    pub text: String,
}

impl ZhSentence {
    /// Processes Chinese text into phonemes and phone IDs based on the specified mode.
    pub fn g2p(&mut self, g2pw: &mut G2PW, mode: ZhMode) {
        match mode {
            ZhMode::Mandarin => self.g2p_mandarin(g2pw),
            ZhMode::Cantonese => self.g2p_cantonese(),
        }
    }

    /// Processes Mandarin text using the G2PW model.
    fn g2p_mandarin(&mut self, g2pw: &mut G2PW) {
        let pinyin = g2pw.g2p(&self.text);
        if pinyin.len() != self.text.chars().count() && !self.text.is_empty() {
            warn!(
                "Pinyin length mismatch: {} (pinyin) vs {} (text chars) for text '{}'",
                pinyin.len(),
                self.text.chars().count(),
                self.text
            );
        }
        self.phones = pinyin;
        debug!("phones: {:?}", self.phones);
        self.build_phone_id_and_word2ph();
    }

    /// Processes Cantonese text using the yue module.
    fn g2p_cantonese(&mut self) {
        let (pinyin, word2ph) = yue::g2p(&self.text);
        debug!("pinyin: {:?}", pinyin);
        self.phones = pinyin.into_iter().map(G2PWOut::Yue).collect();
        self.build_phone_id_and_word2ph();
        self.word2ph = word2ph; // Override Pinnacle if Cantonese provides word2ph
    }

    /// Converts phonemes to phone IDs and generates word-to-phoneme mapping.
    fn build_phone_id_and_word2ph(&mut self) {
        self.phone_ids.clear();
        self.word2ph.clear();
        for p in &self.phones {
            match p {
                G2PWOut::Pinyin(p) => {
                    let (initial, final_) = split_zh_ph(p);
                    self.phone_ids.push(get_phone_symbol(initial));
                    if !final_.is_empty() {
                        self.phone_ids.push(get_phone_symbol(final_));
                        self.word2ph.push(2);
                    } else {
                        self.word2ph.push(1);
                    }
                }
                G2PWOut::Yue(c) => {
                    self.phone_ids.push(get_phone_symbol(c));
                    self.word2ph.push(2);
                }
                G2PWOut::RawChar(c) => {
                    self.phone_ids.push(get_phone_symbol(&c.to_string()));
                    self.word2ph.push(1);
                }
            }
        }
        debug!("phone_id {:?}", self.phone_ids);
    }

    /// Returns the phoneme IDs for the sentence.
    pub fn build_phone(&self) -> Result<Vec<i64>, GSVError> {
        Ok(self.phone_ids.clone())
    }
}
