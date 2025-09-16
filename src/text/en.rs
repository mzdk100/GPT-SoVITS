mod g2p_en;

pub use g2p_en::*;
use {
    crate::{error::GSVError, text::get_phone_symbol},
    log::debug,
    std::borrow::Cow,
};

#[derive(PartialEq, Eq, Clone)]
pub enum EnWord {
    Word(String),
    Punctuation(&'static str),
}

impl std::fmt::Debug for EnWord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnWord::Word(w) => write!(f, "\"{}\"", w),
            EnWord::Punctuation(p) => write!(f, "\"{}\"", p),
        }
    }
}

#[derive(Debug, Default)]
pub struct EnSentence {
    pub phone_ids: Vec<i64>,
    pub phones: Vec<Cow<'static, str>>,
    pub word2ph: Vec<i32>,
    pub text: Vec<EnWord>,
}

impl EnSentence {
    pub fn g2p(&mut self, g2p_en: &mut G2pEn) -> Result<(), GSVError> {
        self.phones.clear();
        self.phone_ids.clear();
        self.word2ph.clear();
        for word in &self.text {
            match word {
                EnWord::Word(w) => {
                    let phonemes = g2p_en.g2p(w)?;
                    let mut cnt = 0;
                    for ph in phonemes {
                        self.phones.push(Cow::Owned(ph.clone()));
                        self.phone_ids.push(get_phone_symbol(&ph));
                        cnt += 1;
                        if ph.contains("0")
                            || ph.contains("1")
                            || ph.contains("2")
                            || ph.contains("3")
                            || ph.contains("4")
                        {
                            self.word2ph.push(cnt);
                            cnt = 0;
                        }
                    }
                    if cnt > 0 {
                        self.word2ph.push(cnt);
                    }
                }
                EnWord::Punctuation(p) => {
                    self.phones.push(Cow::Borrowed(p));
                    self.phone_ids.push(get_phone_symbol(p));
                    self.word2ph.push(1);
                }
            };
        }
        debug!("EnSentence phones: {:?}", self.phones);
        debug!("EnSentence phone_ids: {:?}", self.phone_ids);
        debug!("EnSentence word2ph: {:?}", self.word2ph);
        Ok(())
    }

    pub fn build_phone(&self) -> Result<Vec<i64>, GSVError> {
        Ok(self.phone_ids.clone())
    }

    pub fn get_text_string(&self) -> String {
        let mut result = String::with_capacity(self.text.len() * 5); // Estimate capacity
        for w in &self.text {
            match w {
                EnWord::Word(s) => result.push_str(s),
                EnWord::Punctuation(p) => result.push_str(p),
            }
        }
        result
    }
}
