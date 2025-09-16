use {
    crate::{error::GSVError, text::Lang},
    pest::Parser,
};

#[derive(pest_derive::Parser)]
#[grammar = "src/text/rule.pest"]
pub struct ExprParser;

pub mod zh {
    use super::*;
    use pest::iterators::Pair;

    fn parse_pn(pair: Pair<Rule>, dst_string: &mut String) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::pn);
        match pair.as_str() {
            "+" => dst_string.push_str("加"),
            "-" => dst_string.push_str("减"),
            "*" | "×" => dst_string.push_str("乘"),
            "/" | "÷" => dst_string.push_str("除以"),
            "=" => dst_string.push_str("等于"),
            _ => return Err(GSVError::UnknownOperator(pair.as_str().to_owned())),
        }
        Ok(())
    }

    fn parse_flag(pair: Pair<Rule>, dst_string: &mut String) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::flag);
        match pair.as_str() {
            "+" => dst_string.push_str("正"),
            "-" => dst_string.push_str("负"),
            _ => return Err(GSVError::UnknownFlag(pair.as_str().to_owned())),
        }
        Ok(())
    }

    fn parse_percent(pair: Pair<Rule>, dst_string: &mut String) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::percent);
        dst_string.push_str("百分之");
        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::decimals => parse_decimals(pair, dst_string)?,
                Rule::integer => parse_integer(pair, dst_string, true)?,
                _ => return Err(GSVError::UnknownRuleInPercent(pair.as_str().to_owned())),
            }
        }
        Ok(())
    }

    static UNITS: [&str; 4] = ["", "十", "百", "千"];
    static BASE_UNITS: [&str; 4] = ["", "万", "亿", "万"];

    fn parse_integer(
        pair: Pair<Rule>,
        dst_string: &mut String,
        unit: bool,
    ) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::integer);

        let digits: Vec<_> = pair.into_inner().collect(); // Remove .rev() to process left-to-right
        let mut result = String::new();
        let mut has_non_zero = false;

        for (i, pair) in digits.iter().enumerate() {
            let txt = match pair.as_str() {
                "0" => "零",
                "1" => "一",
                "2" => "二",
                "3" => "三",
                "4" => "四",
                "5" => "五",
                "6" => "六",
                "7" => "七",
                "8" => "八",
                "9" => "九",
                _ => return Err(GSVError::UnknownDigit(pair.as_str().to_owned())),
            };
            // Calculate the position from most significant digit
            let pos = digits.len() - 1 - i;
            let u = if pos % 4 != 0 {
                UNITS[pos % 4]
            } else {
                BASE_UNITS[(pos / 4) % 4]
            };

            if txt != "零" {
                has_non_zero = true;
                // Skip "一" for tens place (pos == 1) when the digit is 1
                if !(pos == 1 && txt == "一") {
                    result.push_str(txt);
                }
                if unit {
                    result.push_str(u);
                }
            } else if has_non_zero && unit && pos > 0 {
                result.push_str(txt);
            }
        }

        if result.is_empty() {
            dst_string.push_str("零");
        } else {
            if result.ends_with("零") {
                result.truncate(result.len() - "零".len());
            }
            dst_string.push_str(&result);
        }

        Ok(())
    }

    fn parse_decimals(pair: Pair<Rule>, dst_string: &mut String) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::decimals);

        let mut inner = pair.into_inner().rev();
        let f_part = inner.next().unwrap();
        if let Some(i_part) = inner.next() {
            parse_integer(i_part, dst_string, true)?;
        } else {
            dst_string.push_str("零");
        }
        dst_string.push_str("点");
        parse_integer(f_part, dst_string, false)?;

        Ok(())
    }

    fn parse_fractional(pair: Pair<Rule>, dst_string: &mut String) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::fractional);

        let mut inner = pair.into_inner();
        let numerator = inner.next().unwrap();
        let denominator = inner.next().unwrap();
        parse_integer(denominator, dst_string, true)?;
        dst_string.push_str("分之");
        parse_integer(numerator, dst_string, true)?;
        Ok(())
    }

    fn parse_num(pair: Pair<Rule>, dst_string: &mut String) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::num);

        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::flag => parse_flag(pair, dst_string)?,
                Rule::percent => parse_percent(pair, dst_string)?,
                Rule::decimals => parse_decimals(pair, dst_string)?,
                Rule::fractional => parse_fractional(pair, dst_string)?,
                Rule::integer => parse_integer(pair, dst_string, true)?,
                _ => return Err(GSVError::UnknownRuleInNum(pair.as_str().to_owned())),
            }
        }
        Ok(())
    }

    fn parse_signs(pair: Pair<Rule>, dst_string: &mut String) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::signs);

        let inner = pair.into_inner();
        for pair in inner {
            log::debug!("{:?}", pair);
            match pair.as_rule() {
                Rule::num => parse_num(pair, dst_string)?,
                Rule::pn => parse_pn(pair, dst_string)?,
                Rule::word => {
                    log::warn!("word: {:?}", pair.as_str());
                }
                _ => return Err(GSVError::UnknownRuleInSigns(pair.as_str().to_owned())),
            }
        }
        Ok(())
    }

    fn parse_link(pair: Pair<Rule>, dst_string: &mut String) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::link);
        if pair.as_str() == "-" {
            dst_string.push_str("杠");
        }
        Ok(())
    }

    fn parse_word(pair: Pair<Rule>, dst_string: &mut String) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::word);
        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::digit => {
                    let txt = match pair.as_str() {
                        "0" => "零",
                        "1" => "一",
                        "2" => "二",
                        "3" => "三",
                        "4" => "四",
                        "5" => "五",
                        "6" => "六",
                        "7" => "七",
                        "8" => "八",
                        "9" => "九",
                        _ => return Err(GSVError::UnknownDigit(pair.as_str().to_owned())),
                    };
                    dst_string.push_str(txt);
                }
                Rule::alpha => {
                    dst_string.push_str(pair.as_str());
                }
                Rule::greek => {
                    let txt = match pair.as_str() {
                        "α" | "Α" => "阿尔法",
                        "β" | "Β" => "贝塔",
                        "γ" | "Γ" => "伽马",
                        "δ" | "Δ" => "德尔塔",
                        "ε" | "Ε" => "艾普西龙",
                        "ζ" | "Ζ" => "泽塔",
                        "η" | "Η" => "艾塔",
                        "θ" | "Θ" => "西塔",
                        "ι" | "Ι" => "约塔",
                        "κ" | "Κ" => "卡帕",
                        "λ" | "Λ" => "兰姆达",
                        "μ" | "Μ" => "缪",
                        "ν" | "Ν" => "纽",
                        "ξ" | "Ξ" => "克西",
                        "ο" | "Ο" => "欧米克戈",
                        "π" | "Π" => "派",
                        "ρ" | "Ρ" => "罗",
                        "σ" | "Σ" => "西格玛",
                        "τ" | "Τ" => "套",
                        "υ" | "Υ" => "宇普西龙",
                        "φ" | "Φ" => "斐",
                        "χ" | "Χ" => "希",
                        "ψ" | "Ψ" => "普西",
                        "ω" | "Ω" => "欧米伽",
                        _ => return Err(GSVError::UnknownGreekLetter(pair.as_str().to_owned())),
                    };
                    dst_string.push_str(txt);
                }
                _ => return Err(GSVError::UnknownRuleWord(pair.as_str().to_owned())),
            }
        }
        Ok(())
    }

    fn parse_ident(pair: Pair<Rule>, dst_string: &mut String) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::ident);
        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::word => parse_word(pair, dst_string)?,
                Rule::link => parse_link(pair, dst_string)?,
                _ => return Err(GSVError::UnknownRuleIdent(pair.as_str().to_owned())),
            }
        }
        Ok(())
    }

    pub fn parse_all(pair: Pair<Rule>, dst_string: &mut String) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::all);
        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::signs => parse_signs(pair, dst_string)?,
                Rule::ident => parse_ident(pair, dst_string)?,
                _ => return Err(GSVError::UnknownRuleAll(pair.as_str().to_owned())),
            }
        }
        Ok(())
    }
}

pub mod en {
    use {super::*, crate::error::GSVError, pest::iterators::Pair};

    const SEPARATOR: &str = " ";

    fn parse_pn(pair: Pair<Rule>, dst_string: &mut String) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::pn);
        if !dst_string.is_empty() {
            dst_string.push_str(SEPARATOR);
        }
        match pair.as_str() {
            "+" => dst_string.push_str("plus"),
            "-" => dst_string.push_str("minus"),
            "*" | "×" => dst_string.push_str("times"),
            "/" | "÷" => {
                dst_string.push_str("divided by");
            }
            "=" => dst_string.push_str("is"),
            _ => return Err(GSVError::UnknownOperator(pair.as_str().to_owned())),
        }
        Ok(())
    }

    fn parse_flag(pair: Pair<Rule>, dst_string: &mut String) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::flag);
        if !dst_string.is_empty() {
            dst_string.push_str(SEPARATOR);
        }
        match pair.as_str() {
            "-" => dst_string.push_str("negative"),
            _ => return Err(GSVError::UnknownFlag(pair.as_str().to_owned())),
        }
        Ok(())
    }

    fn parse_percent(pair: Pair<Rule>, dst_string: &mut String) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::percent);
        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::decimals => parse_decimals(pair, dst_string)?,
                Rule::integer => parse_integer(pair, dst_string, true)?,
                _ => return Err(GSVError::UnknownRuleInPercent(pair.as_str().to_owned())),
            }
        }
        if !dst_string.is_empty() {
            dst_string.push_str(SEPARATOR);
        }
        dst_string.push_str("percent");
        Ok(())
    }

    fn parse_integer(
        pair: Pair<Rule>,
        dst_string: &mut String,
        unit: bool,
    ) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::integer);
        if !dst_string.is_empty() {
            dst_string.push_str(SEPARATOR);
        }

        // Note: Replace with proper num2en::str_to_words if available
        let digits: Vec<_> = pair.into_inner().collect();
        for pair in digits {
            let txt = match pair.as_str() {
                "0" => "zero",
                "1" => "one",
                "2" => "two",
                "3" => "three",
                "4" => "four",
                "5" => "five",
                "6" => "six",
                "7" => "seven",
                "8" => "eight",
                "9" => "nine",
                _ => return Err(GSVError::UnknownDigit(pair.as_str().to_owned())),
            };
            dst_string.push_str(txt);
            if unit && !dst_string.is_empty() {
                dst_string.push_str(SEPARATOR);
            }
        }
        Ok(())
    }

    fn parse_decimals(pair: Pair<Rule>, dst_string: &mut String) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::decimals);
        if !dst_string.is_empty() {
            dst_string.push_str(SEPARATOR);
        }

        let mut inner = pair.into_inner().rev();
        let f_part = inner.next().unwrap();
        if let Some(i_part) = inner.next() {
            parse_integer(i_part, dst_string, true)?;
        } else {
            dst_string.push_str("zero");
        }
        if !dst_string.is_empty() {
            dst_string.push_str(SEPARATOR);
        }
        dst_string.push_str("point");
        if !dst_string.is_empty() {
            dst_string.push_str(SEPARATOR);
        }
        parse_integer(f_part, dst_string, false)?;
        Ok(())
    }

    fn parse_fractional(pair: Pair<Rule>, dst_string: &mut String) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::fractional);
        let mut inner = pair.into_inner();
        let numerator = inner.next().unwrap();
        let denominator = inner.next().unwrap();
        parse_integer(numerator, dst_string, true)?;
        if !dst_string.is_empty() {
            dst_string.push_str(SEPARATOR);
        }
        dst_string.push_str("over");
        if !dst_string.is_empty() {
            dst_string.push_str(SEPARATOR);
        }
        parse_integer(denominator, dst_string, true)?;
        Ok(())
    }

    fn parse_num(pair: Pair<Rule>, dst_string: &mut String) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::num);
        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::flag => parse_flag(pair, dst_string)?,
                Rule::percent => parse_percent(pair, dst_string)?,
                Rule::decimals => parse_decimals(pair, dst_string)?,
                Rule::fractional => parse_fractional(pair, dst_string)?,
                Rule::integer => parse_integer(pair, dst_string, true)?,
                _ => return Err(GSVError::UnknownRuleInNum(pair.as_str().to_owned())),
            }
        }
        Ok(())
    }

    fn parse_signs(pair: Pair<Rule>, dst_string: &mut String) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::signs);
        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::num => parse_num(pair, dst_string)?,
                Rule::pn => parse_pn(pair, dst_string)?,
                Rule::word => {}
                _ => return Err(GSVError::UnknownRuleInSigns(pair.as_str().to_owned())),
            }
        }
        Ok(())
    }

    fn parse_link(pair: Pair<Rule>) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::link);
        Ok(())
    }

    fn parse_word(pair: Pair<Rule>, dst_string: &mut String) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::word);
        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::digit => {
                    let txt = match pair.as_str() {
                        "0" => "zero",
                        "1" => "one",
                        "2" => "two",
                        "3" => "three",
                        "4" => "four",
                        "5" => "five",
                        "6" => "six",
                        "7" => "seven",
                        "8" => "eight",
                        "9" => "nine",
                        _ => return Err(GSVError::UnknownDigit(pair.as_str().to_owned())),
                    };
                    if !dst_string.is_empty() {
                        dst_string.push_str(SEPARATOR);
                    }
                    dst_string.push_str(txt);
                }
                Rule::alpha => {
                    if !dst_string.is_empty() {
                        dst_string.push_str(SEPARATOR);
                    }
                    dst_string.push_str(pair.as_str());
                }
                Rule::greek => {
                    let txt = match pair.as_str() {
                        "α" | "Α" => "alpha",
                        "β" | "Β" => "beta",
                        "γ" | "Γ" => "gamma",
                        "δ" | "Δ" => "delta",
                        "ε" | "Ε" => "epsilon",
                        "ζ" | "Ζ" => "zeta",
                        "η" | "Η" => "eta",
                        "θ" | "Θ" => "theta",
                        "ι" | "Ι" => "iota",
                        "κ" | "Κ" => "kappa",
                        "λ" | "Λ" => "lambda",
                        "μ" | "Μ" => "mu",
                        "ν" | "Ν" => "nu",
                        "ξ" | "Ξ" => "xi",
                        "ο" | "Ο" => "omicron",
                        "π" | "Π" => "pi",
                        "ρ" | "Ρ" => "rho",
                        "σ" | "Σ" => "sigma",
                        "τ" | "Τ" => "tau",
                        "υ" | "Υ" => "upsilon",
                        "φ" | "Φ" => "phi",
                        "χ" | "Χ" => "chi",
                        "ψ" | "Ψ" => "psi",
                        "ω" | "Ω" => "omega",
                        _ => return Err(GSVError::UnknownGreekLetter(pair.as_str().to_owned())),
                    };
                    if !dst_string.is_empty() {
                        dst_string.push_str(SEPARATOR);
                    }
                    dst_string.push_str(txt);
                }
                _ => return Err(GSVError::UnknownRuleWord(pair.as_str().to_owned())),
            }
        }
        Ok(())
    }

    fn parse_ident(pair: Pair<Rule>, dst_string: &mut String) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::ident);
        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::word => parse_word(pair, dst_string)?,
                Rule::link => parse_link(pair)?,
                _ => return Err(GSVError::UnknownRuleIdent(pair.as_str().to_owned())),
            }
        }
        Ok(())
    }

    pub fn parse_all(pair: Pair<Rule>, dst_string: &mut String) -> Result<(), GSVError> {
        assert_eq!(pair.as_rule(), Rule::all);
        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::signs => parse_signs(pair, dst_string)?,
                Rule::ident => parse_ident(pair, dst_string)?,
                _ => return Err(GSVError::UnknownRuleAll(pair.as_str().to_owned())),
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct NumSentence {
    pub text: String,
    pub lang: Lang,
}

static NUM_OP: [char; 8] = ['+', '-', '*', '×', '/', '÷', '=', '%'];

impl NumSentence {
    pub fn need_drop(&self) -> bool {
        let num_text = self.text.trim();
        num_text.is_empty() || num_text.chars().all(|c| NUM_OP.contains(&c))
    }

    pub fn is_link_symbol(&self) -> bool {
        self.text == "-"
    }

    pub fn to_lang_text(&self) -> Result<String, GSVError> {
        let mut dst_string = String::new();
        let pairs = ExprParser::parse(Rule::all, &self.text)?;
        for pair in pairs {
            match self.lang {
                Lang::Zh => zh::parse_all(pair, &mut dst_string)?,
                Lang::En => en::parse_all(pair, &mut dst_string)?,
            }
        }
        Ok(dst_string.trim().to_string())
    }
}

pub fn is_numeric(p: &str) -> bool {
    p.chars().any(|c| c.is_numeric())
        || p.contains(&NUM_OP)
        || p.to_lowercase().contains(&[
            'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ',
            'σ', 'ς', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
        ])
}
