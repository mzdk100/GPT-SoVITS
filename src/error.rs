//noinspection SpellCheckingInspection
use {
    ndarray::ShapeError,
    ort::Error as OrtError,
    pest::error::Error as PestError,
    std::{
        cmp::Ord,
        error::Error,
        fmt::{Debug, Display, Formatter, Result as FmtResult},
        hash::Hash,
        io::Error as IoError,
        time::SystemTimeError,
    },
    voxudio::OperationError as VoxudioError,
};

#[derive(Debug)]
pub enum GSVError {
    Box(Box<dyn Error + Send + Sync>),
    DecodeTokenFailed,
    GeneratePhonemesOrBertFeaturesFailed(String),
    InputEmpty,
    Io(IoError),
    Ort(OrtError),
    Pest(String),
    Shape(ShapeError),
    SystemTime(SystemTimeError),
    UnknownRuleAll(String),
    UnknownRuleIdent(String),
    UnknownRuleWord(String),
    UnknownGreekLetter(String),
    UnknownOperator(String),
    UnknownFlag(String),
    UnknownRuleInPercent(String),
    UnknownDigit(String),
    UnknownRuleInNum(String),
    UnknownRuleInSigns(String),
    Voxudio(VoxudioError),
}

impl Error for GSVError {}

impl Display for GSVError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "GSVError: ")?;
        match self {
            Self::Box(e) => Display::fmt(e, f),
            Self::DecodeTokenFailed => {
                write!(f, "DecodeTokenFailedError: Can't decode output audio.")
            }
            Self::GeneratePhonemesOrBertFeaturesFailed(s) => write!(
                f,
                "GeneratePhonemesOrBertFeaturesFailedError: No phonemes or BERT features could be generated for the text: {}",
                s
            ),
            Self::InputEmpty => write!(f, "InputEmptyError: Input data is empty."),
            Self::Io(e) => Display::fmt(e, f),
            Self::Ort(e) => write!(f, "OrtError: {}", e),
            Self::Pest(s) => write!(f, "PestError: {}", s),
            Self::Shape(e) => Display::fmt(e, f),
            Self::SystemTime(e) => Display::fmt(e, f),
            Self::UnknownRuleAll(s) => {
                write!(f, "UnknownRuleAllError: Unknown rule in all: {:?}", s)
            }
            Self::UnknownRuleIdent(s) => {
                write!(f, "UnknownRuleIdentError: Unknown rule in ident: {:?}", s)
            }
            Self::UnknownRuleWord(s) => {
                write!(f, "UnknownRuleWordError: Unknown rule in word: {:?}", s)
            }
            Self::UnknownGreekLetter(s) => {
                write!(f, "UnknownGreekLetterError: Unknown Greek letter: {:?}", s)
            }
            Self::UnknownOperator(s) => {
                write!(f, "UnknownOperatorError: Unknown operator: {:?}", s)
            }
            Self::UnknownFlag(s) => {
                write!(f, "UnknownFlagError: Unknown flag: {:?}", s)
            }
            Self::UnknownRuleInPercent(s) => {
                write!(
                    f,
                    "UnknownRuleInPercentError: Unknown rule in percent: {:?}",
                    s
                )
            }
            Self::UnknownDigit(s) => {
                write!(f, "UnknownDigitError: Unknown digit: {:?}", s)
            }
            Self::UnknownRuleInNum(s) => {
                write!(f, "UnknownRuleInNumError: Unknown rule in num: {:?}", s)
            }
            Self::UnknownRuleInSigns(s) => {
                write!(f, "UnknownRuleInSignsError: Unknown rule in signs: {:?}", s)
            }
            Self::Voxudio(e) => Display::fmt(e, f),
        }
    }
}

impl<R> From<OrtError<R>> for GSVError
where
    OrtError<R>: Display,
{
    fn from(value: OrtError<R>) -> Self {
        Self::Pest(format!("{value}"))
    }
}

impl From<IoError> for GSVError {
    fn from(value: IoError) -> Self {
        Self::Io(value)
    }
}

impl From<ShapeError> for GSVError {
    fn from(value: ShapeError) -> Self {
        Self::Shape(value)
    }
}

impl From<SystemTimeError> for GSVError {
    fn from(value: SystemTimeError) -> Self {
        Self::SystemTime(value)
    }
}

impl From<Box<dyn Error + Send + Sync>> for GSVError {
    fn from(value: Box<dyn Error + Send + Sync>) -> Self {
        Self::Box(value)
    }
}

impl<R> From<PestError<R>> for GSVError
where
    R: Copy + Debug + Hash + Ord,
{
    fn from(value: PestError<R>) -> Self {
        Self::Pest(format!("{}", value))
    }
}

impl From<VoxudioError> for GSVError {
    fn from(value: VoxudioError) -> Self {
        Self::Pest(format!("{value}"))
    }
}
