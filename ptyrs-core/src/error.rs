/// Errors from building or validating the data model.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("shape mismatch: {0}")]
    Shape(String),
    #[error(
        "value {value} at flat index {index} does not fit the u16 count path (0..={max}); needs the f32 data path"
    )]
    CountOutOfRange { index: usize, value: f64, max: u16 },
    #[error("non-integer count {value} at flat index {index}; needs the f32 data path")]
    NonInteger { index: usize, value: f64 },
    #[error("invalid input: {0}")]
    Invalid(String),
}
