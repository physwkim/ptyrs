//! ptyrs-nn: the neural-network boundary of ptyrs on burn (DESIGN §4). It meets the engine only
//! through arrays (predictor frames in, object patches out) and the object gradient (DIP),
//! invariant 7.

pub mod data;
pub mod interop;
pub mod predictor;
pub mod ptychonn;

/// Errors from ptyrs-nn.
#[derive(Debug, thiserror::Error)]
pub enum NnError {
    #[error(transparent)]
    Core(#[from] ptyrs_core::CoreError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("burn: {0}")]
    Burn(String),
    #[error("invalid input: {0}")]
    Invalid(String),
}

pub(crate) fn burn_err(e: impl std::fmt::Debug) -> NnError {
    NnError::Burn(format!("{e:?}"))
}
