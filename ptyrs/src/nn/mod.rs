//! ptyrs::nn: the neural-network boundary of ptyrs on burn (DESIGN §4). It meets the engine only
//! through arrays (predictor frames in, object patches out) and the object gradient (DIP),
//! invariant 7.
#![warn(clippy::disallowed_methods)] // clippy.toml: CUDA devices only through `cuda`

use burn::prelude::Device;
use cubecl_runtime::config::cache::CacheConfig;
use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig};

pub mod data;
pub mod dip;
pub mod interop;
pub mod live;
pub mod metric;
pub mod norm_relu;
pub mod predictor;
pub mod ptychonn;

/// Errors from ptyrs::nn.
#[derive(Debug, thiserror::Error)]
pub enum NnError {
    #[error(transparent)]
    Core(#[from] crate::base::CoreError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("burn: {0}")]
    Burn(String),
    #[error("invalid input: {0}")]
    Invalid(String),
    #[error("object loss: {0}")]
    Loss(dip::LossError),
}

pub(crate) fn burn_err(e: impl std::fmt::Debug) -> NnError {
    NnError::Burn(format!("{e:?}"))
}

/// Burn's CUDA device `ordinal`, the one way ptyrs names a CUDA device to burn.
///
/// The first call installs cubecl's configuration as cubecl would load it, but with its
/// environment rooted in the user cache directory (`~/.cache/cubecl`) whatever the working
/// directory, so what one process autotuned and compiled the next one reads back. An application
/// that configured cubecl first keeps its own configuration.
#[expect(clippy::disallowed_methods, reason = "the one sanctioned call")]
pub fn cuda(ordinal: usize) -> Device {
    static CONFIGURED: std::sync::Once = std::sync::Once::new();
    CONFIGURED.call_once(|| {
        let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
        config.environment.path = CacheConfig::Global;
        CubeClRuntimeConfig::try_set(config);
    });
    Device::cuda(ordinal)
}
