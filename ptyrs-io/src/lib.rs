//! ptyrs file I/O on the pure-Rust `rust-hdf5` crate (no libhdf5).
//!
//! - [`write_cxi`] / [`read_cxi`]: a [`ptyrs_core::Scan`] in the CXI layout PyNX
//!   `save_ptycho_data_cxi` writes and `pynx-ptycho-cxi` reads.
//! - [`write_modes`] / [`read_modes`]: an object and probe pair (for example the initial
//!   values a run starts from) as complex128 datasets h5py reads as `complex128` arrays.

mod cxi;
mod modes;

pub use cxi::{read_cxi, write_cxi};
pub use modes::{read_modes, write_modes};

/// Errors from reading or writing ptyrs files.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error(transparent)]
    Hdf5(#[from] rust_hdf5::Hdf5Error),
    #[error(transparent)]
    Core(#[from] ptyrs_core::CoreError),
    /// The file does not follow the expected layout, or the data cannot be stored in it.
    #[error("{0}")]
    Layout(String),
}
