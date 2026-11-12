//! ptyrs CPU f64 reference engine: the oracle the GPU kernels are tested against (DESIGN §7 M1).
//!
//! The operators follow PyNX `ptycho/cpu_operator.py` (far field, no background, no
//! multibeam, no direct-beam frames) and run in `f64` / complex `f64` in one fixed order; rPIE
//! follows pty-chi `reconstructors/pie.py`. [`Ptycho`] chains them into AP, DM and rPIE.
//!
//! Contract shared by every operator (the GPU kernels implement the same one):
//! - An exit-wave stack `psi` is `[nview, ny, nx]` with view `k` taken from the
//!   [`ptyrs_core::AddrTable`] (`k = frame * nmode + mode`), stored FFT-shifted (array origin
//!   at `[0, 0]`) in both the object and the detector plane, as PyNX `_psi`. Index `j` of a
//!   shifted plane is natural (centred) index [`shift`]`(j)`; frames are even, so the shift
//!   is its own inverse.
//! - Measured intensities are shifted the same way once, in [`Observed`].
//! - [`Fft2`] is unitary (numpy `norm="ortho"`), so `Σ|psi|²` is the same in both planes.
//! - Every object/probe sum runs over views in table order (frame-major, mode-minor), the
//!   PyNX loop order.

mod fft;
mod obs;
mod ops;
mod ptycho;
pub mod sim;

pub use fft::Fft2;
pub use obs::{Observed, shift};
pub use ops::{
    Llk, accumulate_obj, accumulate_probe, apply_amplitude, llk, obj_probe_to_psi,
    psi_to_obj_probe, quad_phase,
};
pub use ptycho::{Algorithm, ApParams, DmParams, LlkRecord, Ptycho, RpieParams};

/// Errors from building or running the engine.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    Core(#[from] ptyrs_core::CoreError),
    #[error("invalid input: {0}")]
    Invalid(String),
}
