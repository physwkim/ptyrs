//! ptyrs host library (DESIGN §4): one module per former crate. `base` is the data model, a name
//! other than `core`, which would shadow the built-in crate.

pub mod base;
pub mod dist;
pub mod engine;
pub mod fft;
pub mod gpu;
pub mod io;
pub mod live;
// burn builds only with the `nn` feature.
#[cfg(feature = "nn")]
pub mod nn;
pub mod refgen;
pub mod tile;
