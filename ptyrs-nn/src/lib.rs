//! ptyrs-nn: the neural-network boundary of ptyrs on burn (DESIGN §4). It meets the engine only
//! through arrays (predictor frames in, object patches out) and the object gradient (DIP),
//! invariant 7.

pub mod data;
pub mod interop;
