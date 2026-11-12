//! The rPIE parameters of the pty-chi comparison, shared by the tests and examples that run
//! rPIE, so the pty-chi values in `tests/data` stay reproducible from `examples/llk_trace.rs`.
//! Apart from `common` so that a test binary without rPIE carries no dead constant.

use ptyrs_engine::RpieParams;

/// Alpha below 1 keeps the `(1 − α)|·|²` term of both step denominators in play; batches of 4
/// sum overlapping updates and still converge (10 diverge on the 64-pixel scan, on both
/// sides).
pub const RPIE: RpieParams = RpieParams {
    obj_step: Some(1.0),
    probe_step: Some(1.0),
    obj_alpha: 0.5,
    probe_alpha: 0.5,
    batch_size: 4,
    calc_llk: 1,
};
