//! The parity scenario ([`scenario`]), shared by the tests and examples so the PyNX and
//! pty-chi values in `tests/data` stay reproducible from `examples/engine_llk_trace.rs`.

pub mod scenario;

use ptyrs::base::{Object, Probe, Scan};

/// [`scenario::dm_scenario_truth`] without the truth LLK.
pub fn dm_scenario(
    frame: usize,
    npos: usize,
    nprobe: usize,
    photons: f64,
    step: f64,
) -> (Scan, Object, Probe) {
    let (scan, object, probe, _) = scenario::dm_scenario_truth(frame, npos, nprobe, photons, step);
    (scan, object, probe)
}
