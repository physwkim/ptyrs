//! [`near_cases`] and what the GPU tests run on them: the engine's options and the CPU LLK.

use std::sync::Arc;

use cuda_core::CudaContext;
use ptyrs::base::{Object, Probe};
use ptyrs::engine::{GpuOptions, GpuPtycho, llk};

#[path = "near_cases.rs"]
mod near_cases;
pub use near_cases::*;

pub const OPTIONS: GpuOptions = GpuOptions {
    stack_size: Some(8),
    graph: true,
    llk_slots: 3,
    fused: true,
    psi_f16: false,
};

impl Case {
    /// The normalised Poisson LLK of `object` and `probe` over every frame.
    pub fn llk(&self, object: &Object, probe: &Probe) -> f64 {
        let psi = propagated(&self.addr, &self.optics(), object, probe);
        llk(&self.obs, 0..self.addr.nframe(), self.addr.nmode(), &psi)
            .normalised(self.obs.nb_obs())
            .poisson
    }

    pub fn gpu(&self, graph: bool) -> GpuPtycho {
        let ctx: Arc<CudaContext> = CudaContext::new(0).expect("CUDA device 0");
        let options = GpuOptions { graph, ..OPTIONS };
        let (object, probe) = (self.object.clone(), self.probe.clone());
        GpuPtycho::with_optics(&ctx, &self.scan, &self.optics(), object, probe, options).unwrap()
    }
}
