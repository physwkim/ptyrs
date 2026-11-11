//! PyNX DM on the GPU.

use super::GpuPtycho;
use crate::ptycho::{Algorithm, llk_due};
use crate::{DmParams, EngineError};

impl GpuPtycho {
    /// [`crate::Ptycho::dm`]. Cycle 0 of a call projects `PO` itself (the reflection maps
    /// `psi = PO` to itself and adds nothing back); later cycles reflect each stack's carried
    /// exit waves against `PO` in the size's scratch stack, project, and add the scratch's
    /// remainder back (`dm_reflect`, `add_c32`). Then `loop_obj_probe` object-only and
    /// probe-only updates, the probe one from the new object.
    pub fn dm(&mut self, params: &DmParams, ncycle: usize) -> Result<(), EngineError> {
        let alpha = params.alpha as f32;
        for ic in 0..ncycle {
            for s in 0..self.stacks.len() {
                let st = &mut self.stacks[s];
                let sz = &mut self.sizes[st.size];
                let len = super::to_u32("stack length", st.psi.len())?;
                let po = if ic == 0 {
                    &mut st.psi
                } else {
                    &mut sz.scratch
                };
                self.kernels.obj_probe_to_psi(
                    &self.stream,
                    &self.views,
                    st.span.arg(),
                    &self.obj,
                    &self.probe,
                    po,
                )?;
                if ic > 0 {
                    self.kernels.dm_reflect(
                        &self.stream,
                        &mut st.psi,
                        &mut sz.scratch,
                        len,
                        alpha,
                    )?;
                }
                self.project(s, false)?;
                if ic > 0 {
                    let st = &mut self.stacks[s];
                    let scratch = &self.sizes[st.size].scratch;
                    self.kernels
                        .add_c32(&self.stream, &mut st.psi, scratch, len)?;
                }
            }
            for _ in 0..params.loop_obj_probe {
                for (o, p) in [(params.obj_inertia, None), (None, params.probe_inertia)] {
                    if o.is_none() && p.is_none() {
                        continue;
                    }
                    self.zero_sums(o, p)?;
                    for s in 0..self.stacks.len() {
                        self.accumulate(s, o.is_some(), p.is_some())?;
                    }
                    self.apply_sums(o, p)?;
                }
            }
            if llk_due(params.calc_llk, ic, ncycle) {
                let l = self.llk()?;
                self.record(Algorithm::Dm, l);
            }
            self.cycle += 1;
        }
        if params.zero_phase_ramp {
            self.zero_phase_ramp()?;
        }
        Ok(())
    }
}
