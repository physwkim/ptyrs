//! PyNX AP on the GPU.

use super::GpuPtycho;
use crate::ptycho::{Algorithm, llk_due};
use crate::{ApParams, EngineError};

impl GpuPtycho {
    /// [`crate::Ptycho::ap`]: per stack, exit waves of the current estimate, projection (the
    /// LLK of the estimate is taken from the forward stack) and the update sums; after the
    /// last stack, both updates from the sums, which read only the old arrays.
    pub fn ap(&mut self, params: &ApParams, ncycle: usize) -> Result<(), EngineError> {
        let (o, p) = (params.obj_inertia, params.probe_inertia);
        for ic in 0..ncycle {
            let due = llk_due(params.calc_llk, ic, ncycle);
            self.zero_sums(o, p)?;
            for s in 0..self.stacks.len() {
                let st = &mut self.stacks[s];
                self.kernels.obj_probe_to_psi(
                    &self.stream,
                    &self.views,
                    st.span.arg(),
                    &self.obj,
                    &self.probe,
                    &mut st.psi,
                )?;
                self.project(s, due)?;
                self.accumulate(s, o.is_some(), p.is_some())?;
            }
            self.apply_sums(o, p)?;
            if due {
                let l = self.read_llk()?;
                self.record(Algorithm::Ap, l);
            }
            self.cycle += 1;
        }
        if params.zero_phase_ramp {
            self.zero_phase_ramp()?;
        }
        Ok(())
    }
}
