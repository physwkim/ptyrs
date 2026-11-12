//! PyNX AP on the GPU.

use cuda_async::device_operation::ExecutionContext;
use ptyrs_gpu::Error as GpuError;

use super::{Epoch, GpuPtycho};
use crate::ptycho::{Algorithm, llk_due};
use crate::{ApParams, EngineError};

impl GpuPtycho {
    /// [`crate::Ptycho::ap`]: per stack, exit waves of the current estimate, projection (the
    /// LLK of the estimate is taken from the forward stack) and the update sums; after the
    /// last stack, both updates from the sums, which read only the old arrays.
    pub fn ap(&mut self, params: &ApParams, ncycle: usize) -> Result<(), EngineError> {
        for ic in 0..ncycle {
            if llk_due(params.calc_llk, ic, ncycle) {
                self.ap_cycle(&self.context(), params, true)?;
                let l = self.read_llk()?;
                self.record(Algorithm::Ap, l);
            } else {
                self.steady_cycle(Epoch::Ap(*params))?;
            }
            self.cycle += 1;
        }
        if params.zero_phase_ramp {
            self.zero_phase_ramp()?;
        }
        Ok(())
    }

    /// One AP cycle enqueued on `ctx`; with `llk`, every stack's LLK partials too.
    pub(super) fn ap_cycle(
        &mut self,
        ctx: &ExecutionContext,
        params: &ApParams,
        llk: bool,
    ) -> Result<(), GpuError> {
        let (o, p) = (params.obj_inertia, params.probe_inertia);
        self.zero_sums(ctx, o, p)?;
        for s in 0..self.stacks.len() {
            self.kernels.obj_probe_to_psi(
                ctx,
                &self.views,
                self.table.at(s),
                &self.obj,
                &self.probe,
                &mut self.stacks[s].psi,
            )?;
            self.project(ctx, s, llk)?;
            self.accumulate(ctx, s, o.is_some(), p.is_some())?;
        }
        self.apply_sums(ctx, o, p)
    }
}
