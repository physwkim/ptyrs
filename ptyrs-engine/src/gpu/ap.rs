//! PyNX AP on the GPU.

use cuda_async::device_operation::ExecutionContext;
use ptyrs_gpu::{Ended, Error as GpuError, RecordWriter};

use super::{Bufs, Epoch, GpuPtycho};
use crate::ptycho::llk_due;
use crate::{ApParams, EngineError};

impl GpuPtycho {
    /// [`crate::Ptycho::ap`]: per stack, exit waves of the current estimate, projection (the
    /// LLK of the estimate is taken from the forward stack) and the update sums; after the
    /// last stack, both updates from the sums, which read only the old arrays.
    pub fn ap(&mut self, params: &ApParams, ncycle: usize) -> Result<(), EngineError> {
        for ic in 0..ncycle {
            let epoch = Epoch::Ap {
                params: *params,
                llk: llk_due(params.calc_llk, ic, ncycle),
            };
            self.dev.steady(epoch, self.cycle)?;
            self.cycle += 1;
        }
        self.collect()?;
        if params.zero_phase_ramp {
            self.zero_phase_ramp()?;
        }
        Ok(())
    }
}

impl Bufs<'_> {
    /// One AP cycle enqueued on `ctx`; with `llk`, its LLK record too, ended after the last
    /// stack.
    pub(super) fn ap_cycle(
        &mut self,
        ctx: &ExecutionContext,
        params: &ApParams,
        mut llk: Option<RecordWriter<'_>>,
    ) -> Result<Option<Ended>, GpuError> {
        let (o, p) = (params.obj_inertia, params.probe_inertia);
        for s in 0..self.stacks.len() {
            self.kernels.obj_probe_to_psi(
                ctx,
                self.views,
                self.table.at(s),
                self.obj,
                self.probe,
                self.stacks.get(s).psi,
            )?;
            self.project(ctx, s, llk.as_mut())?;
            self.accumulate(ctx, s, o.is_some(), p.is_some())?;
        }
        let ended = llk.map(|w| w.end(ctx)).transpose()?;
        self.apply_sums(ctx, o, p)?;
        Ok(ended)
    }
}
