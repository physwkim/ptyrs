//! PyNX DM on the GPU.

use cuda_async::device_operation::ExecutionContext;
use ptyrs_gpu::Error as GpuError;

use super::{Bufs, Epoch, GpuPtycho};
use crate::ptycho::{Algorithm, llk_due};
use crate::{DmParams, EngineError};

impl GpuPtycho {
    /// [`crate::Ptycho::dm`]. Cycle 0 of a call projects `PO` itself (the reflection maps
    /// `psi = PO` to itself and adds nothing back); later cycles reflect each stack's carried
    /// exit waves against `PO` in the scratch stack, project, and add the scratch's remainder
    /// back (`dm_reflect`, `add_c32`). Then `loop_obj_probe` object-only and probe-only
    /// updates, the probe one from the new object.
    pub fn dm(&mut self, params: &DmParams, ncycle: usize) -> Result<(), EngineError> {
        for ic in 0..ncycle {
            self.dev.steady(Epoch::Dm {
                params: *params,
                first: ic == 0,
            })?;
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

impl Bufs<'_> {
    /// One DM cycle enqueued on `ctx`; `first` for cycle 0 of a call.
    pub(super) fn dm_cycle(
        &mut self,
        ctx: &ExecutionContext,
        params: &DmParams,
        first: bool,
    ) -> Result<(), GpuError> {
        let alpha = params.alpha as f32;
        for s in 0..self.stacks.len() {
            let st = self.stacks.get(s);
            let len = st.frames.len;
            let po = if first {
                &mut *st.psi
            } else {
                &mut *self.scratch
            };
            self.kernels.obj_probe_to_psi(
                ctx,
                self.views,
                self.table.at(s),
                self.obj,
                self.probe,
                po,
            )?;
            if !first {
                self.kernels
                    .dm_reflect(ctx, st.psi, self.scratch, len, alpha)?;
            }
            self.project(ctx, s, false)?;
            if !first {
                self.kernels
                    .add_c32(ctx, self.stacks.get(s).psi, self.scratch, len)?;
            }
        }
        for _ in 0..params.loop_obj_probe {
            for (o, p) in [(params.obj_inertia, None), (None, params.probe_inertia)] {
                if o.is_none() && p.is_none() {
                    continue;
                }
                for s in 0..self.stacks.len() {
                    self.accumulate(ctx, s, o.is_some(), p.is_some())?;
                }
                self.apply_sums(ctx, o, p)?;
            }
        }
        Ok(())
    }
}
