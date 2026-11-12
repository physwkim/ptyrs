//! PyNX DM on the GPU.

use cuda_async::device_operation::ExecutionContext;
use ptyrs_gpu::{Error as GpuError, Exit, FusedInputs, FusedSums};

use super::device::{M1Bufs, PassesMut};
use super::{Bufs, Epoch, GpuPtycho};
use crate::ptycho::{Algorithm, llk_due};
use crate::{DmParams, EngineError};

impl GpuPtycho {
    /// [`crate::Ptycho::dm`]. Cycle 0 of a call projects `PO` itself (the reflection maps
    /// `psi = PO` to itself and adds nothing back); later cycles reflect each stack's carried
    /// exit waves against `PO`, project, and add the remainder back (M1: `dm_reflect` into
    /// the scratch stack, `add_c32`). Then `loop_obj_probe` object-only and probe-only
    /// updates, the probe one from the new object; the fused projection adds the first
    /// object-only update's sums itself.
    pub fn dm(&mut self, params: &DmParams, ncycle: usize) -> Result<(), EngineError> {
        for ic in 0..ncycle {
            let epoch = Epoch::Dm {
                params: *params,
                first: ic == 0,
            };
            self.dev.steady(epoch, self.cycle)?;
            if llk_due(params.calc_llk, ic, ncycle) {
                self.dev.llk_pass(Some((self.cycle, Algorithm::Dm)))?;
            }
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
    /// One DM cycle enqueued on `ctx`; `first` for cycle 0 of a call.
    pub(super) fn dm_cycle(
        &mut self,
        ctx: &ExecutionContext,
        passes: &mut PassesMut<'_>,
        params: &DmParams,
        first: bool,
    ) -> Result<(), GpuError> {
        let alpha = params.alpha as f32;
        // Whether the projection added the first object-only update's sums, which read the
        // projected exit waves and the probe the projection read.
        let mut summed = false;
        match passes {
            PassesMut::M1(m1) => {
                for s in 0..self.stacks.len() {
                    self.dm_m1(ctx, m1, s, alpha, first)?;
                }
            }
            PassesMut::Fused(f) => {
                summed = params.loop_obj_probe > 0 && params.obj_inertia.is_some();
                for s in 0..self.stacks.len() {
                    let stack = self.table.at(s);
                    let psi = self.stacks.get(s).psi;
                    let inputs = FusedInputs {
                        views: self.views,
                        counts: self.counts,
                        obj: self.obj,
                        probe: self.probe,
                    };
                    let exit = if first {
                        Exit::DmFirst { psi }
                    } else {
                        Exit::Dm { psi, alpha }
                    };
                    let sums = FusedSums {
                        obj: summed.then_some(&mut *self.obj_sums),
                        probe: None,
                    };
                    self.kernels
                        .fused_project(ctx, &inputs, stack, exit, f, sums)?;
                }
            }
        }
        for _ in 0..params.loop_obj_probe {
            for (o, p) in [(params.obj_inertia, None), (None, params.probe_inertia)] {
                if o.is_none() && p.is_none() {
                    continue;
                }
                if !std::mem::take(&mut summed) {
                    for s in 0..self.stacks.len() {
                        self.accumulate(ctx, s, o.is_some(), p.is_some())?;
                    }
                }
                self.apply_sums(ctx, o, p)?;
            }
        }
        Ok(())
    }

    /// Stack `s`'s DM exit waves by M1's launches.
    fn dm_m1(
        &mut self,
        ctx: &ExecutionContext,
        m1: &mut M1Bufs<'_>,
        s: usize,
        alpha: f32,
        first: bool,
    ) -> Result<(), GpuError> {
        let st = self.stacks.get(s);
        let len = st.frames.len;
        let po = if first {
            &mut *st.psi
        } else {
            &mut *m1.scratch
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
                .dm_reflect(ctx, st.psi, m1.scratch, len, alpha)?;
        }
        self.project(ctx, m1, s, None)?;
        if !first {
            self.kernels
                .add_c32(ctx, self.stacks.get(s).psi, m1.scratch, len)?;
        }
        Ok(())
    }
}
