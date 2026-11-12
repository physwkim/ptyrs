//! PyNX AP on the GPU.

use crate::gpu::{
    Ended, Error as GpuError, Exit, FarSums, FusedBufs, FusedInputs, FusedSums, RecordWriter,
};
use cuda_async::device_operation::ExecutionContext;

use super::device::PassesMut;
use super::{Bufs, Epoch, GpuPtycho};
use crate::engine::ptycho::llk_due;
use crate::engine::{ApParams, EngineError};

impl GpuPtycho {
    /// [`crate::engine::Ptycho::ap`]: per stack, exit waves of the current estimate, projection (the
    /// LLK of the estimate is taken from the forward stack) and the update sums; after the
    /// last stack, both updates from the sums, which read only the old arrays. With stepped
    /// features, a pass of their own first takes the LLK and the far sums of the estimate
    /// under the cycle-start model and steps them, and the projection models the stepped
    /// values.
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
        passes: &mut PassesMut<'_>,
        params: &ApParams,
        mut llk: Option<RecordWriter<'_>>,
    ) -> Result<Option<Ended>, GpuError> {
        let (o, p) = (params.obj_inertia, params.probe_inertia);
        let stepped = match passes {
            PassesMut::Fused(f) if self.features.stepped() => {
                self.step_features(ctx, f, llk.take())?
            }
            _ => None,
        };
        for s in 0..self.stacks.len() {
            match passes {
                PassesMut::M1(m1) => {
                    self.kernels.obj_probe_to_psi(
                        ctx,
                        self.views,
                        self.table.at(s),
                        self.obj,
                        self.probe,
                        self.stacks.get(s).psi,
                    )?;
                    self.project(ctx, m1, s, llk.as_mut())?;
                    self.accumulate(ctx, s, o.is_some(), p.is_some())?;
                }
                PassesMut::Fused(f) => {
                    self.fused_ap(ctx, f, s, llk.as_mut(), o.is_some(), p.is_some())?
                }
            }
        }
        let ended = llk.map(|w| w.end(ctx)).transpose()?.or(stepped);
        self.apply_sums(ctx, o, p)?;
        Ok(ended)
    }

    /// Stack `s` of an AP cycle by the fused passes: the projection with its LLK partials and
    /// the object sums, then the probe sums from the projected exit waves it leaves in `f`.
    fn fused_ap(
        &mut self,
        ctx: &ExecutionContext,
        f: &mut FusedBufs,
        s: usize,
        llk: Option<&mut RecordWriter<'_>>,
        obj: bool,
        probe: bool,
    ) -> Result<(), GpuError> {
        let dims = *self.views.dims();
        let stack = self.table.at(s);
        let inputs = FusedInputs {
            views: self.views,
            counts: self.counts,
            obj: self.obj,
            probe: self.probe,
            model: self.features.model(),
        };
        let far = FarSums {
            llk: llk.is_some().then_some(&mut *self.llk),
            ..FarSums::default()
        };
        let exit = Exit::Ap { far };
        let sums = FusedSums {
            obj: obj.then_some(&mut *self.obj_sums),
            ..FusedSums::default()
        };
        self.kernels
            .fused_project(ctx, &inputs, stack, exit, f, sums)?;
        if let Some(rec) = llk {
            rec.push(ctx, &dims, stack, self.llk)?;
        }
        if probe {
            self.kernels.psi_to_probe(
                ctx,
                self.views,
                stack,
                f.work(),
                self.obj,
                self.probe_sums,
            )?;
        }
        Ok(())
    }
}
