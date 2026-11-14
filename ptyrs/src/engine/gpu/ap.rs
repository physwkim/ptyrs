//! PyNX AP on the GPU.

use crate::gpu::{
    Ended, Error as GpuError, Exit, FarSums, FusedBufs, FusedInputs, FusedSums, RecordWriter,
    ShiftProbe,
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
    /// values. With a position, every pass models the probe moved by each frame's offsets,
    /// the projection also takes each frame's position sums against the arrays it read, the
    /// probe takes the step of [`crate::gpu::Kernels::probe_step`] against the old object,
    /// and the offsets step after the updates.
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
        let (o, mut p) = (params.obj_inertia, params.probe_inertia);
        let stepped = match passes {
            PassesMut::Fused(f) if self.features.stepped() => {
                self.step_features(ctx, f, llk.take())?
            }
            _ => None,
        };
        match passes {
            PassesMut::M1(m1) => {
                for s in 0..self.scratch.len() {
                    let stack = self.scratch.at(s);
                    let psi = &mut *m1.scratch;
                    self.kernels
                        .obj_probe_to_psi(ctx, self.views, stack, self.obj, self.probe, psi)?;
                    let plan = m1.plans.get(self.scratch.frames[s].plan);
                    self.project(ctx, stack, plan, psi, llk.as_mut())?;
                    self.accumulate(ctx, stack, psi, o.is_some(), p.is_some())?;
                }
            }
            PassesMut::Fused(f) => {
                if self.fused_ap(ctx, f, llk.as_mut(), o.is_some(), p)? {
                    p = None;
                }
            }
        }
        let ended = llk.map(|w| w.end(ctx)).transpose()?.or(stepped);
        self.apply_sums(ctx, o, p)?;
        self.features.position_step(ctx, self.grad)?;
        Ok(ended)
    }

    /// The stacks of an AP cycle by the fused passes: per stack the projection with its LLK
    /// partials and the object sums, then the probe sums from the projected exit waves it
    /// leaves in `f`. With a position, the projection adds the shifted probe's sums and each
    /// frame's position sums instead, and after the last stack the shifted probe's become the
    /// probe's step, which it takes here against the object the projection read: `true` then,
    /// with no probe sums left to apply.
    fn fused_ap(
        &mut self,
        ctx: &ExecutionContext,
        f: &mut FusedBufs,
        mut llk: Option<&mut RecordWriter<'_>>,
        obj: bool,
        probe_inertia: Option<f64>,
    ) -> Result<bool, GpuError> {
        let probe = probe_inertia.is_some();
        let dims = *self.views.dims();
        let mut m =
            self.features
                .lend(ctx, self.kernels, &dims, self.obj, self.probe, self.exit)?;
        for s in 0..self.scratch.len() {
            let stack = self.scratch.at(s);
            let inputs = FusedInputs {
                views: self.views,
                counts: self.counts,
                obj: self.obj,
                probe: self.probe,
                model: m.model,
            };
            let far = FarSums {
                llk: llk.is_some().then_some(&mut *self.llk),
                ..FarSums::default()
            };
            let exit = Exit::Ap { far };
            let sums = FusedSums {
                obj: obj.then_some(&mut *self.obj_sums),
                shift: m.shift_sums(probe.then_some(ShiftProbe::Step as _), true),
                ..FusedSums::default()
            };
            self.kernels
                .fused_project(ctx, &inputs, stack, exit, f, sums)?;
            if let Some(rec) = &mut llk {
                rec.push(ctx, &dims, stack, self.llk)?;
            }
            if probe && !m.shifted() {
                self.kernels.psi_to_probe(
                    ctx,
                    self.views,
                    stack,
                    f.work(),
                    self.obj,
                    self.probe_sums,
                )?;
            }
        }
        let Some(inertia) = probe_inertia else {
            return Ok(false);
        };
        m.probe_sums(ctx, self.kernels, &dims, self.probe_sums)?;
        let model = m.model;
        let Some(ShiftProbe::Step(step)) = m.shift_sums(Some(ShiftProbe::Step), false).probe else {
            return Ok(false);
        };
        let inputs = FusedInputs {
            views: self.views,
            counts: self.counts,
            obj: self.obj,
            probe: self.probe,
            model,
        };
        let stacks = (0..self.scratch.len()).map(|s| self.scratch.at(s));
        let sums = &mut *self.probe_sums;
        self.kernels
            .probe_step(ctx, &inputs, stacks, f, step, sums, inertia as f32)?;
        self.kernels
            .probe_step_add(ctx, self.probe, self.probe_sums)?;
        Ok(true)
    }
}
