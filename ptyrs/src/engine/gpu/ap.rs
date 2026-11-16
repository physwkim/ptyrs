//! PyNX AP on the GPU.

use crate::gpu::{
    Error as GpuError, Exit, FarSums, FusedBufs, FusedInputs, FusedSums, RecordWriter, ShiftProbe,
    ShiftedStepSums,
};
use cuda_async::device_operation::ExecutionContext;

use super::device::{OneSlice, PassesMut, Sample};
use super::{Bufs, Epoch, GpuPtycho, Tag};
use crate::engine::ptycho::llk_due;
use crate::engine::{ApParams, EngineError};
use crate::fault;

impl GpuPtycho {
    /// [`crate::engine::Ptycho::ap`]: per stack, exit waves of the current estimate, projection (the
    /// LLK of the estimate is taken from the forward stack) and the update sums; after the
    /// last stack, both updates from the sums, which read only the old arrays. With stepped
    /// features, a pass of their own first takes the LLK and the far sums of the estimate
    /// under the cycle-start model and steps them, and the projection models the stepped
    /// values. With a position, every pass models the probe moved by each frame's offsets,
    /// the projection also takes each frame's position sums against the arrays it read, the
    /// probe takes the step of [`crate::gpu::Kernels::probe_step`] against the old object,
    /// and the offsets step after the updates; under [`crate::engine::Sampling::FullObject`]
    /// the passes read each stack's windows of the object sampled at the offsets instead, the
    /// object sums are placed back from them, and the object update divides by the
    /// illumination map.
    pub fn ap(&mut self, params: &ApParams, ncycle: usize) -> Result<(), EngineError> {
        self.call(
            params.zero_phase_ramp,
            |dev| dev.one_unitary("Ap"),
            |dev, cycle| {
                for ic in 0..ncycle {
                    let epoch = Epoch::Ap {
                        params: *params,
                        llk: llk_due(params.calc_llk, ic, ncycle),
                    };
                    dev.steady(epoch, *cycle)?;
                    *cycle += 1;
                }
                Ok(())
            },
        )
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
        mut llk: Option<RecordWriter<'_, Tag>>,
        one: OneSlice,
    ) -> Result<(), GpuError> {
        let (o, mut p) = (params.obj_inertia, params.probe_inertia);
        if let PassesMut::Fused(f) = passes
            && self.features.stepped()
        {
            self.step_features(ctx, f, llk.take(), one)?;
        }
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
                if self.fused_ap(ctx, f, llk.as_mut(), o.is_some(), p, one)? {
                    p = None;
                }
            }
        }
        if let Some(w) = llk {
            w.end(ctx)?;
        }
        fault::at(fault::Exit::ApSums)?;
        self.apply_sums(ctx, o, p)?;
        fault::at(fault::Exit::ApPositions)?;
        self.features.position_step(ctx, self.grad)?;
        Ok(())
    }

    /// The stacks of an AP cycle by the fused passes: per stack the projection with its LLK
    /// partials and the object sums, then the probe sums from the projected exit waves it
    /// leaves in `f`. With a position, the projection adds the shifted probe's sums and each
    /// frame's position sums instead, and after the last stack the shifted probe's become the
    /// probe's step, which it takes here against the object the projection read: `true` then,
    /// with no probe sums left to apply. Under its windows, the stack's windows and their
    /// derivatives are sampled first; the projection reads the windows and adds the object
    /// sums over them, which are placed back after the position and probe sums.
    fn fused_ap(
        &mut self,
        ctx: &ExecutionContext,
        f: &mut FusedBufs,
        mut llk: Option<&mut RecordWriter<'_, Tag>>,
        obj: bool,
        probe_inertia: Option<f64>,
        one: OneSlice,
    ) -> Result<bool, GpuError> {
        let probe = probe_inertia.is_some();
        let dims = *self.views.dims();
        let (m, step) = (self.features).lend_shifted::<ShiftedStepSums>(
            ctx,
            self.kernels,
            &dims,
            self.obj,
            self.probe,
            self.exit,
        )?;
        let mut m = m.one_slice(one);
        let mut step = step.filter(|_| probe);
        for s in 0..self.scratch.len() {
            let stack = self.scratch.at(s);
            let windows = self.windows.as_deref_mut();
            if let Some(mut w) = m.windows(ctx, self.views, stack, windows, Sample::Derivatives)? {
                let read = w.read;
                let inputs = FusedInputs {
                    views: read.views,
                    counts: self.counts,
                    obj: read.blocks,
                    probe: self.probe,
                    model: m.model(),
                };
                let far = FarSums {
                    llk: llk.is_some().then_some(&mut *self.llk),
                    ..FarSums::default()
                };
                let sums = FusedSums {
                    obj: obj.then(|| w.zeroed_patches(ctx)).transpose()?,
                    ..FusedSums::default()
                };
                (self.kernels).fused_project(ctx, &inputs, stack, Exit::Ap { far }, f, sums)?;
                if let Some(rec) = &mut llk {
                    rec.push(ctx, &dims, stack, self.llk)?;
                }
                m.window_position_sums(ctx, self.grad, read, stack, f.work(), self.probe)?;
                if probe {
                    (self.kernels).psi_to_probe(
                        ctx,
                        read.views,
                        stack,
                        f.work(),
                        read.blocks,
                        self.probe_sums,
                    )?;
                }
                if obj {
                    m.place(ctx, self.views, stack, w.patches(), &mut self.obj_sums.num)?;
                }
                continue;
            }
            let inputs = FusedInputs {
                views: self.views,
                counts: self.counts,
                obj: self.obj,
                probe: self.probe,
                model: m.model(),
            };
            let far = FarSums {
                llk: llk.is_some().then_some(&mut *self.llk),
                ..FarSums::default()
            };
            let exit = Exit::Ap { far };
            let sums = FusedSums {
                obj: obj.then_some(&mut *self.obj_sums),
                shift: m.shift_sums(step.as_deref_mut().map(ShiftProbe::Step), true),
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
        let (Some(inertia), Some(step)) = (probe_inertia, step) else {
            return Ok(false);
        };
        (self.kernels).shifted_probe_sums(ctx, &dims, &mut *step, self.probe_sums)?;
        let inputs = FusedInputs {
            views: self.views,
            counts: self.counts,
            obj: self.obj,
            probe: self.probe,
            model: m.model(),
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
