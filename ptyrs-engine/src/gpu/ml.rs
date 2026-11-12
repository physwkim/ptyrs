//! PyNX ML (Poisson conjugate gradient) on the GPU. The gradient pass runs as AP's
//! projection does ([`super::GpuOptions::fused`]); the line search always by M1's launches.

use cuda_async::device_operation::ExecutionContext;
use ptyrs_core::{C64, Stack};
use ptyrs_gpu::{
    CG_BLOCKS, Ended, Error as GpuError, Exit, FusedBufs, FusedInputs, FusedSums, ObjProbeScale,
    RecordWriter,
};

use super::device::{MlBufs, PassesMut};
use super::{Bufs, Epoch, GpuPtycho};
use crate::ptycho::llk_due;
use crate::{EngineError, MlParams};

impl GpuPtycho {
    /// [`crate::Ptycho::ml`]: [`crate::Ptycho::scale_obj_probe`] once, then per cycle the
    /// gradient of the estimate (its LLK taken from the same forward stacks), the
    /// Polak–Ribière direction and the Poisson quadratic step along it; each cycle is one
    /// graph and nothing reads back until the call's end.
    pub fn ml(&mut self, params: &MlParams, ncycle: usize) -> Result<(), EngineError> {
        self.dev.scale_to_photons(self.total_obs as f32)?;
        for ic in 0..ncycle {
            let epoch = Epoch::Ml {
                params: *params,
                first: ic == 0,
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

    /// The ML gradients of the current estimate in `f32`, object then probe, as
    /// [`crate::Ptycho::obj_grad`] and [`crate::Ptycho::probe_grad`] give them.
    pub fn ml_grads(&mut self) -> Result<[Stack<C64>; 2], EngineError> {
        let [obj, probe] = self.dev.ml_grads()?;
        let widen = |v: Vec<_>, dims| {
            let v = v
                .into_iter()
                .map(|c: ptyrs_gpu::C32| C64::new(c.re.into(), c.im.into()));
            Stack::from_vec(dims, v.collect())
        };
        Ok([widen(obj, self.obj_dims)?, widen(probe, self.probe_dims)?])
    }
}

impl Bufs<'_> {
    /// One ML cycle enqueued on `ctx`; with `llk`, the record of the estimate it starts from.
    /// Cycle 0 of a call (`first`) starts from no last gradient and no direction.
    pub(super) fn ml_epoch(
        &mut self,
        ctx: &ExecutionContext,
        passes: &mut PassesMut<'_>,
        ml: &mut MlBufs<'_>,
        params: &MlParams,
        first: bool,
        llk: Option<RecordWriter<'_>>,
    ) -> Result<Option<Ended>, GpuError> {
        let update = [params.update_object, params.update_probe];
        if first {
            for b in ml.h.iter_mut().chain(ml.d.iter_mut()) {
                ptyrs_gpu::zero(ctx, b)?;
            }
        }
        let ended = self.ml_grad(ctx, passes, ml, update, llk)?;

        // β from the gradients in the sums and the last ones in h (then h = g), and d.
        let lens = [self.obj_len, self.probe_len];
        ptyrs_gpu::zero(ctx, ml.parts)?;
        for (i, g) in [&*self.obj_sums, &*self.probe_sums].into_iter().enumerate() {
            if update[i] {
                let (h, first_pair) = (&mut ml.h[i], i as u32 * CG_BLOCKS);
                match self.tile_beta {
                    Some(t) => t.beta_parts(ctx, &g.num, h, lens[i], ml.parts, first_pair)?,
                    None => self
                        .grad
                        .beta_parts(ctx, &g.num, h, lens[i], ml.parts, first_pair)?,
                }
            }
        }
        self.grad.sum_parts(ctx, ml.parts, 2 * CG_BLOCKS, ml.beta)?;
        for (i, g) in [&mut *self.obj_sums, &mut *self.probe_sums]
            .into_iter()
            .enumerate()
        {
            if update[i] {
                self.grad
                    .direction(ctx, &mut ml.d[i], g, lens[i], ml.beta)?;
            }
        }

        // γ along d, then the step.
        let dims = *self.views.dims();
        for s in 0..self.stacks.len() {
            let stack = self.table.at(s);
            let x = [&*self.obj, &*self.probe];
            let d = [&ml.d[0], &ml.d[1]];
            self.grad
                .gamma_fields(ctx, self.views, stack, x, d, ml.fields)?;
            let plan = ml.plans.get(self.stacks.get(s).frames.plan);
            for f in [&mut ml.fields.po, &mut ml.fields.a, &mut ml.fields.b] {
                plan.forward_inplace(ctx, f)?;
            }
            self.grad
                .gamma_frames(ctx, &dims, stack, self.counts, ml.fields, ml.gamma_parts)?;
        }
        let npart = ptyrs_gpu::gamma_parts(&dims)?;
        self.grad.sum_parts(ctx, ml.gamma_parts, npart, ml.gamma)?;
        for (i, x) in [&mut *self.obj, &mut *self.probe].into_iter().enumerate() {
            if update[i] {
                self.grad
                    .step(ctx, x, &mut ml.d[i], &ml.h[i], lens[i], ml.gamma)?;
            }
        }
        Ok(ended)
    }

    /// The Poisson gradients of the estimate into the update sums' `num` (object, probe, as
    /// `update` asks), stack by stack, with the LLK partials into `llk` if given; by the
    /// fused passes' [`Exit::Grad`] or by M1's launches, as `passes` holds. Returns the ended
    /// record.
    pub(super) fn ml_grad(
        &mut self,
        ctx: &ExecutionContext,
        passes: &mut PassesMut<'_>,
        ml: &mut MlBufs<'_>,
        update: [bool; 2],
        mut llk: Option<RecordWriter<'_>>,
    ) -> Result<Option<Ended>, GpuError> {
        for s in 0..self.stacks.len() {
            match passes {
                PassesMut::Fused(f) => self.fused_grad(ctx, f, s, update, llk.as_mut())?,
                PassesMut::M1(_) => self.m1_grad(ctx, ml, s, update, llk.as_mut())?,
            }
        }
        llk.map(|w| w.end(ctx)).transpose()
    }

    /// Stack `s` of [`Self::ml_grad`] by one fused pass.
    fn fused_grad(
        &mut self,
        ctx: &ExecutionContext,
        f: &mut FusedBufs,
        s: usize,
        update: [bool; 2],
        llk: Option<&mut RecordWriter<'_>>,
    ) -> Result<(), GpuError> {
        let dims = *self.views.dims();
        let stack = self.table.at(s);
        let inputs = FusedInputs {
            views: self.views,
            counts: self.counts,
            obj: self.obj,
            probe: self.probe,
        };
        let exit = Exit::Grad {
            llk: llk.is_some().then_some(&mut *self.llk),
        };
        let sums = FusedSums {
            obj: update[0].then_some(&mut *self.obj_sums),
            probe: update[1].then_some(&mut *self.probe_sums),
        };
        self.kernels
            .fused_project(ctx, &inputs, stack, exit, f, sums)?;
        if let Some(rec) = llk {
            rec.push(ctx, &dims, stack, self.llk)?;
        }
        Ok(())
    }

    /// Stack `s` of [`Self::ml_grad`] by M1's launches in `ml.fields.po`: exit waves, forward
    /// FFT, the LLK partials, [`ptyrs_gpu::GradKernels::poisson_grad`], inverse FFT,
    /// back-projection.
    fn m1_grad(
        &mut self,
        ctx: &ExecutionContext,
        ml: &mut MlBufs<'_>,
        s: usize,
        update: [bool; 2],
        llk: Option<&mut RecordWriter<'_>>,
    ) -> Result<(), GpuError> {
        let dims = *self.views.dims();
        let stack = self.table.at(s);
        let psi = &mut ml.fields.po;
        self.kernels
            .obj_probe_to_psi(ctx, self.views, stack, self.obj, self.probe, psi)?;
        let plan = ml.plans.get(self.stacks.get(s).frames.plan);
        plan.forward_inplace(ctx, psi)?;
        if let Some(rec) = llk {
            self.kernels
                .llk(ctx, stack, self.counts, psi, self.llk_scratch, self.llk)?;
            rec.push(ctx, &dims, stack, self.llk)?;
        }
        self.grad
            .poisson_grad(ctx, &dims, stack, self.counts, psi)?;
        plan.inverse_inplace(ctx, psi)?;
        if update[0] {
            self.kernels
                .psi_to_obj(ctx, self.views, stack, psi, self.probe, self.obj_sums)?;
        }
        if update[1] {
            self.kernels
                .psi_to_probe(ctx, self.views, stack, psi, self.obj, self.probe_sums)?;
        }
        Ok(())
    }

    /// [`crate::Ptycho::scale_obj_probe`] on the device, `total` being the unmasked observed
    /// photons: the calculated photons over the same unmasked pixels (stack by stack, by the
    /// LLK kernel on the far field in `ml.fields.po`), then [`Bufs::scale_obj_probe`], all
    /// without a host sync.
    pub(super) fn scale_to_photons(
        &mut self,
        ctx: &ExecutionContext,
        ml: &mut MlBufs<'_>,
        total: f32,
    ) -> Result<(), GpuError> {
        ptyrs_gpu::zero(ctx, ml.icalc)?;
        let dims = *self.views.dims();
        let psi = &mut ml.fields.po;
        for s in 0..self.stacks.len() {
            let stack = self.table.at(s);
            self.kernels
                .obj_probe_to_psi(ctx, self.views, stack, self.obj, self.probe, psi)?;
            ml.plans
                .get(self.stacks.get(s).frames.plan)
                .forward_inplace(ctx, psi)?;
            self.kernels
                .llk(ctx, stack, self.counts, psi, self.llk_scratch, self.llk)?;
            self.grad.icalc_add(ctx, &dims, stack, self.llk, ml.icalc)?;
        }
        let icalc = &*ml.icalc;
        self.scale_obj_probe(ctx, ObjProbeScale::Absolute { icalc, total })
    }
}
