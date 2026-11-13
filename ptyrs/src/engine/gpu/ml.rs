//! PyNX ML (Poisson conjugate gradient) on the GPU. The opening scale and the gradient pass
//! run as AP's projection does ([`super::GpuOptions::fused`]); the line search always by
//! M1's launches, under a position by the moved probes of [`Kernels::shifted_rows_fwd`].

use std::sync::Arc;

use crate::base::{C64, Stack};
use crate::gpu::{
    C32, CG_BLOCKS, Ended, Error as GpuError, Exit, FarSums, FusedBufs, FusedInputs, FusedSums,
    GammaCounts, Kernels, Model, PositionKernels, RecordWriter, Shift, SpectrumBuf, ViewDims,
};
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer};

use super::device::{MlBufs, PassesMut};
use super::{Bufs, Epoch, GpuPtycho, plane_len};
use crate::engine::ptycho::llk_due;
use crate::engine::{EngineError, MlParams};

impl GpuPtycho {
    /// [`crate::engine::Ptycho::ml`]: [`crate::engine::Ptycho::scale_obj_probe`] once, then per cycle the
    /// gradient of the estimate (its LLK taken from the same forward stacks), the
    /// Polak–Ribière direction and the Poisson quadratic step along it; each cycle is one
    /// graph and nothing reads back until the call's end. With stepped features, a pass of
    /// their own first takes the LLK and the far sums of the estimate under the cycle-start
    /// model and steps them, and the cycle models the stepped values. With a position, every
    /// pass models the probe moved by each frame's offsets, an AP projection of the estimate
    /// takes each frame's position sums after the features' step, and the offsets step after
    /// the update.
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
    /// [`crate::engine::Ptycho::obj_grad`] and [`crate::engine::Ptycho::probe_grad`] give them.
    pub fn ml_grads(&mut self) -> Result<[Stack<C64>; 2], EngineError> {
        let [obj, probe] = self.dev.ml_grads()?;
        let widen = |v: Vec<_>, dims| {
            let v = v
                .into_iter()
                .map(|c: crate::gpu::C32| C64::new(c.re.into(), c.im.into()));
            Stack::from_vec(dims, v.collect())
        };
        Ok([widen(obj, self.obj_dims)?, widen(probe, self.probe_dims)?])
    }
}

/// The line search's moved probes under the position model, per stack: `w[0]` the
/// estimate's and `w[1]` the probe direction's, `[nf_max * nprobe, ny, nx]` each as
/// [`PositionKernels::shift_w`] leaves them, and the probe direction's spectra.
pub(super) struct MlMoved {
    w: [DeviceBuffer<C32>; 2],
    spectrum: SpectrumBuf,
    kernels: PositionKernels,
}

impl MlMoved {
    /// For frames of `dims` in stacks of up to `nf_max`.
    pub(super) fn new(
        stream: &Arc<CudaStream>,
        kernels: &Kernels,
        dims: &ViewDims,
        nf_max: u32,
    ) -> Result<Self, EngineError> {
        let len = nf_max as usize * dims.nprobe as usize * plane_len(dims);
        Ok(Self {
            w: [
                DeviceBuffer::zeroed(stream, len)?,
                DeviceBuffer::zeroed(stream, len)?,
            ],
            spectrum: SpectrumBuf::zeroed(stream, dims)?,
            kernels: PositionKernels::load(kernels)?,
        })
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
        mut llk: Option<RecordWriter<'_>>,
    ) -> Result<Option<Ended>, GpuError> {
        let update = [params.update_object, params.update_probe];
        if first {
            for b in ml.h.iter_mut().chain(ml.d.iter_mut()) {
                crate::gpu::zero(ctx, b)?;
            }
        }
        let stepped = match passes {
            PassesMut::Fused(f) if self.features.stepped() => {
                self.step_features(ctx, f, llk.take())?
            }
            _ => None,
        };
        if let PassesMut::Fused(f) = passes
            && self.features.shifted()
        {
            self.projected_position_sums(ctx, f)?;
        }
        let ended = self.ml_grad(ctx, passes, ml, update, llk)?.or(stepped);

        // β from the gradients in the sums and the last ones in h (then h = g), and d.
        let lens = [self.obj_len, self.probe_len];
        crate::gpu::zero(ctx, ml.parts)?;
        for (i, g) in [&*self.obj_sums, &*self.probe_sums].into_iter().enumerate() {
            if update[i] {
                let first_pair = i as u32 * CG_BLOCKS;
                self.grad
                    .beta_parts(ctx, &g.num, &mut ml.h[i], lens[i], ml.parts, first_pair)?;
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

        // γ along d, then the step. Under a position, the fields are those of the moved
        // probes W of P and W' of dP: po = O W, a = dO W + O W', b = dO W'.
        let dims = *self.views.dims();
        let m = self.features.lend(ctx, self.kernels, &dims, self.probe)?;
        let obs = GammaCounts {
            counts: self.counts,
            model: Model {
                shift: None,
                ..m.model
            },
        };
        let mut moved = match (m.model.shift, ml.moved.as_deref_mut()) {
            (None, _) => None,
            (Some(shift), Some(mv)) => {
                let MlMoved {
                    w,
                    spectrum,
                    kernels,
                } = mv;
                let dshift = Shift {
                    spectrum: self
                        .kernels
                        .probe_spectrum(ctx, &dims, &ml.d[1], spectrum)?,
                    ..shift
                };
                Some((w, kernels, [shift, dshift]))
            }
            (Some(_), None) => {
                return Err(GpuError::Shape(
                    "ML state built for another position model".into(),
                ));
            }
        };
        for s in 0..self.stacks.len() {
            let stack = self.table.at(s);
            let frames = &self.stacks.get(s).frames;
            let d = [&ml.d[0], &ml.d[1]];
            match &mut moved {
                None => {
                    let x = [&*self.obj, &*self.probe];
                    self.grad
                        .gamma_fields(ctx, self.views, stack, x, d, ml.fields)?;
                }
                Some((w, kernels, shifts)) => {
                    let [w0, w1] = &mut **w;
                    for (w, probe, shift) in
                        [(&mut *w0, &*self.probe, shifts[0]), (w1, d[1], shifts[1])]
                    {
                        let inputs = FusedInputs {
                            views: self.views,
                            counts: self.counts,
                            obj: self.obj,
                            probe,
                            model: Model {
                                shift: Some(shift),
                                ..m.model
                            },
                        };
                        let work = &mut ml.fields.po;
                        self.kernels
                            .shifted_rows_fwd(ctx, &inputs, stack, w, work)?;
                        kernels.shift_w(ctx, &dims, stack, w)?;
                    }
                    let (rk, f) = (self.rpie, &mut *ml.fields);
                    let views = self.views;
                    rk.obj_wave_to_psi(ctx, views, stack, self.obj, w0, &mut f.po)?;
                    rk.obj_wave_to_psi(ctx, views, stack, d[0], w0, &mut f.a)?;
                    rk.obj_wave_to_psi(ctx, views, stack, self.obj, w1, &mut f.b)?;
                    self.kernels.add_c32(ctx, &mut f.a, &f.b, frames.len)?;
                    rk.obj_wave_to_psi(ctx, views, stack, d[0], w1, &mut f.b)?;
                }
            }
            let plan = ml.plans.get(frames.plan);
            for f in [&mut ml.fields.po, &mut ml.fields.a, &mut ml.fields.b] {
                plan.forward_inplace(ctx, f)?;
            }
            self.grad
                .gamma_frames(ctx, &dims, stack, obs, ml.fields, ml.gamma_parts)?;
        }
        let npart = crate::gpu::gamma_parts(&dims)?;
        self.grad.sum_parts(ctx, ml.gamma_parts, npart, ml.gamma)?;
        for (i, x) in [&mut *self.obj, &mut *self.probe].into_iter().enumerate() {
            if update[i] {
                self.grad
                    .step(ctx, x, &mut ml.d[i], &ml.h[i], lens[i], ml.gamma)?;
            }
        }
        self.features.position_step(ctx, self.grad)?;
        Ok(ended)
    }

    /// Each frame's position sums of the estimate under the model, from the AP projection of
    /// its exit waves, stack by stack; the pass writes nothing else the cycle reads.
    fn projected_position_sums(
        &mut self,
        ctx: &ExecutionContext,
        f: &mut FusedBufs,
    ) -> Result<(), GpuError> {
        let dims = *self.views.dims();
        let mut m = self.features.lend(ctx, self.kernels, &dims, self.probe)?;
        for s in 0..self.wide.len() {
            let inputs = FusedInputs {
                views: self.views,
                counts: self.counts,
                obj: self.obj,
                probe: self.probe,
                model: m.model,
            };
            let exit = Exit::Ap {
                far: FarSums::default(),
            };
            let sums = FusedSums {
                shift: m.shift_sums(false, true),
                ..FusedSums::default()
            };
            self.kernels
                .fused_project(ctx, &inputs, self.wide.at(s), exit, f, sums)?;
        }
        Ok(())
    }

    /// The Poisson gradients of the estimate into the update sums' `num` (object, probe, as
    /// `update` asks), stack by stack, with the LLK partials into `llk` if given; by the
    /// fused passes' [`Exit::Grad`] or by M1's launches (which model no features), as
    /// `passes` holds. Returns the ended record.
    pub(super) fn ml_grad(
        &mut self,
        ctx: &ExecutionContext,
        passes: &mut PassesMut<'_>,
        ml: &mut MlBufs<'_>,
        update: [bool; 2],
        mut llk: Option<RecordWriter<'_>>,
    ) -> Result<Option<Ended>, GpuError> {
        match passes {
            PassesMut::Fused(f) => self.fused_grad(ctx, f, update, llk.as_mut())?,
            PassesMut::M1(_) => {
                for s in 0..self.stacks.len() {
                    self.m1_grad(ctx, ml, s, update, llk.as_mut())?;
                }
            }
        }
        llk.map(|w| w.end(ctx)).transpose()
    }

    /// The stacks of [`Self::ml_grad`] by one fused pass each; with a position, the shifted
    /// probe's sums become the probe's after the last stack.
    fn fused_grad(
        &mut self,
        ctx: &ExecutionContext,
        f: &mut FusedBufs,
        update: [bool; 2],
        mut llk: Option<&mut RecordWriter<'_>>,
    ) -> Result<(), GpuError> {
        let dims = *self.views.dims();
        let mut m = self.features.lend(ctx, self.kernels, &dims, self.probe)?;
        for s in 0..self.wide.len() {
            let stack = self.wide.at(s);
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
            let exit = Exit::Grad { far };
            let shifted = m.shifted();
            let sums = FusedSums {
                obj: update[0].then_some(&mut *self.obj_sums),
                probe: (update[1] && !shifted).then_some(&mut *self.probe_sums),
                shift: m.shift_sums(update[1], false),
            };
            self.kernels
                .fused_project(ctx, &inputs, stack, exit, f, sums)?;
            if let Some(rec) = &mut llk {
                rec.push(ctx, &dims, stack, self.llk)?;
            }
        }
        if update[1] {
            m.probe_sums(ctx, self.kernels, &dims, self.probe_sums)?;
        }
        Ok(())
    }

    /// Stack `s` of [`Self::ml_grad`] by M1's launches in `ml.fields.po`: exit waves, forward
    /// FFT, the LLK partials, [`crate::gpu::GradKernels::poisson_grad`], inverse FFT,
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

    /// [`crate::engine::Ptycho::scale_obj_probe`] on the device, `total` being the unmasked observed
    /// photons: the calculated photons of the exit waves over the same unmasked pixels, each
    /// frame's times its intensity scale (stack by stack, by the LLK pass `passes` hold, the
    /// fused one under the position model), then [`Bufs::scale_obj_probe`], all without a
    /// host sync.
    pub(super) fn scale_to_photons(
        &mut self,
        ctx: &ExecutionContext,
        passes: &mut PassesMut<'_>,
        ml: &mut MlBufs<'_>,
        total: f32,
    ) -> Result<(), GpuError> {
        crate::gpu::zero(ctx, ml.icalc)?;
        let dims = *self.views.dims();
        match passes {
            PassesMut::M1(_) => {
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
                    let scale = self.features.scale();
                    self.grad
                        .icalc_add(ctx, &dims, stack, self.llk, scale, ml.icalc)?;
                }
            }
            PassesMut::Fused(f) => {
                let m = self.features.lend(ctx, self.kernels, &dims, self.probe)?;
                // The exit waves' photons alone: icalc_add scales them, and
                // scale_obj_probe takes the background's off the total.
                let model = Model {
                    shift: m.model.shift,
                    ..Model::default()
                };
                for s in 0..self.stacks.len() {
                    let stack = self.table.at(s);
                    let inputs = FusedInputs {
                        views: self.views,
                        counts: self.counts,
                        obj: self.obj,
                        probe: self.probe,
                        model,
                    };
                    let far = FarSums {
                        llk: Some(&mut *self.llk),
                        ..FarSums::default()
                    };
                    self.kernels.fused_llk(ctx, &inputs, stack, f, far)?;
                    self.grad
                        .icalc_add(ctx, &dims, stack, self.llk, m.model.scale, ml.icalc)?;
                }
            }
        }
        self.scale_obj_probe(ctx, Some((&*ml.icalc, total)))
    }
}
