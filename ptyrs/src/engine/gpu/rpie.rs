//! pty-chi rPIE on the GPU, in the call sequence of `crate::gpu::RpieKernels`.

use std::sync::Arc;

use crate::engine::sampling::Sampling;
use crate::gpu::sampling::{Modes, StackPoints, WindowBlock, window_views};
use crate::gpu::{
    C32, EpochOrder, Error as GpuError, FrameWaves, FusedBufs, FusedInputs, Kernels, Model,
    PositionRule, PositionStep, PowerMax, PsiSums, RpieKernels, ShiftProbe, SliceSource, Solve,
    Views, Weight,
};
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer};

use super::device::{
    FeaturesMut, Moved, MovedLent, MovedScratch, PositionLent, RpieBufs, another_model,
    moved_inputs, moved_shift,
};
use super::propagate::Propagate;
use super::{Bufs, Cut, Device, Epoch, GpuPtycho, batch_frames, stack_len};
use crate::engine::features::{POSITION_LIMIT, POSITION_STEP};
use crate::engine::ptycho::{Algorithm, llk_due};
use crate::engine::{EngineError, RpieParams};

impl GpuPtycho {
    /// [`crate::engine::Ptycho::rpie`]: consecutive batches of `batch_size` frames, each
    /// stepping the object and probe from the batch-start arrays, and under
    /// [`crate::engine::Features::position`] its frames' offsets; one LLK record after every
    /// due epoch.
    pub fn rpie(&mut self, params: &RpieParams, nepoch: usize) -> Result<(), EngineError> {
        let check = |dev: &Device| {
            let dims = dev.dims();
            if dims.nobj != 1 {
                return Err(EngineError::Invalid(
                    "rPIE needs exactly one object mode".into(),
                ));
            }
            params.validate()?;
            if dev.models_scale_or_background() {
                return Err(EngineError::Invalid(
                    "rPIE models no intensity scale or background".into(),
                ));
            }
            dev.one_unitary("Rpie")?;
            stack_len(&dims, batch_frames(&dims, params.batch_size) as usize)?;
            Ok(())
        };
        self.call(false, check, |dev, cycle| {
            for ic in 0..nepoch {
                dev.steady(Epoch::Rpie(*params), *cycle)?;
                if llk_due(params.calc_llk, ic, nepoch) {
                    dev.llk_pass(Some((*cycle, Algorithm::Rpie)))?;
                }
                *cycle += 1;
            }
            Ok(())
        })
    }
}

/// rPIE's state under the position model: a batch's moved probe, the scan order its offsets
/// step in, what its batches read by the device's [`Sampling`], and the offsets' step if the
/// position is refined.
pub(super) struct RpieMoved {
    probes: Moved,
    /// Never set: rPIE's batches take the frames in scan order.
    order: EpochOrder,
    read: RpieRead,
    step: Option<PositionStep>,
}

/// What rPIE's batches read under the position model, by the device's [`Sampling`].
enum RpieRead {
    /// [`Sampling::Patch`]: the scan's views, with the shifted sums' scratch, its own as a
    /// batch is one stack, which may outgrow the device's.
    Patch(Box<FusedBufs>),
    /// [`Sampling::FullObject`]: the batches' windows.
    FullObject(Box<RpieWindows>),
}

/// rPIE's windows under [`Sampling::FullObject`]: the views over its batches' windows, a
/// batch's windows `[nobj, nf_max, ny, nx]`, the views' object, and the object's numerator
/// over the windows.
struct RpieWindows {
    views: Views,
    win: DeviceBuffer<C32>,
    patches: DeviceBuffer<C32>,
}

impl RpieMoved {
    /// For `cut`'s batches, one stack each, of `views` under the position model by `moved`.
    pub(super) fn new(
        stream: &Arc<CudaStream>,
        kernels: &Kernels,
        views: &Views,
        cut: &Cut,
        moved: MovedScratch,
    ) -> Result<Self, EngineError> {
        let dims = views.dims();
        let nf_max = cut.table.nf_max();
        let sampling = moved.sampling.into();
        let read = match sampling {
            Sampling::Patch => {
                RpieRead::Patch(Box::new(FusedBufs::with_shift(stream, dims, nf_max)?))
            }
            Sampling::FullObject => {
                let stacks: Vec<_> = cut.frames.iter().map(|f| f.range.clone()).collect();
                let views = window_views(stream, dims, &stacks)?;
                let len = views.dims().obj_len();
                RpieRead::FullObject(Box::new(RpieWindows {
                    win: DeviceBuffer::zeroed(stream, len)?,
                    patches: DeviceBuffer::zeroed(stream, len)?,
                    views,
                }))
            }
        };
        Ok(Self {
            probes: Moved::new(stream, kernels, dims, nf_max, 1, sampling)?,
            order: EpochOrder::new(kernels, stream, views)?,
            read,
            step: (moved.refined)
                .then(|| PositionStep::zeroed(stream, dims, nf_max))
                .transpose()?,
        })
    }

    /// `moved` apart: what [`RpieLend::lend`] lends with the device's `features`, under
    /// [`Sampling::FullObject`] the windows, which a batch reads while the features are lent
    /// to sample them, and the offsets' step as `features` refine the position: an error when
    /// `moved` was built for the other.
    fn split<'m>(
        moved: Option<&'m mut Self>,
        features: &FeaturesMut<'_>,
    ) -> Result<RpieSplit<'m>, GpuError> {
        let Some(m) = moved else {
            return Ok((RpieLend(None, None), None, features.own_step(None)?));
        };
        let (fused, windows) = match &mut m.read {
            RpieRead::Patch(f) => (Some(&mut **f), None),
            RpieRead::FullObject(w) => (None, Some(&mut **w)),
        };
        let step = features.own_step(m.step.as_mut())?;
        Ok((
            RpieLend(Some((&mut m.probes, &m.order)), fused),
            windows,
            step,
        ))
    }
}

/// [`RpieMoved::split`]: its lend, its windows and its step.
type RpieSplit<'m> = (
    RpieLend<'m>,
    Option<&'m mut RpieWindows>,
    Option<&'m mut PositionStep>,
);

/// [`RpieMoved`] but its windows and step: its [`Moved`], its scan order and the shifted sums'
/// scratch.
struct RpieLend<'m>(
    Option<(&'m mut Moved, &'m EpochOrder)>,
    Option<&'m mut FusedBufs>,
);

impl<'m> RpieLend<'m> {
    /// With the device's `features`, its scan order as its own ([`PositionLent::lend`]).
    fn lend(
        self,
        features: &'m mut FeaturesMut<'_>,
    ) -> Result<Option<PositionLent<'m, &'m EpochOrder>>, GpuError> {
        PositionLent::lend(self.0, self.1, features)
    }
}

/// What a batch launches with but the arrays, copied out of [`Bufs`] to leave them free to
/// lend.
#[derive(Clone, Copy)]
struct Launch<'a> {
    k: &'a Kernels,
    rk: &'a RpieKernels,
    prop: Propagate<'a>,
    counts: &'a DeviceBuffer<u16>,
}

/// The offsets' step from a batch's position sums of waves of intensity scale `scale`.
fn solve(scale: f32) -> Solve {
    Solve {
        rule: PositionRule {
            step: POSITION_STEP as f32,
            limit: POSITION_LIMIT as f32,
        },
        scale,
    }
}

impl<'a> Bufs<'a> {
    /// One epoch enqueued on `ctx`, `moved` rPIE's state under the position model.
    pub(super) fn rpie_epoch(
        &mut self,
        ctx: &ExecutionContext,
        params: &RpieParams,
        rb: &mut RpieBufs<'_>,
        mut moved: Option<&mut RpieMoved>,
    ) -> Result<(), GpuError> {
        // Each frame is in one batch, stepped after its batch's passes: the offsets the epoch
        // starts from serve every batch.
        let (lend, ..) = RpieMoved::split(moved.as_deref_mut(), &self.features)?;
        if let Some(mut m) = lend.lend(&mut self.features)? {
            let order = *m.own();
            m.gather(ctx, order)?;
        }
        for b in 0..rb.batches.len() {
            match RpieMoved::split(moved.as_deref_mut(), &self.features)? {
                (lend, None, step) => self.rpie_batch(ctx, params, rb, b, (lend, step))?,
                (lend, Some(w), step) => {
                    self.rpie_window_batch(ctx, params, rb, b, (lend, step), w)?
                }
            }
        }
        Ok(())
    }

    fn launch(&self) -> Launch<'a> {
        Launch {
            k: self.kernels,
            rk: self.rpie,
            prop: Propagate {
                kernels: self.bh,
                stage: self.exit.stage,
                dims: *self.views.dims(),
            },
            counts: self.counts,
        }
    }

    /// One batch of an epoch enqueued on `ctx`, its views on the object; with `lend` under
    /// [`Sampling::Patch`], each frame's probe moved by its offset, which `step` steps.
    fn rpie_batch(
        &mut self,
        ctx: &ExecutionContext,
        params: &RpieParams,
        rb: &mut RpieBufs<'_>,
        b: usize,
        (lend, mut step): (RpieLend<'_>, Option<&mut PositionStep>),
    ) -> Result<(), GpuError> {
        let l = self.launch();
        let Launch { k, rk, .. } = l;
        let (frames, stack) = (&rb.batches[b], rb.table.at(b));
        let dims = l.prop.dims;
        let n = rb.n;
        let nk = frames.nf() * dims.nmode;
        let moved = lend.lend(&mut self.features)?;
        let moved = (moved.map(MovedLent::try_from)).transpose()?;
        let mut moved = (moved.map(|m| m.batch(ctx, k, &dims, self.probe))).transpose()?;
        let base = FusedInputs {
            views: self.views,
            counts: self.counts,
            obj: self.obj,
            probe: self.probe,
            model: Model::default(),
        };

        // Δ = Aᴴ ψ' / n − ψ, A the propagator, ψ = O_v W with W the moved probe under positions.
        match &mut moved {
            Some(m) => {
                let inputs = moved_inputs(&base, moved_shift(m.pos, m.spectrum));
                let w = &mut m.w[0];
                k.shifted_probe(ctx, &inputs, stack, w)?;
                m.kernels.shift_w(ctx, &dims, stack, w)?;
                rk.obj_wave_to_psi(ctx, self.views, stack, self.obj, w, rb.psi)?;
            }
            None => k.obj_probe_to_psi(ctx, self.views, stack, self.obj, self.probe, rb.psi)?,
        }
        rb.delta(ctx, l, b)?;
        if let (Some(m), Some(step)) = (&moved, step.as_deref_mut()) {
            // The slopes of the object the batch started from, against Δ before the probe
            // step weights it.
            let chi_w = (&*rb.dlt, FrameWaves::Probes(&m.w[0]));
            m.kernels
                .sums(ctx, self.views, stack, self.obj, chi_w, step)?;
        }

        if params.obj_step.is_some() {
            match &mut moved {
                // `conj(W) Δ / ((1 − a) |W|² + a max Σ_m |W_m|²)`, the max over the batch.
                Some(m) => {
                    let groups = PowerMax {
                        ngroup: frames.nf(),
                        nsum: dims.nprobe,
                        n,
                    };
                    rk.power_max_all(ctx, &m.w[0], groups, rb.max_chunks, rb.pmax)?;
                    let spec = Weight {
                        nk,
                        n,
                        per_plane: false,
                        alpha: params.obj_alpha as f32,
                    };
                    rk.weight(ctx, rb.dlt, &m.w[0], rb.pmax, rb.psi, spec)?;
                    let sums = PsiSums {
                        obj: Some(&mut self.obj_sums.num),
                        probe: None,
                    };
                    let inputs = moved_inputs(&base, moved_shift(m.pos, m.spectrum));
                    k.shifted_psi_sums(ctx, &inputs, stack, rb.psi, m.fused, sums)?;
                }
                None => {
                    rb.ptilde(ctx, l, self.probe, params.obj_alpha)?;
                    k.psi_to_obj(
                        ctx,
                        self.views,
                        stack,
                        rb.dlt,
                        rb.ptilde,
                        &mut self.obj_sums.num,
                    )?;
                }
            }
        }
        if params.probe_step.is_some() {
            rb.probe_weight(ctx, l, b, (self.views, self.obj), params.probe_alpha)?;
            match &mut moved {
                // `conj(q) T_−d(conj(O) Δ / ((1 − a) |O|² + a max |O|²))`.
                Some(m) => {
                    let sums = PsiSums {
                        obj: None,
                        probe: Some(ShiftProbe::Num(&mut *m.probe_sums)),
                    };
                    let inputs = moved_inputs(&base, moved_shift(m.pos, m.spectrum));
                    k.shifted_psi_sums(ctx, &inputs, stack, rb.dlt, m.fused, sums)?;
                    k.shifted_probe_sums(ctx, &dims, m.probe_sums, &mut self.probe_sums.num)?;
                }
                None => k.psi_to_probe(
                    ctx,
                    self.views,
                    stack,
                    rb.dlt,
                    self.obj,
                    &mut self.probe_sums.num,
                )?,
            }
        }
        // The offsets' step reads neither array, so it may precede their steps, which end
        // the borrow of the batch-start probe.
        if let (Some(m), Some(step)) = (moved, step) {
            let batch = frames.range.clone();
            let solve = solve(self.exit.scale);
            let offsets = m.offsets;
            m.kernels.step(ctx, batch, m.own, offsets, solve, step)?;
        }
        self.rpie_steps(ctx, params, frames.nf())
    }

    /// [`Self::rpie_batch`] under [`Sampling::FullObject`], its views on `w`, the windows of the
    /// batch-start object sampled at the offsets `lend` lends and `step` steps: the probe
    /// stays, the object's sums are placed back from the windows, and the position sums read
    /// their derivatives.
    fn rpie_window_batch(
        &mut self,
        ctx: &ExecutionContext,
        params: &RpieParams,
        rb: &mut RpieBufs<'_>,
        b: usize,
        (lend, mut step): (RpieLend<'_>, Option<&mut PositionStep>),
        w: &mut RpieWindows,
    ) -> Result<(), GpuError> {
        let l = self.launch();
        let (frames, stack) = (&rb.batches[b], rb.table.at(b));
        let dims = l.prop.dims;
        let mut m = (self.features)
            .lend(ctx, l.k, &dims, self.obj, self.probe, self.exit)?
            .one_slice(rb.one);
        let block = |order, modes, block| WindowBlock {
            slice: 0,
            order,
            modes,
            block,
        };
        m.windows_of(
            ctx,
            self.views,
            stack,
            None,
            block([0, 0], Modes::All, 0),
            &mut w.win,
        )?;
        l.k.obj_probe_to_psi(ctx, &w.views, stack, &w.win, self.probe, rb.psi)?;
        rb.delta(ctx, l, b)?;
        if params.obj_step.is_some() {
            rb.ptilde(ctx, l, self.probe, params.obj_alpha)?;
            crate::gpu::zero(ctx, &mut w.patches)?;
            (l.k).psi_to_obj(ctx, &w.views, stack, rb.dlt, rb.ptilde, &mut w.patches)?;
            m.place(ctx, self.views, stack, &w.patches, &mut self.obj_sums.num)?;
        }
        let Some(PositionLent::FullObject(s)) = lend.lend(&mut self.features)? else {
            return Err(another_model());
        };
        if let Some(step) = step.as_deref_mut() {
            // The slopes of the windows the batch started from, against Δ before the probe
            // step weights it.
            let points = StackPoints {
                views: self.views,
                stack,
                offsets: s.offsets,
            };
            let derivatives = |out: &mut DeviceBuffer<C32>| {
                for (axis, order) in (0..).zip([[1, 0], [0, 1]]) {
                    let at = block(order, Modes::First, axis);
                    s.sampler.windows(ctx, points, s.spectra, at, out)?;
                }
                Ok(())
            };
            let at = (&*rb.dlt, SliceSource::Probe(self.probe));
            (s.kernels).derivative_sums(ctx, self.views, stack, derivatives, at, step)?;
        }
        if params.probe_step.is_some() {
            rb.probe_weight(ctx, l, b, (&w.views, &w.win), params.probe_alpha)?;
            (l.k).psi_to_probe(
                ctx,
                &w.views,
                stack,
                rb.dlt,
                &w.win,
                &mut self.probe_sums.num,
            )?;
        }
        if let Some(step) = step {
            let solve = solve(self.exit.scale);
            let offsets = s.offsets;
            (s.kernels).step(ctx, frames.range.clone(), s.own, offsets, solve, step)?;
        }
        self.rpie_steps(ctx, params, frames.nf())
    }

    /// The object's and the probe's steps from their sums, the probe's over the batch's `nf`
    /// frames.
    fn rpie_steps(
        &mut self,
        ctx: &ExecutionContext,
        params: &RpieParams,
        nf: u32,
    ) -> Result<(), GpuError> {
        let rk = self.rpie;
        if let Some(step) = params.obj_step {
            rk.step(
                ctx,
                self.obj,
                &mut self.obj_sums.num,
                self.obj_len,
                step as f32,
            )?;
        }
        if let Some(step) = params.probe_step {
            let s = (step / nf as f64) as f32;
            rk.step(ctx, self.probe, &mut self.probe_sums.num, self.probe_len, s)?;
        }
        Ok(())
    }
}

impl RpieBufs<'_> {
    /// Batch `b`'s `Δ = Aᴴ ψ' / n − ψ` into `dlt`, A the propagator, of its exit waves ψ in
    /// `psi`, which it leaves negated.
    fn delta(&mut self, ctx: &ExecutionContext, l: Launch<'_>, b: usize) -> Result<(), GpuError> {
        let (frames, stack) = (&self.batches[b], self.table.at(b));
        let plan = self.plans.get(frames.plan);
        l.prop.forward_from(ctx, plan, stack, self.psi, self.dlt)?;
        l.rk.amplitude(ctx, &l.prop.dims, stack, l.counts, self.dlt)?;
        l.prop.adjoint(ctx, plan, stack, self.dlt)?;
        l.k.scale_c32(ctx, self.psi, frames.len, -1.0)?;
        l.k.add_c32(ctx, self.dlt, self.psi, frames.len)
    }

    /// `ptilde = P / ((1 − a) |P|² + a max Σ_m |P_m|²)` of `probe`, the max over its pixels:
    /// the object step's weight where the views read the probe unmoved.
    fn ptilde(
        &mut self,
        ctx: &ExecutionContext,
        l: Launch<'_>,
        probe: &DeviceBuffer<C32>,
        alpha: f64,
    ) -> Result<(), GpuError> {
        let (nprobe, n) = (l.prop.dims.nprobe, self.n);
        let groups = PowerMax {
            ngroup: 1,
            nsum: nprobe,
            n,
        };
        l.rk.power_max(ctx, probe, groups, self.max_chunks, self.pmax)?;
        let spec = Weight {
            nk: nprobe,
            n,
            per_plane: false,
            alpha: alpha as f32,
        };
        l.rk.weight(ctx, probe, probe, self.pmax, self.ptilde, spec)
    }

    /// Batch `b`'s `Δ` in `dlt` weighted for the probe step against `obj`, which `views`
    /// read: `Δ / ((1 − a) |O_v|² + a max |O_v|²)` per view, leaving `O_v q` in `psi`.
    fn probe_weight(
        &mut self,
        ctx: &ExecutionContext,
        l: Launch<'_>,
        b: usize,
        (views, obj): (&Views, &DeviceBuffer<C32>),
        alpha: f64,
    ) -> Result<(), GpuError> {
        let (nk, n) = (self.batches[b].nf() * l.prop.dims.nmode, self.n);
        (l.k).obj_probe_to_psi(ctx, views, self.table.at(b), obj, self.ones, self.psi)?;
        let groups = PowerMax {
            ngroup: nk,
            nsum: 1,
            n,
        };
        l.rk.power_max(ctx, self.psi, groups, self.max_chunks, self.omax)?;
        let spec = Weight {
            nk,
            n,
            per_plane: true,
            alpha: alpha as f32,
        };
        l.rk.weight_inplace(ctx, self.dlt, self.psi, self.omax, spec)
    }
}
