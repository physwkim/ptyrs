//! rPIE on the GPU, in the call sequence of `crate::gpu::RpieKernels`: pty-chi's, its object
//! step normalised by the batch's overlap.

use std::sync::Arc;

use crate::engine::sampling::Sampling;
use crate::gpu::sampling::{Modes, StackPoints, WindowBlock, window_views};
use crate::gpu::{
    BhScan, C32, EpochOrder, Error as GpuError, FrameWaves, FusedBufs, FusedInputs, Kernels, Model,
    PositionRule, PositionStep, PowerMax, PsiSums, RpieKernels, ShiftProbe, SliceSource, Solve,
    Views, Weight,
};
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer};

use super::device::{
    FeaturesMut, Moved, MovedLent, MovedScratch, PositionLent, RpieBufs, SampledLent,
    another_model, moved_inputs, moved_shift,
};
use super::propagate::Propagate;
use super::{Bufs, Cut, Device, GpuPtycho, batch_frames, stack_len};
use crate::engine::features::{POSITION_LIMIT, POSITION_STEP};
use crate::engine::ptycho::{Algorithm, llk_due};
use crate::engine::{EngineError, RpieParams, random_batches};

impl GpuPtycho {
    /// [`crate::engine::Ptycho::rpie`]: each epoch's [`random_batches`] of `batch_size` frames
    /// from the params' seed and the epoch's cycle, each batch stepping the object and probe
    /// from the batch-start arrays, and under [`crate::engine::Features::position`] its
    /// frames' offsets; one LLK record after every due epoch.
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
            let nframe = dev.dims().nframe as usize;
            for ic in 0..nepoch {
                let epoch = *cycle as u64;
                let order: Vec<u32> = random_batches(nframe, params.batch_size, params.seed, epoch)
                    .concat()
                    .into_iter()
                    .map(|f| f as u32)
                    .collect();
                dev.rpie_epoch(*params, *cycle, &order)?;
                if llk_due(params.calc_llk, ic, nepoch) {
                    dev.llk_pass(Some((*cycle, Algorithm::Rpie)))?;
                }
                *cycle += 1;
            }
            Ok(())
        })
    }
}

/// rPIE's state under the position model: a batch's moved probe, what its batches read by the
/// device's [`Sampling`], and the offsets' step if the position is refined.
pub(super) struct RpieMoved {
    probes: Moved,
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
            read,
            step: (moved.refined)
                .then(|| PositionStep::zeroed(stream, dims, nf_max))
                .transpose()?,
        })
    }

    /// `moved` apart: what [`RpieLend::lend`] lends with the device's `features` and the
    /// epoch's `order`, under [`Sampling::FullObject`] the windows, and the offsets' step as
    /// `features` refine the position: an error when `moved` was built for the other.
    fn split<'m>(
        moved: Option<&'m mut Self>,
        order: &'m EpochOrder,
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
        Ok((RpieLend(Some((&mut m.probes, order)), fused), windows, step))
    }
}

/// [`RpieMoved::split`]: its lend, its windows and its step.
type RpieSplit<'m> = (
    RpieLend<'m>,
    Option<&'m mut RpieWindows>,
    Option<&'m mut PositionStep>,
);

/// [`RpieMoved`] but its windows and step: its [`Moved`], the epoch's order and the shifted
/// sums' scratch.
struct RpieLend<'m>(
    Option<(&'m mut Moved, &'m EpochOrder)>,
    Option<&'m mut FusedBufs>,
);

impl<'m> RpieLend<'m> {
    /// With the device's `features`, the epoch's order as its own ([`PositionLent::lend`]).
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
        // starts from, by epoch position, serve every batch.
        let order = rb.order;
        let (lend, ..) = RpieMoved::split(moved.as_deref_mut(), order, &self.features)?;
        if let Some(mut m) = lend.lend(&mut self.features)? {
            m.gather(ctx, order)?;
        }
        for b in 0..rb.batches.len() {
            match RpieMoved::split(moved.as_deref_mut(), order, &self.features)? {
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

    /// One batch of an epoch enqueued on `ctx`, the epoch order's views on the object; with
    /// `lend` under [`Sampling::Patch`], each frame's probe moved by its offset, which `step`
    /// steps.
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
        let order = rb.order;
        let views = order.views();
        let dims = l.prop.dims;
        let moved = lend.lend(&mut self.features)?;
        let moved = (moved.map(MovedLent::try_from)).transpose()?;
        let mut moved = (moved.map(|m| m.batch(ctx, k, &dims, self.probe))).transpose()?;
        let base = FusedInputs {
            views,
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
                rk.obj_wave_to_psi(ctx, views, stack, self.obj, w, rb.psi)?;
            }
            None => k.obj_probe_to_psi(ctx, views, stack, self.obj, self.probe, rb.psi)?,
        }
        rb.delta(ctx, l, b)?;
        if let (Some(m), Some(step)) = (&moved, step.as_deref_mut()) {
            // The slopes of the object the batch started from, against Δ before the probe
            // step weights it.
            let chi_w = (&*rb.dlt, FrameWaves::Probes(&m.w[0]));
            m.kernels.sums(ctx, views, stack, self.obj, chi_w, step)?;
        }

        if params.obj_step.is_some() {
            // `conj(W) Δ / ((1 − a) d + a max Σ_m d)`, `d` each mode's `|W|²` placed over the
            // batch and read back per view.
            for m in 0..dims.nprobe {
                match &moved {
                    Some(mv) => rk.overlap_moved(ctx, views, stack, &mv.w[0], m, rb.overlap)?,
                    None => rk.overlap(ctx, views, stack, self.probe, m, rb.overlap)?,
                }
                rk.overlap_read(ctx, views, stack, rb.overlap, m, rb.dens)?;
            }
            rb.obj_weight(ctx, l, b, params.obj_alpha)?;
            match &mut moved {
                Some(m) => {
                    let sums = PsiSums {
                        obj: Some(&mut self.obj_sums.num),
                        probe: None,
                    };
                    let inputs = moved_inputs(&base, moved_shift(m.pos, m.spectrum));
                    k.shifted_psi_sums(ctx, &inputs, stack, rb.psi, m.fused, sums)?;
                }
                None => k.psi_to_obj(
                    ctx,
                    views,
                    stack,
                    rb.psi,
                    self.probe,
                    &mut self.obj_sums.num,
                )?,
            }
        }
        if params.probe_step.is_some() {
            rb.probe_weight(ctx, l, b, (views, self.obj), params.probe_alpha)?;
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
                    views,
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
    /// batch-start object sampled at the offsets by epoch position `lend` lends, which `step`
    /// steps: the probe stays, the object's sums are placed back from the windows, and the
    /// position sums read their derivatives. The object step's overlap is placed and read
    /// back through the sampler.
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
        let views = rb.order.views();
        let (nprobe, n) = (l.prop.dims.nprobe, rb.n);
        let Some(PositionLent::FullObject(s)) = lend.lend(&mut self.features)? else {
            return Err(another_model());
        };
        let SampledLent {
            own,
            pos,
            kernels,
            offsets,
            sampler,
            spectra,
        } = s;
        let at = StackPoints {
            views,
            stack,
            offsets: pos.offsets(),
        };
        let block = |order, modes, block| WindowBlock {
            slice: 0,
            order,
            modes,
            block,
        };
        sampler.spectra(ctx, self.obj, spectra)?;
        sampler.windows(ctx, at, spectra, block([0, 0], Modes::All, 0), &mut w.win)?;
        l.k.obj_probe_to_psi(ctx, &w.views, stack, &w.win, self.probe, rb.psi)?;
        rb.delta(ctx, l, b)?;
        if let Some(step) = step.as_deref_mut() {
            // The slopes of the windows the batch started from, against Δ before the probe
            // step weights it.
            let derivatives = |out: &mut DeviceBuffer<C32>| {
                for (axis, order) in (0..).zip([[1, 0], [0, 1]]) {
                    let blk = block(order, Modes::First, axis);
                    sampler.windows(ctx, at, spectra, blk, out)?;
                }
                Ok(())
            };
            let chi = (&*rb.dlt, SliceSource::Probe(self.probe));
            kernels.derivative_sums(ctx, views, stack, derivatives, chi, step)?;
        }
        if params.obj_step.is_some() {
            // Each mode's `|P|²` placed over the batch and its windows read back, through
            // `spectra`, which the object's windows no longer need.
            for m in 0..nprobe {
                l.rk.mode_power(ctx, self.probe, m, n, &mut w.patches)?;
                crate::gpu::zero(ctx, rb.overlap)?;
                sampler.place(ctx, at, &w.patches, 0, rb.overlap)?;
                sampler.spectra(ctx, rb.overlap, spectra)?;
                sampler.windows(ctx, at, spectra, block([0, 0], Modes::All, 0), rb.psi)?;
                l.rk.overlap_read(ctx, &w.views, stack, rb.psi, m, rb.dens)?;
            }
            rb.obj_weight(ctx, l, b, params.obj_alpha)?;
            crate::gpu::zero(ctx, &mut w.patches)?;
            (l.k).psi_to_obj(ctx, &w.views, stack, rb.psi, self.probe, &mut w.patches)?;
            sampler.place(ctx, at, &w.patches, 0, &mut self.obj_sums.num)?;
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
            kernels.step(ctx, frames.range.clone(), own, offsets, solve, step)?;
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
        let scan = BhScan {
            order: self.order,
            counts: l.counts,
        };
        l.rk.amplitude(ctx, scan, stack, self.dlt)?;
        l.prop.adjoint(ctx, plan, stack, self.dlt)?;
        l.k.scale_c32(ctx, self.psi, frames.len, -1.0)?;
        l.k.add_c32(ctx, self.dlt, self.psi, frames.len)
    }

    /// Batch `b`'s `Δ` in `dlt` weighted for the object step into `psi`: `Δ / ((1 − a) d + a
    /// max Σ_m d)` per view, `d` the overlap in `dens`, the max over the batch.
    fn obj_weight(
        &mut self,
        ctx: &ExecutionContext,
        l: Launch<'_>,
        b: usize,
        alpha: f64,
    ) -> Result<(), GpuError> {
        let (nf, nmode, n) = (self.batches[b].nf(), l.prop.dims.nmode, self.n);
        let groups = PowerMax {
            ngroup: nf,
            nsum: nmode,
            n,
        };
        l.rk.sum_max(ctx, self.dens, groups, self.pmax)?;
        let spec = Weight {
            nk: nf * nmode,
            n,
            alpha: alpha as f32,
        };
        l.rk.overlap_weight(ctx, (self.dlt, self.dens, self.pmax), self.psi, spec)
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
            alpha: alpha as f32,
        };
        l.rk.weight_inplace(ctx, self.dlt, self.psi, self.omax, spec)
    }
}
