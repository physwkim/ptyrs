//! pty-chi BH on the GPU ([`crate::engine::Bh`]), in the call sequence of `crate::gpu::BhKernels`, its
//! dot products by `crate::tile::TileDots`; the step's scalars stay on the device. Under
//! [`crate::engine::Features::position`] the offsets join the step, on the moved probes.

use std::sync::Arc;

use crate::gpu::{
    BhBatch, BhDisp, BhFieldsRef, BhMovedFarRef, BhMovesRef, BhNearMut, BhOffsets, BhPositions,
    BhScan, BhVar, C32, EpochOrder, Error as GpuError, FusedInputs, Kernels, Model, PsiSums,
    SpectrumBuf, ViewDims,
};
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer};

use super::device::{BhBufs, Moved, MovedBatch, MovedLent, PassesMut, moved_inputs, moved_shift};
use super::{Bufs, GpuPtycho, plane_len};
use crate::engine::features::POSITION_STEP;
use crate::engine::ptycho::{Algorithm, llk_due};
use crate::engine::{BhParams, BhStep, EngineError, random_batches};

impl GpuPtycho {
    /// [`crate::engine::Bh::epoch`] `nepoch` times from a fresh [`crate::engine::Bh`], over the same
    /// [`random_batches`]: with one batch of every frame, CG from the second epoch on, else
    /// steepest descent; one LLK record after every due epoch.
    pub fn bh(
        &mut self,
        params: &BhParams,
        nepoch: usize,
        calc_llk: usize,
    ) -> Result<(), EngineError> {
        if params.batch_size == 0 {
            return Err(EngineError::Invalid("BH batch_size must be >= 1".into()));
        }
        if self.dev.has_stepped() {
            return Err(EngineError::Invalid(
                "BH refines no intensity scale or background, as pty-chi's".into(),
            ));
        }
        let nframe = self.dev.dims().nframe as usize;
        let full = params.batch_size >= nframe;
        for ic in 0..nepoch {
            let order: Vec<u32> = random_batches(nframe, params.batch_size, params.seed, ic as u64)
                .concat()
                .into_iter()
                .map(|f| f as u32)
                .collect();
            self.dev
                .bh_epoch(*params, full && ic > 0, self.cycle, &order)?;
            if llk_due(calc_llk, ic, nepoch) {
                self.dev.llk_pass(Some((self.cycle, Algorithm::Bh)))?;
            }
            self.cycle += 1;
        }
        self.collect()?;
        Ok(())
    }

    /// The steps of the last BH epoch, as [`crate::engine::Bh::epoch`] returns them, read back from
    /// the device; none before a BH call.
    pub fn bh_steps(&self) -> Result<Vec<BhStep>, EngineError> {
        self.dev.bh_steps()
    }
}

/// BH's state under the position model: the moved probes, one stack's forward-FFT'd
/// `conj(O) r`, the spectra of the step's probe parts (`u`'s, `v`'s) and the offsets' step
/// buffers.
pub(super) struct BhMoved {
    probes: Moved,
    y: DeviceBuffer<C32>,
    spectra: [SpectrumBuf; 2],
    off: BhPositions,
}

impl BhMoved {
    /// For frames of `dims` in `slots` stacks per batch of up to `nf_max` frames each, and
    /// stack buffers of `stack_len` values.
    pub(super) fn new(
        stream: &Arc<CudaStream>,
        kernels: &Kernels,
        dims: &ViewDims,
        nf_max: u32,
        slots: usize,
        stack_len: usize,
    ) -> Result<Self, EngineError> {
        Ok(Self {
            probes: Moved::new(stream, kernels, dims, nf_max, slots)?,
            y: DeviceBuffer::zeroed(stream, stack_len)?,
            spectra: [
                SpectrumBuf::zeroed(stream, dims)?,
                SpectrumBuf::zeroed(stream, dims)?,
            ],
            off: BhPositions::zeroed(stream, dims)?,
        })
    }

    /// Its [`Moved`] with its other buffers, the offsets a variable of the step over `order`,
    /// the epoch's.
    fn split<'m>(
        &'m mut self,
        order: &'m EpochOrder,
        params: &BhParams,
    ) -> (&'m mut Moved, BhOwn<'m>) {
        let own = BhOwn {
            y: &mut self.y,
            spectra: &mut self.spectra,
            off: BhOffsets {
                bufs: &mut self.off,
                order,
                scale: params.position_rho.powi(2) as f32,
                step: POSITION_STEP as f32,
            },
        };
        (&mut self.probes, own)
    }
}

/// [`BhMoved`]'s buffers besides its [`Moved`], lent to a batch.
struct BhOwn<'m> {
    y: &'m mut DeviceBuffer<C32>,
    spectra: &'m mut [SpectrumBuf; 2],
    off: BhOffsets<'m>,
}

impl Bufs<'_> {
    /// One epoch enqueued on `ctx`, batch by batch over the frames in `bh.order`; with `cg`,
    /// the first batch continues the last direction.
    pub(super) fn bh_epoch(
        &mut self,
        ctx: &ExecutionContext,
        passes: &mut PassesMut<'_>,
        bh: &mut BhBufs<'_>,
        params: &BhParams,
        cg: bool,
    ) -> Result<(), GpuError> {
        // Each frame is in one batch, stepped after its batch's passes: the offsets the epoch
        // starts from serve every batch.
        let moved = (bh.moved.as_deref_mut()).map(|m| (&mut m.probes, ()));
        if let Some(mut m) = MovedLent::lend(moved, passes.fused(), &mut self.features)? {
            m.gather(ctx, bh.order)?;
        }
        for b in 0..bh.batches.len() {
            self.bh_batch(ctx, passes, bh, params, b, cg)?;
        }
        Ok(())
    }

    /// Batch `b` of an epoch: its gradient into the update sums, the step along `−Pg` (with
    /// `cg`, `β D′ − Pg`) and the update, which consumes the sums.
    fn bh_batch(
        &mut self,
        ctx: &ExecutionContext,
        passes: &mut PassesMut<'_>,
        bh: &mut BhBufs<'_>,
        params: &BhParams,
        b: usize,
        cg: bool,
    ) -> Result<(), GpuError> {
        let dims = *self.views.dims();
        let rho2 = (params.probe_rho.powi(2) * plane_len(&dims) as f64) as f32;
        let batch = bh.batches[b].clone();
        let scan = BhScan {
            order: bh.order,
            counts: self.counts,
        };
        let views = bh.order.views();
        let k = self.kernels;
        let moved = (bh.moved.as_deref_mut()).map(|m| m.split(bh.order, params));
        let moved = MovedLent::lend(moved, passes.fused(), &mut self.features)?;
        let mut moved = (moved.map(|m| m.batch(ctx, k, &dims, self.probe))).transpose()?;
        let base = FusedInputs {
            views,
            counts: self.counts,
            obj: self.obj,
            probe: self.probe,
            model: Model::default(),
        };

        for (i, s) in batch.clone().enumerate() {
            let stack = bh.table.at(s);
            let plan = bh.plans.get(bh.stacks[s].plan);
            let (psi, r) = (&mut bh.psi[i], &mut bh.r[i]);
            match &mut moved {
                Some(m) => {
                    let inputs = moved_inputs(&base, moved_shift(m.pos, m.spectrum));
                    let w = &mut m.w[i];
                    k.shifted_rows_fwd(ctx, &inputs, stack, w, bh.lu)?;
                    m.kernels.shift_w(ctx, &dims, stack, w)?;
                    self.rpie
                        .obj_wave_to_psi(ctx, views, stack, self.obj, w, psi)?;
                }
                None => k.obj_probe_to_psi(ctx, views, stack, self.obj, self.probe, psi)?,
            }
            plan.forward_inplace(ctx, psi)?;
            self.bh.residual(ctx, scan, stack, psi, r)?;
            plan.inverse_inplace(ctx, r)?;
            match &mut moved {
                Some(m) => {
                    let shift = moved_shift(m.pos, m.spectrum);
                    let inputs = moved_inputs(&base, shift);
                    let sums = PsiSums {
                        obj: Some(&mut *self.obj_sums),
                        probe: Some(&mut *m.probe_sums),
                    };
                    k.shifted_psi_sums(ctx, &inputs, stack, r, m.fused, sums)?;
                    let BhOwn { y, off, .. } = &mut m.own;
                    self.bh.obj_conj(ctx, views, stack, self.obj, r, y)?;
                    plan.forward_inplace(ctx, y)?;
                    self.bh.pos_grad(ctx, stack, y, &shift, off)?;
                }
                None => {
                    k.psi_to_obj(ctx, views, stack, r, self.probe, self.obj_sums)?;
                    k.psi_to_probe(ctx, views, stack, r, self.obj, self.probe_sums)?;
                }
            }
        }
        if let Some(m) = &mut moved {
            k.shifted_probe_sums(ctx, &dims, m.probe_sums, self.probe_sums)?;
        }

        let d = BhDisp {
            u: [&self.obj_sums.num, &self.probe_sums.num],
            u_scale: rho2,
            v: cg.then_some([&bh.prev[0], &bh.prev[1]]),
        };
        self.tile.dots(ctx, &d, bh.parts)?;
        let frames = bh.stacks[batch.start].range.start..bh.stacks[batch.end - 1].range.end;
        let step = BhBatch {
            frac: frames.len() as f32 / bh.size as f32,
            frames,
            record: b as u32,
            positions: moved.is_some(),
        };
        let mut hess = None;
        if let Some(m) = &mut moved {
            let shift = moved_shift(m.pos, m.spectrum);
            let BhOwn { y, spectra, off } = &mut m.own;
            self.bh.pos_dots(ctx, &step, &d, off, bh.parts)?;
            let [su, sv] = &mut **spectra;
            let au = k.probe_spectrum(ctx, &dims, d.u[1], su)?;
            let av = (d.v.map(|v| k.probe_spectrum(ctx, &dims, v[1], sv))).transpose()?;
            let mv = BhMovesRef::new(&d, shift, au, av, off)?;
            hess = Some((mv, &*m.w, &mut **y));
        }
        for (i, s) in batch.clone().enumerate() {
            let stack = bh.table.at(s);
            let plan = bh.plans.get(bh.stacks[s].plan);
            match &mut hess {
                None => {
                    let x = [&*self.obj, &*self.probe];
                    self.bh
                        .far(ctx, views, stack, x, &d, [&mut *bh.lu, &mut *bh.lv])?;
                    plan.forward_inplace(ctx, bh.lu)?;
                    if cg {
                        plan.forward_inplace(ctx, bh.lv)?;
                    }
                    let f = BhFieldsRef {
                        psi: &bh.psi[i],
                        r: &bh.r[i],
                        lu: bh.lu,
                        lv: bh.lv,
                    };
                    self.bh.hess_frames(ctx, scan, stack, &f, &d, bh.parts)?;
                }
                Some((mv, w, y)) => {
                    self.bh
                        .pos_spec(ctx, stack, mv, [&mut *bh.lu, &mut *bh.lv])?;
                    plan.inverse_inplace(ctx, bh.lu)?;
                    if cg {
                        plan.inverse_inplace(ctx, bh.lv)?;
                    }
                    let near = BhNearMut {
                        obj: self.obj,
                        w: &w[i],
                        r: &bh.r[i],
                        lu: bh.lu,
                        lv: bh.lv,
                    };
                    self.bh.pos_near(ctx, views, stack, near, &d, bh.parts)?;
                    plan.forward_inplace(ctx, bh.lu)?;
                    if cg {
                        plan.forward_inplace(ctx, bh.lv)?;
                    }
                    self.bh.obj_conj(ctx, views, stack, self.obj, &bh.r[i], y)?;
                    plan.forward_inplace(ctx, y)?;
                    let f = BhMovedFarRef {
                        psi: &bh.psi[i],
                        lu: bh.lu,
                        lv: bh.lv,
                        y,
                    };
                    self.bh.pos_far(ctx, self.counts, stack, &f, mv, bh.parts)?;
                }
            }
        }
        self.bh.step(ctx, &dims, bh.parts, &d, &step, bh.steps)?;
        // The offsets' update reads neither array, so it may precede their updates, which end
        // the borrow of the batch-start probe.
        if let Some(MovedBatch {
            own: BhOwn { mut off, .. },
            offsets,
            ..
        }) = moved
        {
            self.bh
                .pos_update(ctx, &step, &mut off, offsets, bh.steps)?;
        }

        let [prev_obj, prev_probe] = &mut *bh.prev;
        for var in [
            BhVar {
                x: self.obj,
                prev: prev_obj,
                g: self.obj_sums,
                n: self.obj_len,
                scale: 1.0,
            },
            BhVar {
                x: self.probe,
                prev: prev_probe,
                g: self.probe_sums,
                n: self.probe_len,
                scale: rho2,
            },
        ] {
            self.bh.update(ctx, var, bh.steps, step.record)?;
        }
        Ok(())
    }
}
