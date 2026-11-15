//! pty-chi BH on the GPU ([`crate::engine::Bh`]), in the call sequence of `crate::gpu::BhKernels`, its
//! dot products by `crate::tile::TileDots`; the step's scalars stay on the device. Under
//! [`crate::engine::Features::position`] the offsets join the step, on the moved probes or under
//! [`crate::engine::Sampling::FullObject`] the object's windows.

use std::sync::Arc;

use crate::gpu::sampling::{Modes, StackPoints, WindowBlock, window_views};
use crate::gpu::{
    BhBatch, BhDisp, BhFieldsRef, BhMovedFarRef, BhMovesRef, BhNearMut, BhOffsets, BhPositions,
    BhScan, BhVar, BhWinRef, BhWinSecond, C32, EpochOrder, Error as GpuError, FusedInputs, Kernels,
    Model, PsiSums, ShiftProbe, SpectrumBuf, ViewDims, Views,
};
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer};

use super::device::{
    BhBufs, Moved, MovedBatch, MovedLent, MovedSampling, PassesMut, PositionLent, SampledLent,
    another_model, moved_inputs, moved_shift,
};
use super::propagate::Propagate;
use super::{Bufs, Cut, GpuPtycho};
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
        params.validate()?;
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

/// BH's state under the position model: the moved probes, the offsets' step buffers and what
/// the offsets move, by the device's [`crate::engine::Sampling`].
pub(super) struct BhMoved {
    probes: Moved,
    by: BhBy,
    off: BhPositions,
}

/// What BH's offsets move.
enum BhBy {
    /// [`crate::engine::Sampling::Patch`]: the probe, with one stack's forward-FFT'd
    /// `conj(O) r` and the spectra of the step's probe parts (`u`'s, `v`'s).
    Shift {
        y: DeviceBuffer<C32>,
        spectra: [SpectrumBuf; 2],
    },
    /// [`crate::engine::Sampling::FullObject`]: the object's windows.
    Windows(Box<BhWindows>),
}

/// BH's windows under [`crate::engine::Sampling::FullObject`]: each three blocks `[nobj,
/// nf_max, ny, nx]` of a stack's windows and their derivatives over `y` and `x`, unless said
/// otherwise.
struct BhWindows {
    /// The views that read a stack's windows.
    views: Views,
    /// The estimate's, one per stack of a batch, kept from the gradient pass to the Hessian's.
    win: Vec<DeviceBuffer<C32>>,
    /// The estimate's second derivatives over `(y, y)`, `(y, x)` and `(x, x)`.
    second: DeviceBuffer<C32>,
    /// `u_O`'s and `v_O`'s, block 0 lifted by [`crate::gpu::BhKernels::win_lift`].
    hu: DeviceBuffer<C32>,
    hv: DeviceBuffer<C32>,
    /// One block: a stack's object gradient patches.
    patch: DeviceBuffer<C32>,
    /// `[nobj, nyo, nxo]` each: the spectra of `u_O` and `v_O`.
    spectra: [DeviceBuffer<C32>; 2],
}

impl BhMoved {
    /// For frames of `dims` in `cut`'s stacks, `slots` per batch, under `sampling`.
    pub(super) fn new(
        stream: &Arc<CudaStream>,
        kernels: &Kernels,
        dims: &ViewDims,
        cut: &Cut,
        slots: usize,
        sampling: MovedSampling,
    ) -> Result<Self, EngineError> {
        let zeroed = |len| DeviceBuffer::<C32>::zeroed(stream, len);
        let by = match sampling {
            MovedSampling::Patch => BhBy::Shift {
                y: zeroed(cut.wave_len(dims))?,
                spectra: [
                    SpectrumBuf::zeroed(stream, dims)?,
                    SpectrumBuf::zeroed(stream, dims)?,
                ],
            },
            MovedSampling::FullObject { .. } => {
                let stacks: Vec<_> = cut.frames.iter().map(|f| f.range.clone()).collect();
                let views = window_views(stream, dims, &stacks)?;
                let blk = views.dims().obj_len();
                BhBy::Windows(Box::new(BhWindows {
                    views,
                    win: (0..slots)
                        .map(|_| zeroed(3 * blk))
                        .collect::<Result<_, _>>()?,
                    second: zeroed(3 * blk)?,
                    hu: zeroed(3 * blk)?,
                    hv: zeroed(3 * blk)?,
                    patch: zeroed(blk)?,
                    spectra: [zeroed(dims.obj_len())?, zeroed(dims.obj_len())?],
                }))
            }
        };
        let nf_max = cut.table.nf_max();
        Ok(Self {
            probes: Moved::new(stream, kernels, dims, nf_max, slots, sampling.into())?,
            by,
            off: BhPositions::zeroed(stream, dims)?,
        })
    }

    fn samples_object(&self) -> bool {
        matches!(self.by, BhBy::Windows(_))
    }

    /// Its [`Moved`], what the offsets move and the offsets, a variable of the step over
    /// `order`, the epoch's.
    fn split<'m>(
        &'m mut self,
        order: &'m EpochOrder,
        params: &BhParams,
    ) -> (&'m mut Moved, &'m mut BhBy, BhOffsets<'m>) {
        let off = BhOffsets {
            bufs: &mut self.off,
            order,
            scale: params.position_rho.powi(2) as f32,
            step: POSITION_STEP as f32,
        };
        (&mut self.probes, &mut self.by, off)
    }

    /// [`Self::split`] under [`crate::engine::Sampling::Patch`]; an error under the other.
    fn split_shift<'m>(
        &'m mut self,
        order: &'m EpochOrder,
        params: &BhParams,
    ) -> Result<(&'m mut Moved, BhOwn<'m>), GpuError> {
        match self.split(order, params) {
            (m, BhBy::Shift { y, spectra }, off) => Ok((m, BhOwn { y, spectra, off })),
            _ => Err(another_model()),
        }
    }

    /// [`Self::split`] under [`crate::engine::Sampling::FullObject`]; an error under the other.
    fn split_windows<'m>(
        &'m mut self,
        order: &'m EpochOrder,
        params: &BhParams,
    ) -> Result<(&'m mut Moved, BhWinOwn<'m>), GpuError> {
        match self.split(order, params) {
            (m, BhBy::Windows(w), off) => Ok((m, BhWinOwn { w, off })),
            _ => Err(another_model()),
        }
    }
}

/// [`BhMoved`]'s buffers besides its [`Moved`] under [`crate::engine::Sampling::Patch`], lent
/// to a batch.
struct BhOwn<'m> {
    y: &'m mut DeviceBuffer<C32>,
    spectra: &'m mut [SpectrumBuf; 2],
    off: BhOffsets<'m>,
}

/// [`BhMoved`]'s buffers besides its [`Moved`] under [`crate::engine::Sampling::FullObject`],
/// lent to a batch.
struct BhWinOwn<'m> {
    w: &'m mut BhWindows,
    off: BhOffsets<'m>,
}

/// The estimate's windows and their derivatives over `y` and `x`, by block.
const FIRST: [[u32; 2]; 3] = [[0, 0], [1, 0], [0, 1]];
/// The estimate's second derivatives over `(y, y)`, `(y, x)` and `(x, x)`, by block.
const SECOND: [[u32; 2]; 3] = [[2, 0], [1, 1], [0, 2]];

/// The window blocks of every object mode of slice 0, block `b` of derivative `orders[b]`.
fn blocks(orders: [[u32; 2]; 3]) -> impl Iterator<Item = WindowBlock> {
    (0..).zip(orders).map(|(block, order)| WindowBlock {
        slice: 0,
        order,
        modes: Modes::All,
        block,
    })
}

/// Batch `b`'s step over `bh`'s epoch order, with the offsets a variable of it if `positions`.
fn batch_step(bh: &BhBufs<'_>, b: usize, positions: bool) -> BhBatch {
    let batch = &bh.batches[b];
    let frames = bh.stacks[batch.start].range.start..bh.stacks[batch.end - 1].range.end;
    BhBatch {
        frac: frames.len() as f32 / bh.size as f32,
        frames,
        record: b as u32,
        positions,
    }
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
        if let Some(mut m) = PositionLent::lend(moved, passes.fused(), &mut self.features)? {
            m.gather(ctx, bh.order)?;
        }
        let sampled = (bh.moved.as_deref()).is_some_and(BhMoved::samples_object);
        for b in 0..bh.batches.len() {
            match sampled {
                true => self.bh_sampled_batch(ctx, bh, params, b, cg)?,
                false => self.bh_batch(ctx, passes, bh, params, b, cg)?,
            }
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
        let rho2 = (params.probe_rho.powi(2) * self.exit.scale as f64) as f32;
        let prop = Propagate {
            kernels: self.bh,
            stage: self.exit.stage,
            dims,
        };
        let batch = bh.batches[b].clone();
        let scan = BhScan {
            order: bh.order,
            counts: self.counts,
        };
        let views = bh.order.views();
        let k = self.kernels;
        let step = batch_step(bh, b, bh.moved.is_some());
        let moved = (bh.moved.as_deref_mut()).map(|m| m.split_shift(bh.order, params));
        let moved = MovedLent::lend(moved.transpose()?, passes.fused(), &mut self.features)?;
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
                    k.shifted_probe(ctx, &inputs, stack, w)?;
                    m.kernels.shift_w(ctx, &dims, stack, w)?;
                    self.rpie
                        .obj_wave_to_psi(ctx, views, stack, self.obj, w, psi)?;
                }
                None => k.obj_probe_to_psi(ctx, views, stack, self.obj, self.probe, psi)?,
            }
            prop.forward(ctx, plan, stack, psi)?;
            self.bh.residual(ctx, scan, stack, psi, r)?;
            prop.adjoint(ctx, plan, stack, r)?;
            match &mut moved {
                Some(m) => {
                    let shift = moved_shift(m.pos, m.spectrum);
                    let inputs = moved_inputs(&base, shift);
                    let sums = PsiSums {
                        obj: Some(&mut *self.obj_sums),
                        probe: Some(ShiftProbe::Sums(&mut *m.probe_sums)),
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
                    prop.forward(ctx, plan, stack, bh.lu)?;
                    if cg {
                        prop.forward(ctx, plan, stack, bh.lv)?;
                    }
                    let f = BhFieldsRef {
                        views,
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
                    prop.forward(ctx, plan, stack, bh.lu)?;
                    if cg {
                        prop.forward(ctx, plan, stack, bh.lv)?;
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
        self.bh_update(ctx, bh, rho2, step.record)
    }

    /// Batch `b` of an epoch under [`crate::engine::Sampling::FullObject`], as
    /// [`Self::bh_batch`] with the offsets sampling the object's windows (`crate::gpu::bh`'s
    /// module docs).
    fn bh_sampled_batch(
        &mut self,
        ctx: &ExecutionContext,
        bh: &mut BhBufs<'_>,
        params: &BhParams,
        b: usize,
        cg: bool,
    ) -> Result<(), GpuError> {
        let dims = *self.views.dims();
        let rho2 = (params.probe_rho.powi(2) * self.exit.scale as f64) as f32;
        let prop = Propagate {
            kernels: self.bh,
            stage: self.exit.stage,
            dims,
        };
        let batch = bh.batches[b].clone();
        let scan = BhScan {
            order: bh.order,
            counts: self.counts,
        };
        let views = bh.order.views();
        let k = self.kernels;
        let step = batch_step(bh, b, true);
        let moved = (bh.moved.as_deref_mut()).map(|m| m.split_windows(bh.order, params));
        let lent = PositionLent::lend(moved.transpose()?, None, &mut self.features)?;
        let Some(PositionLent::FullObject(lent)) = lent else {
            return Err(another_model());
        };
        let SampledLent {
            own: BhWinOwn { w, mut off },
            pos,
            offsets,
            sampler,
            spectra,
            ..
        } = lent;
        let BhWindows {
            views: wv,
            win: wins,
            second,
            hu,
            hv,
            patch,
            spectra: [su, sv],
        } = w;
        let wv = &*wv;
        let at = |stack| StackPoints {
            views,
            stack,
            offsets: pos.offsets(),
        };
        sampler.spectra(ctx, self.obj, spectra)?;
        let spectra = &*spectra;

        for (i, s) in batch.clone().enumerate() {
            let stack = bh.table.at(s);
            let plan = bh.plans.get(bh.stacks[s].plan);
            let (psi, r, win) = (&mut bh.psi[i], &mut bh.r[i], &mut wins[i]);
            for block in blocks(FIRST) {
                sampler.windows(ctx, at(stack), spectra, block, win)?;
            }
            k.obj_probe_to_psi(ctx, wv, stack, win, self.probe, psi)?;
            prop.forward(ctx, plan, stack, psi)?;
            self.bh.residual(ctx, scan, stack, psi, r)?;
            prop.adjoint(ctx, plan, stack, r)?;
            k.psi_to_probe(ctx, wv, stack, r, win, self.probe_sums)?;
            let f = BhWinRef {
                r,
                probe: self.probe,
            };
            self.bh.win_grad(ctx, stack, f, win, patch, &mut off)?;
            sampler.place(ctx, at(stack), patch, 0, &mut self.obj_sums.num)?;
        }

        let d = BhDisp {
            u: [&self.obj_sums.num, &self.probe_sums.num],
            u_scale: rho2,
            v: cg.then_some([&bh.prev[0], &bh.prev[1]]),
        };
        self.tile.dots(ctx, &d, bh.parts)?;
        self.bh.pos_dots(ctx, &step, &d, &mut off, bh.parts)?;
        sampler.spectra(ctx, d.u[0], su)?;
        if let Some(v) = d.v {
            sampler.spectra(ctx, v[0], sv)?;
        }
        for (i, s) in batch.clone().enumerate() {
            let stack = bh.table.at(s);
            let plan = bh.plans.get(bh.stacks[s].plan);
            let win = &wins[i];
            for block in blocks(FIRST) {
                sampler.windows(ctx, at(stack), su, block, hu)?;
            }
            if cg {
                for block in blocks(FIRST) {
                    sampler.windows(ctx, at(stack), sv, block, hv)?;
                }
            }
            self.bh
                .win_lift(ctx, stack, win, hu, cg.then_some(&mut *hv), &off)?;
            let lifted = BhDisp {
                u: [&*hu, d.u[1]],
                v: d.v.map(|v| [&*hv, v[1]]),
                ..d
            };
            let x = [win, &*self.probe];
            self.bh
                .far(ctx, wv, stack, x, &lifted, [&mut *bh.lu, &mut *bh.lv])?;
            prop.forward(ctx, plan, stack, bh.lu)?;
            if cg {
                prop.forward(ctx, plan, stack, bh.lv)?;
            }
            let f = BhFieldsRef {
                views: wv,
                psi: &bh.psi[i],
                r: &bh.r[i],
                lu: bh.lu,
                lv: bh.lv,
            };
            self.bh
                .hess_frames(ctx, scan, stack, &f, &lifted, bh.parts)?;
            for block in blocks(SECOND) {
                sampler.windows(ctx, at(stack), spectra, block, second)?;
            }
            let f = BhWinRef {
                r: &bh.r[i],
                probe: self.probe,
            };
            let w = BhWinSecond {
                second,
                u: hu,
                v: cg.then_some(&*hv),
            };
            self.bh.win_hess(ctx, stack, f, &w, &off, bh.parts)?;
        }
        self.bh.step(ctx, &dims, bh.parts, &d, &step, bh.steps)?;
        self.bh
            .pos_update(ctx, &step, &mut off, offsets, bh.steps)?;
        self.bh_update(ctx, bh, rho2, step.record)
    }

    /// The batch's updates of the object and the probe by its step record `record`, which
    /// consume the gradient sums.
    fn bh_update(
        &mut self,
        ctx: &ExecutionContext,
        bh: &mut BhBufs<'_>,
        rho2: f32,
        record: u32,
    ) -> Result<(), GpuError> {
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
            self.bh.update(ctx, var, bh.steps, record)?;
        }
        Ok(())
    }
}
