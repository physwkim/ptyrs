//! pty-chi BH on the GPU ([`crate::engine::Bh`]), in the call sequence of `crate::gpu::BhKernels`, its
//! dot products by `crate::tile::TileDots`; the step's scalars stay on the device. Under
//! [`crate::engine::Features::position`] the offsets join the step, on the moved probes.

use std::sync::Arc;

use crate::gpu::{
    BhBatch, BhDisp, BhFieldsRef, BhMovedFarRef, BhMovesRef, BhNearMut, BhOffsets, BhPositions,
    BhScan, BhVar, C32, EpochOrder, Error as GpuError, FusedBufs, FusedInputs, Kernels, Model,
    PositionKernels, Positions, PsiSums, Shift, ShiftedProbeSums, SpectrumBuf, ViewDims,
};
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer};

use super::device::{BhBufs, FeaturesMut, PassesMut, PositionLent};
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

/// BH's state under the position model: per batch slot the moved probes `[nf_max * nprobe,
/// ny, nx]`, one stack's forward-FFT'd `conj(O) r`, the spectra of the step's probe parts
/// (`u`'s, `v`'s), the offsets by epoch position, the offsets' step buffers and their kernels.
pub(super) struct BhMoved {
    w: Vec<DeviceBuffer<C32>>,
    y: DeviceBuffer<C32>,
    spectra: [SpectrumBuf; 2],
    pos: Positions,
    off: BhPositions,
    kernels: PositionKernels,
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
        let wlen = nf_max as usize * dims.nprobe as usize * plane_len(dims);
        let zeroed = |len| DeviceBuffer::<C32>::zeroed(stream, len);
        Ok(Self {
            w: (0..slots).map(|_| zeroed(wlen)).collect::<Result<_, _>>()?,
            y: zeroed(stack_len)?,
            spectra: [
                SpectrumBuf::zeroed(stream, dims)?,
                SpectrumBuf::zeroed(stream, dims)?,
            ],
            pos: Positions::zeroed(stream, dims, nf_max)?,
            off: BhPositions::zeroed(stream, dims)?,
            kernels: PositionKernels::load(kernels)?,
        })
    }
}

/// [`BhMoved`] with the device's position buffers and the fused passes' scratch.
struct MovedLent<'m> {
    moved: &'m mut BhMoved,
    lent: PositionLent<'m>,
    fused: &'m mut FusedBufs,
}

impl<'m> MovedLent<'m> {
    /// For BH's `moved` state with the device's `passes` and `features`: none without a
    /// position, an error when the two disagree.
    fn lend(
        moved: Option<&'m mut BhMoved>,
        passes: &'m mut PassesMut<'_>,
        features: &'m mut FeaturesMut<'_>,
    ) -> Result<Option<Self>, GpuError> {
        match (moved, features.position_lent(), passes) {
            (None, None, _) => Ok(None),
            (Some(moved), Some(lent), PassesMut::Fused(fused)) => {
                Ok(Some(Self { moved, lent, fused }))
            }
            _ => Err(GpuError::Shape(
                "BH state built for another position model".into(),
            )),
        }
    }

    /// The lend for one batch over `order`, the epoch's, with the spectra of `probe` it starts
    /// from (two launches).
    fn batch(
        self,
        ctx: &ExecutionContext,
        kernels: &Kernels,
        probe: &'m DeviceBuffer<C32>,
        order: &'m EpochOrder,
        params: &BhParams,
    ) -> Result<MovedBatch<'m>, GpuError> {
        let PositionLent {
            offsets,
            spectrum,
            probe_sums,
        } = self.lent;
        let BhMoved {
            w,
            y,
            spectra,
            pos,
            off,
            kernels: pk,
        } = self.moved;
        let shift = Shift {
            offsets: pos.offsets(),
            spectrum: kernels.probe_spectrum(ctx, order.views().dims(), probe, spectrum)?,
        };
        Ok(MovedBatch {
            w,
            y,
            spectra,
            kernels: pk,
            shift,
            off: BhOffsets {
                bufs: off,
                order,
                scale: params.position_rho.powi(2) as f32,
                step: POSITION_STEP as f32,
            },
            offsets,
            probe_sums,
            fused: self.fused,
        })
    }
}

/// [`MovedLent`] for one batch: the estimate's moved probes, from the batch-start probe they
/// borrow, and the offsets as a variable of the step.
struct MovedBatch<'m> {
    w: &'m mut [DeviceBuffer<C32>],
    y: &'m mut DeviceBuffer<C32>,
    spectra: &'m mut [SpectrumBuf; 2],
    kernels: &'m PositionKernels,
    shift: Shift<'m>,
    off: BhOffsets<'m>,
    /// By frame: the device's offsets, which the batch steps.
    offsets: &'m mut DeviceBuffer<f32>,
    probe_sums: &'m mut ShiftedProbeSums,
    fused: &'m mut FusedBufs,
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
        let moved = bh.moved.as_deref_mut();
        if let Some(m) = MovedLent::lend(moved, passes, &mut self.features)? {
            let BhMoved { pos, kernels, .. } = m.moved;
            kernels.gather(ctx, bh.order, m.lent.offsets, pos)?;
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
        let moved = MovedLent::lend(bh.moved.as_deref_mut(), passes, &mut self.features)?;
        let mut moved =
            (moved.map(|m| m.batch(ctx, k, self.probe, bh.order, params))).transpose()?;
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
                    let inputs = moved_inputs(&base, m.shift);
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
                    let inputs = moved_inputs(&base, m.shift);
                    let sums = PsiSums {
                        obj: Some(&mut *self.obj_sums),
                        probe: Some(&mut *m.probe_sums),
                    };
                    k.shifted_psi_sums(ctx, &inputs, stack, r, m.fused, sums)?;
                    self.bh.obj_conj(ctx, views, stack, self.obj, r, m.y)?;
                    plan.forward_inplace(ctx, m.y)?;
                    self.bh.pos_grad(ctx, stack, m.y, &m.shift, &mut m.off)?;
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
            self.bh.pos_dots(ctx, &step, &d, &mut m.off, bh.parts)?;
            let [su, sv] = &mut *m.spectra;
            let au = k.probe_spectrum(ctx, &dims, d.u[1], su)?;
            let av = (d.v.map(|v| k.probe_spectrum(ctx, &dims, v[1], sv))).transpose()?;
            let mv = BhMovesRef::new(&d, m.shift, au, av, &m.off)?;
            hess = Some((mv, &*m.w, &mut *m.y));
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
            mut off, offsets, ..
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

/// `base` under the batch's position model `shift`.
fn moved_inputs<'a>(base: &FusedInputs<'a>, shift: Shift<'a>) -> FusedInputs<'a> {
    FusedInputs {
        model: Model {
            shift: Some(shift),
            ..Model::default()
        },
        ..*base
    }
}
