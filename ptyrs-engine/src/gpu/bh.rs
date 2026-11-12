//! pty-chi BH on the GPU ([`crate::Bh`]), in the call sequence of `ptyrs_gpu::BhKernels`;
//! the step's scalars stay on the device.

use cuda_async::device_operation::ExecutionContext;
use ptyrs_gpu::{BhBatch, BhDisp, BhFieldsRef, BhScan, BhVar, Error as GpuError};

use super::device::BhBufs;
use super::{Bufs, GpuPtycho, plane_len};
use crate::ptycho::{Algorithm, llk_due};
use crate::{BhParams, BhStep, EngineError, random_batches};

impl GpuPtycho {
    /// [`crate::Bh::epoch`] `nepoch` times from a fresh [`crate::Bh`], over the same
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

    /// The steps of the last BH epoch, as [`crate::Bh::epoch`] returns them, read back from
    /// the device; none before a BH call.
    pub fn bh_steps(&self) -> Result<Vec<BhStep>, EngineError> {
        self.dev.bh_steps()
    }
}

impl Bufs<'_> {
    /// One epoch enqueued on `ctx`, batch by batch over the frames in `bh.order`; with `cg`,
    /// the first batch continues the last direction.
    pub(super) fn bh_epoch(
        &mut self,
        ctx: &ExecutionContext,
        bh: &mut BhBufs<'_>,
        params: &BhParams,
        cg: bool,
    ) -> Result<(), GpuError> {
        let rho2 = (params.probe_rho.powi(2) * plane_len(self.views.dims()) as f64) as f32;
        for b in 0..bh.batches.len() {
            self.bh_batch(ctx, bh, b, rho2, cg)?;
        }
        Ok(())
    }

    /// Batch `b` of an epoch: its gradient into the update sums, the step along `−Pg` (with
    /// `cg`, `β D′ − Pg`) and the update, which consumes the sums.
    fn bh_batch(
        &mut self,
        ctx: &ExecutionContext,
        bh: &mut BhBufs<'_>,
        b: usize,
        rho2: f32,
        cg: bool,
    ) -> Result<(), GpuError> {
        let dims = *self.views.dims();
        let batch = bh.batches[b].clone();
        let scan = BhScan {
            order: bh.order,
            counts: self.counts,
        };
        let views = bh.order.views();
        for (k, s) in batch.clone().enumerate() {
            let stack = bh.table.at(s);
            let plan = bh.plans.get(bh.stacks[s].plan);
            let (psi, r) = (&mut bh.psi[k], &mut bh.r[k]);
            self.kernels
                .obj_probe_to_psi(ctx, views, stack, self.obj, self.probe, psi)?;
            plan.forward_inplace(ctx, psi)?;
            self.bh.residual(ctx, scan, stack, psi, r)?;
            plan.inverse_inplace(ctx, r)?;
            self.kernels
                .psi_to_obj(ctx, views, stack, r, self.probe, self.obj_sums)?;
            self.kernels
                .psi_to_probe(ctx, views, stack, r, self.obj, self.probe_sums)?;
        }

        let d = BhDisp {
            u: [&self.obj_sums.num, &self.probe_sums.num],
            u_scale: rho2,
            v: cg.then_some([&bh.prev[0], &bh.prev[1]]),
        };
        self.bh.dots(ctx, &dims, &d, bh.parts)?;
        for (k, s) in batch.clone().enumerate() {
            let stack = bh.table.at(s);
            let plan = bh.plans.get(bh.stacks[s].plan);
            let x = [&*self.obj, &*self.probe];
            self.bh
                .far(ctx, views, stack, x, &d, [&mut *bh.lu, &mut *bh.lv])?;
            plan.forward_inplace(ctx, bh.lu)?;
            if cg {
                plan.forward_inplace(ctx, bh.lv)?;
            }
            let f = BhFieldsRef {
                psi: &bh.psi[k],
                r: &bh.r[k],
                lu: bh.lu,
                lv: bh.lv,
            };
            self.bh.hess_frames(ctx, scan, stack, &f, &d, bh.parts)?;
        }
        let frames = bh.stacks[batch.start].range.start..bh.stacks[batch.end - 1].range.end;
        let step = BhBatch {
            frac: frames.len() as f32 / bh.size as f32,
            frames,
            record: b as u32,
        };
        self.bh.step(ctx, &dims, bh.parts, &d, &step, bh.steps)?;

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
