//! pty-chi rPIE on the GPU, in the call sequence of `ptyrs_gpu::RpieKernels`.

use cuda_async::device_operation::ExecutionContext;
use ptyrs_gpu::{Error as GpuError, Weight};

use super::device::RpieBufs;
use super::{Bufs, Epoch, GpuPtycho};
use crate::ptycho::{Algorithm, llk_due};
use crate::{EngineError, RpieParams};

impl GpuPtycho {
    /// [`crate::Ptycho::rpie`]: consecutive batches of `batch_size` frames, each stepping the
    /// object and probe from the batch-start arrays; one LLK record after every due epoch.
    pub fn rpie(&mut self, params: &RpieParams, nepoch: usize) -> Result<(), EngineError> {
        if self.dev.dims().nobj != 1 {
            return Err(EngineError::Invalid(
                "rPIE needs exactly one object mode".into(),
            ));
        }
        if params.batch_size == 0 {
            return Err(EngineError::Invalid("rPIE batch_size must be >= 1".into()));
        }
        for ic in 0..nepoch {
            self.dev.steady(Epoch::Rpie(*params), self.cycle)?;
            if llk_due(params.calc_llk, ic, nepoch) {
                self.dev.llk_pass(Some((self.cycle, Algorithm::Rpie)))?;
            }
            self.cycle += 1;
        }
        self.collect()?;
        Ok(())
    }
}

impl Bufs<'_> {
    /// One epoch enqueued on `ctx`.
    pub(super) fn rpie_epoch(
        &mut self,
        ctx: &ExecutionContext,
        params: &RpieParams,
        rb: &mut RpieBufs<'_>,
    ) -> Result<(), GpuError> {
        for b in 0..rb.batches.len() {
            self.rpie_batch(ctx, params, rb, b)?;
        }
        Ok(())
    }

    /// One batch of an epoch enqueued on `ctx`.
    fn rpie_batch(
        &mut self,
        ctx: &ExecutionContext,
        params: &RpieParams,
        rb: &mut RpieBufs<'_>,
        b: usize,
    ) -> Result<(), GpuError> {
        let frames = &rb.batches[b];
        let stack = rb.table.at(b);
        let plan = rb.plans.get(frames.plan);
        let dims = *self.views.dims();
        let n = rb.n;
        let nk = frames.nf() * dims.nmode;
        let len = frames.len;
        let (k, rk) = (self.kernels, self.rpie);

        // Δ = IFT(ψ') − ψ.
        k.obj_probe_to_psi(ctx, self.views, stack, self.obj, self.probe, rb.psi)?;
        plan.forward(ctx, rb.psi, rb.dlt)?;
        rk.amplitude(ctx, &dims, stack, self.counts, rb.dlt)?;
        plan.inverse_inplace(ctx, rb.dlt)?;
        k.scale_c32(ctx, rb.psi, len, -1.0)?;
        k.add_c32(ctx, rb.dlt, rb.psi, len)?;

        if params.obj_step.is_some() {
            rk.power_max(ctx, self.probe, 1, dims.nprobe, n, rb.pmax)?;
            let spec = Weight {
                nk: dims.nprobe,
                n,
                per_plane: false,
                alpha: params.obj_alpha as f32,
            };
            rk.weight(ctx, self.probe, self.probe, rb.pmax, rb.ptilde, spec)?;
            k.psi_to_obj(ctx, self.views, stack, rb.dlt, rb.ptilde, self.obj_sums)?;
        }
        if params.probe_step.is_some() {
            k.obj_probe_to_psi(ctx, self.views, stack, self.obj, rb.ones, rb.psi)?;
            rk.power_max(ctx, rb.psi, nk, 1, n, rb.omax)?;
            let spec = Weight {
                nk,
                n,
                per_plane: true,
                alpha: params.probe_alpha as f32,
            };
            rk.weight_inplace(ctx, rb.dlt, rb.psi, rb.omax, spec)?;
            k.psi_to_probe(ctx, self.views, stack, rb.dlt, self.obj, self.probe_sums)?;
        }
        if let Some(step) = params.obj_step {
            let s = step as f32;
            rk.step(ctx, self.obj, self.obj_sums, self.obj_len, s)?;
        }
        if let Some(step) = params.probe_step {
            let s = (step / frames.nf() as f64) as f32;
            rk.step(ctx, self.probe, self.probe_sums, self.probe_len, s)?;
        }
        Ok(())
    }
}
