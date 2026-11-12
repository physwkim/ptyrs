//! pty-chi rPIE on the GPU, in the call sequence of `ptyrs_gpu::RpieKernels`.

use cuda_async::device_operation::ExecutionContext;
use cuda_core::DeviceBuffer;
use ptyrs_fft::Plan2d;
use ptyrs_gpu::{C32, Error as GpuError, StackTable, Weight};

use super::{Frames, GpuPtycho, plane_len, stack_buf_len, stacks, to_u32};
use crate::ptycho::{Algorithm, llk_due};
use crate::{EngineError, RpieParams};

/// The device buffers of one [`GpuPtycho::rpie`] call, sized for the longest batch.
struct RpieBufs {
    batches: Vec<Frames>,
    /// Values per frame plane.
    n: u32,
    table: StackTable,
    /// One plan per batch length, indexed by [`Frames::plan`].
    plans: Vec<Plan2d>,
    /// The batch's exit waves, then its object windows `O_v q`.
    psi: DeviceBuffer<C32>,
    /// pty-chi's `Δ`, then the probe step's weighted `Δ`.
    dlt: DeviceBuffer<C32>,
    /// Per-view `max |O_v|^2`.
    omax: DeviceBuffer<f32>,
    /// `P / ((1 - a_o) |P|^2 + a_o max_i sum_m |P_m|^2)`.
    ptilde: DeviceBuffer<C32>,
    pmax: DeviceBuffer<f32>,
    /// A probe of ones, so `obj_probe_to_psi` yields the object windows.
    ones: DeviceBuffer<C32>,
}

impl GpuPtycho {
    /// [`crate::Ptycho::rpie`]: consecutive batches of `batch_size` frames, each stepping the
    /// object and probe from the batch-start arrays; one LLK record after every due epoch.
    pub fn rpie(&mut self, params: &RpieParams, nepoch: usize) -> Result<(), EngineError> {
        let dims = *self.views.dims();
        if dims.nobj != 1 {
            return Err(EngineError::Invalid(
                "rPIE needs exactly one object mode".into(),
            ));
        }
        if params.batch_size == 0 {
            return Err(EngineError::Invalid("rPIE batch_size must be >= 1".into()));
        }
        let stream = &self.stream;
        let (batches, table, plans) =
            stacks(stream, &dims, to_u32("batch_size", params.batch_size)?)?;
        let len = stack_buf_len(&dims, &table);
        let nk_max = table.nf_max() as usize * dims.nmode as usize;
        let mut bufs = RpieBufs {
            batches,
            n: to_u32("frame size", plane_len(&dims))?,
            table,
            plans,
            psi: DeviceBuffer::zeroed(stream, len)?,
            dlt: DeviceBuffer::zeroed(stream, len)?,
            omax: DeviceBuffer::zeroed(stream, nk_max)?,
            ptilde: DeviceBuffer::zeroed(stream, self.probe.len())?,
            pmax: DeviceBuffer::zeroed(stream, 1)?,
            ones: DeviceBuffer::from_host(
                stream,
                &vec![C32 { re: 1.0, im: 0.0 }; self.probe.len()],
            )?,
        };
        for ic in 0..nepoch {
            self.rpie_epoch(&self.context(), params, &mut bufs)?;
            if llk_due(params.calc_llk, ic, nepoch) {
                let l = self.llk()?;
                self.record(Algorithm::Rpie, l);
            }
            self.cycle += 1;
        }
        Ok(())
    }

    /// One epoch enqueued on `ctx`.
    fn rpie_epoch(
        &mut self,
        ctx: &ExecutionContext,
        params: &RpieParams,
        bufs: &mut RpieBufs,
    ) -> Result<(), GpuError> {
        for b in 0..bufs.batches.len() {
            self.rpie_batch(ctx, params, bufs, b)?;
        }
        Ok(())
    }

    /// One batch of an epoch enqueued on `ctx`.
    fn rpie_batch(
        &mut self,
        ctx: &ExecutionContext,
        params: &RpieParams,
        bufs: &mut RpieBufs,
        b: usize,
    ) -> Result<(), GpuError> {
        let frames = &bufs.batches[b];
        let stack = bufs.table.at(b);
        let plan = &mut bufs.plans[frames.plan];
        let dims = *self.views.dims();
        let n = bufs.n;
        let nk = frames.nf() * dims.nmode;
        let len = frames.len;
        let (k, rk) = (&self.kernels, &self.rpie);

        // Δ = IFT(ψ') − ψ.
        k.obj_probe_to_psi(
            ctx,
            &self.views,
            stack,
            &self.obj,
            &self.probe,
            &mut bufs.psi,
        )?;
        plan.forward(ctx, &bufs.psi, &mut bufs.dlt)?;
        rk.amplitude(ctx, &dims, stack, &self.counts, &mut bufs.dlt)?;
        plan.inverse_inplace(ctx, &mut bufs.dlt)?;
        k.scale_c32(ctx, &mut bufs.psi, len, -1.0)?;
        k.add_c32(ctx, &mut bufs.dlt, &bufs.psi, len)?;

        if params.obj_step.is_some() {
            rk.power_max(ctx, &self.probe, 1, dims.nprobe, n, &mut bufs.pmax)?;
            let spec = Weight {
                nk: dims.nprobe,
                n,
                per_plane: false,
                alpha: params.obj_alpha as f32,
            };
            rk.weight(
                ctx,
                &self.probe,
                &self.probe,
                &bufs.pmax,
                &mut bufs.ptilde,
                spec,
            )?;
            k.psi_to_obj(
                ctx,
                &self.views,
                stack,
                &bufs.dlt,
                &bufs.ptilde,
                &mut self.obj_sums,
            )?;
        }
        if params.probe_step.is_some() {
            k.obj_probe_to_psi(
                ctx,
                &self.views,
                stack,
                &self.obj,
                &bufs.ones,
                &mut bufs.psi,
            )?;
            rk.power_max(ctx, &bufs.psi, nk, 1, n, &mut bufs.omax)?;
            let spec = Weight {
                nk,
                n,
                per_plane: true,
                alpha: params.probe_alpha as f32,
            };
            rk.weight_inplace(ctx, &mut bufs.dlt, &bufs.psi, &bufs.omax, spec)?;
            k.psi_to_probe(
                ctx,
                &self.views,
                stack,
                &bufs.dlt,
                &self.obj,
                &mut self.probe_sums,
            )?;
        }
        if let Some(step) = params.obj_step {
            let s = step as f32;
            rk.step(ctx, &mut self.obj, &mut self.obj_sums, self.obj_len, s)?;
        }
        if let Some(step) = params.probe_step {
            let s = (step / frames.nf() as f64) as f32;
            rk.step(
                ctx,
                &mut self.probe,
                &mut self.probe_sums,
                self.probe_len,
                s,
            )?;
        }
        Ok(())
    }
}
