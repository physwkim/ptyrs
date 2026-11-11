//! pty-chi rPIE on the GPU, in the call sequence of `ptyrs_gpu::RpieKernels`.

use std::sync::Arc;

use cuda_core::{CudaStream, DeviceBuffer};
use ptyrs_fft::Plan2d;
use ptyrs_gpu::{C32, ViewDims, Weight};

use super::{GpuPtycho, Span, plane_len, split, to_u32};
use crate::ptycho::{Algorithm, llk_due};
use crate::{EngineError, RpieParams};

/// What the batches of one frame count share.
struct BatchBufs {
    plan: Plan2d,
    /// The batch's exit waves, then its object windows `O_v q`.
    psi: DeviceBuffer<C32>,
    /// pty-chi's `Δ`, then the probe step's weighted `Δ`.
    dlt: DeviceBuffer<C32>,
    /// Per-view `max |O_v|^2`.
    omax: DeviceBuffer<f32>,
}

impl BatchBufs {
    fn new(stream: &Arc<CudaStream>, dims: &ViewDims, nf: u32) -> Result<Self, EngineError> {
        let nk = nf as usize * dims.nmode as usize;
        let len = nk * plane_len(dims);
        Ok(Self {
            plan: Plan2d::new(stream, dims.frame[0] as usize, nk)?,
            psi: DeviceBuffer::zeroed(stream, len)?,
            dlt: DeviceBuffer::zeroed(stream, len)?,
            omax: DeviceBuffer::zeroed(stream, nk)?,
        })
    }
}

/// The device buffers of one [`GpuPtycho::rpie`] call.
struct RpieBufs {
    batches: Vec<(Span, usize)>,
    sizes: Vec<BatchBufs>,
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
        let (batches, lens) = split(dims.nframe, to_u32("batch_size", params.batch_size)?);
        let stream = &self.stream;
        let mut bufs = RpieBufs {
            batches,
            sizes: lens
                .iter()
                .map(|&nf| BatchBufs::new(stream, &dims, nf))
                .collect::<Result<Vec<_>, _>>()?,
            ptilde: DeviceBuffer::zeroed(stream, self.probe.len())?,
            pmax: DeviceBuffer::zeroed(stream, 1)?,
            ones: DeviceBuffer::from_host(
                stream,
                &vec![C32 { re: 1.0, im: 0.0 }; self.probe.len()],
            )?,
        };
        for ic in 0..nepoch {
            for b in 0..bufs.batches.len() {
                self.rpie_batch(params, &mut bufs, b)?;
            }
            if llk_due(params.calc_llk, ic, nepoch) {
                let l = self.llk()?;
                self.record(Algorithm::Rpie, l);
            }
            self.cycle += 1;
        }
        Ok(())
    }

    fn rpie_batch(
        &mut self,
        params: &RpieParams,
        bufs: &mut RpieBufs,
        b: usize,
    ) -> Result<(), EngineError> {
        let (span, size) = bufs.batches[b].clone();
        let bb = &mut bufs.sizes[size];
        let dims = *self.views.dims();
        let n = to_u32("frame size", plane_len(&dims))?;
        let nk = span.nf() * dims.nmode;
        let len = to_u32("batch length", bb.psi.len())?;
        let (k, rk, stream) = (&self.kernels, &self.rpie, &*self.stream);

        // Δ = IFT(ψ') − ψ.
        k.obj_probe_to_psi(
            stream,
            &self.views,
            span.arg(),
            &self.obj,
            &self.probe,
            &mut bb.psi,
        )?;
        bb.dlt.copy_from_device_async(&bb.psi, stream)?;
        bb.plan.forward_inplace(&mut bb.dlt)?;
        rk.amplitude(stream, &self.views, span.arg(), &self.counts, &mut bb.dlt)?;
        bb.plan.inverse_inplace(&mut bb.dlt)?;
        k.scale_c32(stream, &mut bb.psi, len, -1.0)?;
        k.add_c32(stream, &mut bb.dlt, &bb.psi, len)?;

        if params.obj_step.is_some() {
            rk.power_max(stream, &self.probe, 1, dims.nprobe, n, &mut bufs.pmax)?;
            let spec = Weight {
                nk: dims.nprobe,
                n,
                per_plane: false,
                alpha: params.obj_alpha as f32,
            };
            rk.weight(
                stream,
                &self.probe,
                &self.probe,
                &bufs.pmax,
                &mut bufs.ptilde,
                spec,
            )?;
            self.obj_sums.zero(stream)?;
            k.psi_to_obj(
                stream,
                &self.views,
                span.arg(),
                &bb.dlt,
                &bufs.ptilde,
                &mut self.obj_sums,
            )?;
        }
        if params.probe_step.is_some() {
            k.obj_probe_to_psi(
                stream,
                &self.views,
                span.arg(),
                &self.obj,
                &bufs.ones,
                &mut bb.psi,
            )?;
            rk.power_max(stream, &bb.psi, nk, 1, n, &mut bb.omax)?;
            let spec = Weight {
                nk,
                n,
                per_plane: true,
                alpha: params.probe_alpha as f32,
            };
            rk.weight_inplace(stream, &mut bb.dlt, &bb.psi, &bb.omax, spec)?;
            self.probe_sums.zero(stream)?;
            k.psi_to_probe(
                stream,
                &self.views,
                span.arg(),
                &bb.dlt,
                &self.obj,
                &mut self.probe_sums,
            )?;
        }
        if let Some(step) = params.obj_step {
            let n = to_u32("object length", self.obj.len())?;
            k.scale_c32(stream, &mut self.obj_sums.num, n, step as f32)?;
            k.add_c32(stream, &mut self.obj, &self.obj_sums.num, n)?;
        }
        if let Some(step) = params.probe_step {
            let n = to_u32("probe length", self.probe.len())?;
            let s = step / span.nf() as f64;
            k.scale_c32(stream, &mut self.probe_sums.num, n, s as f32)?;
            k.add_c32(stream, &mut self.probe, &self.probe_sums.num, n)?;
        }
        Ok(())
    }
}
