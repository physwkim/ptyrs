//! The GPU backend (DESIGN §7 M1): [`GpuPtycho`] runs AP, DM and rPIE in `f32` on one CUDA
//! stream with the `ptyrs-gpu` pixel kernels and `ptyrs-fft` plans, in the call sequences the
//! `ptyrs_gpu` ops and rPIE docs give, and is checked against [`crate::Ptycho`].
//!
//! The scan's frames are cut into stacks of [`GpuOptions::stack_size`] consecutive frames.
//! Each stack keeps its own exit waves (DM carries them across cycles) and LLK partials;
//! stacks of one length share a cuFFT plan and a scratch stack. Within a cycle nothing reads
//! back to the host: LLK partials are downloaded only in the cycles that record one, and the
//! arrays only when asked for. Each algorithm is its own module over the per-stack steps here,
//! so M2's fused passes can replace those steps one algorithm at a time.

mod ap;
mod dm;
mod rpie;

use std::f64::consts::PI;
use std::ops::Range;
use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use ptyrs_core::{AddrTable, C64, Object, Probe, Scan, Stack};
use ptyrs_fft::Plan2d;
use ptyrs_gpu::{
    C32, Kernels, LlkPartial, LlkSums, RpieKernels, UpdateSums, ViewAddr, ViewDims, Views,
};

use crate::ptycho::{Algorithm, LlkRecord, zero_phase_ramp};
use crate::{EngineError, Fft2, Llk};

/// How [`GpuPtycho`] batches its work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuOptions {
    /// Frames per stack (the last stack may be shorter); PyNX's default is 16.
    pub stack_size: usize,
}

impl Default for GpuOptions {
    fn default() -> Self {
        Self { stack_size: 16 }
    }
}

/// Consecutive frames of one stack or rPIE batch. Every launch takes its frame argument from
/// [`Span::arg`], the one place to change when the kernels read their stack on the device.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Span(Range<u32>);

impl Span {
    fn arg(&self) -> Range<u32> {
        self.0.clone()
    }

    fn nf(&self) -> u32 {
        self.0.end - self.0.start
    }
}

/// One stack of consecutive frames.
struct StackBufs {
    span: Span,
    /// Index into [`GpuPtycho::sizes`].
    size: usize,
    /// Exit waves of the stack's views (planes `0 .. nf * nmode`).
    psi: DeviceBuffer<C32>,
    llk: DeviceBuffer<LlkPartial>,
}

/// What the stacks of one frame count share.
struct SizeBufs {
    plan: Plan2d,
    /// One stack's worth of exit waves: DM's `PO`, the LLK pass's forward stack.
    scratch: DeviceBuffer<C32>,
}

impl SizeBufs {
    fn new(stream: &Arc<CudaStream>, dims: &ViewDims, nf: u32) -> Result<Self, EngineError> {
        let len = nf as usize * dims.nmode as usize * plane_len(dims);
        Ok(Self {
            plan: Plan2d::new(
                stream,
                dims.frame[0] as usize,
                nf as usize * dims.nmode as usize,
            )?,
            scratch: DeviceBuffer::zeroed(stream, len)?,
        })
    }
}

/// `0 .. nframe` in consecutive ranges of `size` frames, the last one shorter when `size`
/// does not divide `nframe`, each with the index of its length in the returned lengths.
fn split(nframe: u32, size: u32) -> (Vec<(Span, usize)>, Vec<u32>) {
    let mut lens: Vec<u32> = Vec::new();
    let mut ranges = Vec::new();
    let mut f0 = 0;
    while f0 < nframe {
        let f1 = nframe.min(f0 + size);
        let idx = match lens.iter().position(|&l| l == f1 - f0) {
            Some(i) => i,
            None => {
                lens.push(f1 - f0);
                lens.len() - 1
            }
        };
        ranges.push((Span(f0..f1), idx));
        f0 = f1;
    }
    (ranges, lens)
}

fn plane_len(dims: &ViewDims) -> usize {
    dims.frame[0] as usize * dims.frame[1] as usize
}

fn to_u32(what: &str, v: usize) -> Result<u32, EngineError> {
    u32::try_from(v).map_err(|_| EngineError::Invalid(format!("{what} {v} exceeds u32")))
}

fn upload(stream: &CudaStream, a: &Stack<C64>) -> Result<DeviceBuffer<C32>, EngineError> {
    Ok(DeviceBuffer::from_host(stream, &to_c32(a.as_slice()))?)
}

fn to_c32(a: &[C64]) -> Vec<C32> {
    a.iter()
        .map(|c| C32 {
            re: c.re as f32,
            im: c.im as f32,
        })
        .collect()
}

fn download(
    stream: &CudaStream,
    buf: &DeviceBuffer<C32>,
    dims: [usize; 3],
) -> Result<Stack<C64>, EngineError> {
    let v = buf.to_host_vec(stream)?;
    Ok(Stack::from_vec(
        dims,
        v.into_iter()
            .map(|c| C64::new(c.re as f64, c.im as f64))
            .collect(),
    )?)
}

/// Reconstruction state on one GPU: the object, probe and per-stack exit waves in device
/// memory, and the LLK history. The same scan, starting arrays and parameters give the same
/// records as [`crate::Ptycho`] up to `f32` rounding.
pub struct GpuPtycho {
    stream: Arc<CudaStream>,
    kernels: Kernels,
    rpie: RpieKernels,
    views: Views,
    counts: DeviceBuffer<u16>,
    obj: DeviceBuffer<C32>,
    probe: DeviceBuffer<C32>,
    obj_dims: [usize; 3],
    probe_dims: [usize; 3],
    obj_sums: UpdateSums,
    probe_sums: UpdateSums,
    stacks: Vec<StackBufs>,
    sizes: Vec<SizeBufs>,
    nb_obs: usize,
    /// The probe-frame FFT of [`zero_phase_ramp`], which runs on the host.
    fft: Fft2,
    cycle: usize,
    history: Vec<LlkRecord>,
}

impl GpuPtycho {
    /// Uploads the scan's counts and views (as [`crate::Ptycho::new`] builds them) and the
    /// starting arrays to `ctx`, on a new stream. Frames must be square (cuFFT plans here are
    /// `n x n`).
    pub fn new(
        ctx: &Arc<CudaContext>,
        scan: &Scan,
        object: Object,
        probe: Probe,
        options: GpuOptions,
    ) -> Result<Self, EngineError> {
        let g = scan.geometry;
        let [ny, nx] = g.frame;
        if ny != nx {
            return Err(EngineError::Invalid(format!(
                "the GPU backend needs square frames, got {:?}",
                g.frame
            )));
        }
        if probe.plane_shape() != g.frame {
            return Err(EngineError::Invalid(format!(
                "probe plane {:?} differs from frame {:?}",
                probe.plane_shape(),
                g.frame
            )));
        }
        if options.stack_size == 0 {
            return Err(EngineError::Invalid("stack_size must be >= 1".into()));
        }
        let pixel = g.pixel_size_object();
        let positions = scan.positions.pynx_preprocess(pixel);
        let addr = AddrTable::new(
            &positions,
            pixel,
            g.frame,
            object.plane_shape(),
            probe.nmode(),
            object.nmode(),
        )?;
        let [nyo, nxo] = object.plane_shape();
        let dims = ViewDims {
            nframe: to_u32("frame count", addr.nframe())?,
            nmode: to_u32("mode count", addr.nmode())?,
            nprobe: to_u32("probe modes", addr.nprobe())?,
            nobj: to_u32("object modes", addr.nobj())?,
            frame: [to_u32("frame side", ny)?, to_u32("frame side", nx)?],
            obj: [to_u32("object side", nyo)?, to_u32("object side", nxo)?],
            quad_phase: (PI * pixel[1] * pixel[1] / (g.wavelength * g.distance)) as f32,
        };
        let addrs: Vec<ViewAddr> = addr
            .views()
            .iter()
            .map(|v| ViewAddr {
                oy: v.oy,
                ox: v.ox,
                fy: 0.0,
                fx: 0.0,
                probe: v.probe,
                obj: v.obj,
            })
            .collect();

        let ctx_stream = ctx.new_stream()?;
        let stream = &ctx_stream;
        let kernels = Kernels::load(ctx)?;
        let rpie = RpieKernels::load(&kernels)?;
        let views = Views::new(stream, dims, &addrs)?;
        let (ranges, lens) = split(dims.nframe, to_u32("stack_size", options.stack_size)?);
        let sizes = lens
            .iter()
            .map(|&nf| SizeBufs::new(stream, &dims, nf))
            .collect::<Result<Vec<_>, _>>()?;
        let n = plane_len(&dims);
        let stacks = ranges
            .into_iter()
            .map(|(span, size)| {
                let nf = span.nf() as usize;
                Ok(StackBufs {
                    span,
                    size,
                    psi: DeviceBuffer::zeroed(stream, nf * dims.nmode as usize * n)?,
                    llk: DeviceBuffer::zeroed(stream, nf)?,
                })
            })
            .collect::<Result<Vec<_>, EngineError>>()?;
        Ok(Self {
            counts: DeviceBuffer::from_host(stream, scan.counts.as_slice())?,
            obj: upload(stream, &object)?,
            probe: upload(stream, &probe)?,
            obj_sums: UpdateSums::zeroed(stream, object.as_slice().len())?,
            probe_sums: UpdateSums::zeroed(stream, probe.as_slice().len())?,
            obj_dims: object.dims(),
            probe_dims: probe.dims(),
            nb_obs: scan.counts.nb_obs(),
            fft: Fft2::new(g.frame),
            stream: ctx_stream.clone(),
            kernels,
            rpie,
            views,
            stacks,
            sizes,
            cycle: 0,
            history: Vec::new(),
        })
    }

    /// The object, read back from the device.
    pub fn object(&self) -> Result<Object, EngineError> {
        Ok(Object::new(download(
            &self.stream,
            &self.obj,
            self.obj_dims,
        )?))
    }

    /// The probe, read back from the device.
    pub fn probe(&self) -> Result<Probe, EngineError> {
        Ok(Probe::new(download(
            &self.stream,
            &self.probe,
            self.probe_dims,
        )?))
    }

    /// LLK records in the order they were taken.
    pub fn history(&self) -> &[LlkRecord] {
        &self.history
    }

    /// Cycles (AP, DM) and epochs (rPIE) run so far.
    pub fn cycle(&self) -> usize {
        self.cycle
    }

    /// Waits for every enqueued step to finish (for timing).
    pub fn synchronize(&self) -> Result<(), EngineError> {
        Ok(self.stream.synchronize()?)
    }

    /// Normalised LLK of the current object and probe, as [`crate::Ptycho::llk`]; the
    /// exit-wave stacks are left as they are.
    pub fn llk(&mut self) -> Result<Llk, EngineError> {
        self.forward_llk()?;
        self.read_llk()
    }

    fn record(&mut self, algorithm: Algorithm, llk: Llk) {
        self.history.push(LlkRecord {
            cycle: self.cycle,
            algorithm,
            llk,
        });
    }

    /// LLK partials of the current estimate into every stack's `llk`, through the scratch
    /// stacks.
    fn forward_llk(&mut self) -> Result<(), EngineError> {
        let dims = *self.views.dims();
        for st in &mut self.stacks {
            let sz = &mut self.sizes[st.size];
            self.kernels.obj_probe_to_psi(
                &self.stream,
                &self.views,
                st.span.arg(),
                &self.obj,
                &self.probe,
                &mut sz.scratch,
            )?;
            sz.plan.forward_inplace(&mut sz.scratch)?;
            self.kernels.llk(
                &self.stream,
                &dims,
                st.span.arg(),
                &self.counts,
                &sz.scratch,
                &mut st.llk,
            )?;
        }
        Ok(())
    }

    /// Reads back the stacks' LLK partials (the one host sync of a recording cycle) and sums
    /// them in frame order.
    fn read_llk(&self) -> Result<Llk, EngineError> {
        let mut partials = Vec::with_capacity(self.views.dims().nframe as usize);
        for st in &self.stacks {
            partials.extend(st.llk.to_host_vec(&self.stream)?);
        }
        let s = LlkSums::from_partials(&partials);
        Ok(Llk {
            poisson: s.poisson,
            gaussian: s.gaussian,
            euclidian: s.euclidean,
            nb_photons_calc: s.icalc,
        }
        .normalised(self.nb_obs))
    }

    /// Fourier-modulus projection of stack `s`'s exit waves in place: FFT, `apply_amplitude`,
    /// inverse FFT. With `llk`, the stack's LLK partials of the forward stack are taken first.
    fn project(&mut self, s: usize, llk: bool) -> Result<(), EngineError> {
        let dims = *self.views.dims();
        let st = &mut self.stacks[s];
        let plan = &self.sizes[st.size].plan;
        plan.forward_inplace(&mut st.psi)?;
        if llk {
            self.kernels.llk(
                &self.stream,
                &dims,
                st.span.arg(),
                &self.counts,
                &st.psi,
                &mut st.llk,
            )?;
        }
        self.kernels.apply_amplitude(
            &self.stream,
            &dims,
            st.span.arg(),
            &self.counts,
            &mut st.psi,
        )?;
        plan.inverse_inplace(&mut st.psi)?;
        Ok(())
    }

    /// Adds stack `s`'s object and/or probe update sums (PyNX `Psi2Obj`, `Psi2Probe`) from its
    /// exit waves and the current arrays.
    fn accumulate(&mut self, s: usize, obj: bool, probe: bool) -> Result<(), EngineError> {
        let st = &self.stacks[s];
        if obj {
            self.kernels.psi_to_obj(
                &self.stream,
                &self.views,
                st.span.arg(),
                &st.psi,
                &self.probe,
                &mut self.obj_sums,
            )?;
        }
        if probe {
            self.kernels.psi_to_probe(
                &self.stream,
                &self.views,
                st.span.arg(),
                &st.psi,
                &self.obj,
                &mut self.probe_sums,
            )?;
        }
        Ok(())
    }

    /// Zeroes the sums of the arrays that have an inertia.
    fn zero_sums(&mut self, obj: Option<f64>, probe: Option<f64>) -> Result<(), EngineError> {
        if obj.is_some() {
            self.obj_sums.zero(&self.stream)?;
        }
        if probe.is_some() {
            self.probe_sums.zero(&self.stream)?;
        }
        Ok(())
    }

    /// PyNX `ObjNorm` / `ProbeNorm` from the accumulated sums, for the arrays that have an
    /// inertia.
    fn apply_sums(&mut self, obj: Option<f64>, probe: Option<f64>) -> Result<(), EngineError> {
        if let Some(inertia) = obj {
            let n = to_u32("object length", self.obj.len())?;
            self.kernels.update_from_sums(
                &self.stream,
                &mut self.obj,
                &self.obj_sums,
                n,
                inertia as f32,
            )?;
        }
        if let Some(inertia) = probe {
            let n = to_u32("probe length", self.probe.len())?;
            self.kernels.update_from_sums(
                &self.stream,
                &mut self.probe,
                &self.probe_sums,
                n,
                inertia as f32,
            )?;
        }
        Ok(())
    }

    /// [`zero_phase_ramp`] on the host: one read-back and upload, after an algorithm's last
    /// cycle.
    fn zero_phase_ramp(&mut self) -> Result<(), EngineError> {
        let mut object = self.object()?;
        let mut probe = self.probe()?;
        zero_phase_ramp(&self.fft, &mut object, &mut probe);
        self.obj
            .copy_from_host(&self.stream, &to_c32(object.as_slice()))?;
        self.probe
            .copy_from_host(&self.stream, &to_c32(probe.as_slice()))?;
        Ok(())
    }
}
