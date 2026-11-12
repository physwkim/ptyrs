//! The GPU backend (DESIGN §7 M1): [`GpuPtycho`] runs AP, DM and rPIE in `f32` on one CUDA
//! stream with the `ptyrs-gpu` pixel kernels and `ptyrs-fft` plans, in the call sequences the
//! `ptyrs_gpu` ops and rPIE docs give, and is checked against [`crate::Ptycho`].
//!
//! The scan's frames are cut into stacks of [`GpuOptions::stack_size`] consecutive frames,
//! one [`StackTable`] entry each: every launch reads its stack's frames from the entry on the
//! device. Each stack keeps its own exit waves (DM carries them across cycles) and LLK
//! partials, sized like PyNX's fixed stack for the longest stack; one scratch stack serves
//! all, and stacks of one length share a cuFFT plan, which transforms only their frames.
//! Each cycle enqueues through its own `ExecutionContext`, dropped when the cycle has been
//! enqueued. Within a cycle nothing reads back to the host: LLK partials are downloaded only
//! in the cycles that record one, and the arrays only when asked for. An AP or DM cycle that
//! records no LLK in-cycle replays one captured CUDA graph (DESIGN B2, [`GpuOptions::graph`]),
//! kept until another algorithm or other parameters need a new one. Each algorithm is its
//! own module over the per-stack steps here, so M2's fused passes can replace those steps one
//! algorithm at a time.

mod ap;
mod dm;
mod rpie;

use std::f64::consts::PI;
use std::ops::Range;
use std::sync::Arc;

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use ptyrs_core::{AddrTable, C64, Object, Probe, Scan, Stack};
use ptyrs_fft::Plan2d;
use ptyrs_gpu::{
    C32, Error as GpuError, Kernels, LlkPartial, LlkSums, RpieKernels, StackTable, UpdateSums,
    ViewAddr, ViewDims, Views,
};

use crate::ptycho::{Algorithm, LlkRecord, zero_phase_ramp};
use crate::{ApParams, DmParams, EngineError, Fft2, Llk};

/// How [`GpuPtycho`] batches its work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuOptions {
    /// Frames per stack (the last stack may be shorter); PyNX's default is 16.
    pub stack_size: usize,
    /// Replay steady-state AP and DM cycles as one captured CUDA graph each instead of
    /// launching them kernel by kernel; both give the same bits.
    pub graph: bool,
}

impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            stack_size: 16,
            graph: true,
        }
    }
}

/// A steady-state cycle with its parameters, which cover every value its launches bake in:
/// the key of a captured graph.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Epoch {
    /// An AP cycle that takes no LLK.
    Ap(ApParams),
    /// A DM cycle after the first of a call.
    Dm(DmParams),
}

struct EpochGraph {
    epoch: Epoch,
    graph: CudaGraph<()>,
}

/// One stack (or rPIE batch): the frames its table entry holds, for what the host needs of
/// them (which plan, how many LLK partials, rPIE's `1 / nf`); the kernels read the entry.
#[derive(Clone, Debug)]
struct Frames {
    range: Range<u32>,
    /// Index of the stack's length, and so of its cuFFT plan.
    plan: usize,
    /// `nf * nmode * n`: the values of the stack's exit waves, what the element-wise steps
    /// and its plan cover.
    len: u32,
}

impl Frames {
    fn nf(&self) -> u32 {
        self.range.end - self.range.start
    }
}

/// One stack's buffers, sized like every stack's for the table's `nf_max` frames.
struct StackBufs {
    frames: Frames,
    /// Exit waves of the stack's views; planes past `nf * nmode` stay zero.
    psi: DeviceBuffer<C32>,
    llk: DeviceBuffer<LlkPartial>,
}

/// `0 .. nframe` cut into consecutive ranges of `size` frames, the last one shorter when
/// `size` does not divide `nframe`; the table of their entries and one plan per length.
fn stacks(
    stream: &Arc<CudaStream>,
    dims: &ViewDims,
    size: u32,
) -> Result<(Vec<Frames>, StackTable, Vec<Plan2d>), EngineError> {
    let (ranges, lens) = split(dims.nframe, size);
    let table = StackTable::new(
        stream,
        dims,
        &ranges.iter().map(|(r, _)| r.clone()).collect::<Vec<_>>(),
    )?;
    let nmode = dims.nmode as usize;
    let plans = lens
        .iter()
        .map(|&nf| Plan2d::new(stream, dims.frame[0] as usize, nf as usize * nmode))
        .collect::<Result<Vec<_>, _>>()?;
    let frames = ranges
        .into_iter()
        .map(|(range, plan)| {
            let len = (range.end - range.start) as usize * nmode * plane_len(dims);
            Ok(Frames {
                range,
                plan,
                len: to_u32("stack length", len)?,
            })
        })
        .collect::<Result<Vec<_>, EngineError>>()?;
    Ok((frames, table, plans))
}

/// Values of `nf_max` frames' exit waves, what every stack buffer holds.
fn stack_buf_len(dims: &ViewDims, table: &StackTable) -> usize {
    table.nf_max() as usize * dims.nmode as usize * plane_len(dims)
}

/// `0 .. nframe` in consecutive ranges of `size` frames, the last one shorter when `size`
/// does not divide `nframe`, each with the index of its length in the returned lengths.
fn split(nframe: u32, size: u32) -> (Vec<(Range<u32>, usize)>, Vec<u32>) {
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
        ranges.push((f0..f1, idx));
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
    obj_len: u32,
    probe_len: u32,
    table: StackTable,
    stacks: Vec<StackBufs>,
    /// One plan per stack length, indexed by [`Frames::plan`].
    plans: Vec<Plan2d>,
    /// One stack's worth of exit waves: DM's `PO`, the LLK pass's forward stack.
    scratch: DeviceBuffer<C32>,
    nb_obs: usize,
    /// The probe-frame FFT of [`zero_phase_ramp`], which runs on the host.
    fft: Fft2,
    use_graph: bool,
    /// The last captured steady-state cycle. Its nodes hold the buffers above, which are
    /// allocated once in [`GpuPtycho::new`] and never replaced.
    graph: Option<EpochGraph>,
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
        let (frames, table, plans) =
            stacks(stream, &dims, to_u32("stack_size", options.stack_size)?)?;
        let len = stack_buf_len(&dims, &table);
        let stacks = frames
            .into_iter()
            .map(|frames| {
                Ok(StackBufs {
                    frames,
                    psi: DeviceBuffer::zeroed(stream, len)?,
                    llk: DeviceBuffer::zeroed(stream, table.nf_max() as usize)?,
                })
            })
            .collect::<Result<Vec<_>, EngineError>>()?;
        Ok(Self {
            obj_len: to_u32("object length", object.as_slice().len())?,
            probe_len: to_u32("probe length", probe.as_slice().len())?,
            scratch: DeviceBuffer::zeroed(stream, len)?,
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
            table,
            stacks,
            plans,
            use_graph: options.graph,
            graph: None,
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
        self.forward_llk(&self.context())?;
        self.read_llk()
    }

    /// A fresh eager context on the stream: one per cycle, so leases end with its work.
    fn context(&self) -> ExecutionContext {
        ExecutionContext::new(self.stream.clone())
    }

    /// Enqueues one `epoch` cycle: replays its graph, captured (on a side stream, so nothing
    /// runs) when the cached one is for another epoch, or launches it eagerly without graphs.
    fn steady_cycle(&mut self, epoch: Epoch) -> Result<(), GpuError> {
        if !self.use_graph {
            return self.epoch_cycle(&self.context(), epoch);
        }
        let cached = match self.graph.take() {
            Some(g) if g.epoch == epoch => g,
            _ => {
                let side = self.kernels.context().new_stream()?;
                let op = ptyrs_gpu::record(|ctx| self.epoch_cycle(ctx, epoch));
                EpochGraph {
                    epoch,
                    graph: CudaGraph::capture(side, op)?,
                }
            }
        };
        let ctx = self.context();
        // SAFETY: `ctx` holds the replay's leases on every captured buffer until the stream
        // has run it, so no other stream can touch them meanwhile.
        let launched = unsafe { cached.graph.launch().execute(&ctx) };
        self.graph = Some(cached);
        Ok(launched?)
    }

    fn epoch_cycle(&mut self, ctx: &ExecutionContext, epoch: Epoch) -> Result<(), GpuError> {
        match epoch {
            Epoch::Ap(params) => self.ap_cycle(ctx, &params, false),
            Epoch::Dm(params) => self.dm_cycle(ctx, &params, false),
        }
    }

    fn record(&mut self, algorithm: Algorithm, llk: Llk) {
        self.history.push(LlkRecord {
            cycle: self.cycle,
            algorithm,
            llk,
        });
    }

    /// LLK partials of the current estimate into every stack's `llk`, through the scratch
    /// stack.
    fn forward_llk(&mut self, ctx: &ExecutionContext) -> Result<(), GpuError> {
        let dims = *self.views.dims();
        for (s, st) in self.stacks.iter_mut().enumerate() {
            let stack = self.table.at(s);
            self.kernels.obj_probe_to_psi(
                ctx,
                &self.views,
                stack,
                &self.obj,
                &self.probe,
                &mut self.scratch,
            )?;
            self.plans[st.frames.plan].forward_inplace(ctx, &mut self.scratch)?;
            self.kernels
                .llk(ctx, &dims, stack, &self.counts, &self.scratch, &mut st.llk)?;
        }
        Ok(())
    }

    /// Reads back the stacks' LLK partials (the one host sync of a recording cycle) and sums
    /// them in frame order.
    fn read_llk(&self) -> Result<Llk, EngineError> {
        let mut partials = Vec::with_capacity(self.views.dims().nframe as usize);
        for st in &self.stacks {
            let llk = st.llk.to_host_vec(&self.stream)?;
            partials.extend_from_slice(&llk[..st.frames.nf() as usize]);
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
    fn project(&mut self, ctx: &ExecutionContext, s: usize, llk: bool) -> Result<(), GpuError> {
        let dims = *self.views.dims();
        let stack = self.table.at(s);
        let st = &mut self.stacks[s];
        let plan = &mut self.plans[st.frames.plan];
        plan.forward_inplace(ctx, &mut st.psi)?;
        if llk {
            self.kernels
                .llk(ctx, &dims, stack, &self.counts, &st.psi, &mut st.llk)?;
        }
        self.kernels
            .apply_amplitude(ctx, &dims, stack, &self.counts, &mut st.psi)?;
        plan.inverse_inplace(ctx, &mut st.psi)?;
        Ok(())
    }

    /// Adds stack `s`'s object and/or probe update sums (PyNX `Psi2Obj`, `Psi2Probe`) from its
    /// exit waves and the current arrays.
    fn accumulate(
        &mut self,
        ctx: &ExecutionContext,
        s: usize,
        obj: bool,
        probe: bool,
    ) -> Result<(), GpuError> {
        let stack = self.table.at(s);
        let psi = &self.stacks[s].psi;
        if obj {
            self.kernels.psi_to_obj(
                ctx,
                &self.views,
                stack,
                psi,
                &self.probe,
                &mut self.obj_sums,
            )?;
        }
        if probe {
            self.kernels.psi_to_probe(
                ctx,
                &self.views,
                stack,
                psi,
                &self.obj,
                &mut self.probe_sums,
            )?;
        }
        Ok(())
    }

    /// Zeroes the sums of the arrays that have an inertia.
    fn zero_sums(
        &mut self,
        ctx: &ExecutionContext,
        obj: Option<f64>,
        probe: Option<f64>,
    ) -> Result<(), GpuError> {
        if obj.is_some() {
            self.obj_sums.zero(ctx)?;
        }
        if probe.is_some() {
            self.probe_sums.zero(ctx)?;
        }
        Ok(())
    }

    /// PyNX `ObjNorm` / `ProbeNorm` from the accumulated sums, for the arrays that have an
    /// inertia; the object only on the tiles some view reaches.
    fn apply_sums(
        &mut self,
        ctx: &ExecutionContext,
        obj: Option<f64>,
        probe: Option<f64>,
    ) -> Result<(), GpuError> {
        if let Some(inertia) = obj {
            self.kernels.update_obj_from_sums(
                ctx,
                &self.views,
                &mut self.obj,
                &mut self.obj_sums,
                inertia as f32,
            )?;
        }
        if let Some(inertia) = probe {
            self.kernels.update_from_sums(
                ctx,
                &mut self.probe,
                &mut self.probe_sums,
                self.probe_len,
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
