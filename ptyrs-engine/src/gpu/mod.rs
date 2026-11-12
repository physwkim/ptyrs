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
mod device;
mod dm;
mod rpie;

use std::f64::consts::PI;
use std::ops::Range;
use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use ptyrs_core::{AddrTable, C64, Object, Probe, Scan, Stack};
use ptyrs_fft::Plan2d;
use ptyrs_gpu::{C32, Error as GpuError, LlkSums, StackTable, ViewAddr, ViewDims};

use device::{Bufs, Device};

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
    dev: Device,
    obj_dims: [usize; 3],
    probe_dims: [usize; 3],
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

        let dev = Device::new(
            ctx,
            dims,
            &addrs,
            scan.counts.as_slice(),
            &to_c32(object.as_slice()),
            &to_c32(probe.as_slice()),
            options,
        )?;
        Ok(Self {
            dev,
            obj_dims: object.dims(),
            probe_dims: probe.dims(),
            nb_obs: scan.counts.nb_obs(),
            fft: Fft2::new(g.frame),
            cycle: 0,
            history: Vec::new(),
        })
    }

    /// The object, read back from the device.
    pub fn object(&self) -> Result<Object, EngineError> {
        Ok(Object::new(download(
            self.dev.stream(),
            self.dev.obj(),
            self.obj_dims,
        )?))
    }

    /// The probe, read back from the device.
    pub fn probe(&self) -> Result<Probe, EngineError> {
        Ok(Probe::new(download(
            self.dev.stream(),
            self.dev.probe(),
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
        Ok(self.dev.stream().synchronize()?)
    }

    /// Normalised LLK of the current object and probe, as [`crate::Ptycho::llk`]; the
    /// exit-wave stacks are left as they are.
    pub fn llk(&mut self) -> Result<Llk, EngineError> {
        let ctx = self.dev.context();
        self.dev.lend().forward_llk(&ctx)?;
        self.read_llk()
    }

    fn record(&mut self, algorithm: Algorithm, llk: Llk) {
        self.history.push(LlkRecord {
            cycle: self.cycle,
            algorithm,
            llk,
        });
    }

    /// Reads back the stacks' LLK partials (the one host sync of a recording cycle) and sums
    /// them in frame order.
    fn read_llk(&self) -> Result<Llk, EngineError> {
        let mut partials = Vec::with_capacity(self.dev.dims().nframe as usize);
        for (frames, llk) in self.dev.stack_llk() {
            let llk = llk.to_host_vec(self.dev.stream())?;
            partials.extend_from_slice(&llk[..frames.nf() as usize]);
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

    /// [`zero_phase_ramp`] on the host: one read-back and upload, after an algorithm's last
    /// cycle.
    fn zero_phase_ramp(&mut self) -> Result<(), EngineError> {
        let mut object = self.object()?;
        let mut probe = self.probe()?;
        zero_phase_ramp(&self.fft, &mut object, &mut probe);
        self.dev
            .write_arrays(&to_c32(object.as_slice()), &to_c32(probe.as_slice()))
    }
}

/// The per-stack steps of every algorithm, on the lent device state.
impl Bufs<'_> {
    fn epoch_cycle(&mut self, ctx: &ExecutionContext, epoch: Epoch) -> Result<(), GpuError> {
        match epoch {
            Epoch::Ap(params) => self.ap_cycle(ctx, &params, false),
            Epoch::Dm(params) => self.dm_cycle(ctx, &params, false),
        }
    }

    /// LLK partials of the current estimate into every stack's `llk`, through the scratch
    /// stack.
    fn forward_llk(&mut self, ctx: &ExecutionContext) -> Result<(), GpuError> {
        let dims = *self.views.dims();
        for s in 0..self.stacks.len() {
            let stack = self.table.at(s);
            let st = self.stacks.get(s);
            self.kernels.obj_probe_to_psi(
                ctx,
                self.views,
                stack,
                self.obj,
                self.probe,
                self.scratch,
            )?;
            self.plans
                .get(st.frames.plan)
                .forward_inplace(ctx, self.scratch)?;
            self.kernels
                .llk(ctx, &dims, stack, self.counts, self.scratch, st.llk)?;
        }
        Ok(())
    }

    /// Fourier-modulus projection of stack `s`'s exit waves in place: FFT, `apply_amplitude`,
    /// inverse FFT. With `llk`, the stack's LLK partials of the forward stack are taken first.
    fn project(&mut self, ctx: &ExecutionContext, s: usize, llk: bool) -> Result<(), GpuError> {
        let dims = *self.views.dims();
        let stack = self.table.at(s);
        let st = self.stacks.get(s);
        let plan = self.plans.get(st.frames.plan);
        plan.forward_inplace(ctx, st.psi)?;
        if llk {
            self.kernels
                .llk(ctx, &dims, stack, self.counts, st.psi, st.llk)?;
        }
        self.kernels
            .apply_amplitude(ctx, &dims, stack, self.counts, st.psi)?;
        plan.inverse_inplace(ctx, st.psi)?;
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
        let psi = self.stacks.get(s).psi;
        if obj {
            self.kernels
                .psi_to_obj(ctx, self.views, stack, psi, self.probe, self.obj_sums)?;
        }
        if probe {
            self.kernels
                .psi_to_probe(ctx, self.views, stack, psi, self.obj, self.probe_sums)?;
        }
        Ok(())
    }

    /// PyNX `ObjNorm` / `ProbeNorm` from the accumulated sums, consuming them, for the arrays
    /// that have an inertia; the object only on the tiles some view reaches.
    fn apply_sums(
        &mut self,
        ctx: &ExecutionContext,
        obj: Option<f64>,
        probe: Option<f64>,
    ) -> Result<(), GpuError> {
        if let Some(inertia) = obj {
            self.kernels.update_obj_from_sums(
                ctx,
                self.views,
                self.obj,
                self.obj_sums,
                inertia as f32,
            )?;
        }
        if let Some(inertia) = probe {
            self.kernels.update_from_sums(
                ctx,
                self.probe,
                self.probe_sums,
                self.probe_len,
                inertia as f32,
            )?;
        }
        Ok(())
    }
}
