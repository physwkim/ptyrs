//! The GPU backend (DESIGN §7 M1): [`GpuPtycho`] runs AP, DM, RAAR, rPIE, ML and BH in `f32` on one CUDA
//! stream with the `ptyrs::gpu` pixel kernels and `ptyrs::fft` plans, in the call sequences the
//! `crate::gpu` ops and rPIE docs give, and is checked against [`crate::engine::Ptycho`].
//!
//! The scan's frames are cut into stacks of consecutive frames, of lengths within a frame of
//! each other, one [`StackTable`] entry each: every launch reads its stack's frames from the
//! entry on the device. Each pass cuts its own stacks by the bytes it streams per frame
//! ([`GpuOptions::stack_size`]). DM and RAAR keep
//! each stack's exit waves across cycles (RAAR also its iterate), sized like PyNX's fixed
//! stack for the longest stack. AP, DM and RAAR project a stack, ML
//! takes a stack's gradient, and the LLK pass takes its partials, either by M2's fused passes
//! over one scratch set or by the M1 launch sequence ([`GpuOptions::fused`]), whose one scratch
//! stack serves all and whose stacks of one length share a cuFFT plan, which transforms only
//! their frames; rPIE, BH and ML's line search always run M1's launches. Each cycle enqueues through its own `ExecutionContext`, dropped when the cycle
//! has been enqueued. Nothing in a cycle
//! reads back to the host: an LLK record goes, stack by stack, into the device's
//! [`crate::gpu::LlkRing`], whose copies download while later cycles run, and the host reads
//! the records at the end of each algorithm call (or of [`GpuPtycho::llk`]), the arrays only
//! when asked for. Every cycle, AP's with an LLK and DM's and RAAR's first included, and the LLK pass
//! replay a captured CUDA graph (DESIGN B2, [`GpuOptions::graph`]): one per [`Epoch`], kept
//! with the buffers it captured until another algorithm or other parameters need new ones,
//! and the LLK pass's for the life of the device state. Each algorithm is its own module over
//! the per-stack steps here.

mod ap;
mod bh;
mod device;
mod dm;
mod lsqml;
mod ml;
mod raar;
mod rpie;

use std::f64::consts::PI;
use std::ops::Range;
use std::sync::Arc;

use crate::base::{AddrTable, C64, Object, Probe, Scan, Stack};
use crate::fft::Plan2d;
use crate::gpu::{
    C32, CG_BLOCKS, Ended, Error as GpuError, FarSums, FusedBufs, FusedInputs, ObjProbeScale,
    RecordWriter, StackRef, StackTable, ViewAddr, ViewDims,
};
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};

use device::{Bufs, Device, NormBufs, PassesMut};

use crate::engine::ptycho::{Algorithm, LlkRecord, zero_phase_ramp};
use crate::engine::{
    ApParams, BhParams, DmParams, EngineError, Features, Fft2, Llk, LsqmlParams, MlParams,
    RaarParams, RpieParams,
};

/// How [`GpuPtycho`] batches its work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuOptions {
    /// At most this many frames per stack for every pass but rPIE's, whose batches are its own:
    /// each pass cuts the scan, BH and LSQML each batch, into as few stacks as that takes, of
    /// lengths within a frame of each other; PyNX's default is 16. `None`, the default, cuts each
    /// pass by bytes instead: as many frames as fill 3/8 of the device's L2, the fastest share at
    /// frame sides 256 and 512, with the buffers that more than one launch over a stack touches,
    /// which stay in L2 between them. That is the shared scratch alone in AP, the LLK pass and ML's
    /// gradient, and with it DM's and RAAR's carried exit waves (RAAR's iterate, touched once,
    /// streams); ML's line search, BH and LSQML count their own buffers, and under the position
    /// model BH and LSQML the shift planes they borrow from the shared scratch, whose stacks also
    /// cap theirs.
    pub stack_size: Option<usize>,
    /// Replay every AP, DM and ML cycle, rPIE epoch and LLK pass from a CUDA graph captured once
    /// per kind of cycle instead of launching it kernel by kernel; both give the same bits.
    pub graph: bool,
    /// LLK records the device ring holds, at least 1: a record's download is enqueued at
    /// most half this many records after it, and the host waits for one only when it is
    /// this many records ahead of it, or at the end of an algorithm call.
    pub llk_slots: usize,
    /// Project AP and DM stacks, take ML's gradients and the LLK pass's partials by M2's fused
    /// passes ([`crate::gpu::Kernels::fused_project`]; frame sides in
    /// [`crate::gpu::FUSED_SIDES`], and [`GpuPtycho::new`] fails on others) instead of M1's
    /// launch sequence. The two agree to rounding, not bit for bit.
    pub fused: bool,
}

impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            stack_size: None,
            graph: true,
            llk_slots: 8,
            fused: true,
        }
    }
}

/// A cycle with its parameters, which cover every value its launches bake in: the key of a
/// captured graph. Epochs that differ only in their flag share one [`Epoch::run`].
#[derive(Clone, Copy, Debug, PartialEq)]
enum Epoch {
    /// An AP cycle; with `llk`, it takes the LLK of the estimate it starts from.
    Ap { params: ApParams, llk: bool },
    /// A DM cycle; `first` for cycle 0 of a call.
    Dm { params: DmParams, first: bool },
    /// A RAAR cycle; `first` for cycle 0 of a call.
    Raar { params: RaarParams, first: bool },
    /// An rPIE epoch.
    Rpie(RpieParams),
    /// An ML cycle; `first` for cycle 0 of a call, with `llk` it takes the LLK of the
    /// estimate it starts from.
    Ml {
        params: MlParams,
        first: bool,
        llk: bool,
    },
    /// A BH epoch; with `cg` it continues the last epoch's direction.
    Bh { params: BhParams, cg: bool },
    /// An LSQML epoch; `pre` says what it does to the preconditioner.
    Lsqml {
        params: LsqmlParams,
        pre: lsqml::Pre,
    },
}

impl Epoch {
    fn algorithm(self) -> Algorithm {
        match self {
            Self::Ap { .. } => Algorithm::Ap,
            Self::Dm { .. } => Algorithm::Dm,
            Self::Raar { .. } => Algorithm::Raar,
            Self::Rpie(_) => Algorithm::Rpie,
            Self::Ml { .. } => Algorithm::Ml,
            Self::Bh { .. } => Algorithm::Bh,
            Self::Lsqml { .. } => Algorithm::Lsqml,
        }
    }

    /// The epoch with its flag cleared: the parameters a run's buffers are built for.
    fn run(self) -> Self {
        match self {
            Self::Ap { params, .. } => Self::Ap { params, llk: false },
            Self::Dm { params, .. } => Self::Dm {
                params,
                first: false,
            },
            Self::Raar { params, .. } => Self::Raar {
                params,
                first: false,
            },
            Self::Rpie(p) => Self::Rpie(p),
            Self::Ml { params, .. } => Self::Ml {
                params,
                first: false,
                llk: false,
            },
            Self::Bh { params, .. } => Self::Bh { params, cg: false },
            Self::Lsqml { params, .. } => Self::Lsqml {
                params,
                pre: lsqml::Pre::Keep,
            },
        }
    }
}

/// What an LLK record is filed under: the cycle and algorithm of a history record, or `None`
/// for [`GpuPtycho::llk`]'s.
type Tag = Option<(usize, Algorithm)>;

/// One stack (or rPIE batch): the frames its table entry holds, for what the host needs of
/// them (which plan, rPIE's `1 / nf`); the kernels read the entry.
#[derive(Clone, Debug)]
struct Frames {
    range: Range<u32>,
    /// Index of the stack's length, and so of its M1 cuFFT plan.
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

/// Stacks of consecutive frames: their [`Frames`], the table of their entries and their
/// distinct lengths, in the order [`Frames::plan`] indexes them.
struct Cut {
    frames: Vec<Frames>,
    table: StackTable,
    lens: Vec<u32>,
}

impl Cut {
    fn len(&self) -> usize {
        self.frames.len()
    }

    fn at(&self, s: usize) -> StackRef<'_> {
        self.table.at(s)
    }

    /// Values of the longest stack's exit waves, what each wave buffer of the cut holds.
    fn wave_len(&self, dims: &ViewDims) -> usize {
        self.table.nf_max() as usize * dims.nmode as usize * plane_len(dims)
    }

    /// One cuFFT plan per stack length, in [`Frames::plan`]'s order.
    fn plans(&self, stream: &Arc<CudaStream>, dims: &ViewDims) -> Result<Vec<Plan2d>, EngineError> {
        let nmode = dims.nmode as usize;
        Ok(self
            .lens
            .iter()
            .map(|&nf| Plan2d::new(stream, dims.frame[0] as usize, nf as usize * nmode))
            .collect::<Result<Vec<_>, _>>()?)
    }
}

/// Every cut of the frames into stacks: `0 .. nframe` in batches of `batch` (the last batch
/// may be shorter), each cut by [`even_stacks`] into stacks of at most `max`, and each batch's
/// range of stack indices. A pass over the scan is one batch of every frame; rPIE's batches
/// are a stack each (`max` equal to `batch`).
fn batch_stacks(
    stream: &Arc<CudaStream>,
    dims: &ViewDims,
    batch: u32,
    max: u32,
) -> Result<(Cut, Vec<Range<usize>>), EngineError> {
    let (mut ranges, mut lens, mut batches) = (Vec::new(), Vec::<u32>::new(), Vec::new());
    for f0 in (0..dims.nframe).step_by(batch as usize) {
        let first = ranges.len();
        for r in even_stacks(f0..dims.nframe.min(f0 + batch), max) {
            let nf = r.end - r.start;
            let plan = match lens.iter().position(|&l| l == nf) {
                Some(plan) => plan,
                None => {
                    lens.push(nf);
                    lens.len() - 1
                }
            };
            ranges.push((r, plan));
        }
        batches.push(first..ranges.len());
    }
    let (frames, table) = table_of(stream, dims, ranges)?;
    let cut = Cut {
        frames,
        table,
        lens,
    };
    Ok((cut, batches))
}

/// `range` cut into as few consecutive stacks of at most `max` frames as it takes, longer
/// ones first, their lengths within a frame of each other: a short tail stack would cost a
/// full stack's launches for a few frames.
fn even_stacks(range: Range<u32>, max: u32) -> impl Iterator<Item = Range<u32>> {
    let len = range.end - range.start;
    let k = len.div_ceil(max);
    (0..k).scan(range.start, move |f0, s| {
        let nf = len / k + u32::from(s < len % k);
        *f0 += nf;
        Some(*f0 - nf..*f0)
    })
}

/// The table of `ranges` and their [`Frames`], each range with its plan index.
fn table_of(
    stream: &Arc<CudaStream>,
    dims: &ViewDims,
    ranges: Vec<(Range<u32>, usize)>,
) -> Result<(Vec<Frames>, StackTable), EngineError> {
    let table = StackTable::new(
        stream,
        dims,
        &ranges.iter().map(|(r, _)| r.clone()).collect::<Vec<_>>(),
    )?;
    let nmode = dims.nmode as usize;
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
    Ok((frames, table))
}

/// A per-frame buffer of a loop over stacks: its bytes per frame and how many of the passes
/// over one stack (launches) of the loop's cycle touch it.
#[derive(Clone, Copy, Debug)]
struct Streamed {
    bytes: usize,
    passes: u32,
}

impl Streamed {
    fn new(bytes: usize, passes: u32) -> Self {
        Self { bytes, passes }
    }
}

/// [`GpuOptions::stack_size`] on one device: a loop's frames per stack from the buffers its
/// passes stream. Stacking keeps the buffers a stack's passes share in L2 from one pass to the
/// next, so only the bytes per frame of buffers more than one pass touches fill the budget; a
/// buffer one pass touches streams through either way.
#[derive(Clone, Copy, Debug)]
struct StackRule {
    size: Option<usize>,
    /// 3/8 of the device's L2, in bytes.
    budget: usize,
}

impl StackRule {
    /// For `size` on a device of an `l2`-byte L2.
    fn new(size: Option<usize>, l2: usize) -> Self {
        Self {
            size,
            budget: l2 / 8 * 3,
        }
    }

    /// Frames per stack of a loop over `bufs`: the given size, else as many frames of the
    /// shared buffers as fit the budget, at least one.
    fn frames(self, bufs: &[Streamed]) -> Result<u32, EngineError> {
        let shared: usize = (bufs.iter().filter(|b| b.passes > 1))
            .map(|b| b.bytes)
            .sum();
        let n = self
            .size
            .unwrap_or_else(|| (self.budget / shared.max(1)).max(1));
        to_u32("stack_size", n)
    }

    /// `0 .. nframe` in stacks of at most [`Self::frames`] for `bufs`.
    fn cut(
        self,
        stream: &Arc<CudaStream>,
        dims: &ViewDims,
        bufs: &[Streamed],
    ) -> Result<Cut, EngineError> {
        Ok(batch_stacks(stream, dims, dims.nframe, self.frames(bufs)?)?.0)
    }
}

/// Bytes per frame of `planes` frame planes of `T`.
fn plane_bytes<T>(dims: &ViewDims, planes: u32) -> usize {
    planes as usize * plane_len(dims) * size_of::<T>()
}

/// Bytes per frame of one exit-wave set, its `nmode` planes.
fn wave_bytes(dims: &ViewDims) -> usize {
    plane_bytes::<C32>(dims, dims.nmode)
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
/// records as [`crate::engine::Ptycho`] up to `f32` rounding.
pub struct GpuPtycho {
    dev: Device,
    obj_dims: [usize; 3],
    probe_dims: [usize; 3],
    /// Sum of the unmasked observed counts, for ML's [`crate::engine::Ptycho::scale_obj_probe`].
    total_obs: f64,
    /// The probe-frame FFT of [`zero_phase_ramp`], which runs on the host.
    fft: Fft2,
    cycle: usize,
    history: Vec<LlkRecord>,
}

impl GpuPtycho {
    /// Uploads the scan's counts and views (as [`crate::engine::Ptycho::new`] builds them) and the
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
        if options.stack_size == Some(0) {
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
            total_obs: scan.counts.total(),
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

    /// Replaces the object between algorithm calls, which start from it as from the one
    /// [`Self::new`] uploads; fails, changing nothing, when its dims differ from the current
    /// object's.
    pub fn set_object(&mut self, object: &Object) -> Result<(), EngineError> {
        if object.dims() != self.obj_dims {
            return Err(EngineError::Invalid(format!(
                "object {:?} differs from the reconstruction's {:?}",
                object.dims(),
                self.obj_dims
            )));
        }
        self.dev.write_object(&to_c32(object.as_slice()))
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

    /// The features modelled, read back from the device.
    pub fn features(&self) -> Result<Features, EngineError> {
        self.dev.features()
    }

    /// Models and refines `features` from here on, as [`crate::engine::Ptycho::set_features`], in
    /// `f32` on the device; with [`GpuOptions::fused`] only. An error leaves the current ones.
    pub fn set_features(&mut self, features: &Features) -> Result<(), EngineError> {
        let dims = self.dev.dims();
        let [ny, nx] = dims.frame.map(|n| n as usize);
        features.check(dims.nframe as usize, [ny, nx])?;
        self.dev.set_features(features)
    }

    /// Cycles (AP, DM, ML) and epochs (rPIE) run so far.
    pub fn cycle(&self) -> usize {
        self.cycle
    }

    /// Waits for every enqueued step to finish (for timing).
    pub fn synchronize(&self) -> Result<(), EngineError> {
        Ok(self.dev.stream().synchronize()?)
    }

    /// Normalised LLK of the current object and probe, as [`crate::engine::Ptycho::llk`]; the
    /// exit-wave stacks are left as they are.
    pub fn llk(&mut self) -> Result<Llk, EngineError> {
        self.dev.llk_pass(None)?;
        Ok(self
            .collect()?
            .expect("the LLK pass counts one untagged record"))
    }

    /// Waits for the LLK records enqueued since the last collection (the one host sync of an
    /// algorithm call that takes any), appends the tagged ones to the history, and returns
    /// the untagged one.
    fn collect(&mut self) -> Result<Option<Llk>, EngineError> {
        let mut untagged = None;
        for (tag, t) in self.dev.flush()? {
            let s = t.sums;
            let llk = Llk {
                poisson: s.poisson,
                gaussian: s.gaussian,
                euclidian: s.euclidean,
                nb_photons_calc: s.icalc,
            }
            .normalised(t.nb_obs);
            match tag {
                Some((cycle, algorithm)) => self.history.push(LlkRecord {
                    cycle,
                    algorithm,
                    llk,
                }),
                None => untagged = Some(llk),
            }
        }
        Ok(untagged)
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
    /// The LLK record of the current estimate through `rec`: M1's by the scratch stack.
    fn forward_llk(
        &mut self,
        ctx: &ExecutionContext,
        passes: &mut PassesMut<'_>,
        mut rec: RecordWriter<'_>,
    ) -> Result<Ended, GpuError> {
        let dims = *self.views.dims();
        match passes {
            PassesMut::M1(m1) => {
                for s in 0..self.scratch.len() {
                    let stack = self.scratch.at(s);
                    self.kernels.obj_probe_to_psi(
                        ctx, self.views, stack, self.obj, self.probe, m1.scratch,
                    )?;
                    let plan = self.scratch.frames[s].plan;
                    m1.plans.get(plan).forward_inplace(ctx, m1.scratch)?;
                    self.kernels.llk(
                        ctx,
                        stack,
                        self.counts,
                        m1.scratch,
                        self.llk_scratch,
                        self.llk,
                    )?;
                    rec.push(ctx, &dims, stack, self.llk)?;
                }
            }
            PassesMut::Fused(f) => {
                let m = self.features.lend(ctx, self.kernels, &dims, self.probe)?;
                for s in 0..self.scratch.len() {
                    let stack = self.scratch.at(s);
                    let inputs = FusedInputs {
                        views: self.views,
                        counts: self.counts,
                        obj: self.obj,
                        probe: self.probe,
                        model: m.model,
                    };
                    let far = FarSums {
                        llk: Some(&mut *self.llk),
                        ..FarSums::default()
                    };
                    self.kernels.fused_llk(ctx, &inputs, stack, f, far)?;
                    rec.push(ctx, &dims, stack, self.llk)?;
                }
            }
        }
        rec.end(ctx)
    }

    /// The features' step before a cycle models them: stack by stack, the far sums of the
    /// current estimate under the current model by the fused LLK pass (with `llk`, its LLK
    /// partials too, pushed into that record and ended after the last stack), then the step.
    pub(super) fn step_features(
        &mut self,
        ctx: &ExecutionContext,
        f: &mut FusedBufs,
        mut llk: Option<RecordWriter<'_>>,
    ) -> Result<Option<Ended>, GpuError> {
        let dims = *self.views.dims();
        let mut m = self.features.lend(ctx, self.kernels, &dims, self.probe)?;
        for s in 0..self.scratch.len() {
            let stack = self.scratch.at(s);
            let inputs = FusedInputs {
                views: self.views,
                counts: self.counts,
                obj: self.obj,
                probe: self.probe,
                model: m.model,
            };
            let far = m.far(llk.is_some().then_some(&mut *self.llk));
            self.kernels.fused_llk(ctx, &inputs, stack, f, far)?;
            if let Some(rec) = &mut llk {
                rec.push(ctx, &dims, stack, self.llk)?;
            }
        }
        let ended = llk.map(|w| w.end(ctx)).transpose()?;
        self.features.step(ctx, self.kernels, self.grad)?;
        Ok(ended)
    }

    /// M1 Fourier-modulus projection of `stack`'s exit waves `psi` in place by `plan`, its
    /// length's: FFT, `apply_amplitude`, inverse FFT. With `llk`, the stack's LLK partials of
    /// the forward stack go into that record first.
    fn project(
        &mut self,
        ctx: &ExecutionContext,
        stack: StackRef<'_>,
        plan: &mut Plan2d,
        psi: &mut DeviceBuffer<C32>,
        llk: Option<&mut RecordWriter<'_>>,
    ) -> Result<(), GpuError> {
        let dims = *self.views.dims();
        plan.forward_inplace(ctx, psi)?;
        if let Some(rec) = llk {
            self.kernels
                .llk(ctx, stack, self.counts, psi, self.llk_scratch, self.llk)?;
            rec.push(ctx, &dims, stack, self.llk)?;
        }
        self.kernels
            .apply_amplitude(ctx, &dims, stack, self.counts, psi)?;
        plan.inverse_inplace(ctx, psi)?;
        Ok(())
    }

    /// Adds `stack`'s object and/or probe update sums (PyNX `Psi2Obj`, `Psi2Probe`) from its
    /// exit waves `psi` and the current arrays.
    fn accumulate(
        &mut self,
        ctx: &ExecutionContext,
        stack: StackRef<'_>,
        psi: &DeviceBuffer<C32>,
        obj: bool,
        probe: bool,
    ) -> Result<(), GpuError> {
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

    /// PyNX `ScaleObjProbe`, without a host sync: the norms of the object and the probe, then
    /// the scaling, [`ObjProbeScale::Absolute`] to `absolute`'s `(icalc, total)` photons, the
    /// background's photons left out of both, or [`ObjProbeScale::Relative`] for `None`.
    fn scale_obj_probe(
        &mut self,
        ctx: &ExecutionContext,
        absolute: Option<(&DeviceBuffer<f32>, f32)>,
    ) -> Result<(), GpuError> {
        let NormBufs {
            parts,
            pairs: [obj_norm, probe_norm],
        } = &mut *self.norms;
        for (x, n, out) in [
            (&*self.obj, self.obj_len, &mut *obj_norm),
            (&*self.probe, self.probe_len, &mut *probe_norm),
        ] {
            crate::gpu::zero(ctx, parts)?;
            self.grad.norm_parts(ctx, x, n, parts)?;
            self.grad.sum_parts(ctx, parts, CG_BLOCKS, out)?;
        }
        let norms = [&*obj_norm, &*probe_norm];
        let scale = match absolute {
            Some((icalc, total)) => ObjProbeScale::Absolute {
                icalc,
                total,
                background: self.features.background_photons(ctx, self.grad)?,
            },
            None => ObjProbeScale::Relative,
        };
        self.grad
            .scale_obj_probe(ctx, self.obj, self.probe, norms, scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(side: u32) -> ViewDims {
        ViewDims {
            nframe: 400,
            nmode: 2,
            nprobe: 2,
            nobj: 1,
            frame: [side; 2],
            obj: [1024; 2],
            quad_phase: 0.0,
        }
    }

    /// A batch of 96 at most 19 per stack is 6 stacks of 16, not 5 of 19 and one of 1; 30 at
    /// 7 is 6, 6, 6, 6, 6; lengths that divide stay whole, and 5 frames at 8 are one stack.
    #[test]
    fn even_stacks_cut_a_batch_into_equal_lengths() {
        let lens = |r: Range<u32>, max| {
            let stacks: Vec<_> = even_stacks(r.clone(), max).collect();
            assert_eq!(stacks.first().map(|s| s.start), Some(r.start));
            assert_eq!(stacks.last().map(|s| s.end), Some(r.end));
            assert!(stacks.windows(2).all(|w| w[0].end == w[1].start));
            stacks.iter().map(|s| s.end - s.start).collect::<Vec<_>>()
        };
        assert_eq!(lens(0..96, 19), [16; 6]);
        assert_eq!(lens(96..126, 7), [6; 5]);
        assert_eq!(lens(0..64, 19), [16; 4]);
        assert_eq!(lens(0..64, 18), [16; 4]);
        assert_eq!(lens(0..65, 16), [13; 5]);
        assert_eq!(lens(0..10, 4), [4, 3, 3]);
        assert_eq!(lens(0..96, 16), [16; 6]);
        assert_eq!(lens(3..8, 8), [5]);
    }

    /// 24 MiB of a 64 MiB L2 holds 24, 12, 8 and 6 frames of one to four shared two-mode
    /// exit-wave sets at 256x256, 6, 3, 2 and 1 at 512x512, and 16 of one and a half (a set
    /// stored at half width); a set one pass touches does not count; a frame alone that
    /// overfills it is one, and a given size wins.
    #[test]
    fn stack_rule_fills_three_eighths_of_l2() {
        const L2: usize = 64 << 20;
        let rule = StackRule::new(None, L2);
        let sets = |wave, k, passes| vec![Streamed::new(wave, passes); k];
        for (side, want) in [(256, [24, 12, 8, 6]), (512, [6, 3, 2, 1])] {
            let wave = wave_bytes(&dims(side));
            let frames = [1, 2, 3, 4].map(|k| rule.frames(&sets(wave, k, 2)).unwrap());
            assert_eq!(frames, want);
        }
        let wave = wave_bytes(&dims(256));
        let once = [sets(wave, 2, 3), sets(wave, 1, 1)].concat();
        assert_eq!(rule.frames(&once).unwrap(), 12);
        let half = [Streamed::new(wave, 2), Streamed::new(wave / 2, 2)];
        assert_eq!(rule.frames(&half).unwrap(), 16);
        let big = sets(wave_bytes(&dims(512)), 4, 2);
        assert_eq!(StackRule::new(None, 4 << 20).frames(&big[..1]).unwrap(), 1);
        assert_eq!(StackRule::new(Some(16), L2).frames(&big).unwrap(), 16);
    }
}
