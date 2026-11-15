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
mod propagate;
mod raar;
mod rpie;

use std::ops::Range;
use std::sync::Arc;

use crate::base::{AddrTable, C64, Object, Probe, Scan, Stack};
use crate::fault;
use crate::fft::Plan2d;
use crate::gpu::{
    C32, CG_BLOCKS, Error as GpuError, FarSums, FusedBufs, FusedInputs, Model, ObjProbeScale,
    RecordWriter, StackRef, StackTable, ViewAddr, ViewDims, Waves,
};
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};

use device::{Bufs, Device, NormBufs, OneSlice, PassesMut, SliceUploads, Sliced, Uploads};

use crate::engine::ptycho::{Algorithm, LlkRecord, zero_phase_ramp};
use crate::engine::{
    ApParams, BhParams, DmParams, EngineError, Features, Fft2, Llk, LsqmlParams, MlParams, Optics,
    RaarParams, RpieParams, Sampling,
};

/// How [`GpuPtycho`] batches its work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuOptions {
    /// At most this many frames per stack for every pass but rPIE's, whose batches are its own:
    /// each pass cuts the scan, BH and LSQML each batch, into as few stacks as that takes, of
    /// lengths within a frame of each other; PyNX's default is 16. [`GpuPtycho::new`] refuses a
    /// size whose stack, or the scan if it has fewer frames, holds more exit-wave values than
    /// u32 indexes. `None`, the default, cuts each
    /// pass by bytes instead: as many frames as fill 3/8 of the device's L2, the fastest share at
    /// frame sides 256 and 512, with the most bytes that the launches over a stack leave in L2
    /// for later ones at any one point, but no fewer than keep the device's threads busy, ~3.4
    /// pixels each. That is the shared scratch alone in AP, the LLK pass and ML's gradient, and
    /// with it DM's and RAAR's carried exit waves (RAAR's iterate, touched once, streams); ML's
    /// line search, BH and LSQML count their own buffers, and under the position model BH and
    /// LSQML the shift planes they borrow from the shared scratch, whose stacks also cap theirs.
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
    /// Hold DM's and RAAR's carried exit waves and RAAR's iterate as f16 pairs, each value
    /// times one power of two fixed from the counts at upload (DESIGN B5), and cut their
    /// stacks at those 4 bytes per value. Fused passes only: M1's cuFFT transforms the waves
    /// in place in f32, and [`GpuPtycho::new`] fails on `psi_f16` without [`Self::fused`].
    pub psi_f16: bool,
}

impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            stack_size: None,
            graph: true,
            llk_slots: 8,
            fused: true,
            psi_f16: false,
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
    let frames = ranges
        .into_iter()
        .map(|(range, plan)| {
            Ok(Frames {
                len: stack_len(dims, (range.end - range.start) as usize)?,
                range,
                plan,
            })
        })
        .collect::<Result<Vec<_>, EngineError>>()?;
    Ok((frames, table))
}

/// The values of a stack of `nf` frames' exit waves.
fn stack_len(dims: &ViewDims, nf: usize) -> Result<u32, EngineError> {
    to_u32("stack length", nf * dims.nmode as usize * plane_len(dims))
}

/// A buffer of a loop over stacks: its bytes per frame, its bytes whatever the stack's frames,
/// and the first and last of the passes over one stack (launches, numbered in order) that touch
/// it. A call that transforms in chunks counts as the launches of one chunk, however many it
/// takes, and is declared with its planes by [`Self::chunked`]. The loops sharing a cut number their passes in turn, so a buffer kept from one loop
/// to the next is listed once for each: that reuse spans all of a batch's stacks, which no
/// stack size keeps in L2.
#[derive(Clone, Copy, Debug)]
struct Streamed {
    bytes: usize,
    fixed: usize,
    first: u32,
    last: u32,
}

impl Streamed {
    /// `bytes` per frame, touched by `passes`.
    fn new(bytes: usize, passes: std::ops::RangeInclusive<u32>) -> Self {
        Self {
            bytes,
            ..Self::fixed(0, passes)
        }
    }

    /// `bytes` whatever the stack's frames, touched by `passes`.
    fn fixed(bytes: usize, passes: std::ops::RangeInclusive<u32>) -> Self {
        Self {
            bytes: 0,
            fixed: bytes,
            first: *passes.start(),
            last: *passes.end(),
        }
    }

    /// A call over `passes` that transforms the stack in chunks, keeping `held` bytes whatever
    /// the stack's frames across its launches, and the planes it `writes`, `(bytes per frame,
    /// the last pass to read them)`, and `reads`, `(bytes per frame, the first pass to touch
    /// them)`. Its chunks run one after another through every launch, so those planes span all
    /// of them.
    fn chunked(
        held: usize,
        passes: std::ops::RangeInclusive<u32>,
        writes: &[(usize, u32)],
        reads: &[(usize, u32)],
    ) -> Vec<Self> {
        let (first, last) = (*passes.start(), *passes.end());
        let writes = (writes.iter()).map(|&(b, read)| Self::new(b, first..=read.max(last)));
        let reads = (reads.iter()).map(|&(b, from)| Self::new(b, from.min(first)..=last));
        let held = std::iter::once(Self::fixed(held, passes));
        held.chain(writes).chain(reads).collect()
    }

    /// Whether it is live from pass `p` to the next: touched by `p` or an earlier pass and by a
    /// later one.
    fn live(self, p: u32) -> bool {
        (self.first..self.last).contains(&p)
    }

    /// The shift planes of a [`Kernels::shifted_psi_sums`] from pass `p`: `shift_pre` writes
    /// `T`, `shift_sums` reads it into the adjoint, object planes and power, `shift_post` and
    /// the probe sums take the adjoint, the gather the rest.
    fn shifted_psi_sums(dims: &ViewDims, p: u32) -> [Self; 3] {
        let probe = plane_bytes::<C32>(dims, dims.nprobe);
        let obj = plane_bytes::<C32>(dims, dims.nobj) + plane_bytes::<f32>(dims, 1);
        [
            Self::new(probe, p..=p + 1),
            Self::new(probe, p + 1..=p + 3),
            Self::new(obj, p + 1..=p + 4),
        ]
    }
}

/// [`GpuOptions::stack_size`] on one device: a loop's frames per stack from the buffers its
/// passes stream. Stacking keeps what one pass over a stack leaves for a later one in L2, so
/// the budget holds what is live across every boundary between passes, its fixed bytes and its
/// bytes per frame for each frame; a buffer one pass touches streams through either way. Below
/// a floor of frames a pass leaves the device's threads short of work, whatever L2 holds.
#[derive(Clone, Copy, Debug)]
struct StackRule {
    size: Option<usize>,
    /// 3/8 of the device's L2, in bytes.
    budget: usize,
    /// The fewest frames per stack that keep the device's threads busy.
    floor: usize,
    /// The most frames per stack: the scan's, at most those whose exit waves [`stack_len`]
    /// indexes.
    most: u32,
}

impl StackRule {
    /// For `size` on `ctx`'s device and frames of `dims`.
    fn on(size: Option<usize>, ctx: &CudaContext, dims: &ViewDims) -> Result<Self, EngineError> {
        let sms = ctx.multiprocessor_count()? as usize;
        let threads = sms * ctx.max_threads_per_multiprocessor()? as usize;
        Self::new(size, ctx.l2_cache_size_bytes()?, threads, dims)
    }

    /// For `size` on a device of an `l2`-byte L2 and `threads` resident threads, and frames of
    /// `dims`; refused if a stack of `size` frames, or of one, fewer if the scan has fewer,
    /// overflows [`stack_len`], so no stack a call cuts by it does. The floor holds ~3.4 pixels
    /// per thread, where the measured fastest stacks begin: 2^19 pixels on 100 SMs of 1536
    /// threads, 32, 8 and 2 frames at 128x128, 256x256 and 512x512.
    fn new(
        size: Option<usize>,
        l2: usize,
        threads: usize,
        dims: &ViewDims,
    ) -> Result<Self, EngineError> {
        let nframe = dims.nframe as usize;
        stack_len(dims, size.unwrap_or(1).min(nframe))?;
        let fits = u32::MAX as usize / (dims.nmode as usize * plane_len(dims));
        Ok(Self {
            size,
            budget: l2 / 8 * 3,
            floor: (threads * 17 / 5).div_ceil(dims.n()),
            most: dims.nframe.min(fits as u32),
        })
    }

    /// Frames per stack of a loop over `bufs`: the given size, else the most that fit the
    /// budget beside the fixed bytes at every boundary where frames' bytes are live, at least
    /// the floor; at most [`Self::most`].
    fn frames(self, bufs: &[Streamed]) -> u32 {
        // What is live after a pass is live after the latest first pass up to it, a set that
        // bounds the frames at least as tightly, so the first passes suffice.
        let fit = (bufs.iter().map(|b| b.first))
            .filter_map(|p| {
                let live = || bufs.iter().filter(|b| b.live(p));
                let per_frame: usize = live().map(|b| b.bytes).sum();
                let fixed: usize = live().map(|b| b.fixed).sum();
                (per_frame > 0).then(|| self.budget.saturating_sub(fixed) / per_frame)
            })
            .min()
            .unwrap_or(self.budget);
        let n = self.size.unwrap_or_else(|| fit.max(self.floor));
        n.min(self.most as usize) as u32
    }

    /// `0 .. nframe` in stacks of at most [`Self::frames`] for `bufs`.
    fn cut(
        self,
        stream: &Arc<CudaStream>,
        dims: &ViewDims,
        bufs: &[Streamed],
    ) -> Result<Cut, EngineError> {
        Ok(batch_stacks(stream, dims, dims.nframe, self.frames(bufs))?.0)
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
///
/// A call that fails with [`EngineError::Invalid`] has changed nothing. One that changes the
/// device state and fails otherwise may leave it part-changed, its LLK records unfiled, and
/// every later call refused with [`EngineError::Aborted`] but the read-backs:
/// [`Self::object`], [`Self::objects`], [`Self::probe`], [`Self::features`],
/// [`Self::history`], [`Self::cycle`], [`Self::sampling`] and [`Self::synchronize`].
pub struct GpuPtycho {
    dev: Gated,
    obj_dims: [usize; 3],
    probe_dims: [usize; 3],
    /// Sum of the unmasked observed counts, for ML's [`crate::engine::Ptycho::scale_obj_probe`].
    total_obs: f64,
    /// The probe-frame FFT of [`zero_phase_ramp`], which runs on the host.
    fft: Fft2,
    cycle: usize,
    history: Vec<LlkRecord>,
}

/// The [`Device`] behind the gate of [`GpuPtycho`]'s calls: aborted from the first failure
/// of a call through [`Self::change`] on. A call's checks, with what it builds before it
/// changes anything, run through the gate too, after its refusal, so an aborted engine
/// answers [`EngineError::Aborted`] whatever the arguments.
struct Gated {
    dev: Device,
    aborted: bool,
}

impl Gated {
    /// `f`, which may leave the device part-changed when it fails, on what `check` built:
    /// aborted while `f` runs, so its failure leaves the gate aborted.
    fn change<T, R>(
        &mut self,
        check: impl FnOnce(&Device) -> Result<T, EngineError>,
        f: impl FnOnce(&mut Device, T) -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        let built = check(self.checked()?)?;
        self.aborted = true;
        let r = f(&mut self.dev, built)?;
        self.aborted = false;
        Ok(r)
    }

    /// `f`, which leaves the device's state as it found it, failed or not, after `check`: it
    /// never aborts.
    fn read<R>(
        &mut self,
        check: impl FnOnce(&Device) -> Result<(), EngineError>,
        f: impl FnOnce(&mut Device) -> Result<R, EngineError>,
    ) -> Result<R, EngineError> {
        check(self.checked()?)?;
        f(&mut self.dev)
    }

    /// The device, refused once a call through [`Self::change`] has failed.
    fn checked(&self) -> Result<&Device, EngineError> {
        if self.aborted {
            return Err(EngineError::Aborted);
        }
        Ok(&self.dev)
    }

    /// The device unchecked, for a read-back, which also runs once aborted.
    fn unchecked(&self) -> &Device {
        &self.dev
    }
}

/// Fails unless an object of `dims` held in `len` values is the shape `have` of the current
/// object.
fn object_fits(have: [usize; 3], dims: [usize; 3], len: usize) -> Result<(), EngineError> {
    if dims != have || len != dims.iter().product::<usize>() {
        return Err(EngineError::Invalid(format!(
            "object {dims:?} of {len} values differs from the reconstruction's {have:?}"
        )));
    }
    Ok(())
}

/// Waits for the LLK records enqueued on `dev` since the last collection (the one host sync
/// of an algorithm call that takes any): the tagged ones, for the history, and the untagged
/// one.
fn collect(dev: &mut Device) -> Result<(Vec<LlkRecord>, Option<Llk>), EngineError> {
    let (mut tagged, mut untagged) = (Vec::new(), None);
    for (tag, t) in dev.flush()? {
        let s = t.sums;
        let llk = Llk {
            poisson: s.poisson,
            gaussian: s.gaussian,
            euclidian: s.euclidean,
            nb_photons_calc: s.icalc,
        }
        .normalised(t.nb_obs);
        match tag {
            Some((cycle, algorithm)) => tagged.push(LlkRecord {
                cycle,
                algorithm,
                llk,
            }),
            None => untagged = Some(llk),
        }
    }
    Ok((tagged, untagged))
}

/// [`zero_phase_ramp`] of `dev`'s object of `obj_dims` and probe of `probe_dims`, on the
/// host by `fft`: one read-back and upload, after an algorithm's last cycle.
fn remove_phase_ramp(
    dev: &mut Device,
    fft: &Fft2,
    obj_dims: [usize; 3],
    probe_dims: [usize; 3],
) -> Result<(), EngineError> {
    let mut object = Object::new(download(dev.stream(), dev.obj(), obj_dims)?);
    let mut probe = Probe::new(download(dev.stream(), dev.probe(), probe_dims)?);
    zero_phase_ramp(fft, &mut object, &mut probe);
    dev.write_arrays(&to_c32(object.as_slice()), &to_c32(probe.as_slice()))
}

impl GpuPtycho {
    /// Uploads the scan's counts and views (as [`crate::engine::Ptycho::new`] builds them) and the
    /// starting arrays to `ctx`, on a new stream, for the far field of the scan's geometry.
    /// Frames must be square (cuFFT plans here are `n x n`).
    pub fn new(
        ctx: &Arc<CudaContext>,
        scan: &Scan,
        object: Object,
        probe: Probe,
        options: GpuOptions,
    ) -> Result<Self, EngineError> {
        let optics = Optics::far_field(&scan.geometry);
        Self::with_optics(ctx, scan, &optics, object, probe, options)
    }

    /// [`Self::new`] under `optics`, which give the object pixel of the scan's positions, the
    /// exit waves' plane phase and the propagator in place of the scan geometry's far field;
    /// of the geometry only the frame is read. Fails on optics without a
    /// [`Optics::fourier_form`], and on more than one object slice, which
    /// [`Self::with_slices`] takes. Angular spectrum with an evanescent band runs LSQML, BH
    /// and the LLK pass alone, as the CPU's: every other algorithm fails on it.
    pub fn with_optics(
        ctx: &Arc<CudaContext>,
        scan: &Scan,
        optics: &Optics,
        object: Object,
        probe: Probe,
        options: GpuOptions,
    ) -> Result<Self, EngineError> {
        Self::with_slices(ctx, scan, optics, &[object], probe, options)
    }

    /// [`Self::with_optics`] on a multislice object, `objects` its slices in the order the
    /// wave crosses them, one per slice of `optics` and all of one shape. More than one
    /// slice runs LSQML ([`Self::lsqml`]) and BH ([`Self::bh`]) alone, BH's position model
    /// only under [`crate::engine::Sampling::FullObject`]: every other algorithm fails on it.
    pub fn with_slices(
        ctx: &Arc<CudaContext>,
        scan: &Scan,
        optics: &Optics,
        objects: &[Object],
        probe: Probe,
        options: GpuOptions,
    ) -> Result<Self, EngineError> {
        if objects.len() != optics.nslice() {
            return Err(EngineError::Invalid(format!(
                "{} object slices for optics of {}",
                objects.len(),
                optics.nslice()
            )));
        }
        let object = &objects[0];
        if let Some(o) = objects.iter().find(|o| o.dims() != object.dims()) {
            return Err(EngineError::Invalid(format!(
                "object slice {:?} differs from the first's {:?}",
                o.dims(),
                object.dims()
            )));
        }
        let g = scan.geometry;
        let [ny, nx] = g.frame;
        if optics.exit().shape() != g.frame {
            return Err(EngineError::Invalid(format!(
                "propagator plane {:?} differs from frame {:?}",
                optics.exit().shape(),
                g.frame
            )));
        }
        let pixel = optics.pixel();
        let form = optics.fourier_form().map_err(|e| {
            EngineError::Invalid(format!(
                "the GPU backend cannot run these optics over object pixels {pixel:?}: {e}"
            ))
        })?;
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
            // A multislice object's waves take the phase on the last slice, by
            // `OpticsKernels::slice_phase`; no pass over the views does.
            quad_phase: if objects.len() == 1 {
                form.quad as f32
            } else {
                0.0
            },
        };
        let slices = (objects.len() > 1).then(|| SliceUploads {
            deeper: objects[1..].iter().map(|o| to_c32(o.as_slice())).collect(),
            gaps: (optics.gaps().iter())
                .map(|p| {
                    to_c32(
                        p.transfer()
                            .expect("Optics::with_slices builds angular spectrum"),
                    )
                })
                .collect(),
            quad: form.quad as f32,
        });
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

        let uploads = Uploads {
            stage: form.stage,
            scale: optics.exit().intensity_scale() as f32,
            addrs: &addrs,
            counts: scan.counts.as_slice(),
            obj: &to_c32(object.as_slice()),
            probe: &to_c32(probe.as_slice()),
            slices,
        };
        let sampling = Sampling::auto(optics, object.plane_shape(), g.frame);
        let dev = Device::new(ctx, dims, uploads, options, sampling)?;
        Ok(Self {
            dev: Gated {
                dev,
                aborted: false,
            },
            obj_dims: object.dims(),
            probe_dims: probe.dims(),
            total_obs: scan.counts.total(),
            fft: Fft2::new(g.frame),
            cycle: 0,
            history: Vec::new(),
        })
    }

    /// The object, of a multislice object its first slice, read back from the device.
    pub fn object(&self) -> Result<Object, EngineError> {
        let dev = self.dev.unchecked();
        Ok(Object::new(download(
            dev.stream(),
            dev.obj(),
            self.obj_dims,
        )?))
    }

    /// The object's slices in the order the wave crosses them, read back from the device.
    pub fn objects(&self) -> Result<Vec<Object>, EngineError> {
        let dev = self.dev.unchecked();
        std::iter::once(dev.obj())
            .chain(dev.deeper())
            .map(|b| Ok(Object::new(download(dev.stream(), b, self.obj_dims)?)))
            .collect()
    }

    /// Replaces the object, of a multislice object its first slice, between algorithm calls,
    /// which start from it as from the one [`Self::new`] uploads; fails, changing nothing,
    /// when its dims differ from the current object's.
    pub fn set_object(&mut self, object: &Object) -> Result<(), EngineError> {
        let (have, len) = (self.obj_dims, object.as_slice().len());
        self.dev.change(
            |_| object_fits(have, object.dims(), len),
            |dev, ()| dev.write_object(&to_c32(object.as_slice())),
        )
    }

    /// [`Self::set_object`] from `src`, an object of `dims` in this reconstruction's context
    /// whose writes from other streams have finished; the copy is waited for, so `src` may be
    /// released or rewritten on return.
    pub fn set_object_device(
        &mut self,
        dims: [usize; 3],
        src: &DeviceBuffer<C32>,
    ) -> Result<(), EngineError> {
        let have = self.obj_dims;
        self.dev.change(
            |_| object_fits(have, dims, src.len()),
            |dev, ()| dev.copy_object(src),
        )
    }

    /// The probe, read back from the device.
    pub fn probe(&self) -> Result<Probe, EngineError> {
        let dev = self.dev.unchecked();
        Ok(Probe::new(download(
            dev.stream(),
            dev.probe(),
            self.probe_dims,
        )?))
    }

    /// LLK records in the order they were taken.
    pub fn history(&self) -> &[LlkRecord] {
        &self.history
    }

    /// The features modelled, read back from the device.
    pub fn features(&self) -> Result<Features, EngineError> {
        self.dev.unchecked().features()
    }

    /// How [`Features::position`] enters the exit waves, as
    /// [`crate::engine::Ptycho::sampling`]: [`Sampling::auto`] of the optics, object and frame
    /// the engine was built with, unless set.
    pub fn sampling(&self) -> Sampling {
        self.dev.unchecked().sampling()
    }

    /// Enters [`Features::position`] as `sampling` in every call from here on, as
    /// [`crate::engine::Ptycho::set_sampling`].
    pub fn set_sampling(&mut self, sampling: Sampling) -> Result<(), EngineError> {
        self.dev.change(
            |dev| dev.build_sampling(sampling),
            |dev, new| {
                dev.set_sampling(new);
                Ok(())
            },
        )
    }

    /// Models and refines `features` from here on, as [`crate::engine::Ptycho::set_features`], in
    /// `f32` on the device; with [`GpuOptions::fused`] only.
    pub fn set_features(&mut self, features: &Features) -> Result<(), EngineError> {
        self.dev.change(
            |dev| {
                let dims = dev.dims();
                let [ny, nx] = dims.frame.map(|n| n as usize);
                features.check(dims.nframe as usize, [ny, nx])?;
                dev.build_features(features)
            },
            |dev, new| dev.set_features(new),
        )
    }

    /// Cycles (AP, DM, ML) and epochs (rPIE) run so far.
    pub fn cycle(&self) -> usize {
        self.cycle
    }

    /// Waits for every enqueued step to finish (for timing).
    pub fn synchronize(&self) -> Result<(), EngineError> {
        Ok(self.dev.unchecked().stream().synchronize()?)
    }

    /// Normalised LLK of the current object and probe, as [`crate::engine::Ptycho::llk`]; the
    /// exit-wave stacks are left as they are.
    pub fn llk(&mut self) -> Result<Llk, EngineError> {
        let (records, llk) = self.dev.read(
            |_| Ok(()),
            |dev| {
                dev.llk_pass(None)?;
                collect(dev)
            },
        )?;
        self.history.extend(records);
        Ok(llk.expect("the LLK pass counts one untagged record"))
    }

    /// An algorithm call through [`Gated::change`] after `check`: `f` enqueues its cycles,
    /// counting each in the cycle number it is lent, then the LLK records are collected and,
    /// with `ramp`, the phase ramp removed. The records join the history once all of it has
    /// succeeded.
    fn call(
        &mut self,
        ramp: bool,
        check: impl FnOnce(&Device) -> Result<(), EngineError>,
        f: impl FnOnce(&mut Device, &mut usize) -> Result<(), EngineError>,
    ) -> Result<(), EngineError> {
        let Self {
            dev,
            obj_dims,
            probe_dims,
            fft,
            cycle,
            history,
            ..
        } = self;
        let records = dev.change(check, |dev, ()| {
            f(dev, cycle)?;
            let (records, _) = collect(dev)?;
            if ramp {
                remove_phase_ramp(dev, fft, *obj_dims, *probe_dims)?;
            }
            Ok(records)
        })?;
        history.extend(records);
        Ok(())
    }
}

/// The per-stack steps of every algorithm, on the lent device state.
impl Bufs<'_> {
    /// The LLK record of the current estimate through `rec`: M1's by the scratch stack.
    fn forward_llk(
        &mut self,
        ctx: &ExecutionContext,
        passes: &mut PassesMut<'_>,
        mut rec: RecordWriter<'_, Tag>,
    ) -> Result<(), GpuError> {
        let dims = *self.views.dims();
        match passes {
            PassesMut::M1(m1) => {
                let (k, optics) = (self.kernels, self.exit.kernels);
                for s in 0..self.scratch.len() {
                    let stack = self.scratch.at(s);
                    let psi = match &mut self.slices {
                        None => {
                            k.obj_probe_to_psi(
                                ctx, self.views, stack, self.obj, self.probe, m1.scratch,
                            )?;
                            &mut *m1.scratch
                        }
                        Some(sl) => {
                            let inputs = FusedInputs {
                                views: self.views,
                                counts: self.counts,
                                obj: self.obj,
                                probe: self.probe,
                                model: Model::default(),
                            };
                            sl.exit_waves(ctx, k, optics, &inputs, self.deeper, stack)?
                        }
                    };
                    let plan = self.scratch.frames[s].plan;
                    m1.plans.get(plan).forward_inplace(ctx, psi)?;
                    k.llk(ctx, stack, self.counts, psi, self.llk_scratch, self.llk)?;
                    rec.push(ctx, &dims, stack, self.llk)?;
                }
            }
            PassesMut::Fused(f) => {
                let whole = Sliced {
                    first: self.obj,
                    deeper: self.deeper,
                };
                let mut m = (self.features).lend(
                    ctx,
                    self.kernels,
                    &dims,
                    self.obj,
                    self.probe,
                    self.exit,
                )?;
                let windows = self.windows.as_deref_mut();
                let mut read = m.sliced(ctx, (self.views, whole), windows)?;
                for s in 0..self.scratch.len() {
                    let stack = self.scratch.at(s);
                    let (views, obj) = m.slices(ctx, &mut read, stack)?;
                    let inputs = FusedInputs {
                        views,
                        counts: self.counts,
                        obj: obj.first,
                        probe: self.probe,
                        model: m.model,
                    };
                    let far = FarSums {
                        llk: Some(&mut *self.llk),
                        ..FarSums::default()
                    };
                    match &mut self.slices {
                        None => self.kernels.fused_llk(ctx, &inputs, stack, f, far)?,
                        Some(sl) => {
                            let (k, exit) = (self.kernels, self.exit);
                            let rows =
                                sl.exit_waves(ctx, k, exit.kernels, &inputs, obj.deeper, stack)?;
                            exit.kernels.rows(ctx, &dims, stack, rows, false)?;
                            (exit.kernels).exit(ctx, &dims, stack, exit.stage, rows, false)?;
                            k.rows_llk(ctx, &inputs, stack, rows, f, far)?;
                        }
                    }
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
        mut llk: Option<RecordWriter<'_, Tag>>,
        one: OneSlice,
    ) -> Result<(), GpuError> {
        let dims = *self.views.dims();
        let mut m = (self.features)
            .lend(ctx, self.kernels, &dims, self.obj, self.probe, self.exit)?
            .one_slice(one);
        for s in 0..self.scratch.len() {
            let stack = self.scratch.at(s);
            let windows = self.windows.as_deref_mut();
            let (views, obj) = m.object(ctx, (self.views, self.obj), stack, windows)?;
            let inputs = FusedInputs {
                views,
                counts: self.counts,
                obj,
                probe: self.probe,
                model: m.model,
            };
            let far = m.far(llk.is_some().then_some(&mut *self.llk));
            self.kernels.fused_llk(ctx, &inputs, stack, f, far)?;
            if let Some(rec) = &mut llk {
                rec.push(ctx, &dims, stack, self.llk)?;
            }
        }
        if let Some(w) = llk {
            w.end(ctx)?;
        }
        fault::at(fault::Exit::Feature)?;
        self.features.step(ctx, self.kernels, self.grad)?;
        Ok(())
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
        llk: Option<&mut RecordWriter<'_, Tag>>,
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
        psi: &dyn Waves,
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
    /// that have an inertia; the object only on the tiles some view reaches, or under
    /// [`crate::engine::Sampling::FullObject`] everywhere by the normal equations of the
    /// current probe's windows ([`device::FeaturesMut::solve_obj`]).
    fn apply_sums(
        &mut self,
        ctx: &ExecutionContext,
        obj: Option<f64>,
        probe: Option<f64>,
    ) -> Result<(), GpuError> {
        if let Some(inertia) = obj {
            let (sums, inertia) = (&mut *self.obj_sums, inertia as f32);
            if !(self.features).solve_obj(ctx, self.views, self.probe, self.obj, sums, inertia)? {
                (self.kernels).update_obj_from_sums(ctx, self.views, self.obj, sums, inertia)?;
            }
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

    /// 24 MiB of a 64 MiB L2 holds 24, 12, 8 and 6 frames of one to four two-mode exit-wave
    /// sets live across one pass boundary at 256x256, 6, 3, 2 and 1 at 512x512, and 16 of one
    /// and a half (a set stored at half width). A set one pass touches does not count, nor does
    /// one handed on at a pass, while sets whose passes overlap add up. A frame alone that
    /// overfills it is one; on 153600 threads four sets take the floor, 32, 8 and 2 frames at
    /// 128x128, 256x256 and 512x512, and one set at 128x128 its fit, 96. A given size wins.
    #[test]
    fn stack_rule_fills_three_eighths_of_l2() {
        const L2: usize = 64 << 20;
        type Bufs<'a> = &'a [(usize, std::ops::RangeInclusive<u32>)];
        let frames = |rule: StackRule, bufs: Bufs| {
            let bufs: Vec<_> = bufs
                .iter()
                .map(|(b, p)| Streamed::new(*b, p.clone()))
                .collect();
            rule.frames(&bufs)
        };
        for (side, want) in [(256, [24, 12, 8, 6]), (512, [6, 3, 2, 1])] {
            let (rule, wave) = (
                StackRule::new(None, L2, 1, &dims(side)).unwrap(),
                wave_bytes(&dims(side)),
            );
            let k_sets = |k| frames(rule, &vec![(wave, 0..=1); k]);
            assert_eq!([1, 2, 3, 4].map(k_sets), want);
        }
        let (rule, wave) = (
            StackRule::new(None, L2, 1, &dims(256)).unwrap(),
            wave_bytes(&dims(256)),
        );
        assert_eq!(frames(rule, &[(wave, 0..=2), (wave, 1..=1)]), 24);
        assert_eq!(frames(rule, &[(wave, 0..=1), (wave, 1..=2)]), 24);
        assert_eq!(frames(rule, &[(wave, 0..=2), (wave, 1..=3)]), 12);
        assert_eq!(frames(rule, &[(wave, 0..=1), (wave / 2, 0..=1)]), 16);
        let big = [(wave_bytes(&dims(512)), 0..=1)];
        assert_eq!(
            frames(StackRule::new(None, 4 << 20, 1, &dims(512)).unwrap(), &big),
            1
        );
        let floored = [128, 256, 512].map(|side| {
            let rule = StackRule::new(None, L2, 153600, &dims(side)).unwrap();
            frames(rule, &vec![(wave_bytes(&dims(side)), 0..=1); 4])
        });
        assert_eq!(floored, [32, 8, 2]);
        let rule = StackRule::new(None, L2, 153600, &dims(128)).unwrap();
        assert_eq!(frames(rule, &[(wave_bytes(&dims(128)), 0..=1)]), 96);
        let given = StackRule::new(Some(16), L2, 153600, &dims(512)).unwrap();
        assert_eq!(frames(given, &big), 16);
    }

    /// At 256x256 a two-mode set is 1 MiB a frame against a 24 MiB budget: one set live after
    /// pass 0 and two after pass 2 fit 12 frames. 8 MiB fixed live after both passes binds
    /// after 2, to 8 frames, as it does live there alone, and after 0 alone leaves 12; alone
    /// after a pass it binds nothing.
    #[test]
    fn a_fixed_entry_binds_where_frames_are_live_beside_it() {
        const MIB: usize = 1 << 20;
        let rule = StackRule::new(None, 64 * MIB, 1, &dims(256)).unwrap();
        let wave = wave_bytes(&dims(256));
        assert_eq!(wave, MIB);
        let sets = [0..=1, 2..=3, 2..=3].map(|p| Streamed::new(wave, p));
        let with = |fixed: Streamed| rule.frames(&[sets.as_slice(), &[fixed]].concat());
        assert_eq!(rule.frames(&sets), 12);
        assert_eq!(with(Streamed::fixed(8 * MIB, 0..=3)), 8);
        assert_eq!(with(Streamed::fixed(8 * MIB, 2..=3)), 8);
        assert_eq!(with(Streamed::fixed(8 * MIB, 0..=1)), 12);
        assert_eq!(with(Streamed::fixed(64 * MIB, 1..=2)), 12);
    }

    /// At 256x256 a two-mode set is 1 MiB a frame against a 24 MiB budget: a lone chunked
    /// call holding 23 MiB leaves it 1 frame, whether it writes the set for a later pass, reads
    /// it from an earlier one, or writes it for a pass within its own; holding 8 MiB, 16.
    #[test]
    fn a_chunked_call_holds_its_bytes_beside_the_planes_it_reads_and_writes() {
        const MIB: usize = 1 << 20;
        let rule = StackRule::new(None, 64 * MIB, 1, &dims(256)).unwrap();
        let wave = wave_bytes(&dims(256));
        let frames = |call: Vec<Streamed>| rule.frames(&call);
        assert_eq!(
            frames(Streamed::chunked(23 * MIB, 0..=2, &[(wave, 3)], &[])),
            1
        );
        assert_eq!(
            frames(Streamed::chunked(23 * MIB, 1..=3, &[], &[(wave, 0)])),
            1
        );
        assert_eq!(
            frames(Streamed::chunked(23 * MIB, 0..=2, &[(wave, 1)], &[])),
            1
        );
        assert_eq!(
            frames(Streamed::chunked(8 * MIB, 0..=2, &[(wave, 3)], &[])),
            16
        );
    }

    /// Fixed bytes past the budget beside a set leave no frame to fit, so the floor stands: 8
    /// frames at 256x256 on 153600 threads, 1 on one thread; a given size still wins.
    #[test]
    fn the_floor_holds_under_a_fixed_entry_past_the_budget() {
        const MIB: usize = 1 << 20;
        let bufs = [
            Streamed::new(wave_bytes(&dims(256)), 0..=1),
            Streamed::fixed(32 * MIB, 0..=1),
        ];
        let frames = |size, threads| {
            let rule = StackRule::new(size, 64 * MIB, threads, &dims(256)).unwrap();
            rule.frames(&bufs)
        };
        assert_eq!(frames(None, 153600), 8);
        assert_eq!(frames(None, 1), 1);
        assert_eq!(frames(Some(16), 153600), 16);
    }

    /// At 256x256 and one mode a stack of 65535 frames' exit waves fits u32, one of 65536
    /// does not: a size of 65536 is refused for a scan of 65537 frames, which the passes cut
    /// in two but a batch of 65536 would not, and taken for one of 65535, as any size past
    /// the scan's frames is. Unsized, a fit past it stops at the most that do.
    #[test]
    fn a_stack_size_past_u32_is_refused_by_the_rule() {
        let rule = |size, nframe| {
            let dims = ViewDims {
                nframe,
                nmode: 1,
                ..dims(256)
            };
            StackRule::new(size, 64 << 20, 1, &dims)
        };
        let r = rule(Some(65536), 65537);
        assert!(matches!(r, Err(EngineError::Invalid(_))), "{r:?}");
        assert_eq!(rule(Some(65535), 65537).unwrap().frames(&[]), 65535);
        assert_eq!(rule(Some(1 << 40), 65535).unwrap().frames(&[]), 65535);
        let r = rule(Some(1 << 40), 65536);
        assert!(matches!(r, Err(EngineError::Invalid(_))), "{r:?}");
        let byte = [Streamed::new(1, 0..=1)];
        assert_eq!(rule(None, 70000).unwrap().frames(&byte), 65535);
    }

    /// An engine on a simulated raster of 36 points, its object flat: 32-pixel frames, or the
    /// fused passes' smallest, 64.
    pub(super) fn engine(ctx: &Arc<CudaContext>, graph: bool, fused: bool) -> GpuPtycho {
        use crate::base::{Counts, Geometry, Positions, required_obj_shape};
        use crate::live::farfield::{Ptycho, PtychoSpec, ScanPath};

        let n = if fused { 64 } else { 32 };
        let p = Ptycho::new(PtychoSpec {
            n,
            path: ScanPath::Raster { cols: 6 },
            points: 36,
            step: 6.0,
            photons: 1e6,
            seed: 5,
            wavelength: 1.5498e-10,
            distance: 5.0,
            det_pixel: 75e-6,
        });
        let s = p.spec();
        let g = Geometry::new(s.wavelength, s.distance, s.det_pixel, [n, n]).unwrap();
        let (x, y): (Vec<f64>, Vec<f64>) = (0..p.points()).map(|k| p.position(k)).unzip();
        let counts = (0..p.points()).flat_map(|k| p.frame(k)).collect();
        let scan = Scan::new(
            g,
            Positions::new(y, x).unwrap(),
            Counts::from_raw([p.points(), n, n], counts).unwrap(),
        )
        .unwrap();
        let pixel = g.pixel_size_object();
        let obj = required_obj_shape(&scan.positions.pynx_preprocess(pixel), pixel, [n, n], 0);
        let object = Object::new(Stack::filled([1, obj[0], obj[1]], C64::new(1.0, 0.0)));
        let probe = p.probe().iter().map(|c| C64::new(c.re.into(), c.im.into()));
        let probe = Probe::new(Stack::from_vec([2, n, n], probe.collect()).unwrap());
        let options = GpuOptions {
            stack_size: Some(8),
            graph,
            llk_slots: 4,
            fused,
            psi_f16: false,
        };
        GpuPtycho::new(ctx, &scan, object, probe, options).unwrap()
    }

    /// The gate: a call refused by its checks changes nothing and leaves the engine open, one
    /// that fails past them leaves it refusing every later call that enqueues work but the
    /// read-backs, and a read that fails leaves it open.
    mod gate {
        use cuda_async::error::DeviceError;

        use super::super::*;
        use super::engine;
        use crate::engine::{Cost, Feature};
        use crate::fault::Exit;

        fn ctx() -> Arc<CudaContext> {
            CudaContext::new(0).expect("GPU 0")
        }

        fn ap(zero_phase_ramp: bool) -> ApParams {
            ApParams {
                obj_inertia: Some(0.01),
                probe_inertia: Some(0.01),
                calc_llk: 1,
                zero_phase_ramp,
            }
        }

        /// The object's and probe's bits.
        fn arrays(e: &GpuPtycho) -> Vec<u64> {
            let (o, p) = (e.object().unwrap(), e.probe().unwrap());
            let px = o.as_slice().iter().chain(p.as_slice());
            px.flat_map(|c| [c.re.to_bits(), c.im.to_bits()]).collect()
        }

        /// Whether `r` failed at the armed exit; a capture returns its closure's error as the
        /// device operation's.
        fn refused<T>(r: &Result<T, EngineError>) -> bool {
            matches!(r, Err(EngineError::Gpu(
                GpuError::Driver(d) | GpuError::Device(DeviceError::Driver(d)),
            )) if *d == fault::REFUSAL)
        }

        /// Every call that enqueues work is refused, each read-back runs.
        fn aborted(e: &mut GpuPtycho, what: &str) {
            let dims = e.obj_dims;
            let object = e.object().unwrap();
            let stream = Arc::clone(e.dev.unchecked().stream());
            let mut buf = DeviceBuffer::<C32>::zeroed(&stream, dims.iter().product()).unwrap();
            let bh = BhParams {
                batch_size: 36,
                probe_rho: 1.0,
                position_rho: 0.1,
                seed: 0,
            };
            let lsqml = LsqmlParams {
                batch_size: 36,
                cost: Cost::Poisson,
                seed: 0,
            };
            let calls = [
                ("llk", e.llk().map(drop)),
                ("ap", e.ap(&ap(false), 1)),
                ("dm", e.dm(&DmParams::default(), 1)),
                ("raar", e.raar(&RaarParams::default(), 1)),
                ("rpie", e.rpie(&RpieParams::default(), 1)),
                ("ml", e.ml(&MlParams::default(), 1)),
                ("bh", e.bh(&bh, 1, 1)),
                ("lsqml", e.lsqml(&lsqml, 1, 1)),
                ("ml_grads", e.ml_grads().map(drop)),
                ("ml_obj_grad_into", e.ml_obj_grad_into(&mut buf)),
                ("bh_steps", e.bh_steps().map(drop)),
                ("lsqml_steps", e.lsqml_steps().map(drop)),
                ("set_object", e.set_object(&object)),
                ("set_object_device", e.set_object_device(dims, &buf)),
                ("set_sampling", e.set_sampling(Sampling::Patch)),
                ("set_features", e.set_features(&Features::default())),
            ];
            for (call, r) in calls {
                assert!(
                    matches!(r, Err(EngineError::Aborted)),
                    "{what}: {call} gave {r:?}"
                );
            }
            e.objects().unwrap();
            e.probe().unwrap();
            e.features().unwrap();
            let _ = (e.history(), e.cycle(), e.sampling());
            e.synchronize().unwrap();
        }

        /// Calls refused by their checks, the features' on M1's passes, rPIE's batch and an
        /// object of other dims, leave LLK and AP bit for bit those of a fresh engine.
        #[test]
        fn a_call_refused_by_its_checks_changes_nothing() {
            let ctx = ctx();
            let [mut e, mut fresh] = [(); 2].map(|()| engine(&ctx, true, false));
            let scale = Features {
                scale: Some(Feature::refined(vec![1.0; 36])),
                ..Features::default()
            };
            let rpie = RpieParams {
                batch_size: 1 << 32,
                ..RpieParams::default()
            };
            let other = Object::new(Stack::filled([1, 2, 2], C64::new(1.0, 0.0)));
            let refusals = [
                ("set_features", e.set_features(&scale)),
                ("rpie", e.rpie(&rpie, 1)),
                ("set_object", e.set_object(&other)),
            ];
            for (call, r) in refusals {
                assert!(matches!(r, Err(EngineError::Invalid(_))), "{call}: {r:?}");
            }
            let llk = |e: &mut GpuPtycho| {
                let l = e.llk().unwrap();
                [l.poisson, l.gaussian, l.euclidian, l.nb_photons_calc].map(f64::to_bits)
            };
            assert_eq!(llk(&mut e), llk(&mut fresh));
            for e in [&mut e, &mut fresh] {
                e.ap(&ap(false), 1).unwrap();
            }
            assert!(arrays(&e) == arrays(&fresh), "the arrays");
            assert_eq!(e.history().len(), 1);
        }

        /// A call failing before it enqueues anything still aborts, its arrays as they were.
        #[test]
        fn a_call_failing_before_its_first_launch_aborts() {
            let ctx = ctx();
            let mut e = engine(&ctx, true, false);
            let start = arrays(&e);
            let armed = fault::arm(Exit::RunBuild, 0);
            let r = e.ap(&ap(false), 1);
            assert!(armed.fired());
            assert!(refused(&r), "{r:?}");
            drop(armed);
            aborted(&mut e, "RunBuild");
            assert!(arrays(&e) == start, "the arrays");
        }

        /// An AP call failing at its third cycle's replay keeps the arrays its first two left,
        /// and files none of their records.
        #[test]
        fn a_call_failing_between_its_cycles_files_none_of_their_records() {
            let ctx = ctx();
            let [mut e, mut b] = [(); 2].map(|()| engine(&ctx, true, false));
            e.ap(&ap(false), 1).unwrap();
            let armed = fault::arm(Exit::Launch, 2);
            let r = e.ap(&ap(false), 4);
            assert!(armed.fired());
            assert!(refused(&r), "{r:?}");
            drop(armed);
            assert_eq!(e.history().len(), 1);
            b.ap(&ap(false), 1).unwrap();
            b.ap(&ap(false), 2).unwrap();
            assert!(arrays(&e) == arrays(&b), "the arrays");
            aborted(&mut e, "Launch");
        }

        /// A launched AP cycle failing at its position step, after its updates, aborts.
        #[test]
        fn a_launched_call_failing_after_its_updates_aborts() {
            let ctx = ctx();
            let mut e = engine(&ctx, false, true);
            let offsets = (0..36).map(|f| [0.1 * (f % 3) as f64 - 0.1, 0.05]);
            let position = Features {
                position: Some(Feature::refined(offsets.collect())),
                ..Features::default()
            };
            e.set_features(&position).unwrap();
            let armed = fault::arm(Exit::ApPositions, 0);
            let r = e.ap(&ap(false), 2);
            assert!(armed.fired());
            assert!(refused(&r), "{r:?}");
            drop(armed);
            aborted(&mut e, "ApPositions");
        }

        /// An AP call failing in its phase ramp's removal, after its records are collected,
        /// files none of them.
        #[test]
        fn a_call_failing_after_its_records_are_collected_files_none() {
            let ctx = ctx();
            let mut e = engine(&ctx, true, false);
            let armed = fault::arm(Exit::RampWrite, 0);
            let r = e.ap(&ap(true), 1);
            assert!(armed.fired());
            assert!(refused(&r), "{r:?}");
            drop(armed);
            assert!(e.history().is_empty(), "{:?}", e.history());
            aborted(&mut e, "RampWrite");
        }

        /// New feature values failing part-way through their write in place abort, the
        /// values written before it read back.
        #[test]
        fn a_feature_write_failing_part_way_aborts() {
            let ctx = ctx();
            let mut e = engine(&ctx, true, true);
            let scale = |v| Features {
                scale: Some(Feature::refined(vec![v; 36])),
                ..Features::default()
            };
            e.set_features(&scale(1.0)).unwrap();
            let armed = fault::arm(Exit::FeatureCopy, 0);
            let r = e.set_features(&scale(2.0));
            assert!(armed.fired());
            assert!(refused(&r), "{r:?}");
            drop(armed);
            aborted(&mut e, "FeatureCopy");
            let written = e.features().unwrap().scale.map(|s| s.values);
            assert_eq!(written, Some(vec![2.0; 36]));
        }

        /// A position, then another sampling, whose shared passes fail to build leave the
        /// engine open on what it modelled before: LLK and AP after each are bit for bit those
        /// of an engine never asked, and so after both are entered.
        #[test]
        fn a_build_failing_leaves_the_engine_open() {
            let ctx = ctx();
            let [mut e, mut fresh] = [(); 2].map(|()| engine(&ctx, true, true));
            let offsets = (0..36).map(|f| [0.1 * (f % 3) as f64 - 0.1, 0.05]);
            let position = Features {
                position: Some(Feature::refined(offsets.collect())),
                ..Features::default()
            };
            let other = match e.sampling() {
                Sampling::Patch => Sampling::FullObject,
                Sampling::FullObject => Sampling::Patch,
            };
            let llk = |e: &mut GpuPtycho| {
                let l = e.llk().unwrap();
                [l.poisson, l.gaussian, l.euclidian, l.nb_photons_calc].map(f64::to_bits)
            };
            let same = |e: &mut GpuPtycho, fresh: &mut GpuPtycho, what: &str| {
                assert_eq!(llk(e), llk(fresh), "{what}: the LLK");
                for e in [&mut *e, &mut *fresh] {
                    e.ap(&ap(false), 1).unwrap();
                }
                assert!(arrays(e) == arrays(fresh), "{what}: the arrays");
                let h = |e: &GpuPtycho| {
                    let h = e.history().iter();
                    h.map(|r| r.llk.poisson.to_bits()).collect::<Vec<_>>()
                };
                assert_eq!(h(e), h(fresh), "{what}: the records");
            };
            for e in [&mut e, &mut fresh] {
                e.ap(&ap(false), 1).unwrap();
            }
            let armed = fault::arm(Exit::Reshare, 0);
            let r = e.set_features(&position);
            assert!(armed.fired());
            assert!(refused(&r), "{r:?}");
            drop(armed);
            assert!(e.features().unwrap() == Features::default(), "the features");
            same(&mut e, &mut fresh, "set_features");
            for e in [&mut e, &mut fresh] {
                e.set_features(&position).unwrap();
            }
            let armed = fault::arm(Exit::Reshare, 0);
            let r = e.set_sampling(other);
            assert!(armed.fired());
            assert!(refused(&r), "{r:?}");
            drop(armed);
            assert!(e.sampling() != other, "the sampling");
            same(&mut e, &mut fresh, "set_sampling");
            for e in [&mut e, &mut fresh] {
                e.set_sampling(other).unwrap();
            }
            same(&mut e, &mut fresh, "both");
        }

        /// A call failing past its checks aborts wherever it fails, launched or captured: AP
        /// after a cycle's record and before its sums are applied, the features' step after its
        /// LLK pass, ML after its gradient pass, DM between its projection's stacks and the
        /// full-object solve between its steps.
        #[test]
        fn a_call_failing_part_way_aborts() {
            let ctx = ctx();
            let offsets = (0..36).map(|f| [0.1 * (f % 3) as f64 - 0.1, 0.05]);
            let position = Features {
                position: Some(Feature::refined(offsets.collect())),
                ..Features::default()
            };
            let background = Features {
                background: Some(Feature::refined(vec![1.0; 64 * 64])),
                ..Features::default()
            };
            let exits = [
                (Exit::ApSums, 0),
                (Exit::Feature, 0),
                (Exit::MlGrad, 0),
                (Exit::DmStack, 1),
                (Exit::Solve, 3),
            ];
            for graph in [false, true] {
                for (exit, pass) in exits {
                    let what = format!("{exit:?}, graph {graph}");
                    let fused = matches!(exit, Exit::Feature | Exit::Solve);
                    let mut e = engine(&ctx, graph, fused);
                    match exit {
                        Exit::Feature => e.set_features(&background).unwrap(),
                        Exit::Solve => {
                            e.set_sampling(Sampling::FullObject).unwrap();
                            e.set_features(&position).unwrap();
                        }
                        _ => {}
                    }
                    let armed = fault::arm(exit, pass);
                    let r = match exit {
                        Exit::MlGrad => e.ml(&MlParams::default(), 2),
                        Exit::DmStack => e.dm(&DmParams::default(), 2),
                        _ => e.ap(&ap(false), 2),
                    };
                    assert!(armed.fired(), "{what}");
                    assert!(refused(&r), "{what}: {r:?}");
                    drop(armed);
                    aborted(&mut e, &what);
                }
            }
        }

        /// ML's gradients failing between their pass's stacks leave the engine open, and the
        /// next are bit for bit a fresh engine's: by M1's passes, and by the fused ones under a
        /// held background and a held position, whose shifted probe's sums the failed pass's
        /// first stack filled.
        #[test]
        fn a_read_failing_leaves_the_engine_open() {
            let ctx = ctx();
            let offsets = (0..36).map(|f| [0.1 * (f % 3) as f64 - 0.1, 0.05]);
            let held = Features {
                background: Some(Feature::held(vec![1.0; 64 * 64])),
                position: Some(Feature::held(offsets.collect())),
                ..Features::default()
            };
            let grads = |e: &mut GpuPtycho| {
                let g = e.ml_grads().unwrap();
                let px = g.iter().flat_map(|s| s.as_slice());
                px.flat_map(|c| [c.re.to_bits(), c.im.to_bits()])
                    .collect::<Vec<_>>()
            };
            for fused in [false, true] {
                let [mut e, mut fresh] = [(); 2].map(|()| {
                    let mut e = engine(&ctx, true, fused);
                    if fused {
                        e.set_features(&held).unwrap();
                    }
                    e
                });
                let armed = fault::arm(Exit::MlStack, 1);
                let r = e.ml_grads();
                assert!(armed.fired(), "fused {fused}");
                assert!(refused(&r), "fused {fused}: {r:?}");
                drop(armed);
                assert!(
                    grads(&mut e) == grads(&mut fresh),
                    "fused {fused}: the gradients"
                );
            }
        }
    }
}
