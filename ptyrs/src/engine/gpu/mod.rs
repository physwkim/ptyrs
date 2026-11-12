//! The GPU backend (DESIGN §7 M1): [`GpuPtycho`] runs AP, DM, rPIE, ML and BH in `f32` on one CUDA
//! stream with the `ptyrs::gpu` pixel kernels and `ptyrs::fft` plans, in the call sequences the
//! `crate::gpu` ops and rPIE docs give, and is checked against [`crate::engine::Ptycho`].
//!
//! The scan's frames are cut into stacks of [`GpuOptions::stack_size`] consecutive frames,
//! one [`StackTable`] entry each: every launch reads its stack's frames from the entry on the
//! device. Each stack keeps its own exit waves (DM carries them across cycles), sized like
//! PyNX's fixed stack for the longest stack. AP and DM project a stack, ML takes a stack's
//! gradient, and the LLK pass takes its partials, either by M2's fused passes over one
//! scratch set or by the M1 launch sequence ([`GpuOptions::fused`]), whose one scratch stack
//! serves all and whose stacks of one length share a cuFFT plan, which transforms only their
//! frames; rPIE, BH and ML's line search always run M1's launches. Each cycle enqueues through its own `ExecutionContext`, dropped when the cycle
//! has been enqueued. Nothing in a cycle
//! reads back to the host: an LLK record goes, stack by stack, into the device's
//! [`crate::gpu::LlkRing`], whose copies download while later cycles run, and the host reads
//! the records at the end of each algorithm call (or of [`GpuPtycho::llk`]), the arrays only
//! when asked for. Every cycle, AP's with an LLK and DM's first included, and the LLK pass
//! replay a captured CUDA graph (DESIGN B2, [`GpuOptions::graph`]): one per [`Epoch`], kept
//! with the buffers it captured until another algorithm or other parameters need new ones,
//! and the LLK pass's for the life of the device state. Each algorithm is its own module over
//! the per-stack steps here.

mod ap;
mod bh;
mod device;
mod dm;
mod ml;
mod rpie;

use std::f64::consts::PI;
use std::ops::Range;
use std::sync::Arc;

use crate::base::{AddrTable, C64, Object, Probe, Scan, Stack};
use crate::fft::Plan2d;
use crate::gpu::{
    C32, CG_BLOCKS, Ended, Error as GpuError, FarSums, FusedBufs, FusedInputs, ObjProbeScale,
    RecordWriter, StackTable, ViewAddr, ViewDims,
};
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};

use device::{Bufs, Device, M1Bufs, NormBufs, PassesMut};

use crate::engine::ptycho::{Algorithm, LlkRecord, zero_phase_ramp};
use crate::engine::{
    ApParams, BhParams, DmParams, EngineError, Features, Fft2, Llk, MlParams, Observed, RpieParams,
};

/// How [`GpuPtycho`] batches its work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuOptions {
    /// Frames per stack (the last stack may be shorter); PyNX's default is 16.
    pub stack_size: usize,
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
            stack_size: 16,
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
}

impl Epoch {
    fn algorithm(self) -> Algorithm {
        match self {
            Self::Ap { .. } => Algorithm::Ap,
            Self::Dm { .. } => Algorithm::Dm,
            Self::Rpie(_) => Algorithm::Rpie,
            Self::Ml { .. } => Algorithm::Ml,
            Self::Bh { .. } => Algorithm::Bh,
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
            Self::Rpie(p) => Self::Rpie(p),
            Self::Ml { params, .. } => Self::Ml {
                params,
                first: false,
                llk: false,
            },
            Self::Bh { params, .. } => Self::Bh { params, cg: false },
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

/// Stacks, the table of their entries and their distinct lengths, in the order
/// [`Frames::plan`] indexes them.
type Stacked = (Vec<Frames>, StackTable, Vec<u32>);

/// `0 .. nframe` cut into consecutive ranges of `size` frames, the last one shorter when
/// `size` does not divide `nframe`.
fn stacks(stream: &Arc<CudaStream>, dims: &ViewDims, size: u32) -> Result<Stacked, EngineError> {
    let (ranges, lens) = split(dims.nframe, size);
    let (frames, table) = table_of(stream, dims, ranges)?;
    Ok((frames, table, lens))
}

/// The epoch positions `0 .. nframe` in batches of `size`, each cut into stacks of
/// `stack_size` (the last batch, and a batch's last stack, may be shorter), and each batch's
/// range of stack indices.
fn batch_stacks(
    stream: &Arc<CudaStream>,
    dims: &ViewDims,
    size: u32,
    stack_size: u32,
) -> Result<(Stacked, Vec<Range<usize>>), EngineError> {
    let (mut ranges, mut lens, mut batches) = (Vec::new(), Vec::<u32>::new(), Vec::new());
    for (b, _) in split(dims.nframe, size).0 {
        let first = ranges.len();
        let (sub, sub_lens) = split(b.end - b.start, stack_size);
        for (r, i) in sub {
            let plan = match lens.iter().position(|&l| l == sub_lens[i]) {
                Some(plan) => plan,
                None => {
                    lens.push(sub_lens[i]);
                    lens.len() - 1
                }
            };
            ranges.push((b.start + r.start..b.start + r.end, plan));
        }
        batches.push(first..ranges.len());
    }
    let (frames, table) = table_of(stream, dims, ranges)?;
    Ok(((frames, table, lens), batches))
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

/// One cuFFT plan per stack length of [`stacks`].
fn plans(
    stream: &Arc<CudaStream>,
    dims: &ViewDims,
    lens: &[u32],
) -> Result<Vec<Plan2d>, EngineError> {
    let nmode = dims.nmode as usize;
    Ok(lens
        .iter()
        .map(|&nf| Plan2d::new(stream, dims.frame[0] as usize, nf as usize * nmode))
        .collect::<Result<Vec<_>, _>>()?)
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
            total_obs: Observed::from_counts(&scan.counts).total(),
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
        for s in 0..self.stacks.len() {
            let stack = self.table.at(s);
            match passes {
                PassesMut::M1(m1) => {
                    self.kernels.obj_probe_to_psi(
                        ctx, self.views, stack, self.obj, self.probe, m1.scratch,
                    )?;
                    let plan = self.stacks.get(s).frames.plan;
                    m1.plans.get(plan).forward_inplace(ctx, m1.scratch)?;
                    self.kernels.llk(
                        ctx,
                        stack,
                        self.counts,
                        m1.scratch,
                        self.llk_scratch,
                        self.llk,
                    )?;
                }
                PassesMut::Fused(f) => {
                    let inputs = FusedInputs {
                        views: self.views,
                        counts: self.counts,
                        obj: self.obj,
                        probe: self.probe,
                        model: self.features.model(),
                    };
                    let far = FarSums {
                        llk: Some(&mut *self.llk),
                        ..FarSums::default()
                    };
                    self.kernels.fused_llk(ctx, &inputs, stack, f, far)?;
                }
            }
            rec.push(ctx, &dims, stack, self.llk)?;
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
        for s in 0..self.stacks.len() {
            let stack = self.table.at(s);
            let (model, far) = self.features.far(llk.is_some().then_some(&mut *self.llk));
            let inputs = FusedInputs {
                views: self.views,
                counts: self.counts,
                obj: self.obj,
                probe: self.probe,
                model,
            };
            self.kernels.fused_llk(ctx, &inputs, stack, f, far)?;
            if let Some(rec) = &mut llk {
                rec.push(ctx, &dims, stack, self.llk)?;
            }
        }
        let ended = llk.map(|w| w.end(ctx)).transpose()?;
        self.features.step(ctx, self.kernels, self.grad)?;
        Ok(ended)
    }

    /// M1 Fourier-modulus projection of stack `s`'s exit waves in place: FFT,
    /// `apply_amplitude`, inverse FFT. With `llk`, the stack's LLK partials of the forward
    /// stack go into that record first.
    fn project(
        &mut self,
        ctx: &ExecutionContext,
        m1: &mut M1Bufs<'_>,
        s: usize,
        llk: Option<&mut RecordWriter<'_>>,
    ) -> Result<(), GpuError> {
        let dims = *self.views.dims();
        let stack = self.table.at(s);
        let st = self.stacks.get(s);
        let plan = m1.plans.get(st.frames.plan);
        plan.forward_inplace(ctx, st.psi)?;
        if let Some(rec) = llk {
            self.kernels
                .llk(ctx, stack, self.counts, st.psi, self.llk_scratch, self.llk)?;
            rec.push(ctx, &dims, stack, self.llk)?;
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
