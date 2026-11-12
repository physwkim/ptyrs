//! The device state a captured cycle addresses, and the graph captured from it, under one
//! owner. Nothing outside this module can reach a field of [`Device`] or of a stack's
//! buffers, and nothing here replaces one after [`Device::new`] but
//! [`Device::set_features`], which drops every graph before and recaptures the LLK pass's
//! after: a graph replays on the buffers the eager cycle would use, and other buffers mean
//! another `Device`, or other features, which start with no graph. The device also owns the
//! LLK ring and its count of records: a record is counted once per enqueued run of a cycle
//! or pass that ended it, as the [`Ended`] its body returned shows, eagerly or kept from the
//! capture of its graph.

use std::ops::Range;
use std::sync::Arc;

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use ptyrs_fft::Plan2d;
use ptyrs_gpu::{
    BH_STEP, BhKernels, C32, CG_BLOCKS, Ended, EpochOrder, Error as GpuError, FarSums, FusedBufs,
    GammaStacks, GradKernels, Kernels, LlkPartial, LlkRing, LlkScratch, LlkTotals, Model,
    ObjProbeScale, PowerMax, RingDevice, RpieKernels, ScalePartial, StackTable, UpdateSums,
    ViewAddr, ViewDims, Views,
};
use ptyrs_tile::TileDots;

use super::lsqml::{LsqmlBufs, LsqmlState, Pre};
use super::{
    Epoch, Frames, GpuOptions, Tag, batch_stacks, plane_len, plans, split, stack_buf_len, stacks,
    to_u32,
};
use crate::{
    ApParams, BhParams, BhStep, DmParams, EngineError, Features, LsqmlParams, LsqmlStep, MlParams,
    RpieParams,
};

/// Every buffer, module and plan a cycle launches on, and the graphs captured from them.
pub(super) struct Device {
    stream: Arc<CudaStream>,
    kernels: Kernels,
    rpie: RpieKernels,
    grad: GradKernels,
    bh: BhKernels,
    /// BH's dot products, compiled by [`Device::new`] so no capture compiles a Tile kernel.
    tile: TileDots,
    views: Views,
    counts: DeviceBuffer<u16>,
    obj: DeviceBuffer<C32>,
    probe: DeviceBuffer<C32>,
    obj_sums: UpdateSums,
    probe_sums: UpdateSums,
    obj_len: u32,
    probe_len: u32,
    table: StackTable,
    stacks: Vec<StackBufs>,
    passes: Passes,
    /// One stack's LLK partials, on their way into the ring.
    llk: DeviceBuffer<LlkPartial>,
    /// The chunk sums M1's [`Kernels::llk`] reduces a stack's frames through.
    llk_scratch: LlkScratch,
    norms: NormBufs,
    ring: LlkRing<Tag>,
    use_graph: bool,
    /// The epochs run since the parameters last changed, all of one [`Epoch::run`].
    runs: Vec<EpochRun>,
    /// Frames per stack, for the [`MlState`] plans.
    stack_size: u32,
    /// With graphs, the LLK pass's graph.
    llk_graph: Option<Captured>,
    /// ML's state, built by the first ML call and never replaced.
    ml: Option<Box<MlState>>,
    /// BH's state, built by the first BH call and replaced only for another batch size.
    bh_state: Option<Box<BhState>>,
    /// LSQML's state, built by the first LSQML call and replaced only for another batch size.
    lsqml: Option<Box<LsqmlState>>,
    features: FeatureBufs,
}

/// The features' device state as [`Device::set_features`] last set it: each modelled
/// feature's values and the far sums its step reads.
#[derive(Default)]
struct FeatureBufs {
    scale: Option<ScaleBufs>,
}

/// A per-frame intensity scale `[nframe]` and its far sums `[nframe]`.
struct ScaleBufs {
    values: DeviceBuffer<f32>,
    sums: DeviceBuffer<ScalePartial>,
}

/// [`FeatureBufs`] lent to a cycle.
pub(super) struct FeaturesMut<'a> {
    scale: Option<ScaleMut<'a>>,
}

/// [`ScaleBufs`] lent to a cycle.
struct ScaleMut<'a> {
    values: &'a mut DeviceBuffer<f32>,
    sums: &'a mut DeviceBuffer<ScalePartial>,
}

impl FeaturesMut<'_> {
    /// Whether some feature takes a step from far sums.
    pub fn stepped(&self) -> bool {
        self.scale.is_some()
    }

    /// The model intensity the features set.
    pub fn model(&self) -> Model<'_> {
        Model {
            scale: self.scale.as_ref().map(|s| &*s.values),
            ..Model::default()
        }
    }

    /// The model intensity, and far sums of `llk` and of every stepped feature.
    pub fn far<'b>(
        &'b mut self,
        llk: Option<&'b mut DeviceBuffer<LlkPartial>>,
    ) -> (Model<'b>, FarSums<'b>) {
        let (scale, scale_sums) = match &mut self.scale {
            Some(s) => (Some(&*s.values), Some(&mut *s.sums)),
            None => (None, None),
        };
        let model = Model {
            scale,
            ..Model::default()
        };
        let far = FarSums {
            llk,
            scale: scale_sums,
            ..FarSums::default()
        };
        (model, far)
    }

    /// Every stepped feature's step from the far sums [`Self::far`] lent the passes before.
    pub fn step(&mut self, ctx: &ExecutionContext, grad: &GradKernels) -> Result<(), GpuError> {
        if let Some(s) = &mut self.scale {
            grad.scale_step(ctx, s.values, s.sums)?;
        }
        Ok(())
    }
}

/// How AP and DM project a stack and the LLK pass takes its partials, with the buffers only
/// that way uses ([`GpuOptions::fused`]).
enum Passes {
    M1 {
        /// One plan per stack length, indexed by [`Frames::plan`].
        plans: Vec<Plan2d>,
        /// One stack's worth of exit waves: DM's `PO`, the LLK pass's forward stack.
        scratch: DeviceBuffer<C32>,
    },
    /// M2's fused passes, one scratch set for every stack in turn.
    Fused(Box<FusedBufs>),
}

/// [`Passes`] lent to a cycle.
pub(super) enum PassesMut<'a> {
    M1(M1Bufs<'a>),
    Fused(&'a mut FusedBufs),
}

/// [`Passes::M1`] lent to a cycle.
pub(super) struct M1Bufs<'a> {
    pub plans: Plans<'a>,
    pub scratch: &'a mut DeviceBuffer<C32>,
}

/// A cycle with the buffers only it uses, and with graphs the graph captured for it.
struct EpochRun {
    cycle: Cycle,
    graph: Option<Captured>,
}

/// A graph and the record its body ended when captured, if any: each replay ends it again.
struct Captured {
    graph: CudaGraph<()>,
    ended: Option<Ended>,
}

/// A cycle's [`Epoch`] and the buffers only it uses, built together, so a graph captured
/// from it bakes in sizes that match its buffers.
enum Cycle {
    Ap {
        params: ApParams,
        llk: bool,
    },
    Dm {
        params: DmParams,
        first: bool,
    },
    Rpie(RpieParams, Box<RpieScratch>),
    /// Its buffers are the device's [`MlState`], which every ML epoch shares.
    Ml {
        params: MlParams,
        first: bool,
        llk: bool,
    },
    /// Its buffers are the device's [`BhState`], which every BH epoch of its batch size
    /// shares.
    Bh {
        params: BhParams,
        cg: bool,
    },
    /// Its buffers are the device's [`LsqmlState`], which every LSQML epoch of its batch
    /// size shares.
    Lsqml {
        params: LsqmlParams,
        pre: Pre,
    },
}

impl Cycle {
    fn epoch(&self) -> Epoch {
        match *self {
            Self::Ap { params, llk } => Epoch::Ap { params, llk },
            Self::Dm { params, first } => Epoch::Dm { params, first },
            Self::Rpie(p, _) => Epoch::Rpie(p),
            Self::Ml { params, first, llk } => Epoch::Ml { params, first, llk },
            Self::Bh { params, cg } => Epoch::Bh { params, cg },
            Self::Lsqml { params, pre } => Epoch::Lsqml { params, pre },
        }
    }
}

/// ML's conjugate-gradient state and scratch, sized by the device alone, so one state serves
/// the graphs of every ML epoch; each call's first epoch resets the state.
struct MlState {
    /// One plan per stack length, indexed by [`Frames::plan`].
    plans: Vec<Plan2d>,
    /// The last cycle's gradient, object then probe.
    h: [DeviceBuffer<C32>; 2],
    /// The search direction, object then probe.
    d: [DeviceBuffer<C32>; 2],
    /// Block partial pairs: the β sums of the object then the probe.
    parts: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    /// The line-search pairs of every frame's pixel chunks ([`ptyrs_gpu::gamma_parts`]).
    gamma_parts: DeviceBuffer<f32>,
    gamma: DeviceBuffer<f32>,
    /// The calculated photons over the unmasked pixels of every frame.
    icalc: DeviceBuffer<f32>,
    fields: GammaStacks,
}

/// An [`MlState`] lent to an epoch.
pub(super) struct MlBufs<'a> {
    pub plans: Plans<'a>,
    pub h: &'a mut [DeviceBuffer<C32>; 2],
    pub d: &'a mut [DeviceBuffer<C32>; 2],
    pub parts: &'a mut DeviceBuffer<f32>,
    pub beta: &'a mut DeviceBuffer<f32>,
    pub gamma_parts: &'a mut DeviceBuffer<f32>,
    pub gamma: &'a mut DeviceBuffer<f32>,
    pub icalc: &'a mut DeviceBuffer<f32>,
    pub fields: &'a mut GammaStacks,
}

impl MlState {
    /// For the device's stacks of `stack_size` frames, whose buffers hold `stack_len` values,
    /// and arrays of `lens = [object, probe]` values.
    fn new(
        stream: &Arc<CudaStream>,
        dims: &ViewDims,
        stack_size: u32,
        stack_len: usize,
        lens: [usize; 2],
    ) -> Result<Self, EngineError> {
        let (_, plan_lens) = split(dims.nframe, stack_size);
        let zeroed = |len| DeviceBuffer::<C32>::zeroed(stream, len);
        let pair = || DeviceBuffer::<f32>::zeroed(stream, 2);
        Ok(Self {
            plans: plans(stream, dims, &plan_lens)?,
            h: [zeroed(lens[0])?, zeroed(lens[1])?],
            d: [zeroed(lens[0])?, zeroed(lens[1])?],
            parts: DeviceBuffer::zeroed(stream, 4 * CG_BLOCKS as usize)?,
            beta: pair()?,
            gamma_parts: DeviceBuffer::zeroed(stream, 2 * ptyrs_gpu::gamma_parts(dims)? as usize)?,
            gamma: pair()?,
            icalc: DeviceBuffer::zeroed(stream, 1)?,
            fields: GammaStacks::zeroed(stream, stack_len)?,
        })
    }

    fn lend(&mut self) -> MlBufs<'_> {
        MlBufs {
            plans: Plans(&mut self.plans),
            h: &mut self.h,
            d: &mut self.d,
            parts: &mut self.parts,
            beta: &mut self.beta,
            gamma_parts: &mut self.gamma_parts,
            gamma: &mut self.gamma,
            icalc: &mut self.icalc,
            fields: &mut self.fields,
        }
    }
}

/// The `(sum |x|, sum |x|^2)` pairs of the object and the probe that
/// [`Bufs::scale_obj_probe`] scales them by, and the block partials it sums them through.
pub(super) struct NormBufs {
    pub parts: DeviceBuffer<f32>,
    pub pairs: [DeviceBuffer<f32>; 2],
}

impl NormBufs {
    fn zeroed(stream: &CudaStream) -> Result<Self, EngineError> {
        let pair = || DeviceBuffer::<f32>::zeroed(stream, 2);
        Ok(Self {
            parts: DeviceBuffer::zeroed(stream, 2 * CG_BLOCKS as usize)?,
            pairs: [pair()?, pair()?],
        })
    }
}

/// BH's batches over epoch positions and its state, sized by the device and the batch size
/// alone, so one state serves the graphs of every BH epoch of that size; [`Device::bh_epoch`]
/// sets each epoch's frame order before its cycle.
struct BhState {
    /// Frames per batch, the batch size capped at the frame count.
    size: u32,
    /// Each batch's range of `stacks`.
    batches: Vec<Range<usize>>,
    /// The batches cut into stacks of the device's stack size, one table entry each.
    stacks: Vec<Frames>,
    table: StackTable,
    /// One plan per stack length, indexed by [`Frames::plan`].
    plans: Vec<Plan2d>,
    /// A batch's forward exit waves and inverse-FFT'd residuals, one per stack of a batch.
    psi: Vec<DeviceBuffer<C32>>,
    r: Vec<DeviceBuffer<C32>>,
    /// One stack's forward-FFT'd derivatives along the scaled gradient and the last direction.
    lu: DeviceBuffer<C32>,
    lv: DeviceBuffer<C32>,
    /// The last direction, object then probe.
    prev: [DeviceBuffer<C32>; 2],
    /// The scan frame at each epoch position, and the views in that order.
    order: EpochOrder,
    /// One step's partial sums ([`ptyrs_gpu::bh_parts`]).
    parts: DeviceBuffer<f32>,
    /// The epoch's step records, one per batch.
    steps: DeviceBuffer<f32>,
}

/// A [`BhState`] lent to an epoch.
pub(super) struct BhBufs<'a> {
    pub size: u32,
    pub batches: &'a [Range<usize>],
    pub stacks: &'a [Frames],
    pub table: &'a StackTable,
    pub plans: Plans<'a>,
    pub psi: &'a mut [DeviceBuffer<C32>],
    pub r: &'a mut [DeviceBuffer<C32>],
    pub lu: &'a mut DeviceBuffer<C32>,
    pub lv: &'a mut DeviceBuffer<C32>,
    pub prev: &'a mut [DeviceBuffer<C32>; 2],
    pub order: &'a EpochOrder,
    pub parts: &'a mut DeviceBuffer<f32>,
    pub steps: &'a mut DeviceBuffer<f32>,
}

impl BhState {
    /// For `views` in batches of `size` frames cut into stacks of `stack_size`, and arrays of
    /// `lens = [object, probe]` values; the epoch order is the scan's.
    fn new(
        stream: &Arc<CudaStream>,
        kernels: &Kernels,
        views: &Views,
        size: u32,
        stack_size: u32,
        lens: [usize; 2],
    ) -> Result<Self, EngineError> {
        let dims = views.dims();
        let ((stacks, table, plan_lens), batches) = batch_stacks(stream, dims, size, stack_size)?;
        let len = stack_buf_len(dims, &table);
        let slots = batches.iter().map(|b| b.len()).max().unwrap_or(0);
        let zeroed = |len| DeviceBuffer::<C32>::zeroed(stream, len);
        Ok(Self {
            size,
            plans: plans(stream, dims, &plan_lens)?,
            psi: (0..slots).map(|_| zeroed(len)).collect::<Result<_, _>>()?,
            r: (0..slots).map(|_| zeroed(len)).collect::<Result<_, _>>()?,
            lu: zeroed(len)?,
            lv: zeroed(len)?,
            prev: [zeroed(lens[0])?, zeroed(lens[1])?],
            order: EpochOrder::new(kernels, stream, views)?,
            parts: DeviceBuffer::zeroed(stream, ptyrs_gpu::bh_parts(dims))?,
            steps: DeviceBuffer::zeroed(stream, BH_STEP as usize * batches.len())?,
            batches,
            stacks,
            table,
        })
    }

    fn lend(&mut self) -> BhBufs<'_> {
        BhBufs {
            size: self.size,
            batches: &self.batches,
            stacks: &self.stacks,
            table: &self.table,
            plans: Plans(&mut self.plans),
            psi: &mut self.psi,
            r: &mut self.r,
            lu: &mut self.lu,
            lv: &mut self.lv,
            prev: &mut self.prev,
            order: &self.order,
            parts: &mut self.parts,
            steps: &mut self.steps,
        }
    }
}

/// rPIE's batches (one table entry and plan length each) and buffers, sized for the longest
/// batch.
struct RpieScratch {
    batches: Vec<Frames>,
    /// Values per frame plane.
    n: u32,
    table: StackTable,
    /// One plan per batch length, indexed by [`Frames::plan`].
    plans: Vec<Plan2d>,
    psi: DeviceBuffer<C32>,
    dlt: DeviceBuffer<C32>,
    max_chunks: DeviceBuffer<f32>,
    omax: DeviceBuffer<f32>,
    ptilde: DeviceBuffer<C32>,
    pmax: DeviceBuffer<f32>,
    ones: DeviceBuffer<C32>,
}

/// An [`RpieScratch`] lent to an epoch.
pub(super) struct RpieBufs<'a> {
    pub batches: &'a [Frames],
    /// Values per frame plane.
    pub n: u32,
    pub table: &'a StackTable,
    pub plans: Plans<'a>,
    /// The batch's exit waves, then its object windows `O_v q`.
    pub psi: &'a mut DeviceBuffer<C32>,
    /// pty-chi's `Δ`, then the probe step's weighted `Δ`.
    pub dlt: &'a mut DeviceBuffer<C32>,
    /// [`ptyrs_gpu::RpieKernels::power_max`]'s per-chunk maxes, for up to `nk_max` groups.
    pub max_chunks: &'a mut DeviceBuffer<f32>,
    /// Per-view `max |O_v|^2`.
    pub omax: &'a mut DeviceBuffer<f32>,
    /// `P / ((1 - a_o) |P|^2 + a_o max_i sum_m |P_m|^2)`.
    pub ptilde: &'a mut DeviceBuffer<C32>,
    pub pmax: &'a mut DeviceBuffer<f32>,
    /// A probe of ones, so `obj_probe_to_psi` yields the object windows.
    pub ones: &'a DeviceBuffer<C32>,
}

impl RpieScratch {
    fn new(
        stream: &Arc<CudaStream>,
        dims: &ViewDims,
        params: &RpieParams,
        nprobe_values: usize,
    ) -> Result<Self, EngineError> {
        let (batches, table, lens) =
            stacks(stream, dims, to_u32("batch_size", params.batch_size)?)?;
        let len = stack_buf_len(dims, &table);
        let nk_max = table.nf_max() as usize * dims.nmode as usize;
        let n = to_u32("frame size", plane_len(dims))?;
        let max_groups = PowerMax {
            ngroup: to_u32("planes per batch", nk_max)?,
            nsum: 1,
            n,
        };
        Ok(Self {
            batches,
            n,
            table,
            plans: plans(stream, dims, &lens)?,
            psi: DeviceBuffer::zeroed(stream, len)?,
            dlt: DeviceBuffer::zeroed(stream, len)?,
            max_chunks: DeviceBuffer::zeroed(stream, max_groups.chunks())?,
            omax: DeviceBuffer::zeroed(stream, nk_max)?,
            ptilde: DeviceBuffer::zeroed(stream, nprobe_values)?,
            pmax: DeviceBuffer::zeroed(stream, 1)?,
            ones: DeviceBuffer::from_host(stream, &vec![C32 { re: 1.0, im: 0.0 }; nprobe_values])?,
        })
    }

    fn lend(&mut self) -> RpieBufs<'_> {
        RpieBufs {
            batches: &self.batches,
            n: self.n,
            table: &self.table,
            plans: Plans(&mut self.plans),
            psi: &mut self.psi,
            dlt: &mut self.dlt,
            max_chunks: &mut self.max_chunks,
            omax: &mut self.omax,
            ptilde: &mut self.ptilde,
            pmax: &mut self.pmax,
            ones: &self.ones,
        }
    }
}

/// One stack's buffers, sized like every stack's for the table's `nf_max` frames.
struct StackBufs {
    frames: Frames,
    /// Exit waves of the stack's views; planes past `nf * nmode` stay zero.
    psi: DeviceBuffer<C32>,
}

/// The [`Device`] lent to a cycle, but for its [`PassesMut`]: references only, so a cycle
/// launches on every buffer but rebinding a field leaves the `Device` as it was.
pub(super) struct Bufs<'a> {
    pub kernels: &'a Kernels,
    pub rpie: &'a RpieKernels,
    pub grad: &'a GradKernels,
    pub bh: &'a BhKernels,
    pub tile: &'a TileDots,
    pub views: &'a Views,
    pub counts: &'a DeviceBuffer<u16>,
    pub obj: &'a mut DeviceBuffer<C32>,
    pub probe: &'a mut DeviceBuffer<C32>,
    pub obj_sums: &'a mut UpdateSums,
    pub probe_sums: &'a mut UpdateSums,
    pub obj_len: u32,
    pub probe_len: u32,
    pub table: &'a StackTable,
    pub stacks: Stacks<'a>,
    /// A stack's LLK partials, which [`ptyrs_gpu::RecordWriter::push`] then copies into the
    /// record.
    pub llk: &'a mut DeviceBuffer<LlkPartial>,
    /// The chunk sums of M1's [`Kernels::llk`].
    pub llk_scratch: &'a mut LlkScratch,
    pub norms: &'a mut NormBufs,
    /// ML's state, once an ML call has built it.
    pub ml: Option<MlBufs<'a>>,
    /// BH's state, once a BH call has built it.
    pub bh_state: Option<BhBufs<'a>>,
    /// LSQML's state, once an LSQML call has built it.
    pub lsqml: Option<LsqmlBufs<'a>>,
    pub features: FeaturesMut<'a>,
}

/// The stacks lent to a cycle, one at a time.
pub(super) struct Stacks<'a>(&'a mut [StackBufs]);

/// Stack `s` lent to a cycle.
pub(super) struct StackMut<'a> {
    pub frames: &'a Frames,
    pub psi: &'a mut DeviceBuffer<C32>,
}

impl Stacks<'_> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn get(&mut self, s: usize) -> StackMut<'_> {
        let st = &mut self.0[s];
        StackMut {
            frames: &st.frames,
            psi: &mut st.psi,
        }
    }
}

/// The cuFFT plans lent to a cycle, by [`Frames::plan`].
pub(super) struct Plans<'a>(&'a mut [Plan2d]);

impl Plans<'_> {
    pub fn get(&mut self, i: usize) -> &mut Plan2d {
        &mut self.0[i]
    }
}

impl Device {
    /// Loads the kernels and uploads the views, counts and starting arrays on a new stream of
    /// `ctx`; the stacks cut `0 .. nframe` into `options.stack_size` frames, the LLK ring
    /// holds `options.llk_slots` records of every frame, its unmasked pixels counted here, and
    /// `options.fused` picks the [`Passes`].
    pub fn new(
        ctx: &Arc<CudaContext>,
        dims: ViewDims,
        addrs: &[ViewAddr],
        counts: &[u16],
        obj: &[C32],
        probe: &[C32],
        options: GpuOptions,
    ) -> Result<Self, EngineError> {
        let stream = ctx.new_stream()?;
        let kernels = Kernels::load(ctx)?;
        let rpie = RpieKernels::load(&kernels)?;
        let grad = GradKernels::load(&kernels)?;
        let bh = BhKernels::load(&kernels)?;
        let views = Views::new(&stream, dims, addrs)?;
        let tile = TileDots::new(&stream, &dims)?;
        let (frames, table, lens) =
            stacks(&stream, &dims, to_u32("stack_size", options.stack_size)?)?;
        let len = stack_buf_len(&dims, &table);
        let passes = if options.fused {
            let f = FusedBufs::new(&stream, &dims, table.nf_max()).map_err(|e| match e {
                GpuError::Shape(m) => EngineError::Invalid(format!(
                    "{m}; GpuOptions {{ fused: false, .. }} runs M1's launches instead"
                )),
                e => e.into(),
            })?;
            Passes::Fused(Box::new(f))
        } else {
            Passes::M1 {
                plans: plans(&stream, &dims, &lens)?,
                scratch: DeviceBuffer::zeroed(&stream, len)?,
            }
        };
        let stacks = frames
            .into_iter()
            .map(|frames| {
                Ok(StackBufs {
                    frames,
                    psi: DeviceBuffer::zeroed(&stream, len)?,
                })
            })
            .collect::<Result<Vec<_>, EngineError>>()?;
        let mut dev = Self {
            obj_len: to_u32("object length", obj.len())?,
            probe_len: to_u32("probe length", probe.len())?,
            llk: DeviceBuffer::zeroed(&stream, table.nf_max() as usize)?,
            llk_scratch: LlkScratch::new(&stream, &dims, table.nf_max())?,
            norms: NormBufs::zeroed(&stream)?,
            ring: LlkRing::new(
                &kernels,
                &stream,
                &dims,
                to_u32("llk_slots", options.llk_slots)?,
            )?,
            counts: DeviceBuffer::from_host(&stream, counts)?,
            obj: DeviceBuffer::from_host(&stream, obj)?,
            probe: DeviceBuffer::from_host(&stream, probe)?,
            obj_sums: UpdateSums::zeroed(&stream, obj.len())?,
            probe_sums: UpdateSums::zeroed(&stream, probe.len())?,
            kernels,
            rpie,
            grad,
            bh,
            tile,
            views,
            table,
            stacks,
            passes,
            use_graph: options.graph,
            runs: Vec::new(),
            llk_graph: None,
            ml: None,
            bh_state: None,
            lsqml: None,
            features: FeatureBufs::default(),
            stack_size: to_u32("stack_size", options.stack_size)?,
            stream,
        };
        let ctx = dev.context();
        dev.ring.device().count(&ctx, 0..dims.nframe, &dev.counts)?;
        dev.capture_llk()?;
        Ok(dev)
    }

    /// With graphs, captures the LLK pass's graph.
    fn capture_llk(&mut self) -> Result<(), EngineError> {
        if self.use_graph {
            self.llk_graph = Some(self.capture(|bufs, passes, ring, ctx| {
                bufs.forward_llk(ctx, passes, ring.record()).map(Some)
            })?);
        }
        Ok(())
    }

    /// Models `features`, which the caller checked against the scan, from here on: the
    /// values of the features already set are written in place; other features replace the
    /// buffers, and so first drop every graph, which addresses them, and then recapture the
    /// LLK pass's. Only the fused passes model features.
    pub fn set_features(&mut self, features: &Features) -> Result<(), EngineError> {
        if matches!(self.passes, Passes::M1 { .. }) && *features != Features::default() {
            return Err(EngineError::Invalid(
                "features need GpuOptions { fused: true, .. }".into(),
            ));
        }
        let scale: Option<Vec<f32>> = features
            .scale
            .as_ref()
            .map(|s| s.iter().map(|&s| s as f32).collect());
        match (&mut self.features.scale, scale) {
            (Some(b), Some(v)) => b.values.copy_from_host(&self.stream, &v)?,
            (None, None) => {}
            (_, scale) => {
                let scale = scale
                    .map(|v| {
                        Ok::<_, EngineError>(ScaleBufs {
                            values: DeviceBuffer::from_host(&self.stream, &v)?,
                            sums: DeviceBuffer::zeroed(&self.stream, v.len())?,
                        })
                    })
                    .transpose()?;
                self.runs.clear();
                self.llk_graph = None;
                self.features.scale = scale;
                self.capture_llk()?;
            }
        }
        Ok(())
    }

    /// The modelled features, read back.
    pub fn features(&self) -> Result<Features, EngineError> {
        let scale = match &self.features.scale {
            Some(b) => Some(b.values.to_host_vec(&self.stream)?),
            None => None,
        };
        Ok(Features {
            scale: scale.map(|v| v.into_iter().map(f64::from).collect()),
        })
    }

    /// Whether some feature is modelled.
    pub fn has_features(&self) -> bool {
        self.features.scale.is_some()
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// A fresh eager context on the stream: one per cycle, so leases end with its work.
    pub fn context(&self) -> ExecutionContext {
        ExecutionContext::new(self.stream.clone())
    }

    pub fn dims(&self) -> ViewDims {
        *self.views.dims()
    }

    pub fn obj(&self) -> &DeviceBuffer<C32> {
        &self.obj
    }

    pub fn probe(&self) -> &DeviceBuffer<C32> {
        &self.probe
    }

    /// Overwrites the object in place, so every captured graph reads the new values.
    pub fn write_object(&mut self, obj: &[C32]) -> Result<(), EngineError> {
        Ok(self.obj.copy_from_host(&self.stream, obj)?)
    }

    /// Overwrites the object and probe in place.
    pub fn write_arrays(&mut self, obj: &[C32], probe: &[C32]) -> Result<(), EngineError> {
        self.write_object(obj)?;
        self.probe.copy_from_host(&self.stream, probe)?;
        Ok(())
    }

    /// Lends every buffer, and apart the passes' own and the ring the cycle writes its records
    /// through.
    fn lend(&mut self) -> (Bufs<'_>, PassesMut<'_>, &mut RingDevice) {
        let bufs = Bufs {
            kernels: &self.kernels,
            rpie: &self.rpie,
            grad: &self.grad,
            bh: &self.bh,
            tile: &self.tile,
            views: &self.views,
            counts: &self.counts,
            obj: &mut self.obj,
            probe: &mut self.probe,
            obj_sums: &mut self.obj_sums,
            probe_sums: &mut self.probe_sums,
            obj_len: self.obj_len,
            probe_len: self.probe_len,
            table: &self.table,
            stacks: Stacks(&mut self.stacks),
            llk: &mut self.llk,
            llk_scratch: &mut self.llk_scratch,
            norms: &mut self.norms,
            ml: self.ml.as_deref_mut().map(MlState::lend),
            bh_state: self.bh_state.as_deref_mut().map(BhState::lend),
            lsqml: self.lsqml.as_deref_mut().map(LsqmlState::lend),
            features: FeaturesMut {
                scale: self.features.scale.as_mut().map(|s| ScaleMut {
                    values: &mut s.values,
                    sums: &mut s.sums,
                }),
            },
        };
        let passes = match &mut self.passes {
            Passes::M1 { plans, scratch } => PassesMut::M1(M1Bufs {
                plans: Plans(plans),
                scratch,
            }),
            Passes::Fused(f) => PassesMut::Fused(f),
        };
        (bufs, passes, self.ring.device())
    }

    /// Enqueues one `epoch` cycle, numbered `cycle`: replays the graph captured for it, or
    /// launches it eagerly without graphs, and counts the LLK record it ends, if any. The
    /// epochs of one [`Epoch::run`] keep their buffers and graphs until an epoch of another
    /// run replaces them.
    pub fn steady(&mut self, epoch: Epoch, cycle: usize) -> Result<(), EngineError> {
        self.steady_with(epoch, cycle, |_| Ok(()))
    }

    /// One BH epoch as [`Self::steady`] enqueues it, after setting its frame order: `order`
    /// holds the scan frame at each epoch position.
    pub fn bh_epoch(
        &mut self,
        params: BhParams,
        cg: bool,
        cycle: usize,
        order: &[u32],
    ) -> Result<(), EngineError> {
        self.steady_with(Epoch::Bh { params, cg }, cycle, |dev| {
            let ctx = dev.context();
            let bh = dev
                .bh_state
                .as_deref_mut()
                .expect("new_run built the BH state");
            Ok(bh.order.set(&ctx, &dev.kernels, &dev.views, order)?)
        })
    }

    /// The last BH epoch's steps, one per batch, read back: each batch's frames from the
    /// device's epoch order and its `(β, α, top, bottom)` record; none before a BH call.
    pub fn bh_steps(&self) -> Result<Vec<BhStep>, EngineError> {
        let Some(bh) = self.bh_state.as_deref() else {
            return Ok(Vec::new());
        };
        let ids = bh.order.ids().to_host_vec(&self.stream)?;
        let records = bh.steps.to_host_vec(&self.stream)?;
        let steps = bh
            .batches
            .iter()
            .zip(records.as_chunks::<{ BH_STEP as usize }>().0);
        Ok(steps
            .map(|(b, r)| {
                let at = bh.stacks[b.start].range.start as usize
                    ..bh.stacks[b.end - 1].range.end as usize;
                let [beta, alpha, top, bottom] = r.map(f64::from);
                BhStep {
                    frames: ids[at].iter().map(|&f| f as usize).collect(),
                    beta,
                    alpha,
                    top,
                    bottom,
                }
            })
            .collect())
    }

    /// One LSQML epoch as [`Self::steady`] enqueues it, after setting its frame order:
    /// `order` holds the scan frame at each epoch position.
    pub fn lsqml_epoch(
        &mut self,
        params: LsqmlParams,
        pre: Pre,
        cycle: usize,
        order: &[u32],
    ) -> Result<(), EngineError> {
        self.steady_with(Epoch::Lsqml { params, pre }, cycle, |dev| {
            let ctx = dev.context();
            let l = dev
                .lsqml
                .as_deref_mut()
                .expect("new_run built the LSQML state");
            Ok(l.set_order(&ctx, &dev.kernels, &dev.views, order)?)
        })
    }

    /// The last LSQML epoch's steps, one per batch; none before an LSQML call.
    pub fn lsqml_steps(&self) -> Result<Vec<LsqmlStep>, EngineError> {
        match self.lsqml.as_deref() {
            Some(l) => l.steps(&self.stream),
            None => Ok(Vec::new()),
        }
    }

    /// [`Self::steady`] with `stage` enqueued just before the cycle, once its buffers are
    /// built.
    fn steady_with(
        &mut self,
        epoch: Epoch,
        cycle: usize,
        stage: impl FnOnce(&mut Self) -> Result<(), EngineError>,
    ) -> Result<(), EngineError> {
        let mut runs = std::mem::take(&mut self.runs);
        runs.retain(|r| r.cycle.epoch().run() == epoch.run());
        let at = match runs.iter().position(|r| r.cycle.epoch() == epoch) {
            Some(at) => Ok(at),
            None => self.new_run(epoch).map(|run| {
                runs.push(run);
                runs.len() - 1
            }),
        };
        let done = at.and_then(|at| {
            stage(self)?;
            let run = &mut runs[at];
            let tag = Some((cycle, epoch.algorithm()));
            match &run.graph {
                Some(c) => self.replay(c, tag),
                None => {
                    let ctx = self.context();
                    let (mut bufs, mut passes, ring) = self.lend();
                    match bufs.cycle(&ctx, &mut passes, ring, &mut run.cycle) {
                        Ok(ended) => self.count(ended.as_ref(), tag),
                        Err(e) => Err(e.into()),
                    }
                }
            }
        });
        self.runs = runs;
        done
    }

    /// Enqueues the LLK record of the current estimate under `tag`, from the graph captured
    /// in [`Self::new`] or eagerly without graphs.
    pub fn llk_pass(&mut self, tag: Tag) -> Result<(), EngineError> {
        let graph = self.llk_graph.take();
        let done = match &graph {
            Some(c) => self.replay(c, tag),
            None => {
                let ctx = self.context();
                let (mut bufs, mut passes, ring) = self.lend();
                match bufs.forward_llk(&ctx, &mut passes, ring.record()) {
                    Ok(ended) => self.count(Some(&ended), tag),
                    Err(e) => Err(e.into()),
                }
            }
        };
        self.llk_graph = graph;
        done
    }

    /// The LLK records counted since the last flush, in order; waits for their copies.
    pub fn flush(&mut self) -> Result<Vec<(Tag, LlkTotals)>, EngineError> {
        Ok(self.ring.flush(&self.context())?)
    }

    /// The buffers of `epoch`'s cycle, and with graphs its graph, captured from them.
    fn new_run(&mut self, epoch: Epoch) -> Result<EpochRun, EngineError> {
        let mut cycle = match epoch {
            Epoch::Ap { params, llk } => Cycle::Ap { params, llk },
            Epoch::Dm { params, first } => Cycle::Dm { params, first },
            Epoch::Rpie(p) => {
                let scratch = RpieScratch::new(&self.stream, &self.dims(), &p, self.probe.len())?;
                Cycle::Rpie(p, Box::new(scratch))
            }
            Epoch::Ml { params, first, llk } => {
                self.build_ml()?;
                Cycle::Ml { params, first, llk }
            }
            Epoch::Bh { params, cg } => {
                self.build_bh(&params)?;
                Cycle::Bh { params, cg }
            }
            Epoch::Lsqml { params, pre } => {
                self.build_lsqml(&params)?;
                Cycle::Lsqml { params, pre }
            }
        };
        let graph = if self.use_graph {
            Some(self.capture(|bufs, passes, ring, ctx| bufs.cycle(ctx, passes, ring, &mut cycle))?)
        } else {
            None
        };
        Ok(EpochRun { cycle, graph })
    }

    /// Builds the [`MlState`] if no ML call has yet.
    fn build_ml(&mut self) -> Result<(), EngineError> {
        if self.ml.is_none() {
            let dims = self.dims();
            let stack_len = stack_buf_len(&dims, &self.table);
            let lens = [self.obj.len(), self.probe.len()];
            let state = MlState::new(&self.stream, &dims, self.stack_size, stack_len, lens)?;
            self.ml = Some(Box::new(state));
        }
        Ok(())
    }

    /// Builds the [`BhState`] for `params`' batch size unless it is built. A state of another
    /// size is replaced only here, when [`Self::steady`] has already dropped the runs of
    /// other parameters, so no graph captured from it survives.
    fn build_bh(&mut self, params: &BhParams) -> Result<(), EngineError> {
        let dims = self.dims();
        let size = to_u32("batch_size", params.batch_size.min(dims.nframe as usize))?;
        if self.bh_state.as_ref().is_none_or(|s| s.size != size) {
            let lens = [self.obj.len(), self.probe.len()];
            let state = BhState::new(
                &self.stream,
                &self.kernels,
                &self.views,
                size,
                self.stack_size,
                lens,
            )?;
            self.bh_state = Some(Box::new(state));
        }
        Ok(())
    }

    /// Builds the [`LsqmlState`] for `params`' batch size unless it is built, replaced as
    /// [`Self::build_bh`] replaces BH's.
    fn build_lsqml(&mut self, params: &LsqmlParams) -> Result<(), EngineError> {
        let size = to_u32(
            "batch_size",
            params.batch_size.min(self.dims().nframe as usize),
        )?;
        if self.lsqml.as_ref().is_none_or(|s| s.size() != size) {
            let state = LsqmlState::new(
                &self.stream,
                &self.kernels,
                &self.views,
                size,
                self.stack_size,
            )?;
            self.lsqml = Some(Box::new(state));
        }
        Ok(())
    }

    /// [`Bufs::scale_to_photons`] enqueued eagerly, with `total` the unmasked observed photons.
    pub fn scale_to_photons(&mut self, total: f32) -> Result<(), EngineError> {
        self.build_ml()?;
        let ctx = self.context();
        let (mut bufs, _, _) = self.lend();
        Ok(bufs.with_ml(|b, ml| b.scale_to_photons(&ctx, ml, total))?)
    }

    /// PyNX `ScaleObjProbe(absolute=False)` enqueued eagerly: `Σ|O|` and `Σ|P|` balanced,
    /// `O P` unchanged.
    pub fn balance_obj_probe(&mut self) -> Result<(), EngineError> {
        let ctx = self.context();
        let (mut bufs, _, _) = self.lend();
        Ok(bufs.scale_obj_probe(&ctx, ObjProbeScale::Relative)?)
    }

    /// The ML gradients of the current estimate, object then probe, as
    /// [`Bufs::ml_grad`] leaves them in the update sums; downloads them and zeroes the sums.
    pub fn ml_grads(&mut self) -> Result<[Vec<C32>; 2], EngineError> {
        self.build_ml()?;
        {
            let ctx = self.context();
            let (mut bufs, mut passes, _) = self.lend();
            bufs.with_ml(|b, ml| b.ml_grad(&ctx, &mut passes, ml, [true, true], None))?;
        }
        let grads = [
            self.obj_sums.num.to_host_vec(&self.stream)?,
            self.probe_sums.num.to_host_vec(&self.stream)?,
        ];
        let ctx = self.context();
        for sums in [&mut self.obj_sums, &mut self.probe_sums] {
            ptyrs_gpu::zero(&ctx, &mut sums.num)?;
            ptyrs_gpu::zero(&ctx, &mut sums.den)?;
        }
        Ok(grads)
    }

    /// `f` on the lent buffers, captured on a side stream, so nothing runs; with the record
    /// it ended.
    fn capture(
        &mut self,
        f: impl FnOnce(
            &mut Bufs<'_>,
            &mut PassesMut<'_>,
            &mut RingDevice,
            &ExecutionContext,
        ) -> Result<Option<Ended>, GpuError>
        + Send,
    ) -> Result<Captured, EngineError> {
        let side = self.kernels.context().new_stream()?;
        let (mut bufs, mut passes, ring) = self.lend();
        let mut ended = None;
        let op = ptyrs_gpu::record(|ctx| {
            ended = Some(f(&mut bufs, &mut passes, ring, ctx)?);
            Ok(())
        });
        let graph = CudaGraph::capture(side, op).map_err(GpuError::from)?;
        let ended = ended.expect("a capture that succeeded ran its body once");
        Ok(Captured { graph, ended })
    }

    /// Enqueues `c`'s graph on the stream and counts the record it ends, if any, under `tag`.
    fn replay(&mut self, c: &Captured, tag: Tag) -> Result<(), EngineError> {
        // SAFETY: the context holds the replay's leases on every captured buffer until the
        // stream has run it, so no other stream can touch them meanwhile.
        unsafe { c.graph.launch().execute(&self.context()) }.map_err(GpuError::from)?;
        self.count(c.ended.as_ref(), tag)
    }

    /// Counts under `tag` the record a run just enqueued ended, if it ended one.
    fn count(&mut self, ended: Option<&Ended>, tag: Tag) -> Result<(), EngineError> {
        if let Some(ended) = ended {
            let nframe = self.dims().nframe;
            self.ring.enqueued(&self.context(), ended, tag, nframe)?;
        }
        Ok(())
    }
}

impl<'a> Bufs<'a> {
    /// `f` on these buffers and ML's state, lent apart.
    pub(super) fn with_ml<R>(&mut self, f: impl FnOnce(&mut Self, &mut MlBufs<'a>) -> R) -> R {
        let mut ml = self.ml.take().expect("Device::build_ml built the ML state");
        let r = f(self, &mut ml);
        self.ml = Some(ml);
        r
    }

    /// `f` on these buffers and BH's state, lent apart.
    pub(super) fn with_bh<R>(&mut self, f: impl FnOnce(&mut Self, &mut BhBufs<'a>) -> R) -> R {
        let mut bh = self
            .bh_state
            .take()
            .expect("Device::build_bh built the BH state");
        let r = f(self, &mut bh);
        self.bh_state = Some(bh);
        r
    }

    /// One cycle of `cycle` enqueued on `ctx`, and the LLK record it ends: an AP cycle with
    /// `llk` writes one through `ring`, no other cycle touches it.
    fn cycle(
        &mut self,
        ctx: &ExecutionContext,
        passes: &mut PassesMut<'_>,
        ring: &mut RingDevice,
        cycle: &mut Cycle,
    ) -> Result<Option<Ended>, GpuError> {
        match cycle {
            Cycle::Ap { params, llk } => {
                self.ap_cycle(ctx, passes, params, llk.then(|| ring.record()))
            }
            Cycle::Dm { params, first } => {
                self.dm_cycle(ctx, passes, params, *first).map(|()| None)
            }
            Cycle::Rpie(p, scratch) => self.rpie_epoch(ctx, p, &mut scratch.lend()).map(|()| None),
            Cycle::Ml { params, first, llk } => self.with_ml(|b, ml| {
                b.ml_epoch(ctx, passes, ml, params, *first, llk.then(|| ring.record()))
            }),
            Cycle::Bh { params, cg } => self
                .with_bh(|b, bh| b.bh_epoch(ctx, bh, params, *cg))
                .map(|()| None),
            Cycle::Lsqml { params, pre } => self.lsqml_epoch(ctx, params, *pre).map(|()| None),
        }
    }
}
