//! The device state a captured cycle addresses, and the graph captured from it, under one
//! owner. Nothing outside this module can reach a field of [`Device`] or of a stack's
//! buffers, and nothing here replaces one after [`Device::new`]: a graph replays on the
//! buffers the eager cycle would use, and other buffers mean another `Device`, which starts
//! with no graph. The device also owns the LLK ring and its count of records: a record is
//! counted once per enqueued run of a cycle or pass that ended it, as the [`Ended`] its body
//! returned shows, eagerly or kept from the capture of its graph.

use std::sync::Arc;

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use ptyrs_fft::Plan2d;
use ptyrs_gpu::{
    C32, Ended, Error as GpuError, FusedBufs, Kernels, LlkPartial, LlkRing, LlkSums, RingDevice,
    RpieKernels, StackTable, UpdateSums, ViewAddr, ViewDims, Views,
};
use ptyrs_tile::TileProbeUpdate;

use super::{Epoch, Frames, GpuOptions, Tag, plane_len, plans, stack_buf_len, stacks, to_u32};
use crate::{ApParams, DmParams, EngineError, RpieParams};

/// Every buffer, module and plan a cycle launches on, and the graphs captured from them.
pub(super) struct Device {
    stream: Arc<CudaStream>,
    kernels: Kernels,
    rpie: RpieKernels,
    /// With [`GpuOptions::tile_probe_update`], the Tile probe update.
    tile: Option<TileProbeUpdate>,
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
    ring: LlkRing<Tag>,
    use_graph: bool,
    /// The epochs run since the parameters last changed, all of one [`Epoch::run`].
    runs: Vec<EpochRun>,
    /// With graphs, the LLK pass's graph.
    llk_graph: Option<Captured>,
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
    Fused(FusedBufs),
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
    Ap { params: ApParams, llk: bool },
    Dm { params: DmParams, first: bool },
    Rpie(RpieParams, Box<RpieScratch>),
}

impl Cycle {
    fn epoch(&self) -> Epoch {
        match *self {
            Self::Ap { params, llk } => Epoch::Ap { params, llk },
            Self::Dm { params, first } => Epoch::Dm { params, first },
            Self::Rpie(p, _) => Epoch::Rpie(p),
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
        Ok(Self {
            batches,
            n: to_u32("frame size", plane_len(dims))?,
            table,
            plans: plans(stream, dims, &lens)?,
            psi: DeviceBuffer::zeroed(stream, len)?,
            dlt: DeviceBuffer::zeroed(stream, len)?,
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
    pub tile: Option<&'a TileProbeUpdate>,
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
    /// holds `options.llk_slots` records, and `options.fused` picks the [`Passes`].
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
        let views = Views::new(&stream, dims, addrs)?;
        let tile = options
            .tile_probe_update
            .then(|| TileProbeUpdate::new(&stream, &views))
            .transpose()?;
        let (frames, table, lens) =
            stacks(&stream, &dims, to_u32("stack_size", options.stack_size)?)?;
        let len = stack_buf_len(&dims, &table);
        let passes = if options.fused {
            Passes::Fused(FusedBufs::new(&stream, &dims, table.nf_max())?)
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
            ring: LlkRing::new(
                &kernels,
                &stream,
                dims.nframe,
                to_u32("llk_slots", options.llk_slots)?,
            )?,
            counts: DeviceBuffer::from_host(&stream, counts)?,
            obj: DeviceBuffer::from_host(&stream, obj)?,
            probe: DeviceBuffer::from_host(&stream, probe)?,
            obj_sums: UpdateSums::zeroed(&stream, obj.len())?,
            probe_sums: UpdateSums::zeroed(&stream, probe.len())?,
            kernels,
            rpie,
            tile,
            views,
            table,
            stacks,
            passes,
            use_graph: options.graph,
            runs: Vec::new(),
            llk_graph: None,
            stream,
        };
        if dev.use_graph {
            dev.llk_graph = Some(dev.capture(|bufs, passes, ring, ctx| {
                bufs.forward_llk(ctx, passes, ring.record()).map(Some)
            })?);
        }
        Ok(dev)
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

    /// Overwrites the object and probe in place.
    pub fn write_arrays(&mut self, obj: &[C32], probe: &[C32]) -> Result<(), EngineError> {
        self.obj.copy_from_host(&self.stream, obj)?;
        self.probe.copy_from_host(&self.stream, probe)?;
        Ok(())
    }

    /// Lends every buffer, and apart the passes' own and the ring the cycle writes its records
    /// through.
    fn lend(&mut self) -> (Bufs<'_>, PassesMut<'_>, &mut RingDevice) {
        let bufs = Bufs {
            kernels: &self.kernels,
            rpie: &self.rpie,
            tile: self.tile.as_ref(),
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
        let mut runs = std::mem::take(&mut self.runs);
        runs.retain(|r| r.cycle.epoch().run() == epoch.run());
        let at = match runs.iter().position(|r| r.cycle.epoch() == epoch) {
            Some(at) => at,
            None => {
                runs.push(self.new_run(epoch)?);
                runs.len() - 1
            }
        };
        let run = &mut runs[at];
        let tag = Some((cycle, epoch.algorithm()));
        let done = match &run.graph {
            Some(c) => self.replay(c, tag),
            None => {
                let ctx = self.context();
                let (mut bufs, mut passes, ring) = self.lend();
                match bufs.cycle(&ctx, &mut passes, ring, &mut run.cycle) {
                    Ok(ended) => self.count(ended.as_ref(), tag),
                    Err(e) => Err(e.into()),
                }
            }
        };
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
    pub fn flush(&mut self) -> Result<Vec<(Tag, LlkSums)>, EngineError> {
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
        };
        let graph = if self.use_graph {
            Some(self.capture(|bufs, passes, ring, ctx| bufs.cycle(ctx, passes, ring, &mut cycle))?)
        } else {
            None
        };
        Ok(EpochRun { cycle, graph })
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
            self.ring.enqueued(&self.context(), ended, tag)?;
        }
        Ok(())
    }
}

impl Bufs<'_> {
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
        }
    }
}
