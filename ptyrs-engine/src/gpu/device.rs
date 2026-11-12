//! The device state a captured cycle addresses, and the graph captured from it, under one
//! owner. Nothing outside this module can reach a field of [`Device`] or of a stack's
//! buffers, and nothing here replaces one after [`Device::new`]: a graph replays on the
//! buffers the eager cycle would use, and other buffers mean another `Device`, which starts
//! with no graph.

use std::sync::Arc;

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use ptyrs_fft::Plan2d;
use ptyrs_gpu::{
    C32, Error as GpuError, Kernels, LlkPartial, RpieKernels, StackTable, UpdateSums, ViewAddr,
    ViewDims, Views,
};

use super::{Epoch, Frames, GpuOptions, stack_buf_len, stacks, to_u32};
use crate::EngineError;

/// Every buffer, module and plan a cycle launches on, and the graph of the last steady
/// cycle captured from them.
pub(super) struct Device {
    stream: Arc<CudaStream>,
    kernels: Kernels,
    rpie: RpieKernels,
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
    /// One plan per stack length, indexed by [`Frames::plan`].
    plans: Vec<Plan2d>,
    /// One stack's worth of exit waves: DM's `PO`, the LLK pass's forward stack.
    scratch: DeviceBuffer<C32>,
    use_graph: bool,
    run: Option<EpochRun>,
}

/// A steady cycle and the graph captured for it.
struct EpochRun {
    epoch: Epoch,
    graph: CudaGraph<()>,
}

/// One stack's buffers, sized like every stack's for the table's `nf_max` frames.
struct StackBufs {
    frames: Frames,
    /// Exit waves of the stack's views; planes past `nf * nmode` stay zero.
    psi: DeviceBuffer<C32>,
    llk: DeviceBuffer<LlkPartial>,
}

/// The [`Device`] lent to a cycle: references only, so a cycle launches on every buffer but
/// rebinding a field leaves the `Device` as it was.
pub(super) struct Bufs<'a> {
    pub kernels: &'a Kernels,
    pub rpie: &'a RpieKernels,
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
    pub plans: Plans<'a>,
    pub scratch: &'a mut DeviceBuffer<C32>,
}

/// The stacks lent to a cycle, one at a time.
pub(super) struct Stacks<'a>(&'a mut [StackBufs]);

/// Stack `s` lent to a cycle.
pub(super) struct StackMut<'a> {
    pub frames: &'a Frames,
    pub psi: &'a mut DeviceBuffer<C32>,
    pub llk: &'a mut DeviceBuffer<LlkPartial>,
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
            llk: &mut st.llk,
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
    /// `ctx`; the stacks cut `0 .. nframe` into `options.stack_size` frames.
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
        let (frames, table, plans) =
            stacks(&stream, &dims, to_u32("stack_size", options.stack_size)?)?;
        let len = stack_buf_len(&dims, &table);
        let stacks = frames
            .into_iter()
            .map(|frames| {
                Ok(StackBufs {
                    frames,
                    psi: DeviceBuffer::zeroed(&stream, len)?,
                    llk: DeviceBuffer::zeroed(&stream, table.nf_max() as usize)?,
                })
            })
            .collect::<Result<Vec<_>, EngineError>>()?;
        Ok(Self {
            obj_len: to_u32("object length", obj.len())?,
            probe_len: to_u32("probe length", probe.len())?,
            scratch: DeviceBuffer::zeroed(&stream, len)?,
            counts: DeviceBuffer::from_host(&stream, counts)?,
            obj: DeviceBuffer::from_host(&stream, obj)?,
            probe: DeviceBuffer::from_host(&stream, probe)?,
            obj_sums: UpdateSums::zeroed(&stream, obj.len())?,
            probe_sums: UpdateSums::zeroed(&stream, probe.len())?,
            kernels,
            rpie,
            views,
            table,
            stacks,
            plans,
            use_graph: options.graph,
            run: None,
            stream,
        })
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

    /// Each stack's frames and LLK partials.
    pub fn stack_llk(&self) -> impl Iterator<Item = (&Frames, &DeviceBuffer<LlkPartial>)> {
        self.stacks.iter().map(|st| (&st.frames, &st.llk))
    }

    /// Overwrites the object and probe in place.
    pub fn write_arrays(&mut self, obj: &[C32], probe: &[C32]) -> Result<(), EngineError> {
        self.obj.copy_from_host(&self.stream, obj)?;
        self.probe.copy_from_host(&self.stream, probe)?;
        Ok(())
    }

    /// Lends every buffer to a cycle enqueued outside a graph.
    pub fn lend(&mut self) -> Bufs<'_> {
        Bufs {
            kernels: &self.kernels,
            rpie: &self.rpie,
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
            plans: Plans(&mut self.plans),
            scratch: &mut self.scratch,
        }
    }

    /// Enqueues one `epoch` cycle: replays its graph, captured (on a side stream, so nothing
    /// runs) when the cached one is for another epoch, or launches it eagerly without graphs.
    pub fn steady(&mut self, epoch: Epoch) -> Result<(), GpuError> {
        if !self.use_graph {
            let ctx = self.context();
            return self.lend().epoch_cycle(&ctx, epoch);
        }
        let run = match self.run.take() {
            Some(run) if run.epoch == epoch => run,
            _ => {
                let side = self.kernels.context().new_stream()?;
                let mut bufs = self.lend();
                let op = ptyrs_gpu::record(|ctx| bufs.epoch_cycle(ctx, epoch));
                EpochRun {
                    epoch,
                    graph: CudaGraph::capture(side, op)?,
                }
            }
        };
        let ctx = self.context();
        // SAFETY: `ctx` holds the replay's leases on every captured buffer until the stream
        // has run it, so no other stream can touch them meanwhile.
        let launched = unsafe { run.graph.launch().execute(&ctx) };
        self.run = Some(run);
        Ok(launched?)
    }
}
