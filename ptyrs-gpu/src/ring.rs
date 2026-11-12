//! The LLK ring (`ptyrs_kernels::ring`): LLK records written on the device by the epochs that
//! take them, read on the host after those epochs, with no host sync in between.
//!
//! A record is one LLK of the whole scan, written through one [`RecordWriter`]: per stack,
//! [`Kernels::llk`] into a staging buffer and [`RecordWriter::push`] of it, then
//! [`RecordWriter::end`], the only advance of the cursor, which consumes the writer and
//! returns the [`Ended`] the host count takes. These are ordinary launches, legal in a
//! capture. The host counts each record with [`LlkRing::enqueued`] once its launches, or the
//! graph holding them, are enqueued, and [`LlkRing::flush`] returns the totals of every
//! record counted since the last flush, in order, each from [`LlkSums::from_partials`] over
//! its frames. Every frame of record `k` must carry stamp `k`: a frame not pushed, a record
//! ended but not counted or counted but not ended, or a slot overwritten before its copy, is
//! a [`RingError::Stamp`], not a value.

use std::collections::VecDeque;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer, DriverError, Event, LaunchConfig, PinnedHostBuffer};
use ptyrs_kernels::ring::{RingRecord, ring_kernels};

use crate::{
    Error, Kernels, LlkPartial, LlkSums, StackCtl, StackRef, ViewDims, check_len, dptr, lease,
};

/// The ring's device half, what a cycle launches on: `nslot` records of `nframe` stamped
/// partials and the cursor numbering the current record. Only [`LlkRing::new`] builds one,
/// and only a [`RecordWriter`] writes it.
pub struct RingDevice {
    base: Kernels,
    module: ring_kernels::LoadedModule,
    records: DeviceBuffer<RingRecord>,
    cursor: DeviceBuffer<u64>,
    nframe: u32,
    nslot: u32,
}

impl RingDevice {
    /// Starts writing the current record.
    pub fn record(&mut self) -> RecordWriter<'_> {
        RecordWriter { dev: self }
    }

    fn push(
        &mut self,
        ctx: &ExecutionContext,
        dims: &ViewDims,
        stack: StackRef<'_>,
        partials: &DeviceBuffer<LlkPartial>,
    ) -> Result<(), Error> {
        let g = dims.geom(stack)?;
        if g.nframe != self.nframe {
            return Err(Error::Shape(format!(
                "a stack of {} frames, the ring holds {}",
                g.nframe, self.nframe
            )));
        }
        check_len("partials", partials, g.nf_max as usize)?;
        let config = LaunchConfig::for_num_elems(g.nf_max);
        let reads = [partials.storage(), self.cursor.storage(), stack.storage()];
        self.base
            .launch(ctx, &reads, &[self.records.storage()], |stream| {
                // SAFETY: `partials` checked above; `records` holds `nslot * nframe` and
                // `cursor` one entry since `LlkRing::new`.
                unsafe {
                    self.module.llk_push(
                        stream,
                        config,
                        dptr(partials),
                        dptr(&self.records),
                        dptr(&self.cursor),
                        stack.ctl() as *const StackCtl,
                        g,
                        self.nslot,
                    )
                }
            })
    }

    fn advance(&mut self, ctx: &ExecutionContext) -> Result<(), Error> {
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        self.base
            .launch(ctx, &[], &[self.cursor.storage()], |stream| {
                // SAFETY: `cursor` holds one entry.
                unsafe { self.module.ring_advance(stream, config, dptr(&self.cursor)) }
            })
    }
}

/// The current record being written: its pushes, then its one advance by [`Self::end`]. The
/// writer holds the ring for the whole record, so no other record interleaves with it.
#[must_use = "a record ends only through RecordWriter::end"]
pub struct RecordWriter<'r> {
    dev: &'r mut RingDevice,
}

impl RecordWriter<'_> {
    /// Copies the stack's partials, as [`Kernels::llk`] writes them (`nf_max` records), into
    /// the record, stamped with it.
    pub fn push(
        &mut self,
        ctx: &ExecutionContext,
        dims: &ViewDims,
        stack: StackRef<'_>,
        partials: &DeviceBuffer<LlkPartial>,
    ) -> Result<(), Error> {
        self.dev.push(ctx, dims, stack, partials)
    }

    /// Ends the record: the cursor numbers the next.
    pub fn end(self, ctx: &ExecutionContext) -> Result<Ended, Error> {
        self.dev.advance(ctx)?;
        Ok(Ended(()))
    }
}

/// What [`RecordWriter::end`] returns: the launches that wrote it end one record each time
/// they run, eagerly or as a graph captured from them, and [`LlkRing::enqueued`] counts one
/// record per run.
#[derive(Debug)]
#[must_use = "a record ended is counted with LlkRing::enqueued"]
pub struct Ended(());

/// An LLK ring of `nslot` records: the [`RingDevice`] the epochs write and its pinned host
/// mirror, read in record order. The copy of record `k` is enqueued at most
/// `max(nslot / 2, 1)` records after it, and the host waits for it only when it would
/// otherwise overwrite `k`'s host slot, `nslot` records on, or on [`Self::flush`]; until
/// then epochs keep enqueueing while earlier records download. `T` is the caller's tag of a
/// record.
pub struct LlkRing<T> {
    dev: RingDevice,
    /// Slot `k % nslot` holds record `k` once its copy has completed.
    host: PinnedHostBuffer<RingRecord>,
    /// Records counted, copied (the copy enqueued) and read: `read <= copied <= next`, and
    /// `next - copied < max(nslot / 2, 1)` between calls, so the slot the next record
    /// overwrites has been copied.
    next: u64,
    copied: u64,
    read: u64,
    /// The tags of records `read .. next`.
    tags: VecDeque<T>,
    /// The copies not yet read, oldest first: each one's end record and the event after it;
    /// their ranges tile `read .. copied`.
    copies: VecDeque<(u64, Event)>,
    /// Records read since the last flush.
    done: Vec<(T, LlkSums)>,
}

impl<T> LlkRing<T> {
    /// A ring of `nslot >= 1` records for a scan of `nframe` frames, on the kernels
    /// `kernels` loaded, initialised on `stream`.
    pub fn new(
        kernels: &Kernels,
        stream: &CudaStream,
        nframe: u32,
        nslot: u32,
    ) -> Result<Self, Error> {
        if nframe == 0 || nslot == 0 {
            return Err(Error::Shape(format!(
                "a ring of {nslot} records of {nframe} frames"
            )));
        }
        if stream.context().as_ref() != kernels.context().as_ref() {
            return Err(Error::ContextMismatch);
        }
        let len = nslot as usize * nframe as usize;
        // A stamp no record has, so a slot never pushed reads as not written.
        let unwritten = RingRecord {
            llk: LlkPartial::default(),
            seq: u64::MAX,
        };
        let module = ring_kernels::from_module(kernels.module.as_cuda_module().clone())?;
        Ok(Self {
            dev: RingDevice {
                base: kernels.clone(),
                module,
                records: DeviceBuffer::from_host(stream, &vec![unwritten; len])?,
                cursor: DeviceBuffer::from_host(stream, &[0u64])?,
                nframe,
                nslot,
            },
            host: PinnedHostBuffer::zeroed(kernels.context(), len)?,
            next: 0,
            copied: 0,
            read: 0,
            tags: VecDeque::new(),
            copies: VecDeque::new(),
            done: Vec::new(),
        })
    }

    /// The device half, for the cycles that write records.
    pub fn device(&mut self) -> &mut RingDevice {
        &mut self.dev
    }

    /// Counts the next record under `tag`, once per run of the launches that returned
    /// `ended`, after that run is enqueued on `ctx`'s stream; enqueues the copy of the records
    /// not yet copied when they make a batch.
    pub fn enqueued(
        &mut self,
        ctx: &ExecutionContext,
        _ended: &Ended,
        tag: T,
    ) -> Result<(), RingError> {
        self.next += 1;
        self.tags.push_back(tag);
        if self.next - self.copied >= self.batch() {
            self.copy_out(ctx)?;
        }
        Ok(())
    }

    /// The tags and totals of every record counted since the last flush, in order: enqueues
    /// the copy of the records not yet copied and waits for every copy.
    pub fn flush(&mut self, ctx: &ExecutionContext) -> Result<Vec<(T, LlkSums)>, RingError> {
        if self.copied < self.next {
            self.copy_out(ctx)?;
        }
        self.read_until(self.next)?;
        Ok(std::mem::take(&mut self.done))
    }

    fn batch(&self) -> u64 {
        (self.dev.nslot as u64 / 2).max(1)
    }

    /// Enqueues the copy of records `copied .. next` to their host slots, after reading the
    /// records those slots hold.
    fn copy_out(&mut self, ctx: &ExecutionContext) -> Result<(), RingError> {
        let (start, end) = (self.copied, self.next);
        let nslot = self.dev.nslot as u64;
        self.read_until(end.saturating_sub(nslot))?;
        if ctx.context().as_ref() != self.dev.base.context().as_ref() {
            return Err(Error::ContextMismatch.into());
        }
        lease(ctx, &[self.dev.records.storage()], &[])?;
        ctx.context().bind_to_thread()?;
        let stream = ctx.get_cuda_stream();
        let nframe = self.dev.nframe as usize;
        let size = size_of::<RingRecord>();
        let s0 = (start % nslot) as usize;
        let count = (end - start) as usize;
        let first = count.min(nslot as usize - s0);
        for (slot, n) in [(s0, first), (0, count - first)] {
            if n == 0 {
                continue;
            }
            let at = slot * nframe;
            // SAFETY: slots `slot .. slot + n` lie in both buffers; the host ones hold no
            // record left to read (read_until above), and none is read before this copy's
            // event completes.
            unsafe {
                cuda_core::memory::memcpy_dtoh_async(
                    self.host.as_mut_ptr().add(at),
                    self.dev.records.cu_deviceptr() + (at * size) as u64,
                    n * nframe * size,
                    stream.cu_stream(),
                )?
            };
        }
        self.copies.push_back((end, stream.record_event(None)?));
        self.copied = end;
        Ok(())
    }

    /// Reads every copy holding a record before `k`, oldest first; `k <= copied`. A stamp
    /// error drops every record read since the last flush: a record whose slot a later one
    /// of the same copy overwrote carries its own stamp, and only the later one's shows it.
    fn read_until(&mut self, k: u64) -> Result<(), RingError> {
        while self.read < k {
            let (end, event) = self.copies.front().expect("copies tile read .. copied");
            event.synchronize()?;
            let (first, end) = (self.read, *end);
            self.copies.pop_front();
            self.read = end;
            let tags: Vec<T> = self.tags.drain(..(end - first) as usize).collect();
            for (k, tag) in (first..end).zip(tags) {
                match self.parse(k) {
                    Ok(sums) => self.done.push((tag, sums)),
                    Err(e) => {
                        self.done.clear();
                        return Err(e);
                    }
                }
            }
        }
        Ok(())
    }

    /// Record `k`'s totals from its host slot, whose copy has completed.
    fn parse(&self, k: u64) -> Result<LlkSums, RingError> {
        let nframe = self.dev.nframe as usize;
        let slot = (k % self.dev.nslot as u64) as usize;
        // SAFETY: the slot lies in `host`, its copy has completed, and no later copy into it
        // is enqueued before it is read.
        let recs =
            unsafe { std::slice::from_raw_parts(self.host.as_ptr().add(slot * nframe), nframe) };
        let mut partials = Vec::with_capacity(nframe);
        for (frame, r) in recs.iter().enumerate() {
            if r.seq != k {
                return Err(RingError::Stamp {
                    record: k,
                    frame,
                    found: r.seq,
                });
            }
            partials.push(r.llk);
        }
        Ok(LlkSums::from_partials(&partials))
    }
}

impl<T> Drop for LlkRing<T> {
    fn drop(&mut self) {
        // The copies write `host`, freed after this.
        for (_, event) in &self.copies {
            let _ = event.synchronize();
        }
    }
}

#[derive(Debug)]
pub enum RingError {
    Gpu(Error),
    /// Frame `frame` of record `record` holds stamp `found`, not `record`: a frame not pushed
    /// for it, a record ended but not counted or counted but not ended, or a slot overwritten
    /// before its copy.
    Stamp {
        record: u64,
        frame: usize,
        found: u64,
    },
}

impl std::fmt::Display for RingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gpu(e) => write!(f, "{e}"),
            Self::Stamp {
                record,
                frame,
                found,
            } => write!(
                f,
                "LLK record {record}, frame {frame}: stamp {found}, so not written for it"
            ),
        }
    }
}

impl std::error::Error for RingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gpu(e) => Some(e),
            Self::Stamp { .. } => None,
        }
    }
}

impl From<Error> for RingError {
    fn from(e: Error) -> Self {
        Self::Gpu(e)
    }
}

impl From<DriverError> for RingError {
    fn from(e: DriverError) -> Self {
        Self::Gpu(e.into())
    }
}
