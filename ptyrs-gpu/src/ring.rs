//! The LLK ring (`ptyrs_kernels::ring`): LLK records written on the device by the epochs that
//! take them, read on the host after those epochs, with no host sync in between.
//!
//! A record is one LLK of frames `0 .. n` of the scan, written through one [`RecordWriter`]:
//! per stack, [`Kernels::llk`] into a staging buffer and [`RecordWriter::push`] of it, then
//! [`RecordWriter::end`], the only advance of the cursor, which consumes the writer and
//! returns the [`Ended`] the host count takes. These are ordinary launches, legal in a
//! capture. The host counts each record with [`LlkRing::enqueued`], giving its `n`, once its
//! launches, or the graph holding them, are enqueued. [`LlkRing::flush`] waits for and returns
//! the [`LlkTotals`] of every record counted since the last read, in order, and
//! [`LlkRing::poll`], without waiting, those of them whose copies have completed; each from
//! [`LlkSums::from_partials`] over its frames with their unmasked pixels, which
//! [`RingDevice::count`] counts once per frame. The same count adds each frame to the frame
//! counts of its unmasked pixels, [`RingDevice::pixel_frames`], in integer adds, so they come
//! out the same in any frame order. Every frame of record `k` must carry stamp
//! `k`: a frame not pushed, a record ended but not counted or counted but not ended, or a
//! slot overwritten before its copy, is a [`RingError::Stamp`], not a value; a frame pushed
//! before it was counted is a [`RingError::Uncounted`].

use std::collections::VecDeque;
use std::ops::Range;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer, DriverError, Event, LaunchConfig, PinnedHostBuffer};
use ptyrs_kernels::pixel::PIXEL_THREADS;
use ptyrs_kernels::ring::{COUNT_FRAMES, RingRecord, UNCOUNTED, ring_kernels};

use crate::{
    Error, Kernels, LlkPartial, LlkSums, StackCtl, StackGeom, StackRef, ViewDims, check_len, dptr,
    lease,
};

/// The ring's device half, what a cycle launches on: `nslot` records of `nframe` stamped
/// partials, the cursor numbering the current record, each frame's unmasked pixels and each
/// detector pixel's unmasked frames. Only [`LlkRing::new`] builds one, and only a
/// [`RecordWriter`] writes its records.
pub struct RingDevice {
    base: Kernels,
    module: ring_kernels::LoadedModule,
    records: DeviceBuffer<RingRecord>,
    cursor: DeviceBuffer<u64>,
    /// Frame `f`'s unmasked pixels once [`Self::count`] ran on it, [`UNCOUNTED`] before.
    nobs: DeviceBuffer<u32>,
    /// Zero between counts: within one, the chunks of frame `f` done and their pixels so far.
    partial: DeviceBuffer<u64>,
    /// Detector pixel `p`'s unmasked frames among those [`Self::count`] ran on.
    pixel_frames: DeviceBuffer<u32>,
    nframe: u32,
    /// The detector's `[ny, nx]`.
    frame: [u32; 2],
    nslot: u32,
}

impl RingDevice {
    /// Starts writing the current record.
    pub fn record(&mut self) -> RecordWriter<'_> {
        RecordWriter { dev: self }
    }

    /// Counts the unmasked pixels of frames `frames` in `counts` (`nframe` frames of the
    /// scan), which every later push of those frames carries, and adds each frame to
    /// [`Self::pixel_frames`] of its unmasked pixels; after the frames' counts are in place
    /// and before their first push. A frame counted before is skipped, so counting it again
    /// changes neither. One launch, of every pixel chunk of every frame of the range.
    pub fn count(
        &mut self,
        ctx: &ExecutionContext,
        frames: Range<u32>,
        counts: &DeviceBuffer<u16>,
    ) -> Result<(), Error> {
        // `LlkRing::new` checked that `n + PIXEL_THREADS` fits.
        let n = self.frame[0] * self.frame[1];
        let nf = (frames.end.checked_sub(frames.start))
            .filter(|_| frames.end <= self.nframe)
            .ok_or_else(|| {
                Error::Shape(format!(
                    "frames {frames:?} of a ring of {} frames",
                    self.nframe
                ))
            })?;
        check_len("counts", counts, self.nframe as usize * n as usize)?;
        if nf == 0 {
            return Ok(());
        }
        let blocks = (n.div_ceil(PIXEL_THREADS))
            .checked_mul(nf.div_ceil(COUNT_FRAMES))
            .filter(|&b| b <= i32::MAX as u32)
            .ok_or_else(|| Error::Shape(format!("{nf} frames of {n} pixels in one launch")))?;
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let reads = [counts.storage()];
        let writes = [
            self.nobs.storage(),
            self.partial.storage(),
            self.pixel_frames.storage(),
        ];
        self.base.launch(ctx, &reads, &writes, |stream| {
            // SAFETY: `counts` checked above, `frames` within `nframe`, `n + PIXEL_THREADS`
            // within u32; `nobs` and `partial` hold `nframe` and `pixel_frames` `n` since
            // `LlkRing::new`, and every count leaves `partial` zero.
            unsafe {
                self.module.ring_count(
                    stream,
                    config,
                    dptr(counts),
                    dptr(&self.nobs),
                    dptr(&self.partial),
                    dptr(&self.pixel_frames),
                    frames.start,
                    nf,
                    n,
                )
            }
        })
    }

    /// Detector pixel `p`'s unmasked frames, `ny * nx` in the detector order of the counts,
    /// over the frames [`Self::count`] has counted: the `nobs` of the background update.
    pub fn pixel_frames(&self) -> &DeviceBuffer<u32> {
        &self.pixel_frames
    }

    /// The stack's geometry, checked against the ring's frames and detector.
    fn geom(&self, dims: &ViewDims, stack: StackRef<'_>) -> Result<StackGeom, Error> {
        let g = dims.geom(stack)?;
        if (g.nframe, dims.frame) != (self.nframe, self.frame) {
            return Err(Error::Shape(format!(
                "a stack of {} frames of {:?} pixels, the ring holds {} of {:?}",
                g.nframe, dims.frame, self.nframe, self.frame
            )));
        }
        Ok(g)
    }

    fn push(
        &mut self,
        ctx: &ExecutionContext,
        dims: &ViewDims,
        stack: StackRef<'_>,
        partials: &DeviceBuffer<LlkPartial>,
    ) -> Result<(), Error> {
        let g = self.geom(dims, stack)?;
        check_len("partials", partials, g.nf_max as usize)?;
        let config = LaunchConfig::for_num_elems(g.nf_max);
        let reads = [
            partials.storage(),
            self.nobs.storage(),
            self.cursor.storage(),
            stack.storage(),
        ];
        self.base
            .launch(ctx, &reads, &[self.records.storage()], |stream| {
                // SAFETY: `partials` checked above; `records` holds `nslot * nframe`, `nobs`
                // `nframe` and `cursor` one entry since `LlkRing::new`.
                unsafe {
                    self.module.llk_push(
                        stream,
                        config,
                        dptr(partials),
                        dptr(&self.nobs),
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
    /// the record with each frame's unmasked pixels, stamped with it.
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

/// One record read back: the totals of its frames `0 .. nframe` and their unmasked pixels,
/// so `sums / nb_obs` is a per-pixel LLK that compares across records of different `nframe`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LlkTotals {
    pub sums: LlkSums,
    pub nframe: u32,
    pub nb_obs: usize,
}

/// What [`RecordWriter::end`] returns: the launches that wrote it end one record each time
/// they run, eagerly or as a graph captured from them, and [`LlkRing::enqueued`] counts one
/// record per run.
#[derive(Debug)]
#[must_use = "a record ended is counted with LlkRing::enqueued"]
pub struct Ended(());

/// An LLK ring of `nslot` records: the [`RingDevice`] the epochs write and its pinned host
/// mirror, read in record order. The copy of record `k` is enqueued at most
/// `max(nslot / 2, 1)` records after it, or by an earlier [`Self::poll`], and the host waits
/// for it only when it would otherwise overwrite `k`'s host slot, `nslot` records on, or on
/// [`Self::flush`]; until then epochs keep enqueueing while earlier records download. `T` is
/// the caller's tag of a record.
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
    /// The tags and frame counts of records `read .. next`.
    tags: VecDeque<(T, u32)>,
    /// The copies not yet read, oldest first: each one's end record and the event after it;
    /// their ranges tile `read .. copied`.
    copies: VecDeque<(u64, Event)>,
    /// Records read since the last flush or poll.
    done: Vec<(T, LlkTotals)>,
}

impl<T> LlkRing<T> {
    /// A ring of `nslot >= 1` records for a scan of `dims.nframe` frames of `dims.frame`
    /// pixels, on the kernels `kernels` loaded, initialised on `stream`.
    pub fn new(
        kernels: &Kernels,
        stream: &CudaStream,
        dims: &ViewDims,
        nslot: u32,
    ) -> Result<Self, Error> {
        let nframe = dims.nframe;
        // `RingDevice::count` takes pixel indices up to `n + PIXEL_THREADS` in u32.
        let n_fits = u32::try_from(dims.n() + PIXEL_THREADS as usize).is_ok();
        if nframe == 0 || nslot == 0 || dims.n() == 0 || !n_fits {
            return Err(Error::Shape(format!(
                "a ring of {nslot} records of {nframe} frames of {:?} pixels",
                dims.frame
            )));
        }
        if stream.context().as_ref() != kernels.context().as_ref() {
            return Err(Error::ContextMismatch);
        }
        let len = nslot as usize * nframe as usize;
        // A stamp no record has, so a slot never pushed reads as not written.
        let unwritten = RingRecord {
            seq: u64::MAX,
            ..RingRecord::default()
        };
        let module = ring_kernels::from_module(kernels.module.as_cuda_module().clone())?;
        Ok(Self {
            dev: RingDevice {
                base: kernels.clone(),
                module,
                records: DeviceBuffer::from_host(stream, &vec![unwritten; len])?,
                cursor: DeviceBuffer::from_host(stream, &[0u64])?,
                nobs: DeviceBuffer::from_host(stream, &vec![UNCOUNTED; nframe as usize])?,
                partial: DeviceBuffer::zeroed(stream, nframe as usize)?,
                pixel_frames: DeviceBuffer::zeroed(stream, dims.n())?,
                nframe,
                frame: dims.frame,
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

    /// Counts the next record, of frames `0 .. nframe`, under `tag`, once per run of the
    /// launches that returned `ended`, after that run is enqueued on `ctx`'s stream; enqueues
    /// the copy of the records not yet copied when they make a batch.
    pub fn enqueued(
        &mut self,
        ctx: &ExecutionContext,
        _ended: &Ended,
        tag: T,
        nframe: u32,
    ) -> Result<(), RingError> {
        if nframe > self.dev.nframe {
            return Err(Error::Shape(format!(
                "a record of {nframe} frames, the ring holds {}",
                self.dev.nframe
            ))
            .into());
        }
        self.next += 1;
        self.tags.push_back((tag, nframe));
        if self.next - self.copied >= self.batch() {
            self.copy_out(ctx)?;
        }
        Ok(())
    }

    /// The tags and totals of every record counted since the last flush or poll, in order:
    /// enqueues the copy of the records not yet copied and waits for every copy.
    pub fn flush(&mut self, ctx: &ExecutionContext) -> Result<Vec<(T, LlkTotals)>, RingError> {
        if self.copied < self.next {
            self.copy_out(ctx)?;
        }
        self.read_until(self.next)?;
        Ok(std::mem::take(&mut self.done))
    }

    /// The tags and totals of the records counted since the last flush or poll whose copies
    /// have completed, in order, without waiting for any copy. Then enqueues the copy of the
    /// records not yet copied if their host slots hold no record left to read, so that a later
    /// poll returns them however few records follow.
    pub fn poll(&mut self, ctx: &ExecutionContext) -> Result<Vec<(T, LlkTotals)>, RingError> {
        while let Some((_, event)) = self.copies.front() {
            if !event.query()? {
                break;
            }
            self.read_front()?;
        }
        if self.copied < self.next && self.next - self.read <= self.dev.nslot as u64 {
            self.copy_out(ctx)?;
        }
        Ok(std::mem::take(&mut self.done))
    }

    fn batch(&self) -> u64 {
        (self.dev.nslot as u64 / 2).max(1)
    }

    /// Enqueues the copy of records `copied .. next`, each of its frames `0 .. n`, to their host
    /// slots, after reading the records those slots hold.
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
        let size = size_of::<RingRecord>();
        let pending = self.tags.range((start - self.read) as usize..);
        for (k, &(_, n)) in (start..end).zip(pending) {
            if n == 0 {
                continue;
            }
            let at = (k % nslot) as usize * self.dev.nframe as usize;
            // SAFETY: frames `0 .. n <= nframe` of slot `k % nslot` lie in both buffers; the
            // host slot holds no record left to read (read_until above), and it is not read
            // before this copy's event completes.
            unsafe {
                cuda_core::memory::memcpy_dtoh_async(
                    self.host.as_mut_ptr().add(at),
                    self.dev.records.cu_deviceptr() + (at * size) as u64,
                    n as usize * size,
                    stream.cu_stream(),
                )?
            };
        }
        self.copies.push_back((end, stream.record_event(None)?));
        self.copied = end;
        Ok(())
    }

    /// Waits for and reads every copy holding a record before `k`, oldest first; `k <= copied`.
    fn read_until(&mut self, k: u64) -> Result<(), RingError> {
        while self.read < k {
            let (_, event) = self.copies.front().expect("copies tile read .. copied");
            event.synchronize()?;
            self.read_front()?;
        }
        Ok(())
    }

    /// Reads the oldest copy not read, whose event has completed. A stamp error drops every
    /// record read since the last flush or poll: a record whose slot a later one of the same
    /// copy overwrote carries its own stamp, and only the later one's shows it.
    fn read_front(&mut self) -> Result<(), RingError> {
        let (end, _) = self.copies.pop_front().expect("a copy not read");
        let first = self.read;
        self.read = end;
        let tags: Vec<(T, u32)> = self.tags.drain(..(end - first) as usize).collect();
        for (k, (tag, nframe)) in (first..end).zip(tags) {
            match self.parse(k, nframe) {
                Ok(totals) => self.done.push((tag, totals)),
                Err(e) => {
                    self.done.clear();
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    /// Record `k`'s totals over frames `0 .. nframe` from its host slot, whose copy has
    /// completed.
    fn parse(&self, k: u64, nframe: u32) -> Result<LlkTotals, RingError> {
        let at = (k % self.dev.nslot as u64) as usize * self.dev.nframe as usize;
        // SAFETY: the slot lies in `host`, its copy has completed, and no later copy into it
        // is enqueued before it is read; `nframe` is at most the slot's frames.
        let recs =
            unsafe { std::slice::from_raw_parts(self.host.as_ptr().add(at), nframe as usize) };
        let mut partials = Vec::with_capacity(recs.len());
        let mut nb_obs = 0;
        for (frame, r) in recs.iter().enumerate() {
            if r.seq != k {
                return Err(RingError::Stamp {
                    record: k,
                    frame,
                    found: r.seq,
                });
            }
            if r.nobs == UNCOUNTED {
                return Err(RingError::Uncounted { record: k, frame });
            }
            partials.push(r.llk);
            nb_obs += r.nobs as usize;
        }
        Ok(LlkTotals {
            sums: LlkSums::from_partials(&partials),
            nframe,
            nb_obs,
        })
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
    /// Frame `frame` of record `record` was pushed before [`RingDevice::count`] counted it.
    Uncounted {
        record: u64,
        frame: usize,
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
            Self::Uncounted { record, frame } => write!(
                f,
                "LLK record {record}, frame {frame}: pushed before its unmasked pixels were counted"
            ),
        }
    }
}

impl std::error::Error for RingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gpu(e) => Some(e),
            Self::Stamp { .. } | Self::Uncounted { .. } => None,
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

#[cfg(test)]
mod tests {
    use cuda_core::CudaContext;

    use super::*;

    /// Each frame's unmasked pixels and each pixel's unmasked frames come out as on the host
    /// in any layout of ranges, over several pixel chunks, the last one part full or not, and
    /// a last frame group part full; frame 1 is all masked and frame 2 not at all. A frame
    /// counted twice adds once, and every count leaves `partial` zero.
    #[test]
    fn range_counts_match_the_host_in_any_layout() {
        let cuda = CudaContext::new(0).expect("CUDA device 0");
        let kernels = Kernels::load(&cuda).expect("load ptyrs-gpu kernels");
        let stream = cuda.new_stream().unwrap();
        let ctx = ExecutionContext::new(stream.clone());
        for (frame, nframe) in [([100u32, 100u32], 61u32), ([256, 256], 19)] {
            let dims = ViewDims {
                nframe,
                nmode: 1,
                nprobe: 1,
                nobj: 1,
                frame,
                obj: frame,
                quad_phase: 0.0,
            };
            let n = dims.n();
            let mut st = 0x1234_5678_9abc_def1u64 ^ n as u64;
            let mut counts = vec![0u16; nframe as usize * n];
            for (f, frame) in counts.chunks_mut(n).enumerate() {
                let rate = (f * 37 % 100) as u64;
                for c in frame {
                    st ^= st << 13;
                    st ^= st >> 7;
                    st ^= st << 17;
                    let masked = match f {
                        1 => true,
                        2 => false,
                        _ => st % 100 < rate,
                    };
                    *c = if masked { crate::MASKED } else { 1 };
                }
            }
            let unmasked = |c: &u16| *c != crate::MASKED;
            let want_nobs: Vec<u32> = counts
                .chunks(n)
                .map(|f| f.iter().filter(|c| unmasked(c)).count() as u32)
                .collect();
            let want_pixel_frames: Vec<u32> = (0..n)
                .map(|p| {
                    (0..nframe as usize)
                        .filter(|f| unmasked(&counts[f * n + p]))
                        .count() as u32
                })
                .collect();
            let dev_counts = DeviceBuffer::from_host(&stream, &counts).unwrap();
            let layouts: [Vec<Range<u32>>; 3] = [
                std::iter::once(0..nframe).collect(),
                vec![0..5, 3..nframe / 2, nframe / 2..nframe, 0..nframe],
                (0..nframe).rev().map(|f| f..f + 1).collect(),
            ];
            for ranges in layouts {
                let mut ring = LlkRing::<()>::new(&kernels, &stream, &dims, 1).unwrap();
                for r in &ranges {
                    ring.device().count(&ctx, r.clone(), &dev_counts).unwrap();
                }
                let dev = &ring.dev;
                let at = format!(
                    "frame {frame:?}, ranges {:?}",
                    &ranges[..ranges.len().min(4)]
                );
                assert_eq!(dev.nobs.to_host_vec(&stream).unwrap(), want_nobs, "{at}");
                let got = dev.pixel_frames.to_host_vec(&stream).unwrap();
                assert_eq!(got, want_pixel_frames, "{at}");
                let partial = dev.partial.to_host_vec(&stream).unwrap();
                assert!(partial.iter().all(|&x| x == 0), "{at}");
            }
        }
    }

    /// A record of 4 frames copies those into its host slot and leaves the slot's other 6 as
    /// they were.
    #[test]
    fn a_record_copies_only_its_frames() {
        let cuda = CudaContext::new(0).expect("CUDA device 0");
        let kernels = Kernels::load(&cuda).expect("load ptyrs-gpu kernels");
        let stream = cuda.new_stream().unwrap();
        let ctx = ExecutionContext::new(stream.clone());
        let dims = ViewDims {
            nframe: 10,
            nmode: 1,
            nprobe: 1,
            nobj: 1,
            frame: [8, 8],
            obj: [8, 8],
            quad_phase: 0.0,
        };
        let mut ring = LlkRing::new(&kernels, &stream, &dims, 1).unwrap();
        stream.synchronize().unwrap();
        let old = RingRecord {
            seq: 7,
            ..RingRecord::default()
        };
        ring.host.as_mut_slice().fill(old);
        ring.enqueued(&ctx, &Ended(()), (), 4).unwrap();
        stream.synchronize().unwrap();
        let seqs: Vec<u64> = ring.host.as_slice().iter().map(|r| r.seq).collect();
        assert_eq!(
            seqs[..4],
            [u64::MAX; 4],
            "copied from the unwritten device slot"
        );
        assert_eq!(seqs[4..], [7; 6], "left as they were");
    }
}
