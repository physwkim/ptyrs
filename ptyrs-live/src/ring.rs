//! Pinned host ring of cropped frames and the thread that streams it into
//! the device store.
//!
//! Sequence numbers are handed out by [`Ring::reserve`]; a slot is written
//! only by the holder of its reservation and becomes visible to the uploader
//! when `done[slot] == seq + 1`. The uploader is the only writer of
//! `released`, and advances it only after the event recorded behind that
//! slot's H2D copy has completed, so a worker never overwrites host memory
//! the copy engine may still be reading.
//!
//! On the device every sequence number has its own place in a
//! [`DeviceStore`], so an uploaded frame is never overwritten (invariant 3)
//! and its address is fixed and backed before the first frame (invariant 9).

use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicU64, Ordering};
use std::time::Duration;

use crossbeam_channel::Sender;
use cuda_bindings::CUdeviceptr;
use cuda_core::{Context, DeviceBuffer, DriverError, Event, IntoResult, PinnedHostBuffer};

use crate::grow::{GrowBuf, GrowError};
use crate::stats::{Histogram, thread_cpu_ns, wall_ns};

#[derive(Default)]
struct Meta {
    uid: AtomicI32,
    t_rx: AtomicI64,
    t_pub: AtomicI64,
    t_commit: AtomicI64,
}

pub struct Ring {
    slots: usize,
    slot_elems: usize,
    frames: PinnedHostBuffer<u16>,
    pos: PinnedHostBuffer<f64>,
    frames_ptr: *mut u16,
    pos_ptr: *mut f64,
    next: AtomicU64,
    released: AtomicU64,
    done: Box<[AtomicU64]>,
    meta: Box<[Meta]>,
    stalls: AtomicU64,
}

// SAFETY: the raw pointers address the pinned buffers this struct owns. A
// slot is written only between `reserve` and `commit` by the one holder of
// that sequence number, and read by the uploader only after observing the
// commit (Acquire on `done`) and before it advances `released` (Release),
// which is what lets the next reservation of the slot proceed.
unsafe impl Send for Ring {}
unsafe impl Sync for Ring {}

pub struct Slot<'a> {
    ring: &'a Ring,
    seq: u64,
}

impl Slot<'_> {
    pub fn seq(&self) -> u64 {
        self.seq
    }

    pub fn frame(&mut self) -> &mut [u16] {
        let r = self.ring;
        let i = (self.seq % r.slots as u64) as usize;
        // SAFETY: this reservation is the only writer of slot `i` (see `Ring`).
        unsafe { std::slice::from_raw_parts_mut(r.frames_ptr.add(i * r.slot_elems), r.slot_elems) }
    }

    pub fn commit(self, uid: i32, x: f64, y: f64, t_rx: i64, t_pub: i64) {
        let r = self.ring;
        let i = (self.seq % r.slots as u64) as usize;
        // SAFETY: as in `frame`; slot `i` owns positions `2i` and `2i + 1`.
        unsafe {
            *r.pos_ptr.add(2 * i) = x;
            *r.pos_ptr.add(2 * i + 1) = y;
        }
        let m = &r.meta[i];
        m.uid.store(uid, Ordering::Relaxed);
        m.t_rx.store(t_rx, Ordering::Relaxed);
        m.t_pub.store(t_pub, Ordering::Relaxed);
        m.t_commit.store(wall_ns(), Ordering::Relaxed);
        r.done[i].store(self.seq + 1, Ordering::Release);
    }
}

impl Ring {
    pub fn new(ctx: &Arc<Context>, slots: usize, slot_elems: usize) -> Result<Self, DriverError> {
        let mut frames = PinnedHostBuffer::<u16>::zeroed(ctx, slots * slot_elems)?;
        let mut pos = PinnedHostBuffer::<f64>::zeroed(ctx, slots * 2)?;
        let frames_ptr = frames.as_mut_ptr();
        let pos_ptr = pos.as_mut_ptr();
        Ok(Self {
            slots,
            slot_elems,
            frames,
            pos,
            frames_ptr,
            pos_ptr,
            next: AtomicU64::new(0),
            released: AtomicU64::new(0),
            done: (0..slots).map(|_| AtomicU64::new(0)).collect(),
            meta: (0..slots).map(|_| Meta::default()).collect(),
            stalls: AtomicU64::new(0),
        })
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    pub fn slot_elems(&self) -> usize {
        self.slot_elems
    }

    /// Next sequence number; waits while the ring is full.
    pub fn reserve(&self) -> Slot<'_> {
        let seq = self.next.fetch_add(1, Ordering::Relaxed);
        let mut stalled = false;
        while seq >= self.released.load(Ordering::Acquire) + self.slots as u64 {
            if !stalled {
                self.stalls.fetch_add(1, Ordering::Relaxed);
                stalled = true;
            }
            std::thread::sleep(Duration::from_micros(20));
        }
        Slot { ring: self, seq }
    }

    /// Reservations that found the ring full.
    pub fn stalls(&self) -> u64 {
        self.stalls.load(Ordering::Relaxed)
    }

    /// Host copy of the ring, for end-of-run checks once every worker and
    /// the uploader have stopped.
    pub fn host_frames(&self) -> &[u16] {
        self.frames.as_slice()
    }

    pub fn host_positions(&self) -> &[f64] {
        self.pos.as_slice()
    }

    /// `(seq, uid)` last committed to slot `i`, for end-of-run checks.
    pub fn slot(&self, i: usize) -> Option<(u64, i32)> {
        let d = self.done[i].load(Ordering::Acquire);
        (d != 0).then(|| (d - 1, self.meta[i].uid.load(Ordering::Relaxed)))
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        // An uploader that returned an error left its copies out of the slots queued.
        let _ = self.frames.context().synchronize();
    }
}

/// Device side of the input: frame `seq` at `frames_ptr() + seq * frame
/// bytes`, its `(x, y)` as two f64 at `positions_ptr() + seq * 16`, for
/// `seq < max_frames`. Backed whole when made, before the first frame: the
/// driver's physical allocation now and then blocks its caller for over
/// 100 ms (cuMemCreate, about once per 900 MiB handed out on GPU0), and on
/// the uploader that held up every frame behind it.
///
/// Shared through an `Arc`: whoever holds a pointer into the store holds
/// the store, and its drop waits for queued work ([`GrowBuf`]), so no
/// mapping is released under a reader.
pub struct DeviceStore {
    frames: GrowBuf,
    positions: GrowBuf,
    frame_elems: usize,
    max_frames: u64,
    ctx: Arc<Context>,
}

impl DeviceStore {
    pub fn new(
        ctx: &Arc<Context>,
        max_frames: usize,
        frame_elems: usize,
    ) -> Result<Self, GrowError> {
        let backed = |bytes: usize| -> Result<GrowBuf, GrowError> {
            let mut b = GrowBuf::new(ctx, bytes, bytes)?;
            b.map(0..b.capacity())?;
            Ok(b)
        };
        Ok(Self {
            frames: backed(max_frames * frame_elems * 2)?,
            positions: backed(max_frames * 16)?,
            frame_elems,
            max_frames: max_frames as u64,
            ctx: ctx.clone(),
        })
    }

    pub fn frames_ptr(&self) -> CUdeviceptr {
        self.frames.ptr()
    }

    pub fn positions_ptr(&self) -> CUdeviceptr {
        self.positions.ptr()
    }

    pub fn frame_elems(&self) -> usize {
        self.frame_elems
    }

    pub fn max_frames(&self) -> u64 {
        self.max_frames
    }

    /// All `max_frames` frames as one buffer, frame `seq` from element
    /// `seq * frame_elems`, sharing the store's tracker: the counts the
    /// kernels take.
    ///
    /// # Safety
    /// Work through the view may only read frames of batches already sent.
    pub unsafe fn frames_view(&self) -> Result<DeviceBuffer<u16>, DriverError> {
        let n = self.max_frames as usize * self.frame_elems;
        // SAFETY: forwarded to the caller.
        unsafe { self.frames.typed(n) }
    }

    /// Bytes of device memory backing the store.
    pub fn mapped_bytes(&self) -> usize {
        self.frames.mapped_bytes() + self.positions.mapped_bytes()
    }

    /// Where frames `seqs` and their positions go.
    fn dst(&self, seqs: Range<u64>) -> Result<(CUdeviceptr, CUdeviceptr), GrowError> {
        let fb = (self.frame_elems * 2) as u64;
        if seqs.end > self.max_frames {
            return Err(GrowError::Capacity {
                end: (seqs.end * fb) as usize,
                capacity: (self.max_frames * fb) as usize,
            });
        }
        Ok((
            self.frames_ptr() + seqs.start * fb,
            self.positions_ptr() + seqs.start * 16,
        ))
    }

    /// Synchronous copy of frames `seqs` to `out`, for checks.
    pub fn read_frames(&self, seqs: Range<u64>, out: &mut [u16]) -> Result<(), DriverError> {
        let n = (seqs.end - seqs.start) as usize * self.frame_elems;
        assert_eq!(out.len(), n);
        let fb = (self.frame_elems * 2) as u64;
        let range = (seqs.start * fb) as usize..(seqs.end * fb) as usize;
        assert!(self.frames.is_ready(range));
        self.ctx.bind_to_thread()?;
        // SAFETY: the source range is mapped (checked above; mappings are only
        // added while the store lives) and `out` holds its element count.
        unsafe {
            cuda_bindings::cuMemcpyDtoH_v2(
                out.as_mut_ptr().cast(),
                self.frames_ptr() + seqs.start * fb,
                n * 2,
            )
        }
        .result()
    }

    /// Synchronous copy of the positions of frames `0..n`, for checks.
    pub fn read_positions(&self, n: u64) -> Result<Vec<f64>, DriverError> {
        let mut v = vec![0f64; 2 * n as usize];
        assert!(self.positions.is_ready(0..16 * n as usize));
        self.ctx.bind_to_thread()?;
        // SAFETY: as in `read_frames`.
        unsafe {
            cuda_bindings::cuMemcpyDtoH_v2(v.as_mut_ptr().cast(), self.positions_ptr(), v.len() * 8)
        }
        .result()?;
        Ok(v)
    }
}

/// Frames `seq0..seq0 + n` are in the [`DeviceStore`]. Batches are sent in
/// sequence order and without gaps.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameBatch {
    pub seq0: u64,
    pub n: u64,
    /// `(x, y)` of each frame, m, read from the ring before its slots were
    /// released: the host's copy of what the store holds on the device.
    pub positions: Vec<(f64, f64)>,
}

#[derive(Debug, Clone, Copy)]
pub struct UploaderConfig {
    /// Flush as soon as this many contiguous slots are committed.
    pub batch: NonZeroUsize,
    /// Flush a smaller batch once its oldest slot has waited this long.
    pub flush_ns: i64,
    /// Sleep when a pass did nothing.
    pub idle: Duration,
    /// Events, i.e. batches in flight.
    pub events: usize,
}

pub struct UploadReport {
    pub frames: u64,
    pub batches: u64,
    pub bytes: u64,
    pub max_inflight: usize,
    /// Monitor callback → H2D completion seen by the uploader.
    pub rx_to_gpu: Histogram,
    /// Simulator `dataTimeStamp` → H2D completion.
    pub pub_to_gpu: Histogram,
    /// Worker commit → H2D completion (batching + copy + poll).
    pub commit_to_gpu: Histogram,
    pub cpu_ns: i64,
    /// uid of every uploaded frame, by sequence number.
    pub uids: Vec<i32>,
}

/// Run until `stop` is set and everything reserved has landed in `store`;
/// each landed batch is sent to `batches` if given, which should be
/// unbounded: a full channel stalls the uploader.
pub fn run_uploader(
    ring: Arc<Ring>,
    ctx: Arc<Context>,
    store: Arc<DeviceStore>,
    cfg: UploaderConfig,
    stop: Arc<AtomicBool>,
    batches: Option<Sender<FrameBatch>>,
) -> Result<UploadReport, GrowError> {
    ctx.bind_to_thread()?;
    let stream = ctx.new_stream()?;
    let slots = ring.slots;
    let se = ring.slot_elems;
    assert_eq!(store.frame_elems, se);
    let events: Vec<Event> = (0..cfg.events)
        .map(|_| ctx.new_event_with_flags(None))
        .collect::<Result<_, _>>()?;
    let mut free: Vec<usize> = (0..cfg.events).collect();
    let mut inflight: VecDeque<(usize, u64, u64)> = VecDeque::with_capacity(cfg.events);
    let mut rep = UploadReport {
        frames: 0,
        batches: 0,
        bytes: 0,
        max_inflight: 0,
        rx_to_gpu: Histogram::new(200_000),
        pub_to_gpu: Histogram::new(200_000),
        commit_to_gpu: Histogram::new(200_000),
        cpu_ns: 0,
        uids: Vec::new(),
    };
    let cpu0 = thread_cpu_ns();
    let (mut issued, mut frontier) = (0u64, 0u64);
    loop {
        let mut progressed = false;
        while ring.done[(frontier % slots as u64) as usize].load(Ordering::Acquire) == frontier + 1
        {
            frontier += 1;
            progressed = true;
        }
        let stopping = stop.load(Ordering::Acquire);
        while frontier > issued && !free.is_empty() {
            let pending = frontier - issued;
            let s0 = (issued % slots as u64) as usize;
            let waited = wall_ns() - ring.meta[s0].t_commit.load(Ordering::Relaxed);
            if pending < cfg.batch.get() as u64 && !stopping && waited < cfg.flush_ns {
                break;
            }
            let n = (pending as usize).min(cfg.batch.get()).min(slots - s0);
            let (frames_dst, positions_dst) = store.dst(issued..issued + n as u64)?;
            // SAFETY: slots `s0..s0 + n` are committed and not yet released,
            // so no worker writes them until the event below completes; `dst`
            // placed sequence numbers `issued..issued + n` in the store.
            unsafe {
                cuda_core::memcpy_htod_async(
                    frames_dst,
                    ring.frames_ptr.add(s0 * se),
                    n * se,
                    &stream,
                )?;
                cuda_core::memcpy_htod_async(
                    positions_dst,
                    ring.pos_ptr.add(s0 * 2),
                    n * 2,
                    &stream,
                )?;
            }
            let ev = free.pop().expect("checked non-empty");
            events[ev].record(&stream)?;
            inflight.push_back((ev, issued, n as u64));
            rep.max_inflight = rep.max_inflight.max(inflight.len());
            issued += n as u64;
            rep.batches += 1;
            rep.bytes += (n * (se * 2 + 16)) as u64;
            progressed = true;
        }
        while let Some(&(ev, seq0, n)) = inflight.front() {
            if !events[ev].query()? {
                break;
            }
            let t = wall_ns();
            let at = |seq: u64| (seq % slots as u64) as usize;
            let positions = match batches {
                // SAFETY: the slots are committed and not yet released, so
                // no worker writes their positions (see `Ring`).
                Some(_) => (seq0..seq0 + n)
                    .map(|seq| unsafe {
                        let p = ring.pos_ptr.add(2 * at(seq));
                        (*p, *p.add(1))
                    })
                    .collect(),
                None => Vec::new(),
            };
            for seq in seq0..seq0 + n {
                let m = &ring.meta[at(seq)];
                rep.rx_to_gpu.record(t - m.t_rx.load(Ordering::Relaxed));
                rep.pub_to_gpu.record(t - m.t_pub.load(Ordering::Relaxed));
                rep.commit_to_gpu
                    .record(t - m.t_commit.load(Ordering::Relaxed));
                rep.uids.push(m.uid.load(Ordering::Relaxed));
            }
            rep.frames += n;
            ring.released.store(seq0 + n, Ordering::Release);
            if let Some(tx) = &batches {
                // A consumer that has gone away does not stop the input.
                let _ = tx.send(FrameBatch { seq0, n, positions });
            }
            inflight.pop_front();
            free.push(ev);
            progressed = true;
        }
        if stopping
            && inflight.is_empty()
            && issued == frontier
            && frontier == ring.next.load(Ordering::Acquire)
        {
            break;
        }
        if !progressed {
            std::thread::sleep(cfg.idle);
        }
    }
    rep.cpu_ns = thread_cpu_ns() - cpu0;
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> UploaderConfig {
        UploaderConfig {
            batch: NonZeroUsize::new(5).unwrap(),
            flush_ns: 100_000,
            idle: Duration::from_micros(20),
            events: 4,
        }
    }

    fn produce(ring: &Ring, n: usize) {
        for _ in 0..n {
            let mut s = ring.reserve();
            let seq = s.seq();
            for (j, v) in s.frame().iter_mut().enumerate() {
                *v = (seq as usize * 7 + j) as u16;
            }
            s.commit(1000 + seq as i32, seq as f64, -(seq as f64), 0, 0);
        }
    }

    /// The store is backed whole before the first frame and the uploader
    /// maps nothing.
    #[test]
    fn frames_land_at_their_sequence_number_in_a_store_backed_beforehand() {
        let ctx = Context::new(0).expect("GPU 0");
        let (se, total) = (65536, 200u64);
        // 8 slots wrap 25 times.
        let ring = Arc::new(Ring::new(&ctx, 8, se).unwrap());
        let store = Arc::new(DeviceStore::new(&ctx, total as usize, se).unwrap());
        let backed = store.mapped_bytes();
        assert!(backed >= total as usize * (se * 2 + 16));
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = crossbeam_channel::unbounded();
        let up = {
            let (ring, ctx, stop, store) = (ring.clone(), ctx.clone(), stop.clone(), store.clone());
            std::thread::spawn(move || run_uploader(ring, ctx, store, cfg(), stop, Some(tx)))
        };
        // Four producers commit out of order.
        std::thread::scope(|sc| {
            for _ in 0..4 {
                sc.spawn(|| produce(&ring, total as usize / 4));
            }
        });
        stop.store(true, Ordering::Release);
        let rep = up.join().unwrap().unwrap();

        assert_eq!(rep.frames, total);
        assert_eq!(store.mapped_bytes(), backed);
        let mut next = 0;
        for b in rx.iter() {
            assert_eq!(b.seq0, next);
            let want: Vec<(f64, f64)> = (next..next + b.n)
                .map(|s| (s as f64, -(s as f64)))
                .collect();
            assert_eq!(b.positions, want);
            next += b.n;
        }
        assert_eq!(next, total);
        let uids: Vec<i32> = (0..total as i32).map(|s| 1000 + s).collect();
        assert_eq!(rep.uids, uids);
        let mut dev = vec![0u16; total as usize * se];
        store.read_frames(0..total, &mut dev).unwrap();
        for (seq, f) in dev.chunks(se).enumerate() {
            assert!(
                f.iter()
                    .enumerate()
                    .all(|(j, &v)| v == (seq * 7 + j) as u16),
                "frame {seq}"
            );
        }
        let pos = store.read_positions(total).unwrap();
        for seq in 0..total as usize {
            assert_eq!(
                (pos[2 * seq], pos[2 * seq + 1]),
                (seq as f64, -(seq as f64))
            );
        }
    }

    #[test]
    fn a_frame_past_the_store_fails_the_uploader() {
        let ctx = Context::new(0).expect("GPU 0");
        let se = 65536;
        let ring = Arc::new(Ring::new(&ctx, 64, se).unwrap());
        // 32 frames fill the store; the 33rd has no place.
        let store = Arc::new(DeviceStore::new(&ctx, 32, se).unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let up = {
            let (ring, ctx, stop) = (ring.clone(), ctx.clone(), stop.clone());
            std::thread::spawn(move || run_uploader(ring, ctx, store, cfg(), stop, None))
        };
        produce(&ring, 33);
        stop.store(true, Ordering::Release);
        match up.join().unwrap() {
            Err(GrowError::Capacity { end, capacity }) => {
                assert_eq!(capacity, 32 * se * 2);
                assert!(end > capacity);
            }
            Err(e) => panic!("{e}"),
            Ok(r) => panic!("stored {} frames", r.frames),
        }
        // The uploader dropped the store with batches still queued.
        ctx.synchronize().expect("a queued copy faulted");
    }

    /// The uploader's own copy, held behind a gate while the ring and the store
    /// are dropped: the drop must wait for it, not unmap or free under it.
    #[test]
    fn dropping_the_ring_and_store_waits_for_a_queued_copy() {
        let ctx = Context::new(0).expect("GPU 0");
        let se = 65536;
        let ring = Ring::new(&ctx, 4, se).unwrap();
        let store = DeviceStore::new(&ctx, 4, se).unwrap();
        let s = ctx.new_stream().unwrap();
        let mut gate = PinnedHostBuffer::<u32>::zeroed(&ctx, 1).unwrap();
        // SAFETY: the gate and both ranges outlive the copy (the point of the test).
        unsafe {
            cuda_bindings::cuStreamWaitValue32_v2(
                s.cu_stream(),
                gate.as_ptr() as u64,
                1,
                cuda_bindings::CUstreamWaitValue_flags_enum_CU_STREAM_WAIT_VALUE_GEQ,
            )
            .result()
            .unwrap();
            cuda_core::memcpy_htod_async(store.frames_ptr(), ring.frames_ptr, 4 * se, &s).unwrap();
        }
        let dropped = AtomicBool::new(false);
        let early = std::thread::scope(|sc| {
            let dropped = &dropped;
            sc.spawn(move || {
                drop((ring, store));
                dropped.store(true, Ordering::Release);
            });
            std::thread::sleep(Duration::from_millis(200));
            let early = dropped.load(Ordering::Acquire);
            // SAFETY: the gate lives until the end of the test.
            unsafe { gate.as_mut_ptr().write_volatile(1) };
            early
        });
        assert!(!early, "dropped while a copy between them was queued");
        ctx.synchronize().expect("the queued copy faulted");
    }
}
