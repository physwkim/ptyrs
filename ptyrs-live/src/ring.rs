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
//! Workers commit frames without their positions. The uploader joins each
//! committed frame, in sequence order, with its position from the
//! [`PosTable`]: the frame lands once its position is in. One still without
//! its position is dropped `pos_wait_ns` after it arrived, or sooner, as soon
//! as a worker needs its slot: a missing position holds the frames behind it
//! only while the ring has room for them, so it never costs another frame.
//!
//! On the device every landed frame has its own place in a [`DeviceStore`],
//! the next one in landing order, so an uploaded frame is never overwritten
//! (invariant 3) and its address is fixed and backed before the first frame
//! (invariant 9).

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
use crate::pos::PosTable;
use crate::stats::{Histogram, thread_cpu_ns, wall_ns};

#[derive(Default)]
struct Meta {
    uid: AtomicI32,
    t_rx: AtomicI64,
    t_pub: AtomicI64,
    t_commit: AtomicI64,
    /// Store place + 1 of the slot's frame once it landed, 0 once it dropped;
    /// written by the uploader only.
    place: AtomicU64,
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

    /// Hands the frame to the uploader, which lands it once the position of
    /// `uid` is in.
    pub fn commit(self, uid: i32, t_rx: i64, t_pub: i64) {
        let r = self.ring;
        let i = (self.seq % r.slots as u64) as usize;
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

    /// `(store place, uid)` of the frame last committed to slot `i` if it
    /// landed, for end-of-run checks once the uploader has stopped.
    pub fn slot(&self, i: usize) -> Option<(u64, i32)> {
        let m = &self.meta[i];
        let place = m.place.load(Ordering::Relaxed);
        (self.done[i].load(Ordering::Acquire) != 0 && place != 0)
            .then(|| (place - 1, m.uid.load(Ordering::Relaxed)))
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        // An uploader that returned an error left its copies out of the slots queued.
        let _ = self.frames.context().synchronize();
    }
}

/// Device side of the input: the frame at place `f` at `frames_ptr() + f *
/// frame bytes`, its `(x, y)` as two f64 at `positions_ptr() + f * 16`, for
/// `f < max_frames`. Backed whole when made, before the first frame: the
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

    /// All `max_frames` frames as one buffer, place `f` from element
    /// `f * frame_elems`, sharing the store's tracker: the counts the
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

    /// Where the frames at places `places` and their positions go.
    fn dst(&self, places: Range<u64>) -> Result<(CUdeviceptr, CUdeviceptr), GrowError> {
        let fb = (self.frame_elems * 2) as u64;
        if places.end > self.max_frames {
            return Err(GrowError::Capacity {
                end: (places.end * fb) as usize,
                capacity: (self.max_frames * fb) as usize,
            });
        }
        Ok((
            self.frames_ptr() + places.start * fb,
            self.positions_ptr() + places.start * 16,
        ))
    }

    /// Synchronous copy of the frames at places `places` to `out`, for checks.
    pub fn read_frames(&self, places: Range<u64>, out: &mut [u16]) -> Result<(), DriverError> {
        let n = (places.end - places.start) as usize * self.frame_elems;
        assert_eq!(out.len(), n);
        let fb = (self.frame_elems * 2) as u64;
        let range = (places.start * fb) as usize..(places.end * fb) as usize;
        assert!(self.frames.is_ready(range));
        self.ctx.bind_to_thread()?;
        // SAFETY: the source range is mapped (checked above; mappings are only
        // added while the store lives) and `out` holds its element count.
        unsafe {
            cuda_bindings::cuMemcpyDtoH_v2(
                out.as_mut_ptr().cast(),
                self.frames_ptr() + places.start * fb,
                n * 2,
            )
        }
        .result()
    }

    /// Synchronous copy of the positions at places `0..n`, for checks.
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

/// The frames at places `f0..f0 + n` are in the [`DeviceStore`]. Batches are
/// sent in place order and without gaps.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameBatch {
    pub f0: u64,
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
    /// A frame whose position is still missing this long after the frame
    /// arrived is dropped, if a worker has not needed its slot before.
    pub pos_wait_ns: i64,
}

pub struct UploadReport {
    /// Frames landed.
    pub frames: u64,
    /// Frames dropped because their position did not come in time.
    pub pos_missing: u64,
    /// Of `pos_missing`, those dropped before `pos_wait_ns` because a worker
    /// needed their slot.
    pub pos_crowded: u64,
    /// Frames that reached the head of the ring before their position did.
    pub pos_waits: u64,
    /// Frames whose position had already arrived when the frame did.
    pub pos_first: u64,
    /// Frame arrival → position arrival, for the rest that landed.
    pub pos_after_frame: Histogram,
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
    /// uid of every landed frame, by store place.
    pub uids: Vec<i32>,
}

/// A batch copy in flight: frames `seq0..seq0 + n` of the ring to places
/// `f0..f0 + n`; its completion releases the ring up to `release`, past
/// any frames dropped after it.
struct Flight {
    ev: usize,
    seq0: u64,
    f0: u64,
    n: u64,
    release: u64,
}

/// Run until `stop` is set and everything reserved has landed in `store` or
/// dropped; each landed batch is sent to `batches` if given, which should be
/// unbounded: a full channel stalls the uploader. Frames land in sequence
/// order, each once its position is in `pos`.
pub fn run_uploader(
    ring: Arc<Ring>,
    ctx: Arc<Context>,
    store: Arc<DeviceStore>,
    pos: Arc<PosTable>,
    cfg: UploaderConfig,
    stop: Arc<AtomicBool>,
    batches: Option<Sender<FrameBatch>>,
) -> Result<UploadReport, GrowError> {
    ctx.bind_to_thread()?;
    let stream = ctx.new_stream()?;
    let slots = ring.slots;
    let se = ring.slot_elems;
    assert_eq!(store.frame_elems, se);
    let at = |seq: u64| (seq % slots as u64) as usize;
    let events: Vec<Event> = (0..cfg.events)
        .map(|_| ctx.new_event_with_flags(None))
        .collect::<Result<_, _>>()?;
    let mut free: Vec<usize> = (0..cfg.events).collect();
    let mut inflight: VecDeque<Flight> = VecDeque::with_capacity(cfg.events);
    let mut rep = UploadReport {
        frames: 0,
        pos_missing: 0,
        pos_crowded: 0,
        pos_waits: 0,
        pos_first: 0,
        pos_after_frame: Histogram::new(1_000_000),
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
    // Sequence numbers below `joined` are committed and landing or dropped,
    // as `lands` says for those from `issued` on; `placed` places are taken.
    let (mut joined, mut issued, mut placed) = (0u64, 0u64, 0u64);
    let mut lands = vec![false; slots];
    // The frame last counted in `pos_waits`.
    let mut waiting = None;
    loop {
        let mut progressed = false;
        while ring.done[at(joined)].load(Ordering::Acquire) == joined + 1 {
            let i = at(joined);
            let m = &ring.meta[i];
            let t_rx = m.t_rx.load(Ordering::Relaxed);
            match pos.get(m.uid.load(Ordering::Relaxed)) {
                Some(p) => {
                    // SAFETY: slot `i` is committed and not yet released, so
                    // no worker writes it (see `Ring`).
                    unsafe {
                        *ring.pos_ptr.add(2 * i) = p.x;
                        *ring.pos_ptr.add(2 * i + 1) = p.y;
                    }
                    match p.t_rx - t_rx {
                        lag if lag > 0 => rep.pos_after_frame.record(lag),
                        _ => rep.pos_first += 1,
                    }
                    lands[i] = true;
                }
                None => {
                    // Sequence `joined + slots` is reserved: a worker waits
                    // for this frame's slot.
                    let crowded = ring.next.load(Ordering::Acquire) > joined + slots as u64;
                    if !crowded && wall_ns() - t_rx <= cfg.pos_wait_ns {
                        if waiting != Some(joined) {
                            rep.pos_waits += 1;
                            waiting = Some(joined);
                        }
                        break;
                    }
                    rep.pos_missing += 1;
                    rep.pos_crowded += crowded as u64;
                    m.place.store(0, Ordering::Relaxed);
                    lands[i] = false;
                }
            }
            joined += 1;
            progressed = true;
        }
        let stopping = stop.load(Ordering::Acquire);
        while joined > issued && !free.is_empty() {
            let s0 = at(issued);
            if !lands[s0] {
                // No place in the store; the slot goes back once the copies
                // before it are done.
                issued += 1;
                match inflight.back_mut() {
                    Some(f) => f.release = issued,
                    None => ring.released.store(issued, Ordering::Release),
                }
                progressed = true;
                continue;
            }
            let most = ((joined - issued) as usize)
                .min(cfg.batch.get())
                .min(slots - s0);
            let n = lands[s0..s0 + most].iter().take_while(|&&l| l).count();
            let grows = issued + n as u64 == joined && n < cfg.batch.get();
            let waited = wall_ns() - ring.meta[s0].t_commit.load(Ordering::Relaxed);
            if grows && !stopping && waited < cfg.flush_ns {
                break;
            }
            let (frames_dst, positions_dst) = store.dst(placed..placed + n as u64)?;
            // SAFETY: slots `s0..s0 + n` are committed and not yet released,
            // so no worker writes them until the event below completes; `dst`
            // gave places `placed..placed + n` in the store.
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
            for (k, m) in ring.meta[s0..s0 + n].iter().enumerate() {
                m.place.store(placed + k as u64 + 1, Ordering::Relaxed);
            }
            let ev = free.pop().expect("checked non-empty");
            events[ev].record(&stream)?;
            let n = n as u64;
            inflight.push_back(Flight {
                ev,
                seq0: issued,
                f0: placed,
                n,
                release: issued + n,
            });
            rep.max_inflight = rep.max_inflight.max(inflight.len());
            issued += n;
            placed += n;
            rep.batches += 1;
            rep.bytes += n * (se as u64 * 2 + 16);
            progressed = true;
        }
        while let Some(f) = inflight.front() {
            if !events[f.ev].query()? {
                break;
            }
            let t = wall_ns();
            let seqs = f.seq0..f.seq0 + f.n;
            let positions = match batches {
                // SAFETY: the slots are committed and not yet released, so
                // no worker writes them (see `Ring`).
                Some(_) => seqs
                    .clone()
                    .map(|seq| unsafe {
                        let p = ring.pos_ptr.add(2 * at(seq));
                        (*p, *p.add(1))
                    })
                    .collect(),
                None => Vec::new(),
            };
            for seq in seqs {
                let m = &ring.meta[at(seq)];
                rep.rx_to_gpu.record(t - m.t_rx.load(Ordering::Relaxed));
                rep.pub_to_gpu.record(t - m.t_pub.load(Ordering::Relaxed));
                rep.commit_to_gpu
                    .record(t - m.t_commit.load(Ordering::Relaxed));
                rep.uids.push(m.uid.load(Ordering::Relaxed));
            }
            rep.frames += f.n;
            ring.released.store(f.release, Ordering::Release);
            if let Some(tx) = &batches {
                // A consumer that has gone away does not stop the input.
                let _ = tx.send(FrameBatch {
                    f0: f.f0,
                    n: f.n,
                    positions,
                });
            }
            free.push(f.ev);
            inflight.pop_front();
            progressed = true;
        }
        if stopping
            && inflight.is_empty()
            && issued == joined
            && joined == ring.next.load(Ordering::Acquire)
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
    use crate::pos::Pos;

    fn cfg() -> UploaderConfig {
        UploaderConfig {
            batch: NonZeroUsize::new(5).unwrap(),
            flush_ns: 100_000,
            idle: Duration::from_micros(20),
            events: 4,
            pos_wait_ns: 1_000_000_000,
        }
    }

    /// Trigger `uid`'s frame and position.
    fn pixel(uid: i32, j: usize) -> u16 {
        (uid as usize * 7 + j) as u16
    }

    fn at(uid: i32) -> (f64, f64) {
        (uid as f64, -(uid as f64))
    }

    fn post(pos: &PosTable, uid: i32) {
        let (x, y) = at(uid);
        pos.insert(
            uid,
            Pos {
                x,
                y,
                t_rx: wall_ns(),
            },
        );
    }

    /// Commits triggers `uids`, each arriving now; their positions are posted
    /// elsewhere.
    fn commit(ring: &Ring, uids: impl IntoIterator<Item = i32>) {
        for uid in uids {
            let mut s = ring.reserve();
            for (j, v) in s.frame().iter_mut().enumerate() {
                *v = pixel(uid, j);
            }
            s.commit(uid, wall_ns(), 0);
        }
    }

    fn produce(ring: &Ring, pos: &PosTable, n: usize) {
        for _ in 0..n {
            let mut s = ring.reserve();
            let uid = 1000 + s.seq() as i32;
            for (j, v) in s.frame().iter_mut().enumerate() {
                *v = pixel(uid, j);
            }
            post(pos, uid);
            s.commit(uid, 0, 0);
        }
    }

    fn uploader(
        ring: &Arc<Ring>,
        ctx: &Arc<Context>,
        store: &Arc<DeviceStore>,
        pos: &Arc<PosTable>,
        cfg: UploaderConfig,
        stop: &Arc<AtomicBool>,
        tx: Option<Sender<FrameBatch>>,
    ) -> std::thread::JoinHandle<Result<UploadReport, GrowError>> {
        let (ring, ctx, store, pos, stop) = (
            ring.clone(),
            ctx.clone(),
            store.clone(),
            pos.clone(),
            stop.clone(),
        );
        std::thread::spawn(move || run_uploader(ring, ctx, store, pos, cfg, stop, tx))
    }

    /// Every landed frame against what it was made from: the batches sent, in
    /// place order without gaps, with the positions of `uids`, the uid of
    /// each place, and the store's frames and positions; and the ring's own
    /// record of where each slot's frame landed.
    fn check_landed(
        ring: &Ring,
        store: &DeviceStore,
        rep: &UploadReport,
        rx: crossbeam_channel::Receiver<FrameBatch>,
        uids: &[i32],
    ) {
        let se = store.frame_elems();
        assert_eq!(rep.uids, uids);
        let mut next = 0;
        for b in rx.iter() {
            assert_eq!(b.f0, next);
            let want: Vec<(f64, f64)> = uids[next as usize..(next + b.n) as usize]
                .iter()
                .map(|&u| at(u))
                .collect();
            assert_eq!(b.positions, want);
            next += b.n;
        }
        assert_eq!(next, uids.len() as u64);
        let total = uids.len() as u64;
        let mut dev = vec![0u16; total as usize * se];
        store.read_frames(0..total, &mut dev).unwrap();
        for (f, (d, &uid)) in dev.chunks(se).zip(uids).enumerate() {
            assert!(
                d.iter().enumerate().all(|(j, &v)| v == pixel(uid, j)),
                "place {f}"
            );
        }
        let p = store.read_positions(total).unwrap();
        for (f, &uid) in uids.iter().enumerate() {
            assert_eq!((p[2 * f], p[2 * f + 1]), at(uid), "place {f}");
        }
        for i in 0..ring.slots() {
            if let Some((f, uid)) = ring.slot(i) {
                assert_eq!(uids[f as usize], uid, "slot {i}");
                let host = &ring.host_frames()[i * se..(i + 1) * se];
                assert!(
                    host.iter().enumerate().all(|(j, &v)| v == pixel(uid, j)),
                    "slot {i}"
                );
            }
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
        let pos = Arc::new(PosTable::new(1 << 10));
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = crossbeam_channel::unbounded();
        let up = uploader(&ring, &ctx, &store, &pos, cfg(), &stop, Some(tx));
        // Four producers commit out of order.
        std::thread::scope(|sc| {
            for _ in 0..4 {
                sc.spawn(|| produce(&ring, &pos, total as usize / 4));
            }
        });
        stop.store(true, Ordering::Release);
        let rep = up.join().unwrap().unwrap();

        assert_eq!((rep.frames, rep.pos_missing), (total, 0));
        assert_eq!(store.mapped_bytes(), backed);
        let uids: Vec<i32> = (0..total as i32).map(|s| 1000 + s).collect();
        check_landed(&ring, &store, &rep, rx, &uids);
    }

    /// Frames committed before their positions wait in the ring and land in
    /// sequence order once each position is in, the late ones behind the
    /// first holding the rest of the ring; nothing lands before its position.
    #[test]
    fn a_frame_lands_once_its_position_is_in() {
        let ctx = Context::new(0).expect("GPU 0");
        let se = 4096;
        let ring = Arc::new(Ring::new(&ctx, 8, se).unwrap());
        let store = Arc::new(DeviceStore::new(&ctx, 8, se).unwrap());
        let pos = Arc::new(PosTable::new(64));
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = crossbeam_channel::unbounded();
        let up = uploader(&ring, &ctx, &store, &pos, cfg(), &stop, Some(tx));
        // 2 lands at once, 0 holds 1..=5 until its position is in, 4 and 5
        // posted before 3 land behind it.
        post(&pos, 2);
        commit(&ring, 0..6);
        std::thread::sleep(Duration::from_millis(20));
        assert!(
            rx.try_recv().is_err(),
            "a frame landed before frame 0's position"
        );
        post(&pos, 0);
        post(&pos, 1);
        post(&pos, 4);
        post(&pos, 5);
        std::thread::sleep(Duration::from_millis(20));
        stop.store(true, Ordering::Release);
        std::thread::sleep(Duration::from_millis(20));
        post(&pos, 3);
        let rep = up.join().unwrap().unwrap();
        assert_eq!((rep.frames, rep.pos_missing), (6, 0));
        // Frames 0 and 3 found the ring head before their positions.
        assert_eq!(rep.pos_waits, 2);
        assert_eq!(rep.pos_first + rep.pos_after_frame.count(), 6);
        check_landed(&ring, &store, &rep, rx, &[0, 1, 2, 3, 4, 5]);
    }

    /// A frame whose position never comes is dropped once `pos_wait_ns` has
    /// passed since it arrived, when no worker needs its slot before: it
    /// takes no place in the store and its slot goes back to the ring; the
    /// frames around it, first and last among them, land in order.
    #[test]
    fn a_frame_without_a_position_is_dropped_after_the_wait() {
        let ctx = Context::new(0).expect("GPU 0");
        let se = 4096;
        let total = 120;
        let lost = |uid: i32| uid % 7 == 0 || uid == total - 1;
        let landed: Vec<i32> = (0..total).filter(|&u| !lost(u)).collect();
        // Room for every frame: no worker waits for a slot.
        let ring = Arc::new(Ring::new(&ctx, total as usize, se).unwrap());
        let store = Arc::new(DeviceStore::new(&ctx, landed.len(), se).unwrap());
        let pos = Arc::new(PosTable::new(256));
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = crossbeam_channel::unbounded();
        let cfg = UploaderConfig {
            pos_wait_ns: 2_000_000,
            ..cfg()
        };
        let up = uploader(&ring, &ctx, &store, &pos, cfg, &stop, Some(tx));
        for &uid in &landed {
            post(&pos, uid);
        }
        commit(&ring, 0..total);
        stop.store(true, Ordering::Release);
        let rep = up.join().unwrap().unwrap();
        let missing = (0..total).filter(|&u| lost(u)).count() as u64;
        assert_eq!(
            (rep.frames, rep.pos_missing, rep.pos_crowded),
            (landed.len() as u64, missing, 0)
        );
        assert_eq!(ring.stalls(), 0);
        check_landed(&ring, &store, &rep, rx, &landed);
    }

    /// A full ring behind a frame still without its position holds it until
    /// its position comes within the wait, and it lands: no slot is needed
    /// past the ring.
    #[test]
    fn a_position_within_the_wait_lands_its_frame_behind_a_full_ring() {
        let ctx = Context::new(0).expect("GPU 0");
        let se = 4096;
        let ring = Arc::new(Ring::new(&ctx, 4, se).unwrap());
        let store = Arc::new(DeviceStore::new(&ctx, 4, se).unwrap());
        let pos = Arc::new(PosTable::new(64));
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = crossbeam_channel::unbounded();
        let up = uploader(&ring, &ctx, &store, &pos, cfg(), &stop, Some(tx));
        for uid in 1..4 {
            post(&pos, uid);
        }
        commit(&ring, 0..4);
        std::thread::sleep(Duration::from_millis(20));
        assert!(rx.try_recv().is_err(), "a frame landed before frame 0");
        post(&pos, 0);
        stop.store(true, Ordering::Release);
        let rep = up.join().unwrap().unwrap();
        assert_eq!((rep.frames, rep.pos_missing), (4, 0));
        assert_eq!(ring.stalls(), 0);
        check_landed(&ring, &store, &rep, rx, &[0, 1, 2, 3]);
    }

    /// A frame still without its position is dropped as soon as a worker
    /// needs its slot, long before `pos_wait_ns`: a missing position costs
    /// only its own frame. Here the ring wraps many times past such frames,
    /// singly and in a run longer than the ring, and every frame with a
    /// position lands in order.
    #[test]
    fn a_frame_without_a_position_is_dropped_once_a_worker_needs_its_slot() {
        let ctx = Context::new(0).expect("GPU 0");
        let se = 4096;
        // More frames than the ring holds follow the last one without a
        // position, 119, so a worker needs its slot.
        let total = 126;
        let lost = |uid: i32| uid % 7 == 0 || (40..50).contains(&uid);
        let landed: Vec<i32> = (0..total).filter(|&u| !lost(u)).collect();
        let ring = Arc::new(Ring::new(&ctx, 4, se).unwrap());
        let store = Arc::new(DeviceStore::new(&ctx, landed.len(), se).unwrap());
        let pos = Arc::new(PosTable::new(256));
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = crossbeam_channel::unbounded();
        let cfg = UploaderConfig {
            pos_wait_ns: 3_600_000_000_000,
            ..cfg()
        };
        let up = uploader(&ring, &ctx, &store, &pos, cfg, &stop, Some(tx));
        for &uid in &landed {
            post(&pos, uid);
        }
        commit(&ring, 0..total);
        stop.store(true, Ordering::Release);
        let rep = up.join().unwrap().unwrap();
        let missing = (0..total).filter(|&u| lost(u)).count() as u64;
        assert_eq!(
            (rep.frames, rep.pos_missing, rep.pos_crowded),
            (landed.len() as u64, missing, missing)
        );
        check_landed(&ring, &store, &rep, rx, &landed);
    }

    #[test]
    fn a_frame_past_the_store_fails_the_uploader() {
        let ctx = Context::new(0).expect("GPU 0");
        let se = 65536;
        let ring = Arc::new(Ring::new(&ctx, 64, se).unwrap());
        // 32 frames fill the store; the 33rd has no place.
        let store = Arc::new(DeviceStore::new(&ctx, 32, se).unwrap());
        let pos = Arc::new(PosTable::new(64));
        let stop = Arc::new(AtomicBool::new(false));
        let up = uploader(&ring, &ctx, &store, &pos, cfg(), &stop, None);
        drop(store);
        produce(&ring, &pos, 33);
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
