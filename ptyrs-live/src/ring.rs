//! Pinned host ring of cropped frames and the thread that streams it to the
//! GPU.
//!
//! Sequence numbers are handed out by [`Ring::reserve`]; a slot is written
//! only by the holder of its reservation and becomes visible to the uploader
//! when `done[slot] == seq + 1`. The uploader is the only writer of
//! `released`, and advances it only after the event recorded behind that
//! slot's H2D copy has completed, so a worker never overwrites host memory
//! the copy engine may still be reading.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicU64, Ordering};
use std::time::Duration;

use cuda_core::{Context, DeviceBuffer, DriverError, Event, PinnedHostBuffer};

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

    /// `(uid, ever written)` of slot `i`, for end-of-run checks.
    pub fn slot_uid(&self, i: usize) -> Option<i32> {
        (self.done[i].load(Ordering::Acquire) != 0)
            .then(|| self.meta[i].uid.load(Ordering::Relaxed))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UploaderConfig {
    /// Flush as soon as this many contiguous slots are committed.
    pub batch: usize,
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
    /// Device ring and positions read back after the last copy.
    pub device_frames: Vec<u16>,
    pub device_positions: Vec<f64>,
}

/// Run until `stop` is set and everything reserved has landed on the device.
pub fn run_uploader(
    ring: Arc<Ring>,
    ctx: Arc<Context>,
    cfg: UploaderConfig,
    stop: Arc<AtomicBool>,
) -> Result<UploadReport, DriverError> {
    ctx.bind_to_thread()?;
    let stream = ctx.new_stream()?;
    let slots = ring.slots;
    let se = ring.slot_elems;
    let dev_frames = DeviceBuffer::<u16>::zeroed(&stream, slots * se)?;
    let dev_pos = DeviceBuffer::<f64>::zeroed(&stream, slots * 2)?;
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
        device_frames: Vec::new(),
        device_positions: Vec::new(),
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
            if pending < cfg.batch as u64 && !stopping && waited < cfg.flush_ns {
                break;
            }
            let n = (pending as usize).min(cfg.batch).min(slots - s0);
            // SAFETY: slots `s0..s0 + n` are committed and not yet released,
            // so no worker writes them until the event below completes; both
            // device buffers cover `slots` slots.
            unsafe {
                cuda_core::memcpy_htod_async(
                    dev_frames.cu_deviceptr() + (s0 * se * 2) as u64,
                    ring.frames_ptr.add(s0 * se),
                    n * se,
                    &stream,
                )?;
                cuda_core::memcpy_htod_async(
                    dev_pos.cu_deviceptr() + (s0 * 16) as u64,
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
            for seq in seq0..seq0 + n {
                let m = &ring.meta[(seq % slots as u64) as usize];
                rep.rx_to_gpu.record(t - m.t_rx.load(Ordering::Relaxed));
                rep.pub_to_gpu.record(t - m.t_pub.load(Ordering::Relaxed));
                rep.commit_to_gpu
                    .record(t - m.t_commit.load(Ordering::Relaxed));
            }
            rep.frames += n;
            ring.released.store(seq0 + n, Ordering::Release);
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
    rep.device_frames = vec![0u16; slots * se];
    rep.device_positions = vec![0f64; slots * 2];
    // SAFETY: the host vectors hold exactly the device buffers' element counts.
    unsafe {
        cuda_core::memcpy_dtoh_async(
            rep.device_frames.as_mut_ptr(),
            dev_frames.cu_deviceptr(),
            slots * se,
            &stream,
        )?;
        cuda_core::memcpy_dtoh_async(
            rep.device_positions.as_mut_ptr(),
            dev_pos.cu_deviceptr(),
            slots * 2,
            &stream,
        )?;
    }
    stream.synchronize()?;
    Ok(rep)
}
