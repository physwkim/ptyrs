//! Source-independent half of the receive path. Every input (PVA, ZMQ)
//! turns an update into a [`Work`] and hands it to [`SourceStats::accept`];
//! decode workers then match the position, crop and commit into the pinned
//! [`Ring`], whose uploader emits the one "new frame batch" event.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;

use crate::codec::{
    Bslz4Scratch, Codec, bslz4_decompress_into_u16, bslz4_decompress_range_u16, bytemuck_u16,
    bytemuck_u16_mut, lz4_decompress_into,
};
use crate::handoff::{WorkRx, WorkTx};
use crate::pos::PosTable;
use crate::ring::Ring;
use crate::sched::{parse_cpu_list, pin_current, set_util_min};
use crate::stats::{Histogram, mono_ns, process_cpu_ns, thread_cpu_ns};
use crate::synth::{Region, Synth, SynthScratch};

/// One compressed frame on its way from a source to a decode worker.
pub struct Work {
    pub payload: Bytes,
    pub codec: Codec,
    pub uid: i32,
    /// Arrival at the source callback, wall ns.
    pub t_rx: i64,
    /// Publisher's timestamp, wall ns.
    pub t_pub: i64,
}

/// What the receiver checks frames and positions against.
pub trait Truth: Send + Sync {
    /// Expected pixels of frame `uid` over `r`, row stride `r.w`.
    fn fill(&self, uid: i32, r: Region, out: &mut [u16]);
    /// Scan position of trigger `uid`, metres.
    fn position(&self, uid: i32) -> (f64, f64);
    /// `(width, height)` of a frame.
    fn frame_size(&self) -> (usize, usize);
    /// Distinct frames: trigger `uid` repeats frame `uid % period`.
    fn period(&self) -> usize;
}

impl Truth for Synth {
    fn fill(&self, uid: i32, r: Region, out: &mut [u16]) {
        self.fill_region(self.pool_index(uid), r, out, &mut SynthScratch::default());
    }

    fn position(&self, uid: i32) -> (f64, f64) {
        crate::synth::position(uid)
    }

    fn frame_size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn period(&self) -> usize {
        self.pool
    }
}

/// Detector frame and the crop that reaches the GPU.
#[derive(Debug, Clone, Copy)]
pub struct Geometry {
    pub width: usize,
    pub height: usize,
    pub crop: usize,
    pub x0: usize,
    pub y0: usize,
    /// Decode only the bslz4 blocks overlapping the crop rows.
    pub crop_decode: bool,
}

impl Geometry {
    /// A `crop`-square window centred on `(cx, cy)`.
    pub fn centred(width: usize, height: usize, crop: usize, cx: usize, cy: usize) -> Self {
        let g = Self {
            width,
            height,
            crop,
            x0: cx - crop / 2,
            y0: cy - crop / 2,
            crop_decode: true,
        };
        assert!(
            g.x0 + crop <= width && g.y0 + crop <= height,
            "crop outside the frame"
        );
        g
    }

    pub fn crop_region(&self) -> Region {
        Region {
            x0: self.x0,
            y0: self.y0,
            w: self.crop,
            h: self.crop,
        }
    }

    pub fn pixels(&self) -> usize {
        self.width * self.height
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WorkerCfg {
    pub geom: Geometry,
    /// Regenerate and compare the crop of every uid divisible by this; 0 off.
    pub verify_every: i32,
    /// How long a frame waits for its position before it is dropped.
    pub pos_wait_ns: i64,
}

#[derive(Default)]
pub struct WorkerReport {
    pub frames: u64,
    pub decode_err: u64,
    pub pos_missing: u64,
    pub pos_waited: u64,
    pub pos_mismatch: u64,
    pub verified: u64,
    pub verify_bad: u64,
    pub cpu_ns: i64,
    pub decode: Option<Histogram>,
    /// Frames whose position had already arrived when the frame did.
    pub pos_first: u64,
    /// Frame arrival → position arrival, for the rest.
    pub pos_after_frame: Option<Histogram>,
}

impl WorkerReport {
    /// Sum of several workers' reports.
    pub fn merge(reports: Vec<WorkerReport>) -> (WorkerReport, Histogram, Histogram) {
        let mut wr = WorkerReport::default();
        let mut decode = Histogram::new(100_000);
        let mut pos_after = Histogram::new(1_000_000);
        for r in reports {
            wr.pos_first += r.pos_first;
            wr.frames += r.frames;
            wr.decode_err += r.decode_err;
            wr.pos_missing += r.pos_missing;
            wr.pos_waited += r.pos_waited;
            wr.pos_mismatch += r.pos_mismatch;
            wr.verified += r.verified;
            wr.verify_bad += r.verify_bad;
            wr.cpu_ns += r.cpu_ns;
            if let Some(h) = r.decode {
                decode.merge(&h);
            }
            if let Some(h) = r.pos_after_frame {
                pos_after.merge(&h);
            }
        }
        (wr, decode, pos_after)
    }
}

/// Decode → position match → crop → ring, until every sender is gone.
/// One image-sized scratch per worker, allocated once.
pub fn worker(
    rx: WorkRx,
    ring: Arc<Ring>,
    pos: Arc<PosTable>,
    cfg: WorkerCfg,
    truth: Arc<dyn Truth>,
) -> WorkerReport {
    let g = cfg.geom;
    let mut r = WorkerReport {
        decode: Some(Histogram::new(100_000)),
        pos_after_frame: Some(Histogram::new(1_000_000)),
        ..Default::default()
    };
    let mut img = vec![0u16; g.pixels()];
    let mut bs = Bslz4Scratch::default();
    let mut vbuf = vec![0u16; g.crop * g.crop];
    let rows = g.y0 * g.width..(g.y0 + g.crop) * g.width;
    let row_bytes = g.crop * 2;
    let cpu0 = thread_cpu_ns();
    'frames: while let Some(w) = rx.recv() {
        let t = mono_ns();
        let res = match w.codec {
            Codec::None if w.payload.len() == g.pixels() * 2 => Ok(()),
            Codec::None => Err(crate::codec::CodecError::Framing),
            Codec::Lz4 => lz4_decompress_into(&w.payload, bytemuck_u16_mut(&mut img)),
            Codec::Bslz4 if g.crop_decode => {
                bslz4_decompress_range_u16(&w.payload, &mut img, &mut bs, rows.clone())
            }
            Codec::Bslz4 => bslz4_decompress_into_u16(&w.payload, &mut img, &mut bs),
        };
        if res.is_err() {
            r.decode_err += 1;
            continue;
        }
        let src: &[u8] = match w.codec {
            Codec::None => &w.payload,
            _ => bytemuck_u16(&img),
        };
        r.decode.as_mut().unwrap().record(mono_ns() - t);
        let deadline = mono_ns() + cfg.pos_wait_ns;
        let p = loop {
            if let Some(p) = pos.get(w.uid) {
                break p;
            }
            if mono_ns() > deadline {
                r.pos_missing += 1;
                continue 'frames;
            }
            r.pos_waited += 1;
            std::thread::sleep(Duration::from_micros(20));
        };
        match p.t_rx - w.t_rx {
            lag if lag > 0 => r.pos_after_frame.as_mut().unwrap().record(lag),
            _ => r.pos_first += 1,
        }
        let (x, y) = (p.x, p.y);
        r.pos_mismatch += ((x, y) != truth.position(w.uid)) as u64;
        let mut slot = ring.reserve();
        let dst = bytemuck_u16_mut(slot.frame());
        for (i, d) in dst.chunks_exact_mut(row_bytes).enumerate() {
            let off = ((g.y0 + i) * g.width + g.x0) * 2;
            d.copy_from_slice(&src[off..off + row_bytes]);
        }
        slot.commit(w.uid, x, y, w.t_rx, w.t_pub);
        r.frames += 1;
        if cfg.verify_every > 0 && w.uid % cfg.verify_every == 0 {
            truth.fill(w.uid, g.crop_region(), &mut vbuf);
            r.verified += 1;
            let crop_ok = (0..g.crop).all(|i| {
                let off = ((g.y0 + i) * g.width + g.x0) * 2;
                src[off..off + row_bytes] == *bytemuck_u16(&vbuf[i * g.crop..(i + 1) * g.crop])
            });
            r.verify_bad += (!crop_ok) as u64;
        }
    }
    r.cpu_ns = thread_cpu_ns() - cpu0;
    r
}

/// Frames received that never reached the GPU, by where they dropped out;
/// a run counts only without any (DESIGN.md §7 M5). Past the source's
/// rejects, the loss is a difference, frames received minus frames the GPU
/// consumed, so a drop no counter sees still shows, as `unaccounted`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameLoss {
    /// Updates the source could not read as a frame.
    pub rejected: u64,
    /// Frames the full work queue turned away.
    pub queue_full: u64,
    pub decode_err: u64,
    /// Frames whose position did not arrive in time.
    pub pos_missing: u64,
    /// Received, not consumed on the GPU and in none of the counts above.
    pub unaccounted: u64,
}

impl FrameLoss {
    /// From the source's and the workers' counts and the frames the GPU
    /// consumed, once every worker and the uploader have drained.
    pub fn new(st: &SourceStats, wr: &WorkerReport, on_gpu: u64) -> Self {
        let counted = st.queue_full + wr.decode_err + wr.pos_missing;
        Self {
            rejected: st.rejected,
            queue_full: st.queue_full,
            decode_err: wr.decode_err,
            pos_missing: wr.pos_missing,
            unaccounted: st.unique.saturating_sub(on_gpu).saturating_sub(counted),
        }
    }

    pub fn total(&self) -> u64 {
        self.rejected + self.queue_full + self.decode_err + self.pos_missing + self.unaccounted
    }
}

/// Per-source accounting of what arrived, shared by every input kind.
#[derive(Default)]
pub struct SourceStats {
    frames: u64,
    seen: Vec<u64>,
    pub unique: u64,
    pub dup: u64,
    pub beyond: u64,
    last_uid: i64,
    pub gap_events: u64,
    pub gap_frames: u64,
    pub out_of_order: u64,
    pub queue_full: u64,
    pub rejected: u64,
    pub placeholders: u64,
    /// `None` until the first frame arrives.
    pub rx: Option<RxSpan>,
    /// Arrival of the last of the run's frames, wall ns: the scan's end for
    /// [`crate::converge::Clock`]. `None` while any is missing.
    pub scan_end: Option<i64>,
    pub first_error: Option<String>,
}

/// Receive times and process CPU from the first frame on, in ns.
#[derive(Clone, Copy, Debug)]
pub struct RxSpan {
    pub first: i64,
    pub last: i64,
    pub cpu_first: i64,
    /// `(t, process cpu)` at the latest multiple of 1024 unique frames, for
    /// the steady CPU rate.
    pub cpu_sample: Option<(i64, i64)>,
}

impl SourceStats {
    /// Accounting for a run of `frames` triggers, uids `0..frames`.
    pub fn new(frames: u64) -> Self {
        Self {
            frames,
            seen: vec![0; (frames as usize).div_ceil(64)],
            last_uid: -1,
            ..Default::default()
        }
    }

    pub fn reject(&mut self, why: String) {
        self.rejected += 1;
        self.first_error.get_or_insert(why);
    }

    /// Count `w` and pass it to the workers unless it repeats a uid or lies
    /// past the run. A full queue drops it (counted, never blocks the source).
    pub fn accept(&mut self, w: Work, tx: &WorkTx) {
        let (uid, t_rx) = (w.uid, w.t_rx);
        let rx = self.rx.get_or_insert_with(|| RxSpan {
            first: t_rx,
            last: t_rx,
            cpu_first: process_cpu_ns(),
            cpu_sample: None,
        });
        rx.last = t_rx;
        if uid < 0 || uid as u64 >= self.frames {
            self.beyond += 1;
            return;
        }
        let u = uid as usize;
        let (word, b) = (&mut self.seen[u / 64], 1u64 << (u % 64));
        if *word & b != 0 {
            self.dup += 1;
            return;
        }
        *word |= b;
        self.unique += 1;
        if self.unique == self.frames {
            self.scan_end = Some(t_rx);
        }
        if self.unique.is_multiple_of(1024) {
            rx.cpu_sample = Some((t_rx, process_cpu_ns()));
        }
        let next = self.last_uid + 1;
        match (uid as i64).cmp(&next) {
            std::cmp::Ordering::Greater => {
                self.gap_events += 1;
                self.gap_frames += uid as u64 - next as u64;
            }
            std::cmp::Ordering::Less => self.out_of_order += 1,
            std::cmp::Ordering::Equal => {}
        }
        self.last_uid = self.last_uid.max(uid as i64);
        if tx.try_send(w).is_err() {
            self.queue_full += 1;
        }
    }
}

/// Where each receive-path thread runs: the runtime threads share `rt`, the
/// uploader, every worker and every file reader get a CPU of their own.
/// Empty without a list.
pub struct Placement {
    /// The whole list, for work after the run.
    pub all: Option<Vec<usize>>,
    pub rt: Option<Vec<usize>>,
    pub h2d: Option<usize>,
    pub workers: Vec<Option<usize>>,
    pub readers: Vec<Option<usize>>,
    pub util_min: u32,
}

impl Placement {
    pub fn new(
        cpus: &str,
        rt_threads: usize,
        workers: usize,
        readers: usize,
        util_min: u32,
    ) -> Self {
        if cpus.is_empty() {
            return Self {
                all: None,
                rt: None,
                h2d: None,
                workers: vec![None; workers],
                readers: vec![None; readers],
                util_min,
            };
        }
        let c = parse_cpu_list(cpus).unwrap_or_else(|e| panic!("--cpus: {e}"));
        let w0 = rt_threads + 1;
        let need = w0 + workers + readers;
        assert!(
            c.len() >= need,
            "--cpus {cpus}: {need} needed ({rt_threads} runtime, 1 h2d, {workers} workers, \
             {readers} readers)"
        );
        let some = |s: &[usize]| s.iter().map(|&x| Some(x)).collect();
        Self {
            all: Some(c.clone()),
            rt: Some(c[..rt_threads].to_vec()),
            h2d: Some(c[rt_threads]),
            workers: some(&c[w0..w0 + workers]),
            readers: some(&c[w0 + workers..need]),
            util_min,
        }
    }

    /// Apply to the calling thread.
    pub fn enter(&self, cpus: Option<&[usize]>) {
        if let Some(c) = cpus {
            pin_current(c).unwrap_or_else(|e| panic!("pin to {c:?}: {e}"));
        }
        if self.util_min > 0 {
            set_util_min(self.util_min).unwrap_or_else(|e| {
                panic!("uclamp min {}: {e} (--util-min 0 skips it)", self.util_min)
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn work(uid: i32) -> Work {
        at(uid, 1)
    }

    fn at(uid: i32, t_rx: i64) -> Work {
        Work {
            payload: Bytes::new(),
            codec: Codec::None,
            uid,
            t_rx,
            t_pub: 0,
        }
    }

    /// The scan ends at the arrival that completes it, out of order here;
    /// repeats and strays before or after it move nothing.
    #[test]
    fn the_scan_ends_when_its_last_missing_frame_arrives() {
        let (tx, _rx) = crate::handoff::work_queue(std::num::NonZeroUsize::new(8).unwrap());
        let mut s = SourceStats::new(3);
        for (uid, t) in [(0, 10), (2, 20), (2, 25), (3, 27)] {
            s.accept(at(uid, t), &tx);
            assert_eq!(s.scan_end, None, "after uid {uid}");
        }
        s.accept(at(1, 30), &tx);
        assert_eq!(s.scan_end, Some(30));
        for (uid, t) in [(1, 40), (5, 41)] {
            s.accept(at(uid, t), &tx);
        }
        assert_eq!(s.scan_end, Some(30));
    }

    #[test]
    fn accounting_classifies_gaps_repeats_and_strays() {
        let (tx, rx) = crate::handoff::work_queue(std::num::NonZeroUsize::new(2).unwrap());
        let mut s = SourceStats::new(10);
        for uid in [0, 1, 4, 3, 3, 12, -1] {
            s.accept(work(uid), &tx);
        }
        assert_eq!(
            (s.unique, s.dup, s.beyond, s.gap_events, s.gap_frames),
            (4, 1, 2, 1, 2)
        );
        assert_eq!((s.out_of_order, s.queue_full), (1, 2));
        assert_eq!(rx.try_iter().map(|w| w.uid).collect::<Vec<_>>(), [0, 1]);
    }

    /// 10 unique frames received: every counted drop shows at its stage,
    /// a frame missing from the GPU with no count shows as unaccounted, and
    /// a reject alone is a loss though every frame reached the GPU.
    #[test]
    fn frame_loss_is_received_minus_consumed_by_stage() {
        let st = |queue_full, rejected| SourceStats {
            unique: 10,
            queue_full,
            rejected,
            ..SourceStats::new(10)
        };
        let wr = |decode_err, pos_missing| WorkerReport {
            decode_err,
            pos_missing,
            ..Default::default()
        };
        let none = FrameLoss::new(&st(0, 0), &wr(0, 0), 10);
        assert_eq!((none, none.total()), (FrameLoss::default(), 0));
        let counted = FrameLoss::new(&st(1, 0), &wr(2, 3), 4);
        assert_eq!(
            (counted.queue_full, counted.decode_err, counted.pos_missing),
            (1, 2, 3)
        );
        assert_eq!((counted.unaccounted, counted.total()), (0, 6));
        let silent = FrameLoss::new(&st(1, 0), &wr(0, 0), 7);
        assert_eq!((silent.unaccounted, silent.total()), (2, 3));
        let rejected = FrameLoss::new(&st(0, 1), &wr(0, 0), 10);
        assert_eq!((rejected.rejected, rejected.total()), (1, 1));
    }
}
