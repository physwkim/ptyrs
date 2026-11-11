//! Source-independent half of the receive path. Every input (PVA, ZMQ)
//! turns an update into a [`Work`] and hands it to [`SourceStats::accept`];
//! decode workers then match the position, crop and commit into the pinned
//! [`Ring`], whose uploader emits the one "new frame batch" event.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use crossbeam_channel::{Receiver, Sender, TrySendError};

use crate::codec::{
    Bslz4Scratch, Codec, bslz4_decompress_into_u16, bslz4_decompress_range_u16, bytemuck_u16,
    bytemuck_u16_mut, lz4_decompress_into,
};
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
}

impl Truth for Synth {
    fn fill(&self, uid: i32, r: Region, out: &mut [u16]) {
        self.fill_region(self.pool_index(uid), r, out, &mut SynthScratch::default());
    }

    fn position(&self, uid: i32) -> (f64, f64) {
        crate::synth::position(uid)
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
    rx: Receiver<Work>,
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
    'frames: for w in rx.iter() {
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
    pub first_rx: Option<i64>,
    pub last_rx: i64,
    pub cpu_first: i64,
    /// `(t, process cpu)` every 1024 unique frames, for the steady CPU rate.
    pub cpu_sample: (i64, i64),
    pub first_error: Option<String>,
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
    pub fn accept(&mut self, w: Work, tx: &Sender<Work>) {
        let (uid, t_rx) = (w.uid, w.t_rx);
        if self.first_rx.is_none() {
            self.first_rx = Some(t_rx);
            self.cpu_first = process_cpu_ns();
        }
        self.last_rx = t_rx;
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
        if self.unique.is_multiple_of(1024) {
            self.cpu_sample = (t_rx, process_cpu_ns());
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
        match tx.try_send(w) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => self.queue_full += 1,
            Err(TrySendError::Disconnected(_)) => {}
        }
    }
}

/// Where each receive-path thread runs: the runtime threads share `rt`, the
/// uploader and every worker get a CPU of their own. Empty without a list.
pub struct Placement {
    pub rt: Option<Vec<usize>>,
    pub h2d: Option<usize>,
    pub workers: Vec<Option<usize>>,
    pub util_min: u32,
}

impl Placement {
    pub fn new(cpus: &str, rt_threads: usize, workers: usize, util_min: u32) -> Self {
        if cpus.is_empty() {
            return Self {
                rt: None,
                h2d: None,
                workers: vec![None; workers],
                util_min,
            };
        }
        let c = parse_cpu_list(cpus).unwrap_or_else(|e| panic!("--cpus: {e}"));
        let need = rt_threads + 1 + workers;
        assert!(
            c.len() >= need,
            "--cpus {cpus}: {need} needed ({rt_threads} runtime, 1 h2d, {workers} workers)"
        );
        Self {
            rt: Some(c[..rt_threads].to_vec()),
            h2d: Some(c[rt_threads]),
            workers: c[rt_threads + 1..need].iter().map(|&x| Some(x)).collect(),
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
        Work {
            payload: Bytes::new(),
            codec: Codec::None,
            uid,
            t_rx: 1,
            t_pub: 0,
        }
    }

    #[test]
    fn accounting_classifies_gaps_repeats_and_strays() {
        let (tx, rx) = crossbeam_channel::bounded(2);
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
}
