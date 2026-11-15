//! The fill hook wired end to end: frames through the uploader into the
//! consumer, each run against one GPU's replay of what became of each
//! call's fill.

use std::collections::VecDeque;
use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};
use cuda_core::{Context, DeviceBuffer};

use super::standin::Wrong;
use super::{PatchFill, PatchPredictor, patch_corner};
use crate::dist::Scheme;
use crate::live::farfield::{Ptycho, PtychoSpec, ScanPath};
use crate::live::follow::{Fate, FillHook, FollowCfg, FollowReport, ReconCfg, run_dist_consumer};
use crate::live::pos::{Pos, PosTable};
use crate::live::recon::LiveAp;
use crate::live::ring::{FrameBatch, Ring, UploaderConfig, run_region_uploader};
use crate::live::sent::sent;

const PATCH: [usize; 2] = [32, 32];
const BATCH: usize = 4;
const PACK: usize = 3 * BATCH;

/// [`Wrong`], refusing the calls `fail` (counted from 1) before predicting;
/// its first call waits until `gate` fires or closes.
struct Refusing {
    wrong: Wrong,
    calls: u64,
    fail: &'static [u64],
    gate: Option<Receiver<()>>,
}

impl Refusing {
    fn new(fail: &'static [u64], gate: Option<Receiver<()>>) -> Self {
        Self {
            wrong: Wrong {
                patch: PATCH,
                issued: 0,
            },
            calls: 0,
            fail,
            gate,
        }
    }
}

impl PatchPredictor for Refusing {
    fn patch(&self) -> [usize; 2] {
        self.wrong.patch
    }

    fn predict(
        &mut self,
        counts: &[u16],
        frame: [usize; 2],
        out: &mut [[f32; 2]],
    ) -> Result<(), String> {
        if let Some(g) = self.gate.take() {
            // Closed when the relay ends: the run never hangs here.
            let _ = g.recv();
        }
        self.calls += 1;
        if self.fail.contains(&self.calls) {
            return Err(format!("call {} refused", self.calls));
        }
        self.wrong.predict(counts, frame, out)
    }
}

struct Case {
    ngpu: usize,
    frames: usize,
    /// [`FillHook::counts`] and [`FillHook::fills`]: no fill buffer, or
    /// one per batch at least, so the pool runs dry always or never.
    counts: usize,
    fills: usize,
    fail: &'static [u64],
    /// The predictor's first call is held until a batch went by without
    /// counts.
    gate: bool,
    /// Positions planned; all when `None`.
    plan: Option<usize>,
    /// Between scan points, pixels.
    step: f64,
}

impl Case {
    const fn new(ngpu: usize, frames: usize) -> Self {
        Self {
            ngpu,
            frames,
            counts: frames,
            fills: frames,
            fail: &[],
            gate: false,
            plan: None,
            step: 8.0,
        }
    }
}

/// Streams a 12-column raster of `case.frames`, one frame short, through
/// the uploader, in full batches of [`BATCH`] but the last, into the
/// consumer filling with [`Refusing`] on `case.ngpu` GPUs, a relay between
/// them recording which batches came with counts. Short, the scan ends once
/// the uploader stops, so every call's fill reaches the run. Replays the
/// landings on one GPU, each reported call taking the next batches with
/// counts, predicted, refused, or stitched and queued, at the landing and
/// epoch the run reports, and checks each call's fate and the run's last
/// kept object against the replay's, bit for bit. Returns the run's report.
fn run(case: Case) -> FollowReport {
    let p = Ptycho::new(PtychoSpec {
        n: 64,
        path: ScanPath::Raster { cols: 12 },
        points: 240,
        step: case.step,
        photons: 1e6,
        seed: 5,
        wavelength: 1.5498e-10,
        distance: 5.0,
        det_pixel: 75e-6,
    });
    let (n, total) = (64usize, case.frames);
    let pos: Vec<(f64, f64)> = (0..total).map(|k| p.position(k)).collect();
    let counts: Vec<u16> = (0..total).flat_map(|k| p.frame(k)).collect();
    let rc = ReconCfg {
        nf_max: NonZeroU32::new(8).unwrap(),
        snap_every: NonZeroU32::new(5).unwrap(),
        snaps: 3,
        ..ReconCfg::default()
    };
    let mut cfg = FollowCfg::plan(&p, NonZeroU32::new(total as u32).unwrap(), &rc).unwrap();
    if let Some(k) = case.plan {
        cfg.plan.truncate(k);
    }
    let ctx = Context::new(0).expect("GPU 0");
    let ring = Arc::new(Ring::new(&ctx, 16, n * n).unwrap());
    let stop = Arc::new(AtomicBool::new(false));
    let (up_tx, up_rx) = crossbeam_channel::unbounded::<FrameBatch>();
    let (tx, rx) = crossbeam_channel::unbounded();
    let (ready_tx, ready_rx) = crossbeam_channel::bounded(1);
    let (gate_tx, gate_rx) = crossbeam_channel::bounded::<()>(1);
    let relay = std::thread::spawn(move || {
        let mut gate: Option<Sender<()>> = Some(gate_tx);
        let mut had = Vec::new();
        for b in up_rx {
            had.push(b.counts.is_some());
            if b.counts.is_none()
                && let Some(g) = gate.take()
            {
                let _ = g.send(());
            }
            // The consumer may finish before the uploader closes.
            let _ = tx.send(b);
        }
        had
    });
    let consumer = {
        let cfg = cfg.clone();
        let ordinals: Vec<usize> = (0..case.ngpu).collect();
        let hook = FillHook {
            predictor: Box::new(Refusing::new(case.fail, case.gate.then_some(gate_rx))),
            batch: BATCH,
            pack: PACK,
            counts: case.counts,
            fills: case.fills,
        };
        std::thread::spawn(move || {
            run_dist_consumer(
                &ordinals,
                Scheme::Strips,
                cfg,
                rx,
                ready_tx,
                None,
                Some(hook),
            )
        })
    };
    let (landing, consumer) = sent(&ready_rx, consumer);
    let table = Arc::new(PosTable::new(1 << 16));
    let uploader = {
        let (ring, stop, table) = (ring.clone(), stop.clone(), table.clone());
        let up = UploaderConfig {
            batch: NonZeroUsize::new(BATCH).unwrap(),
            // Never by time: batches are whole but the one stop flushes.
            flush_ns: 60_000_000_000,
            idle: Duration::from_micros(20),
            events: 4,
            pos_wait_ns: 1_000_000_000,
        };
        std::thread::spawn(move || run_region_uploader(ring, landing, table, up, stop, Some(up_tx)))
    };
    let streamed = total - 1;
    let mut k = 0;
    for burst in [50, 1, 90, 0, 99] {
        for _ in 0..burst.min(streamed - k) {
            let mut s = ring.reserve();
            s.frame()
                .copy_from_slice(&counts[k * n * n..(k + 1) * n * n]);
            let (x, y) = pos[k];
            table.insert(k as i32, Pos::new(x, y, 0));
            s.commit(k as i32, 0, 0);
            k += 1;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(k, streamed);
    stop.store(true, Ordering::Release);
    let up = uploader.join().unwrap().unwrap();
    let rep = consumer.join().unwrap().unwrap();
    let had = relay.join().unwrap();
    assert_eq!((up.frames, up.pos_missing), (streamed as u64, 0));
    assert_eq!((rep.stored, rep.landed), (streamed as u64, streamed as u64));
    let batches = rep.landings.len();
    assert_eq!((batches as u64, had.len()), (up.batches, batches));
    assert_eq!(batches, streamed.div_ceil(BATCH));
    let dry = had.iter().filter(|h| !**h).count() as u64;
    assert_eq!(dry, up.counts_dry);
    assert!(
        case.fills == 0 || case.fills >= batches,
        "a pool run dry by chance"
    );
    let fill = rep.fill.as_ref().unwrap();
    assert_eq!(fill.no_counts, dry);

    ctx.bind_to_thread().unwrap();
    let st = ctx.new_stream().unwrap();
    let dev = DeviceBuffer::from_host(&st, &counts).unwrap();
    let mut live = LiveAp::new(&st, &cfg.ap, &cfg.probe, dev, false).unwrap();
    let mut backer = live.backer();
    let mut pf = PatchFill::new(&ctx, rep.shape, PATCH, PACK, fill.calls.len()).unwrap();
    let mut pred = Refusing::new(case.fail, None);
    let (nmode, frame) = (cfg.ap.shape.nmode() as usize, [n, n]);
    let (nprobe, nobj) = (cfg.ap.shape.nprobe(), cfg.ap.shape.nobj());
    let (mut error, mut not_ready) = (None, 0);
    // The frames and patch corners of the batches landed with counts that
    // no call took yet, in landing order.
    let mut queued = VecDeque::new();
    let mut calls = fill.calls.iter().peekable();
    let mut at = 0;
    for (i, &(epochs, upto)) in rep.landings.iter().enumerate() {
        while live.epochs() < epochs {
            live.epoch().unwrap();
        }
        let (a, b) = (at as usize, upto as usize);
        let mut v = Vec::new();
        for &q in &pos[a..b] {
            cfg.place.views(q, nprobe, nobj, &mut v);
        }
        backer.back(&v).unwrap();
        assert_eq!(u64::from(live.push(&v).unwrap()), upto - at);
        at = upto;
        if had[i] {
            let corners: Vec<[u32; 2]> = v
                .chunks(nmode)
                .map(|v| patch_corner([v[0].oy, v[0].ox], frame, PATCH))
                .collect();
            queued.push_back((a..b, corners));
        }
        // The fills that reached the run before the next batch.
        while let Some(c) = calls.next_if(|c| c.landings == i + 1) {
            while live.epochs() < c.epochs {
                live.epoch().unwrap();
            }
            assert!(
                c.batches as usize <= queued.len(),
                "{c:?} before its batches"
            );
            let held: Vec<_> = queued.drain(..c.batches as usize).collect();
            let frames: usize = held.iter().map(|(r, _)| r.len()).sum();
            let end = held.last().map(|(r, _)| r.end as u64);
            assert_eq!((c.frames as usize, Some(c.upto)), (frames, end), "{c:?}");
            assert!(frames <= PACK, "{c:?}");
            let cc: Vec<u16> = held
                .iter()
                .flat_map(|(r, _)| &counts[r.start * n * n..r.end * n * n])
                .copied()
                .collect();
            let corners: Vec<[u32; 2]> = held.iter().flat_map(|(_, c)| c).copied().collect();
            let mut out = vec![[0.0; 2]; frames * PATCH[0] * PATCH[1]];
            let fate = match pred.predict(&cc, frame, &mut out) {
                Err(e) => {
                    error.get_or_insert(e);
                    Fate::Failed
                }
                Ok(()) if case.fills == 0 => Fate::NoBuffer,
                Ok(()) => {
                    let f = pf.stitch(&corners, &out).unwrap();
                    not_ready += live.fill(f).unwrap();
                    Fate::Written
                }
            };
            assert_eq!(c.fate, fate, "{} GPUs: {c:?}", case.ngpu);
        }
    }
    assert_eq!(calls.next(), None);
    assert!(queued.is_empty(), "{} batches in no call", queued.len());
    assert_eq!((&fill.error, fill.not_ready), (&error, not_ready));
    let ncalls = fill.calls.len() as u64;
    let stitched = ncalls - fill.of(Fate::Failed).0 as u64;
    assert_eq!(
        (fill.copy.count(), fill.predict.count(), fill.stitch.count()),
        (ncalls, ncalls, stitched)
    );

    let last = rep.kept.last().unwrap();
    while live.epochs() < last.epochs {
        live.epoch().unwrap();
    }
    let mut snap = live.new_snapshot().unwrap();
    live.snapshot(&mut snap).unwrap();
    assert_eq!(rep.shape, snap.shape());
    let px = snap.wait().unwrap();
    let diff = px[..last.pixels.len()]
        .iter()
        .zip(&last.pixels)
        .filter(|(a, b)| a.map(f32::to_bits) != b.map(f32::to_bits))
        .count();
    assert_eq!(diff, 0, "{} GPUs: pixels unlike the replay's", case.ngpu);
    rep
}

/// With one tap buffer, lent to a batch queued behind the predictor's
/// first call, held until a batch went by without counts, batches land
/// unfilled while it is lent, and the tap lends again once the call ended.
#[test]
fn batches_sent_without_counts_land_unfilled() {
    let rep = run(Case {
        counts: 1,
        gate: true,
        ..Case::new(1, 240)
    });
    let fill = rep.fill.unwrap();
    let written = fill.of(Fate::Written).0;
    assert!(fill.no_counts >= 1 && written >= 2, "{written}");
}

/// With no fill buffer, every call is predicted, finds none free, and
/// leaves its batches' frames unfilled.
#[test]
fn a_fill_finding_no_buffer_free_is_skipped() {
    let rep = run(Case {
        fills: 0,
        ..Case::new(1, 2 * BATCH)
    });
    let fill = rep.fill.unwrap();
    assert_eq!(fill.of(Fate::NoBuffer).1, rep.landings.len() as u64);
}

/// A predictor failing on its first and fourth calls leaves those calls'
/// batches unfilled and the run going, its first error reported.
#[test]
fn a_failing_predictor_skips_its_batches_and_reports_its_first_error() {
    let rep = run(Case {
        fail: &[1, 4],
        ..Case::new(1, 240)
    });
    let fill = rep.fill.unwrap();
    assert_eq!(fill.of(Fate::Failed).0, 2);
    assert_eq!(fill.error.as_deref(), Some("call 1 refused"));
}

/// Planned to its first row only, an object of several chunks is backed
/// by the mapper as the scan goes, ahead of each batch: every fill lands on
/// tiles its batch's push readied, none of its pixels skipped.
#[test]
fn fills_land_whole_on_tiles_backed_mid_scan() {
    let rep = run(Case {
        plan: Some(12),
        step: 48.0,
        ..Case::new(1, 240)
    });
    assert!(
        rep.obj_mapped_bytes > rep.plan_mapped_bytes,
        "{} bytes mapped by the plan, {} by the scan, of a {:?} object",
        rep.plan_mapped_bytes,
        rep.obj_mapped_bytes,
        rep.shape
    );
    assert_eq!(rep.fill.unwrap().not_ready, 0);
}

#[test]
fn a_filling_run_on_four_gpus_is_one_gpus_replay() {
    run(Case::new(4, 240));
}
