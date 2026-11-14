//! The fill hook wired end to end: frames through the uploader into the
//! consumer, each run against one GPU's replay of what became of each
//! batch's fill.

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
use crate::live::follow::{FillHook, FollowCfg, FollowReport, ReconCfg, run_dist_consumer};
use crate::live::pos::{Pos, PosTable};
use crate::live::recon::LiveAp;
use crate::live::ring::{FrameBatch, Ring, UploaderConfig, run_region_uploader};
use crate::live::sent::sent;

const PATCH: [usize; 2] = [32, 32];
const BATCH: usize = 4;

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
    /// [`FillHook::counts`] and [`FillHook::fills`].
    counts: usize,
    fills: usize,
    fail: &'static [u64],
    /// The predictor's first call holds its counts until a batch went by
    /// without any.
    gate: bool,
    /// Batches, from 0, whose fill finds every buffer in flight.
    no_buffer: &'static [usize],
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
            no_buffer: &[],
            plan: None,
            step: 8.0,
        }
    }
}

/// What each count of a [`crate::live::follow::FillReport`] should be.
#[derive(Debug, Default, PartialEq)]
struct Tally {
    written: u64,
    no_counts: u64,
    failed: u64,
    no_buffer: u64,
    dropped: u64,
    not_ready: usize,
    predicted: u64,
    error: Option<String>,
}

/// Streams `case.frames` of a 12-column raster through the uploader, in
/// full batches of [`BATCH`] but the last, into the consumer filling with
/// [`Refusing`] on `case.ngpu` GPUs, a relay between them recording which
/// batches came with counts. Replays the landings on one GPU, each batch's
/// fill predicted, refused, skipped or stitched as the relay and `case`
/// say, and checks the run's fill report and its last kept object against
/// the replay's, bit for bit. Returns the run's report.
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
    let mut cfg = FollowCfg::plan(&p, NonZeroU32::new(total as u32).unwrap(), &rc);
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
    let mut k = 0;
    for burst in [50, 1, 90, 0, 99] {
        for _ in 0..burst.min(total - k) {
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
    assert_eq!(k, total);
    stop.store(true, Ordering::Release);
    let up = uploader.join().unwrap().unwrap();
    let rep = consumer.join().unwrap().unwrap();
    let had = relay.join().unwrap();
    assert_eq!((up.frames, up.pos_missing), (total as u64, 0));
    assert_eq!((rep.stored, rep.landed), (total as u64, total as u64));
    let batches = rep.landings.len();
    assert_eq!((batches as u64, had.len()), (up.batches, batches));
    assert_eq!(batches, total.div_ceil(BATCH));
    let dry = had.iter().filter(|h| !**h).count() as u64;
    assert_eq!(dry, up.counts_dry);

    ctx.bind_to_thread().unwrap();
    let st = ctx.new_stream().unwrap();
    let dev = DeviceBuffer::from_host(&st, &counts).unwrap();
    let mut live = LiveAp::new(&st, &cfg.ap, &cfg.probe, dev, false).unwrap();
    let mut backer = live.backer();
    // Never dry: `case.no_buffer` names the batches the run found it so.
    let mut pf = PatchFill::new(&ctx, rep.shape, PATCH, BATCH, batches).unwrap();
    let mut pred = Refusing::new(case.fail, None);
    let (nmode, frame) = (cfg.ap.shape.nmode as usize, [n, n]);
    let (nprobe, nobj) = (cfg.ap.shape.nprobe, cfg.ap.shape.nobj);
    let mut want = Tally::default();
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
        let whole = u64::from(live.push(&v).unwrap()) == upto - at;
        at = upto;
        if !had[i] {
            want.no_counts += 1;
            continue;
        }
        let corners: Vec<[u32; 2]> = v
            .chunks(nmode)
            .map(|v| patch_corner([v[0].oy, v[0].ox], frame, PATCH))
            .collect();
        let mut out = vec![[0.0; 2]; corners.len() * PATCH[0] * PATCH[1]];
        if let Err(e) = pred.predict(&counts[a * n * n..b * n * n], frame, &mut out) {
            want.failed += 1;
            want.error.get_or_insert(e);
            continue;
        }
        want.predicted += 1;
        if case.no_buffer.contains(&i) {
            want.no_buffer += 1;
            continue;
        }
        let f = pf.stitch(&corners, &out).unwrap();
        if whole {
            want.written += 1;
            want.not_ready += live.fill(f).unwrap();
        } else {
            want.dropped += 1;
        }
    }
    let fill = rep.fill.as_ref().unwrap();
    let got = Tally {
        written: fill.written,
        no_counts: fill.no_counts,
        failed: fill.failed,
        no_buffer: fill.no_buffer,
        dropped: fill.dropped,
        not_ready: fill.not_ready,
        predicted: fill.predict.count(),
        error: fill.error.clone(),
    };
    assert_eq!(got, want, "{} GPUs: the fill report", case.ngpu);
    assert_eq!(fill.stitch.count(), want.predicted);

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

/// With one tap buffer, held by the predictor until a batch went by
/// without counts, batches land unfilled while it is lent, and the tap
/// lends again once the predictor let go.
#[test]
fn batches_sent_without_counts_land_unfilled() {
    let rep = run(Case {
        counts: 1,
        gate: true,
        ..Case::new(1, 240)
    });
    let fill = rep.fill.unwrap();
    assert!(fill.no_counts >= 1 && fill.written >= 2, "{}", fill.written);
}

/// With one fill buffer, the second of two batches finds it on the object
/// still, under the first batch's copies, and lands unfilled.
#[test]
fn a_fill_finding_no_buffer_free_is_skipped() {
    let rep = run(Case {
        fills: 1,
        no_buffer: &[1],
        ..Case::new(1, 2 * BATCH)
    });
    assert_eq!(rep.fill.unwrap().written, 1);
}

/// A predictor failing on its first and fourth calls leaves those batches
/// unfilled and the run going, its first error reported.
#[test]
fn a_failing_predictor_skips_its_batches_and_reports_its_first_error() {
    let rep = run(Case {
        fail: &[1, 4],
        ..Case::new(1, 240)
    });
    let fill = rep.fill.unwrap();
    assert_eq!(fill.failed, 2);
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
