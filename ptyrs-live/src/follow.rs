//! The live AP consumer (DESIGN.md §3 B6, §7 M5): lands each frame batch
//! the uploader sends into a [`LiveAp`] and runs epochs over the landed
//! frames between batches, from graphs captured before the first frame.
//!
//! Once every frame of the scan has landed, or the uploader stopped with
//! some missing, it goes on iterating and keeps a host copy of the object
//! every `snap_every` epochs, stamped with the wall time the host had it,
//! for [`Clock::check`]; a copy from before then lacks frames the offline
//! run has. At most two epochs are queued ahead of the host, so frames that
//! arrive land behind at most one epoch already running.
//! [`offline_ap`] runs the same AP on the same stored frames until it
//! settles, the reference the clock compares against.

use std::num::NonZeroU32;
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender, TryRecvError};
use cuda_core::Context;
use ptyrs_core::{C64, CoreError};
use ptyrs_gpu::{C32, ViewAddr};

use crate::converge::{Clock, ClockError};
use crate::grow::GrowError;
use crate::object::{Snapshot, sleeping_event};
use crate::recon::{ApConfig, LiveAp, ViewPlacement};
use crate::ring::{DeviceStore, FrameBatch};
use crate::stats::wall_ns;

/// What the live run reconstructs with, fixed before the first frame.
#[derive(Clone, Debug)]
pub struct FollowCfg {
    /// `shape.nframe` is the scan's frame count.
    pub ap: ApConfig,
    /// `[nprobe, n, n]`, natural order.
    pub probe: Vec<C32>,
    pub place: ViewPlacement,
    /// Epochs between kept snapshots.
    pub snap_every: NonZeroU32,
    /// Snapshots kept; the run stops after the last.
    pub snaps: usize,
}

/// Object 0 of the live run some epochs after its scan landed.
pub struct Kept {
    /// Wall ns once the host had it.
    pub taken: i64,
    /// Epochs it follows.
    pub epochs: u64,
    /// `[ny, nx]` row-major.
    pub pixels: Vec<[f32; 2]>,
}

pub struct FollowReport {
    /// Frames the uploader sent.
    pub stored: u64,
    /// Frames the live scan took, the first `landed` stored ones.
    pub landed: u64,
    pub epochs: u64,
    /// Epochs queued before the scan had landed; `None` if no frame came.
    pub scan_epochs: Option<u64>,
    /// `[ny, nx]` of the object.
    pub shape: [usize; 2],
    /// `(x, y)` m of each stored frame, by sequence number.
    pub positions: Vec<(f64, f64)>,
    pub kept: Vec<Kept>,
    pub obj_mapped_bytes: usize,
    /// Per batch, the epochs queued before it landed and the frames stored
    /// with it: the schedule that, replayed, gives the same object.
    pub landings: Vec<(u64, u64)>,
}

/// The views of the frames at `positions`.
fn views_of(cfg: &ApConfig, place: &ViewPlacement, positions: &[(f64, f64)]) -> Vec<ViewAddr> {
    let (nprobe, nobj) = (cfg.shape.nprobe, cfg.shape.nobj);
    let mut out = Vec::with_capacity(positions.len() * (nprobe * nobj) as usize);
    for &p in positions {
        place.views(p, nprobe, nobj, &mut out);
    }
    out
}

/// Live AP on `store`'s frames as `batches` say they landed, until the
/// uploader drops its sender and `cfg.snaps` snapshots are kept. Sends on
/// `captured` once its graphs are captured, and drops it unsent if that
/// fails.
pub fn run_ap_consumer(
    ctx: Arc<Context>,
    store: Arc<DeviceStore>,
    cfg: FollowCfg,
    batches: Receiver<FrameBatch>,
    captured: Sender<()>,
) -> Result<FollowReport, GrowError> {
    ctx.bind_to_thread()?;
    let stream = ctx.new_stream()?;
    let shape = cfg.ap.shape;
    let elems = shape.frame[0] as usize * shape.frame[1] as usize;
    if store.frame_elems() != elems || store.max_frames() < u64::from(shape.nframe) {
        return Err(ptyrs_gpu::Error::Shape(format!(
            "a store of {} frames of {} pixels for a scan of {} frames of {:?}",
            store.max_frames(),
            store.frame_elems(),
            shape.nframe,
            shape.frame
        ))
        .into());
    }
    // SAFETY: an epoch reads only landed frames, whose copies into the
    // store completed before the uploader sent their batch.
    let counts = unsafe { store.frames_view()? };
    let mut live = LiveAp::new(&stream, &cfg.ap, &cfg.probe, counts)?;
    let mut snaps = [live.new_snapshot()?, live.new_snapshot()?];
    let fence = [sleeping_event(&ctx)?, sleeping_event(&ctx)?];
    let _ = captured.send(());

    let mut positions: Vec<(f64, f64)> = Vec::with_capacity(shape.nframe as usize);
    let mut kept = Vec::with_capacity(cfg.snaps);
    let (mut open, mut scan_epochs) = (true, None);
    // The snapshot buffer queued last and the epochs it follows.
    let mut pending: Option<(usize, u64)> = None;
    let mut next_snap = 0usize;
    let mut landings = Vec::new();
    let fed = (|| -> Result<(), GrowError> {
        let mut got = Vec::new();
        while kept.len() + usize::from(pending.is_some()) < cfg.snaps {
            // Block for frames only while none has come.
            if open && positions.is_empty() {
                match batches.recv() {
                    Ok(b) => got.push(b),
                    Err(_) => open = false,
                }
            }
            while open {
                match batches.try_recv() {
                    Ok(b) => got.push(b),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => open = false,
                }
            }
            for b in got.drain(..) {
                assert_eq!(
                    b.seq0,
                    positions.len() as u64,
                    "batches arrive in order and without gaps"
                );
                live.push(&views_of(&cfg.ap, &cfg.place, &b.positions))?;
                positions.extend(b.positions);
                landings.push((live.epochs(), positions.len() as u64));
            }
            if positions.is_empty() {
                return Ok(());
            }
            let done = positions.len() as u64 == u64::from(shape.nframe) || !open;
            if done && scan_epochs.is_none() {
                scan_epochs = Some(live.epochs());
            }
            let f = &fence[(live.epochs() % 2) as usize];
            // The epoch two before this one has finished.
            f.synchronize()?;
            live.epoch()?;
            f.record(&stream)?;
            if let Some(k) = scan_epochs.map(|s| live.epochs() - s)
                && k.is_multiple_of(u64::from(cfg.snap_every.get()))
            {
                let i = next_snap;
                next_snap ^= 1;
                live.snapshot(&mut snaps[i])?;
                if let Some((j, epochs)) = pending.replace((i, live.epochs())) {
                    kept.push(keep(&mut snaps[j], epochs)?);
                }
            }
        }
        if let Some((i, epochs)) = pending.take() {
            kept.push(keep(&mut snaps[i], epochs)?);
        }
        Ok(())
    })();
    // Nothing may still read the store once this returns.
    stream.synchronize()?;
    fed?;
    let [ny, nx] = snaps[0].shape();
    Ok(FollowReport {
        stored: positions.len() as u64,
        landed: u64::from(live.scan().landed()),
        epochs: live.epochs(),
        scan_epochs,
        shape: [ny, nx],
        positions,
        kept,
        obj_mapped_bytes: live.scan().obj().mapped_bytes(),
        landings,
    })
}

/// Waits for `snap` and copies its object 0, stamped now.
fn keep(snap: &mut Snapshot, epochs: u64) -> Result<Kept, GrowError> {
    let [ny, nx] = snap.shape();
    let px = snap.wait()?;
    let taken = wall_ns();
    Ok(Kept {
        taken,
        epochs,
        pixels: px[..ny * nx].to_vec(),
    })
}

/// How long [`offline_ap`] iterates: `every` epochs at a time, at least
/// two such steps and at most `max` epochs in all.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settle {
    pub every: NonZeroU32,
    pub max: u64,
}

pub struct Offline {
    pub epochs: u64,
    /// The clock against the run's last object, or why it would not start.
    pub clock: Result<Clock, ClockError>,
}

/// AP as the live run does it on `store`'s frames `0..positions.len()` at
/// `positions`, landed at once, `settle.every` epochs at a time until its
/// object matches the one `every` epochs before over `region` to the
/// clock's threshold, or `settle.max` epochs have run.
pub fn offline_ap(
    ctx: &Arc<Context>,
    store: &DeviceStore,
    cfg: &FollowCfg,
    positions: &[(f64, f64)],
    region: [Range<usize>; 2],
    scan_end: i64,
    settle: Settle,
) -> Result<Offline, GrowError> {
    ctx.bind_to_thread()?;
    let stream = ctx.new_stream()?;
    let mut ap = cfg.ap;
    ap.shape.nframe = u32::try_from(positions.len())
        .ok()
        .filter(|&n| n <= cfg.ap.shape.nframe)
        .ok_or_else(|| {
            ptyrs_gpu::Error::Shape(format!(
                "{} frames for a scan of {}",
                positions.len(),
                cfg.ap.shape.nframe
            ))
        })?;
    // SAFETY: every frame read is stored, and nothing writes the store now.
    let counts = unsafe { store.frames_view()? };
    let mut run = LiveAp::new(&stream, &ap, &cfg.probe, counts)?;
    run.push(&views_of(&ap, &cfg.place, positions))?;
    let mut snap = run.new_snapshot()?;
    let mut earlier: Option<Vec<C64>> = None;
    loop {
        for _ in 0..settle.every.get() {
            run.epoch()?;
        }
        run.snapshot(&mut snap)?;
        let shape = snap.shape();
        let now = plane(snap.wait()?, shape);
        if let Some(e) = &earlier {
            match Clock::new(&now, e, shape, region.clone(), scan_end) {
                Err(ClockError::Moving { .. }) if run.epochs() < settle.max => {}
                clock => {
                    return Ok(Offline {
                        epochs: run.epochs(),
                        clock,
                    });
                }
            }
        }
        earlier = Some(now);
    }
}

/// Object 0 of a snapshot's pixels, `[ny, nx]` = `shape`.
pub fn plane(px: &[[f32; 2]], shape: [usize; 2]) -> Vec<C64> {
    px[..shape[0] * shape[1]]
        .iter()
        .map(|&[re, im]| C64::new(re.into(), im.into()))
        .collect()
}

/// The first of `kept`, in order, that `clock` finds converged, with its
/// time from the scan's end; checked on `threads` threads.
pub fn first_converged(
    clock: &Clock,
    kept: &[Kept],
    shape: [usize; 2],
    threads: usize,
) -> Result<Option<(usize, Duration)>, CoreError> {
    let per = kept.len().div_ceil(threads.max(1)).max(1);
    std::thread::scope(|sc| {
        let hs: Vec<_> = kept
            .chunks(per)
            .enumerate()
            .map(|(c, chunk)| {
                sc.spawn(move || -> Result<Option<(usize, Duration)>, CoreError> {
                    for (i, k) in chunk.iter().enumerate() {
                        if let Some(d) = clock.check(&plane(&k.pixels, shape), k.taken)? {
                            return Ok(Some((c * per + i, d)));
                        }
                    }
                    Ok(None)
                })
            })
            .collect();
        let mut first = None;
        for h in hs {
            if let (None, Some(hit)) = (first, h.join().unwrap()?) {
                first = Some(hit);
            }
        }
        Ok(first)
    })
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;
    use crate::farfield::{Ptycho, PtychoSpec, ScanPath};
    use crate::graph::WarmUp;
    use crate::ring::{Ring, UploaderConfig, run_uploader};
    use crate::scan::ScanShape;

    struct Streamed {
        ctx: Arc<Context>,
        store: Arc<DeviceStore>,
        cfg: FollowCfg,
        rep: FollowReport,
        /// Wall ns just before the last frame was committed.
        scan_end: i64,
        recaptures: u64,
    }

    /// A 6x6 raster of 64x64 frames, two probe modes, streamed through the
    /// ring and the uploader in bursts 20 ms apart into the consumer.
    fn streamed() -> Streamed {
        let p = Ptycho::new(PtychoSpec {
            n: 64,
            path: ScanPath::Raster { cols: 6 },
            points: 36,
            step: 12.0,
            photons: 1e6,
            seed: 5,
            wavelength: 1.5498e-10,
            distance: 5.0,
            det_pixel: 75e-6,
        });
        let (n, total) = (64usize, p.points());
        let pos: Vec<(f64, f64)> = (0..total).map(|k| p.position(k)).collect();
        let dx = p.spec().dx();
        let (place, reach) = ViewPlacement::plan([dx, dx], &pos);
        let shape = ScanShape {
            nframe: total as u32,
            nmode: 2,
            nprobe: 2,
            nobj: 1,
            frame: [n as u32; 2],
            reach,
            quad_phase: 0.0,
        };
        let mut ap = ApConfig {
            shape,
            nf_max: NonZeroU32::new(16).unwrap(),
            cap: 0,
            fill: [1.0, 0.0],
            chunk: 1,
            inertia: 0.01,
        };
        ap.cap = shape.tile_cap(&views_of(&ap, &place, &pos));
        let cfg = FollowCfg {
            ap,
            probe: p
                .probe()
                .iter()
                .map(|c| C32 { re: c.re, im: c.im })
                .collect(),
            place,
            snap_every: NonZeroU32::new(25).unwrap(),
            snaps: 80,
        };

        let ctx = Context::new(0).expect("GPU 0");
        let ring = Arc::new(Ring::new(&ctx, 16, n * n).unwrap());
        let store = Arc::new(DeviceStore::new(&ctx, total, n * n, 1).unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = crossbeam_channel::unbounded();
        let (cap_tx, cap_rx) = crossbeam_channel::bounded(1);
        let consumer = {
            let (ctx, store, cfg) = (ctx.clone(), store.clone(), cfg.clone());
            std::thread::spawn(move || run_ap_consumer(ctx, store, cfg, rx, cap_tx))
        };
        cap_rx.recv().expect("the graphs are captured");
        let warm = WarmUp::end();
        let uploader = {
            let (ring, ctx, stop, store) = (ring.clone(), ctx.clone(), stop.clone(), store.clone());
            let up = UploaderConfig {
                batch: NonZeroUsize::new(4).unwrap(),
                flush_ns: 100_000,
                idle: Duration::from_micros(20),
                events: 4,
            };
            std::thread::spawn(move || run_uploader(ring, ctx, store, up, stop, Some(tx)))
        };
        let (mut k, mut scan_end) = (0, 0);
        for burst in [5, 1, 13, 0, 17] {
            for _ in 0..burst {
                if k == total - 1 {
                    scan_end = wall_ns();
                }
                let mut s = ring.reserve();
                s.frame().copy_from_slice(&p.frame(k));
                s.commit(k as i32, pos[k].0, pos[k].1, 0, 0);
                k += 1;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(k, total);
        stop.store(true, Ordering::Release);
        assert_eq!(uploader.join().unwrap().unwrap().frames, total as u64);
        let rep = consumer.join().unwrap().unwrap();
        assert_eq!(rep.positions, pos);
        Streamed {
            ctx,
            store,
            cfg,
            rep,
            scan_end,
            recaptures: warm.recaptures(),
        }
    }

    /// The consumer lands every streamed frame at its streamed position from
    /// graphs captured before the first, with epochs between landings, and
    /// keeps snapshots every 25 epochs once the scan has landed, each
    /// stamped after the scan's end. Its last one is the object of its
    /// landings replayed on an idle stream, each landing and epoch waited
    /// for: no landing raced an epoch in flight.
    ///
    /// The live object is not held to the offline one here: after thousands
    /// of epochs over a partial scan, AP on these frames stops at 0.98 in
    /// the outer rings against the settled offline run.
    #[test]
    fn the_live_object_is_its_landings_replayed_one_at_a_time() {
        let Streamed {
            ctx,
            store,
            cfg,
            rep,
            scan_end,
            recaptures,
        } = streamed();
        assert_eq!(recaptures, 0);
        let total = rep.positions.len() as u64;
        assert_eq!((rep.stored, rep.landed), (total, total));
        let scan_epochs = rep.scan_epochs.unwrap();
        assert!(scan_epochs > 0, "no epoch ran while the scan landed");
        assert_eq!(rep.landings.last().unwrap(), &(scan_epochs, total));
        assert_eq!(rep.kept.len(), cfg.snaps);
        for (i, kept) in rep.kept.iter().enumerate() {
            assert_eq!(kept.epochs, scan_epochs + 25 * (i as u64 + 1));
            assert!(kept.taken > scan_end);
        }
        let last = rep.kept.last().unwrap();
        assert_eq!(rep.epochs, last.epochs);

        ctx.bind_to_thread().unwrap();
        let st = ctx.new_stream().unwrap();
        // SAFETY: the consumer has returned; nothing writes the store.
        let counts = unsafe { store.frames_view().unwrap() };
        let mut run = LiveAp::new(&st, &cfg.ap, &cfg.probe, counts).unwrap();
        let mut at = 0;
        for &(epochs, upto) in &rep.landings {
            while run.epochs() < epochs {
                run.epoch().unwrap();
                st.synchronize().unwrap();
            }
            let p = &rep.positions[at as usize..upto as usize];
            run.push(&views_of(&cfg.ap, &cfg.place, p)).unwrap();
            st.synchronize().unwrap();
            at = upto;
        }
        while run.epochs() < last.epochs {
            run.epoch().unwrap();
        }
        let mut snap = run.new_snapshot().unwrap();
        run.snapshot(&mut snap).unwrap();
        let got = snap.wait().unwrap();
        assert!(got[..last.pixels.len()] == last.pixels[..]);
    }

    /// The offline run settles within its cap and refuses as still moving
    /// when the cap stops it first. Against a clock on the live run's own
    /// last snapshot, the search on one thread per snapshot finds the same
    /// first converged one as an in-order search, timed from the scan's end.
    #[test]
    fn the_offline_run_settles_and_the_search_finds_the_first_match() {
        let s = streamed();
        let rep = &s.rep;
        let region = s.cfg.place.scanned(&rep.positions, s.cfg.ap.shape.frame);
        let offline = |max| {
            let settle = Settle {
                every: NonZeroU32::new(100).unwrap(),
                max,
            };
            let r = region.clone();
            offline_ap(
                &s.ctx,
                &s.store,
                &s.cfg,
                &rep.positions,
                r,
                s.scan_end,
                settle,
            )
            .unwrap()
        };
        let settled = offline(3000);
        assert!(settled.clock.is_ok());
        assert!((200..=3000).contains(&settled.epochs));
        let capped = offline(200);
        assert_eq!(capped.epochs, 200);
        assert!(matches!(capped.clock, Err(ClockError::Moving { .. })));

        let n = rep.kept.len();
        let obj = |i: usize| plane(&rep.kept[i].pixels, rep.shape);
        let clock = Clock::new(&obj(n - 1), &obj(n - 2), rep.shape, region, s.scan_end).unwrap();
        let first = (0..n)
            .find(|&i| clock.check(&obj(i), rep.kept[i].taken).unwrap().is_some())
            .unwrap();
        let (i, after) = first_converged(&clock, &rep.kept, rep.shape, n)
            .unwrap()
            .unwrap();
        assert_eq!(i, first);
        assert_eq!(after.as_nanos() as i64, rep.kept[i].taken - s.scan_end);
    }
}
