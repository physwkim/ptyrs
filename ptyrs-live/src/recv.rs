//! The receive path: frames from a PVA monitor of `<prefix>FRAME`, an Eiger
//! stream v1 PULL socket or an HDF5 SWMR file → decode workers → uniqueId
//! position match → crop → pinned ring → async H2D into a device store
//! indexed by sequence number, checked against the simulator's model.
//! `pty-recv` builds its [`RecvCfg`] from `--key value` flags, `ptyrs live`
//! from a TOML file with the same keys.
//!
//! The source only parses the frame metadata and hands a zero-copy slice
//! of the received buffer to the workers. Workers reuse one image-sized
//! scratch each; the ring, the store's address range and the events are
//! allocated before the first frame, and the store maps device memory in
//! `store-chunk-mib` steps as frames arrive. A graph captured once before
//! the first frame stands in for the engine and checksums every stored
//! frame, `graph-batch` frames per replay; with `recon` set, the live
//! reconstruction of [`crate::follow`] takes the frames instead and is
//! timed from the scan's end against its offline run.

use std::fmt::{self, Write as _};
use std::net::SocketAddr;
use std::num::{NonZeroU32, NonZeroUsize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cuda_core::Context;
use epics_pva_rs::client_native::PvaClient;
use serde::Deserialize;

use crate::args::Args;
use crate::consumer::{ConsumerReport, checksum, run_checksum_consumer};
use crate::eiger::{EigerFrames, Progress, PullReport, run_pull};
use crate::follow::{
    FollowCfg, FollowReport, ReconCfg, first_converged, offline_ap, plane, run_ap_consumer,
};
use crate::graph::WarmUp;
use crate::handoff::work_queue;
use crate::model::{Model, ModelCfg};
use crate::pipeline::{
    FrameLoss, Geometry, Placement, SourceStats, Truth, WorkerCfg, WorkerReport, worker,
};
use crate::pos::PosTable;
use crate::pva_source::{NtndFrames, PosStats, monitor_positions};
use crate::ring::{DeviceStore, Ring, UploaderConfig, run_uploader};
use crate::stats::{mono_ns, process_cpu_ns, wall_ns};
use crate::swmr::{SwmrCfg, SwmrReport, run_swmr};
use crate::synth::Region;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct RecvCfg {
    /// Triggers in the scan, uids `0..frames`.
    pub frames: u64,
    /// CUDA device ordinal among `CUDA_VISIBLE_DEVICES`.
    pub gpu: usize,
    pub source: SourceCfg,
    /// PVA server of `<prefix>FRAME` (source pva) and `<prefix>POS` (pva, zmq).
    pub addr: SocketAddr,
    pub prefix: String,
    /// FRAME updates the server may send unacknowledged. It squashes the rest;
    /// lz4 at 3 kHz runs the reader near its ceiling, and 256 (85 ms) did not
    /// always cover a reader stall.
    pub pipeline: u32,
    pub pos_queue: u32,
    /// What the frames and positions are checked against; its frame size is
    /// the detector's.
    pub model: ModelCfg,
    pub crop: usize,
    /// Crop centre, frame pixels; the frame centre when unset.
    pub cx: Option<usize>,
    pub cy: Option<usize>,
    /// Decode only the bslz4 blocks overlapping the crop rows.
    pub crop_decode: bool,
    pub workers: NonZeroUsize,
    /// Frames between the source and the workers; a full queue drops.
    pub queue: NonZeroUsize,
    /// Regenerate and compare the crop of every uid divisible by this; 0 off.
    pub verify_every: i32,
    /// Only a position that never comes waits this long, holding one worker
    /// throughout; arrival lag on its own connection is < 1 ms.
    pub pos_wait_ms: i64,
    pub batch: NonZeroUsize,
    pub flush_us: i64,
    pub idle_us: u64,
    pub ring_slots: usize,
    /// Device store capacity; `frames` when unset.
    pub store_frames: Option<usize>,
    pub store_chunk_mib: usize,
    pub graph_batch: NonZeroU32,
    pub rt_threads: usize,
    /// CPU list (`2-23,48-71`) the threads are pinned to; unpinned when unset.
    pub cpus: Option<String>,
    pub util_min: u32,
    /// Stop once no frame arrived for this long.
    pub idle_ms: i64,
    /// Stop if no frame arrived within this long.
    pub connect_timeout_s: i64,
    /// Reconstruct live instead of checksumming; needs the `ptycho` model
    /// and the whole frame as the crop.
    pub recon: Option<ReconCfg>,
}

impl Default for RecvCfg {
    fn default() -> Self {
        Self {
            frames: 30_000,
            gpu: 0,
            source: SourceCfg::Pva,
            addr: ([127, 0, 0, 1], 15075).into(),
            prefix: "PTY:".into(),
            pipeline: 1024,
            pos_queue: 4096,
            model: ModelCfg::default(),
            crop: 256,
            cx: None,
            cy: None,
            crop_decode: true,
            workers: NonZeroUsize::new(16).unwrap(),
            queue: NonZeroUsize::new(256).unwrap(),
            verify_every: 0,
            pos_wait_ms: 1000,
            batch: NonZeroUsize::new(16).unwrap(),
            flush_us: 500,
            idle_us: 50,
            ring_slots: 1024,
            store_frames: None,
            store_chunk_mib: 32,
            graph_batch: NonZeroU32::new(32).unwrap(),
            rt_threads: 1,
            cpus: None,
            util_min: 1024,
            idle_ms: 3000,
            connect_timeout_s: 120,
            recon: None,
        }
    }
}

/// Where the frames come from. TOML: `source = "pva"`, or a `[source.zmq]`
/// or `[source.h5]` table.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub enum SourceCfg {
    /// NTNDArray monitor of `<prefix>FRAME`.
    Pva,
    /// Eiger stream v1; positions still from `<prefix>POS`.
    Zmq(ZmqCfg),
    /// The file carries its own positions.
    H5(H5Cfg),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ZmqCfg {
    pub endpoint: String,
    pub rcvhwm: i32,
}

impl Default for ZmqCfg {
    fn default() -> Self {
        Self {
            endpoint: format!("tcp://127.0.0.1:{}", crate::eiger::PORT),
            rcvhwm: 10_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct H5Cfg {
    pub path: PathBuf,
    #[serde(default = "H5Cfg::readers")]
    pub readers: usize,
    /// Sleep between refreshes that found nothing new.
    #[serde(default = "H5Cfg::poll_us")]
    pub poll_us: u64,
    /// Frames visible but not yet taken by a reader.
    #[serde(default = "H5Cfg::queue")]
    pub queue: usize,
}

impl H5Cfg {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            readers: Self::readers(),
            poll_us: Self::poll_us(),
            queue: Self::queue(),
        }
    }

    fn readers() -> usize {
        4
    }

    fn poll_us() -> u64 {
        200
    }

    fn queue() -> usize {
        1024
    }
}

impl RecvCfg {
    /// pty-recv's flags: every field under its TOML key, plus `--source
    /// zmq --zmq ENDPOINT --rcvhwm N`, `--source h5 --h5 PATH --h5-readers
    /// --h5-poll-us --h5-queue`, `--crop-decode 0|1`, [`ModelCfg::from_args`]
    /// and [`ReconCfg::from_args`].
    pub fn from_args(a: &Args) -> Self {
        let d = Self::default();
        let source = match a.str("source", "pva").as_str() {
            "pva" => SourceCfg::Pva,
            "zmq" => {
                let z = ZmqCfg::default();
                SourceCfg::Zmq(ZmqCfg {
                    endpoint: a.str("zmq", &z.endpoint),
                    rcvhwm: a.get("rcvhwm", z.rcvhwm),
                })
            }
            "h5" => {
                let path: PathBuf = a
                    .opt("h5")
                    .unwrap_or_else(|| panic!("--source h5 needs --h5 PATH"));
                let h = H5Cfg::new(path);
                SourceCfg::H5(H5Cfg {
                    readers: a.get("h5-readers", h.readers),
                    poll_us: a.get("h5-poll-us", h.poll_us),
                    queue: a.get("h5-queue", h.queue),
                    ..h
                })
            }
            s => panic!("--source {s}: expected pva, zmq or h5"),
        };
        Self {
            frames: a.get("frames", d.frames),
            gpu: a.get("gpu", d.gpu),
            source,
            addr: a.get("addr", d.addr),
            prefix: a.str("prefix", &d.prefix),
            pipeline: a.get("pipeline", d.pipeline),
            pos_queue: a.get("pos-queue", d.pos_queue),
            model: ModelCfg::from_args(a),
            crop: a.get("crop", d.crop),
            cx: a.opt("cx"),
            cy: a.opt("cy"),
            crop_decode: a.get::<u8>("crop-decode", d.crop_decode.into()) != 0,
            // With no worker every queued frame is dropped, so 0 fails here.
            workers: a.get("workers", d.workers),
            queue: a.get("queue", d.queue),
            verify_every: a.get("verify-every", d.verify_every),
            pos_wait_ms: a.get("pos-wait-ms", d.pos_wait_ms),
            batch: a.get("batch", d.batch),
            flush_us: a.get("flush-us", d.flush_us),
            idle_us: a.get("idle-us", d.idle_us),
            ring_slots: a.get("ring-slots", d.ring_slots),
            store_frames: a.opt("store-frames"),
            store_chunk_mib: a.get("store-chunk-mib", d.store_chunk_mib),
            graph_batch: a.get("graph-batch", d.graph_batch),
            rt_threads: a.get("rt-threads", d.rt_threads),
            cpus: a.opt("cpus"),
            util_min: a.get("util-min", d.util_min),
            idle_ms: a.get("idle-ms", d.idle_ms),
            connect_timeout_s: a.get("connect-timeout-s", d.connect_timeout_s),
            recon: ReconCfg::from_args(a),
        }
    }
}

/// A step of [`run`] that failed, with its cause.
#[derive(Debug)]
pub struct RecvError {
    what: &'static str,
    cause: String,
}

impl fmt::Display for RecvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.what, self.cause)
    }
}

impl std::error::Error for RecvError {}

fn fail<E: fmt::Display>(what: &'static str) -> impl FnOnce(E) -> RecvError {
    move |e| RecvError {
        what,
        cause: e.to_string(),
    }
}

/// A report figure to the given decimals, or `none` when it is undefined.
struct Fig(Option<f64>, usize);

impl fmt::Display for Fig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(v) => write!(f, "{v:.*}", self.1),
            None => f.write_str("none"),
        }
    }
}

pub struct RecvReport {
    /// Triggers that never reached the device store, received or not.
    pub lost: u64,
    /// Frames received that the GPU never consumed.
    pub rx_lost: FrameLoss,
    /// Graphs captured after the warm-up.
    pub recaptures: u64,
    /// Frames, positions and checksums that reached the GPU wrong, all checks.
    pub bad: usize,
    /// With `recon`, whether a live object matched the settled offline run.
    pub converged: Option<bool>,
    /// The `recv ...` lines, one per check group.
    pub lines: String,
}

impl RecvReport {
    /// 2 lost, 3 wrong, 4 a graph captured after the warm-up, 5 no live
    /// object matched, else 0.
    pub fn exit_code(&self) -> u8 {
        if self.lost > 0 || self.rx_lost.total() > 0 {
            2
        } else if self.bad > 0 {
            3
        } else if self.recaptures > 0 {
            4
        } else if self.converged == Some(false) {
            5
        } else {
            0
        }
    }
}

/// What the graph thread did with the stored frames.
enum Consumed {
    Sums(ConsumerReport),
    Recon(Box<Recon>),
}

struct Recon {
    rep: FollowReport,
    follow: FollowCfg,
    settle_max: u64,
}

/// The live run `rc` asks for on `cfg`'s scan: it starts from the
/// simulator's probe and planned positions, and takes whole frames.
fn follow_cfg(cfg: &RecvCfg, model: &Model, rc: &ReconCfg) -> Result<FollowCfg, RecvError> {
    let refuse = |cause: String| RecvError {
        what: "recon",
        cause,
    };
    let Model::Ptycho(p) = model else {
        return Err(refuse("needs --model ptycho for its probe and scan".into()));
    };
    let n = p.spec().n;
    if cfg.crop != n
        || [cfg.cx, cfg.cy]
            .iter()
            .any(|&c| c.is_some_and(|c| c != n / 2))
    {
        return Err(refuse(format!("needs the whole {n}x{n} frame as the crop")));
    }
    let nframe = u32::try_from(cfg.frames)
        .ok()
        .and_then(NonZeroU32::new)
        .ok_or_else(|| refuse(format!("{} frames", cfg.frames)))?;
    Ok(FollowCfg::plan(p, nframe, rc))
}

/// Stored frames, sequence number `s` holding trigger `uids[s]`, that differ
/// from their synthesis over `r`; read back a slab at a time and compared
/// on `threads` threads.
fn stored_bad(
    store: &DeviceStore,
    uids: &[i32],
    truth: &dyn Truth,
    r: Region,
    threads: usize,
) -> Result<usize, RecvError> {
    let se = store.frame_elems();
    let slab = 1024;
    let mut dev = vec![0u16; slab * se];
    let mut bad = 0;
    for (k, uids) in uids.chunks(slab).enumerate() {
        let s0 = (k * slab) as u64;
        let dev = &mut dev[..uids.len() * se];
        store
            .read_frames(s0..s0 + uids.len() as u64, dev)
            .map_err(fail("store readback"))?;
        let per = uids.len().div_ceil(threads).max(1);
        bad += std::thread::scope(|sc| {
            uids.chunks(per)
                .zip(dev.chunks(per * se))
                .map(|(uids, dev)| {
                    sc.spawn(move || {
                        let mut buf = vec![0u16; se];
                        uids.iter()
                            .zip(dev.chunks(se))
                            .filter(|&(&uid, d)| {
                                truth.fill(uid, r, &mut buf);
                                buf[..] != *d
                            })
                            .count()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().unwrap())
                .sum::<usize>()
        });
    }
    Ok(bad)
}

/// The `recv recon` line and whether a live object matched: the offline run
/// on the stored frames, the clock from `scan_end` over the scanned region,
/// and the first kept snapshot it finds converged.
fn recon_line(
    ctx: &Arc<Context>,
    store: &DeviceStore,
    follow: &FollowCfg,
    settle_max: u64,
    rep: &FollowReport,
    scan_end: Option<i64>,
    threads: usize,
) -> Result<(String, bool), RecvError> {
    let [ny, nx] = rep.shape;
    let region = follow.place.scanned(&rep.positions, follow.ap.shape.frame);
    let [rows, cols] = &region;
    let scan_epochs = rep.scan_epochs.unwrap_or(0);
    // Epoch time once the scan landed, from the kept snapshots' stamps.
    let epoch_ms = match (rep.kept.first(), rep.kept.last()) {
        (Some(a), Some(b)) if b.epochs > a.epochs => {
            Some((b.taken - a.taken) as f64 / 1e6 / (b.epochs - a.epochs) as f64)
        }
        _ => None,
    };
    let mut l = format!(
        "recv recon algo=ap nf_max={} tile_cap={} stored={} landed={} scan_epochs={scan_epochs} \
         epochs={} snap_every={} kept={} epoch_ms={} object={ny}x{nx} obj_mapped_MiB={} \
         crop=window_centre_bbox region_rows={}..{} region_cols={}..{}",
        follow.ap.nf_max,
        follow.ap.cap,
        rep.stored,
        rep.landed,
        rep.epochs,
        follow.snap_every,
        rep.kept.len(),
        Fig(epoch_ms, 3),
        rep.obj_mapped_bytes >> 20,
        rows.start,
        rows.end,
        cols.start,
        cols.end,
    );
    let Some(scan_end) = scan_end else {
        l += " clock=none no_scan_end";
        return Ok((l, false));
    };
    let off = offline_ap(
        ctx,
        store,
        follow,
        &rep.positions,
        region,
        scan_end,
        settle_max,
    )
    .map_err(fail("offline reconstruction"))?;
    // Writing to a String cannot fail.
    write!(l, " offline_epochs={}", off.epochs).unwrap();
    let clock = match off.clock {
        Ok(c) => {
            write!(l, " offline_settle_frc={:.4}", c.settled()).unwrap();
            c
        }
        Err(e) => {
            write!(l, " clock=none offline: {e}").unwrap();
            return Ok((l, false));
        }
    };
    let frc = |px: &[[f32; 2]]| -> Result<(f64, f64), RecvError> {
        let o = plane(px, rep.shape);
        let both = clock.region_frc(&o).and_then(|r| Ok((r, clock.whole(&o)?)));
        both.map_err(fail("clock"))
    };
    let hit = first_converged(&clock, &rep.kept, rep.shape, threads).map_err(fail("clock"))?;
    match hit {
        Some((i, after)) => {
            let k = &rep.kept[i];
            let (r, w) = frc(&k.pixels)?;
            write!(
                l,
                " converged_ms={:.1} converged_epochs={} frc_region={r:.4} frc_whole={w:.4}",
                after.as_secs_f64() * 1e3,
                k.epochs - scan_epochs
            )
            .unwrap();
        }
        None => l += " converged_ms=none",
    }
    if let Some(k) = rep.kept.last() {
        let (r, w) = frc(&k.pixels)?;
        write!(l, " last_frc_region={r:.4} last_frc_whole={w:.4}").unwrap();
    }
    Ok((l, hit.is_some()))
}

/// Receive one scan as `cfg` says, then check what reached the GPU. Pins
/// the calling thread to the runtime CPUs while frames arrive and to every
/// listed CPU for the checks.
pub fn run(cfg: &RecvCfg) -> Result<RecvReport, RecvError> {
    let frames = cfg.frames;
    let workers = cfg.workers.get();
    let model = cfg.model.build();
    let (width, height) = model.frame_size();
    let crop = cfg.crop;
    let mut geom = Geometry::centred(
        width,
        height,
        crop,
        cfg.cx.unwrap_or(width / 2),
        cfg.cy.unwrap_or(height / 2),
    );
    // Only the crop rows reach the GPU; bslz4 blocks outside them are skipped.
    geom.crop_decode = cfg.crop_decode;
    let wcfg = WorkerCfg {
        geom,
        verify_every: cfg.verify_every,
        pos_wait_ns: cfg.pos_wait_ms * 1_000_000,
    };
    let follow = cfg
        .recon
        .as_ref()
        .map(|rc| Ok::<_, RecvError>((follow_cfg(cfg, &model, rc)?, rc.settle_max)))
        .transpose()?;
    let truth: Arc<dyn Truth> = Arc::new(model);
    let ucfg = UploaderConfig {
        batch: cfg.batch,
        flush_ns: cfg.flush_us * 1000,
        idle: Duration::from_micros(cfg.idle_us),
        events: 64,
    };
    let h5_readers = match &cfg.source {
        SourceCfg::H5(h) => h.readers,
        _ => 0,
    };
    let place = Arc::new(Placement::new(
        cfg.cpus.as_deref().unwrap_or(""),
        cfg.rt_threads,
        workers,
        h5_readers,
        cfg.util_min,
    ));
    // The calling thread only polls for completion; it shares the runtime CPUs.
    place.enter(place.rt.as_deref());

    let ctx = Context::new(cfg.gpu).map_err(fail("CUDA context"))?;
    let ring = Arc::new(Ring::new(&ctx, cfg.ring_slots, crop * crop).map_err(fail("pinned ring"))?);
    let store = Arc::new(
        DeviceStore::new(
            &ctx,
            cfg.store_frames.unwrap_or(frames as usize),
            crop * crop,
            cfg.store_chunk_mib << 20,
        )
        .map_err(fail("device store"))?,
    );
    let (btx, brx) = crossbeam_channel::unbounded();
    let (cap_tx, cap_rx) = crossbeam_channel::bounded(1);
    let consumer = {
        let (ctx, store, place) = (ctx.clone(), store.clone(), place.clone());
        let batch = cfg.graph_batch;
        std::thread::Builder::new()
            .name("graph".into())
            .spawn(move || {
                place.enter(place.rt.as_deref());
                match follow {
                    None => {
                        run_checksum_consumer(ctx, store, batch, brx, cap_tx).map(Consumed::Sums)
                    }
                    Some((follow, settle_max)) => {
                        run_ap_consumer(ctx, store, follow.clone(), brx, cap_tx).map(|rep| {
                            Consumed::Recon(Box::new(Recon {
                                rep,
                                follow,
                                settle_max,
                            }))
                        })
                    }
                }
            })
            .unwrap()
    };
    // The warm-up ends once the graph is captured, before any source starts.
    let warm = match cap_rx.recv() {
        Ok(()) => WarmUp::end(),
        Err(_) => {
            let failed = consumer.join().unwrap().err();
            return Err(fail("graph capture")(failed.map_or_else(
                || "the consumer ended before its capture".to_string(),
                |e| e.to_string(),
            )));
        }
    };
    // From here on every exit, failed or not, goes through the one shutdown
    // below: the source and its Sender go, the workers drain, the uploader
    // lands what they committed and the consumer sees its last batch.
    let stop = Arc::new(AtomicBool::new(false));
    let uploader = {
        let (ring, ctx, stop, place) = (ring.clone(), ctx.clone(), stop.clone(), place.clone());
        let store = store.clone();
        std::thread::Builder::new()
            .name("h2d".into())
            .spawn(move || {
                place.enter(place.h2d.as_ref().map(std::slice::from_ref));
                run_uploader(ring, ctx, store, ucfg, stop, Some(btx))
            })
            .unwrap()
    };
    let (tx, rx) = work_queue(cfg.queue);
    let pos = Arc::new(PosTable::new(1 << 16));
    let handles: Vec<_> = (0..workers)
        .map(|i| {
            let (rx, ring, pos, place) = (rx.clone(), ring.clone(), pos.clone(), place.clone());
            let truth = truth.clone();
            std::thread::Builder::new()
                .name(format!("decode{i}"))
                .spawn(move || {
                    place.enter(place.workers[i].as_ref().map(std::slice::from_ref));
                    worker(rx, ring, pos, wcfg, truth)
                })
                .unwrap()
        })
        .collect();
    drop(rx);

    let state = Arc::new(Mutex::new(SourceStats::new(frames)));
    let progress = Arc::new(Progress::default());
    let pos_stats = Arc::new(Mutex::new(PosStats::default()));
    let rt = {
        let place = place.clone();
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(cfg.rt_threads)
            .on_thread_start(move || place.enter(place.rt.as_deref()))
            .enable_all()
            .build()
            .unwrap()
    };
    let pull_stop = Arc::new(AtomicBool::new(false));
    let t_start = mono_ns();
    let streamed: Result<(Option<PullReport>, Option<SwmrReport>), RecvError> =
        rt.block_on(async {
            let client = PvaClient::builder()
                .server_addr(cfg.addr)
                .timeout(Duration::from_secs(10))
                .pipeline_size(cfg.pipeline)
                .build();
            let pos_client = PvaClient::builder()
                .server_addr(cfg.addr)
                .timeout(Duration::from_secs(10))
                .build();
            let pos_sub = match &cfg.source {
                SourceCfg::H5(_) => None,
                _ => Some(
                    monitor_positions(
                        &pos_client,
                        &format!("{}POS", cfg.prefix),
                        cfg.pos_queue,
                        pos.clone(),
                        pos_stats.clone(),
                    )
                    .await
                    .map_err(fail("POS monitor"))?,
                ),
            };
            // Frames from exactly one source; the others hold no Sender.
            let (frame_sub, pull, follow) = match &cfg.source {
                SourceCfg::Pva => {
                    let (st, pr) = (state.clone(), progress.clone());
                    let mut src = NtndFrames::new(geom);
                    let sub = client
                        .pvmonitor_raw_frames_handle(
                            &format!("{}FRAME", cfg.prefix),
                            move |desc, body, order| {
                                let mut s = st.lock().unwrap();
                                src.on_update(&mut s, desc, body, order, &tx);
                                pr.publish(&s);
                            },
                            |ev| eprintln!("recv FRAME {ev:?}"),
                        )
                        .await
                        .map_err(fail("FRAME monitor"))?;
                    (Some(sub), None, None)
                }
                SourceCfg::Zmq(z) => {
                    let (st, pr, z) = (state.clone(), progress.clone(), z.clone());
                    let (stop, place) = (pull_stop.clone(), place.clone());
                    let h = std::thread::Builder::new()
                        .name("zmq-pull".into())
                        .spawn(move || {
                            // libzmq's I/O thread starts with the first socket
                            // and inherits this thread's CPUs.
                            place.enter(place.rt.as_deref());
                            run_pull(
                                &z.endpoint,
                                z.rcvhwm,
                                EigerFrames::new(geom),
                                st,
                                tx,
                                &stop,
                                &pr,
                            )
                        })
                        .unwrap();
                    (None, Some(h), None)
                }
                SourceCfg::H5(h) => {
                    let scfg = SwmrCfg {
                        path: h.path.clone(),
                        readers: h.readers,
                        width,
                        height,
                        poll: Duration::from_micros(h.poll_us),
                        queue: h.queue,
                    };
                    let (st, pr, pos) = (state.clone(), progress.clone(), pos.clone());
                    let (stop, place) = (pull_stop.clone(), place.clone());
                    let h = std::thread::Builder::new()
                        .name("h5-index".into())
                        .spawn(move || {
                            place.enter(place.rt.as_deref());
                            run_swmr(&scfg, &st, &pos, tx, &place, &stop, &pr)
                        })
                        .unwrap();
                    (None, None, Some(h))
                }
            };
            let (idle_ns, connect_ns) = (
                cfg.idle_ms * 1_000_000,
                cfg.connect_timeout_s * 1_000_000_000,
            );
            loop {
                tokio::time::sleep(Duration::from_millis(20)).await;
                let lr = progress.last_rx.load(Ordering::Relaxed);
                // A ZMQ series ends with dseries_end; PVA and the file have no
                // end marker.
                if !matches!(cfg.source, SourceCfg::Zmq(_))
                    && progress.unique.load(Ordering::Relaxed) >= frames
                {
                    break;
                }
                // The PULL and SWMR threads end unasked only at the series end
                // or on a failure, which their joins below report.
                if pull.as_ref().is_some_and(|h| h.is_finished())
                    || follow.as_ref().is_some_and(|h| h.is_finished())
                {
                    eprintln!("recv source ended");
                    break;
                }
                if lr != 0 && wall_ns() - lr > idle_ns {
                    eprintln!("recv idle for {} ms, stopping", cfg.idle_ms);
                    break;
                }
                if lr == 0 && mono_ns() - t_start > connect_ns {
                    eprintln!("recv no frame within {} s, stopping", cfg.connect_timeout_s);
                    break;
                }
            }
            pull_stop.store(true, Ordering::Release);
            let pulled = pull.map(|h| h.join().unwrap().map_err(fail("ZMQ PULL")));
            let followed = follow.map(|h| h.join().unwrap().map_err(fail("HDF5 SWMR")));
            drop(frame_sub);
            drop(pos_sub);
            Ok((pulled.transpose()?, followed.transpose()?))
        });
    // Dropping the runtime drops the monitor callbacks and with them the
    // only Sender, so the workers drain the queue and return.
    rt.shutdown_timeout(Duration::from_secs(2));
    let reports: Vec<WorkerReport> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    stop.store(true, Ordering::Release);
    let up = uploader.join().unwrap();
    // Every frame is stored; a live reconstruction iterates on past here.
    let t_end = wall_ns();
    let cpu_end = process_cpu_ns();
    let cons = consumer.join().unwrap();
    // Before the offline run captures graphs of its own.
    let recaptures = warm.recaptures();
    let (pulled, followed) = streamed?;
    let up = up.map_err(fail("uploader"))?;
    let cons = cons.map_err(fail("graph consumer"))?;

    let st = state.lock().unwrap();
    let ps = pos_stats.lock().unwrap();
    let (wr, decode, pos_after) = WorkerReport::merge(reports);
    // The checks below spread over every listed CPU, not the runtime's.
    place.enter(place.all.as_deref());

    // End-to-end check of the frames the ring still holds: their place in
    // the device store against the host ring slot and against a fresh
    // synthesis of the same crop; the position and uid of every stored frame.
    let se = crop * crop;
    let host = ring.host_frames();
    let filled: Vec<(usize, u64, i32)> = (0..ring.slots())
        .filter_map(|i| ring.slot(i).map(|(seq, uid)| (i, seq, uid)))
        .collect();
    let mut dev = vec![0u16; filled.len() * se];
    for (k, &(_, seq, _)) in filled.iter().enumerate() {
        store
            .read_frames(seq..seq + 1, &mut dev[k * se..(k + 1) * se])
            .map_err(fail("store readback"))?;
    }
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let per = filled.len().div_ceil(threads).max(1);
    let (h2d_bad, e2e_bad): (usize, usize) = std::thread::scope(|sc| {
        filled
            .chunks(per)
            .zip(dev.chunks(per * se))
            .map(|(chunk, dev)| {
                let truth = &truth;
                sc.spawn(move || {
                    let mut buf = vec![0u16; se];
                    let (mut h2d, mut e2e) = (0, 0);
                    for (&(i, _, uid), d) in chunk.iter().zip(dev.chunks(se)) {
                        h2d += usize::from(host[i * se..(i + 1) * se] != *d);
                        truth.fill(uid, geom.crop_region(), &mut buf);
                        e2e += usize::from(buf[..] != *d);
                    }
                    (h2d, e2e)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    });
    let dpos = store
        .read_positions(up.frames)
        .map_err(fail("store readback"))?;
    let store_pos_bad = up
        .uids
        .iter()
        .enumerate()
        .filter(|&(s, &uid)| (dpos[2 * s], dpos[2 * s + 1]) != truth.position(uid))
        .count();
    // Every stored frame as the graph saw it, or as the store holds it for
    // the reconstruction, against its synthesis.
    let sum_bad: usize = match &cons {
        Consumed::Sums(cons) => {
            let per = up.uids.len().div_ceil(threads).max(1);
            std::thread::scope(|sc| {
                up.uids
                    .chunks(per)
                    .zip(cons.checksums.chunks(per))
                    .map(|(uids, sums)| {
                        let truth = &truth;
                        sc.spawn(move || {
                            let mut buf = vec![0u16; se];
                            uids.iter()
                                .zip(sums)
                                .filter(|&(&uid, &sum)| {
                                    truth.fill(uid, geom.crop_region(), &mut buf);
                                    checksum(&buf) != sum
                                })
                                .count()
                        })
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .sum()
            })
        }
        Consumed::Recon(_) => stored_bad(&store, &up.uids, &*truth, geom.crop_region(), threads)?,
    };
    let recon = match &cons {
        Consumed::Sums(_) => None,
        Consumed::Recon(r) => Some(recon_line(
            &ctx,
            &store,
            &r.follow,
            r.settle_max,
            &r.rep,
            st.scan_end,
            threads,
        )?),
    };
    let mut sorted = up.uids.clone();
    sorted.sort_unstable();
    sorted.dedup();
    let store_dup_uids = up.uids.len() - sorted.len();

    // Rates over the receive span; none before a frame arrived or over an
    // empty span.
    let rate = |x: f64, dt: i64| (dt > 0).then(|| x / dt as f64);
    let rx = st.rx;
    let rx_span = |x: f64| rx.and_then(|r| rate(x * 1e9, r.last - r.first));
    let since_first = |x: f64| rx.and_then(|r| rate(x, t_end - r.first));
    let steady = rx.and_then(|r| {
        let (t, cpu) = r.cpu_sample?;
        rate((cpu - r.cpu_first) as f64, t - r.first)
    });
    let lost = frames - up.frames.min(frames);
    let consumed = match &cons {
        Consumed::Sums(c) => c.consumed,
        Consumed::Recon(r) => r.rep.landed,
    };
    let rx_lost = FrameLoss::new(&st, &wr, consumed);
    let mut s = String::new();
    // Writing to a String cannot fail.
    let mut line = |args: fmt::Arguments| {
        s.write_fmt(args).unwrap();
        s.push('\n');
    };
    line(format_args!(
        "recv expected={frames} received_unique={} gpu_frames={} lost={lost} rx_lost={} \
         unaccounted={} dup={} beyond={} \
         gap_events={} gap_frames={} out_of_order={} queue_full={} rejected={} placeholders={} \
         decode_err={} pos_missing={} pos_mismatch={} pos_waits={} pos_updates={} pos_rejected={}",
        st.unique,
        up.frames,
        rx_lost.total(),
        rx_lost.unaccounted,
        st.dup,
        st.beyond,
        st.gap_events,
        st.gap_frames,
        st.out_of_order,
        st.queue_full,
        st.rejected,
        st.placeholders,
        wr.decode_err,
        wr.pos_missing,
        wr.pos_mismatch,
        wr.pos_waited,
        ps.updates,
        ps.rejected,
    ));
    if let Some(p) = &pulled {
        line(format_args!(
            "recv zmq messages={} headers={} series_end={} rejected={}",
            p.messages, p.headers, p.ends, p.rejected
        ));
    }
    if let Some(f) = &followed {
        line(format_args!(
            "recv h5 readers={h5_readers} open_ms={} refreshes={} retries={} frames={} \
             max_backlog={}",
            f.open_ms, f.refreshes, f.retries, f.frames, f.max_backlog
        ));
        line(format_args!("recv h5_read {}", f.read.summary()));
    }
    if let Some(e) = st.first_error.as_ref().or(ps.first_error.as_ref()) {
        line(format_args!("recv first_error={e}"));
    }
    line(format_args!(
        "recv rx_rate_hz={} cpu_cores_steady={} cpu_cores_total={} workers={} \
         worker_cpu_cores={} h2d_cpu_cores={} ring_stalls={} batches={} mean_batch={} \
         max_inflight={} h2d_MBps={} scan_end_ms={}",
        Fig(rx_span(st.unique as f64), 1),
        Fig(steady, 2),
        Fig(
            rx.and_then(|r| rate((cpu_end - r.cpu_first) as f64, t_end - r.first)),
            2
        ),
        workers,
        Fig(since_first(wr.cpu_ns as f64), 2),
        Fig(since_first(up.cpu_ns as f64), 2),
        ring.stalls(),
        up.batches,
        Fig(
            (up.batches > 0).then(|| up.frames as f64 / up.batches as f64),
            2
        ),
        up.max_inflight,
        Fig(rx_span(up.bytes as f64 / 1e6), 1),
        // From the first arrival.
        Fig(
            rx.zip(st.scan_end).map(|(r, e)| (e - r.first) as f64 / 1e6),
            1
        ),
    ));
    line(format_args!("recv rx_to_gpu {}", up.rx_to_gpu.summary()));
    line(format_args!("recv pub_to_gpu {}", up.pub_to_gpu.summary()));
    line(format_args!(
        "recv commit_to_gpu {}",
        up.commit_to_gpu.summary()
    ));
    line(format_args!("recv decode {}", decode.summary()));
    line(format_args!(
        "recv pos_before_frame={} pos_after_frame {}",
        wr.pos_first,
        pos_after.summary()
    ));
    line(format_args!(
        "recv verify sampled={} sampled_bad={} ring_slots_checked={} h2d_mismatch_slots={h2d_bad} \
         e2e_mismatch_slots={e2e_bad} store_frames={} store_mapped_MiB={} store_pos_bad={store_pos_bad} \
         store_dup_uids={store_dup_uids}",
        wr.verified,
        wr.verify_bad,
        filled.len(),
        up.uids.len(),
        store.mapped_bytes() >> 20,
    ));
    let consumed_bad = usize::from(match &cons {
        Consumed::Sums(c) => {
            line(format_args!(
                "recv graph launches={} consumed={} checksummed={} checksum_bad={sum_bad} \
                 out_mapped_MiB={} recaptures={recaptures}",
                c.launches,
                c.consumed,
                c.checksums.len(),
                c.out_mapped_bytes >> 20,
            ));
            c.consumed != up.frames || c.checksums.len() != up.uids.len()
        }
        Consumed::Recon(r) => {
            line(format_args!(
                "recv recon_store frames_bad={sum_bad} recaptures={recaptures}"
            ));
            r.rep.stored != up.frames || r.rep.landed != r.rep.stored
        }
    });
    if let Some((l, _)) = &recon {
        line(format_args!("{l}"));
    }
    let bad = (wr.verify_bad + wr.pos_mismatch) as usize
        + h2d_bad
        + e2e_bad
        + store_pos_bad
        + store_dup_uids
        + sum_bad
        + consumed_bad;
    Ok(RecvReport {
        lost,
        rx_lost,
        recaptures,
        bad,
        converged: recon.map(|(_, c)| c),
        lines: s,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::follow::Algo;
    use crate::model::{PathKind, PtychoCfg};
    use crate::synth::Synth;

    fn flags(f: &[&str]) -> RecvCfg {
        RecvCfg::from_args(&f.iter().map(|s| s.to_string()).collect())
    }

    fn toml(t: &str) -> RecvCfg {
        ::toml::from_str(t).unwrap()
    }

    #[test]
    fn no_flags_and_an_empty_file_are_the_defaults() {
        assert_eq!(flags(&[]), RecvCfg::default());
        assert_eq!(toml(""), RecvCfg::default());
        assert_eq!(toml("source = \"pva\""), RecvCfg::default());
        let recon = RecvCfg {
            recon: Some(ReconCfg::default()),
            ..RecvCfg::default()
        };
        assert_eq!(flags(&["--recon", "ap"]), recon);
        assert_eq!(toml("[recon]"), recon);
    }

    #[test]
    fn every_field_reads_the_same_from_flags_and_toml() {
        // No `..`: a new field must be given a flag and a key here.
        let want = RecvCfg {
            frames: 1234,
            gpu: 1,
            source: SourceCfg::H5(H5Cfg {
                path: "/x/scan.h5".into(),
                readers: 2,
                poll_us: 300,
                queue: 99,
            }),
            addr: "10.0.0.2:5076".parse().unwrap(),
            prefix: "SIM:".into(),
            pipeline: 512,
            pos_queue: 100,
            model: ModelCfg::Ptycho(PtychoCfg {
                n: 128,
                path: PathKind::Raster,
                points: 150,
                cols: Some(13),
                step_px: 12.5,
                photons: 1e5,
                seed: 7,
                wavelength_m: 1e-10,
                distance_m: 3.0,
                det_pixel_m: 55e-6,
            }),
            crop: 128,
            cx: Some(500),
            cy: Some(510),
            crop_decode: false,
            workers: NonZeroUsize::new(4).unwrap(),
            queue: NonZeroUsize::new(64).unwrap(),
            verify_every: 10,
            pos_wait_ms: 50,
            batch: NonZeroUsize::new(8).unwrap(),
            flush_us: 250,
            idle_us: 20,
            ring_slots: 256,
            store_frames: Some(2000),
            store_chunk_mib: 16,
            graph_batch: NonZeroU32::new(8).unwrap(),
            rt_threads: 2,
            cpus: Some("2-5".into()),
            util_min: 0,
            idle_ms: 1000,
            connect_timeout_s: 5,
            recon: Some(ReconCfg {
                algo: Algo::Ap,
                nf_max: NonZeroU32::new(8).unwrap(),
                inertia: 0.05,
                chunk_mib: 4,
                snap_every: NonZeroU32::new(3).unwrap(),
                snaps: 50,
                settle_max: 900,
            }),
        };
        let f = flags(&[
            "--frames",
            "1234",
            "--gpu",
            "1",
            "--source",
            "h5",
            "--h5",
            "/x/scan.h5",
            "--h5-readers",
            "2",
            "--h5-poll-us",
            "300",
            "--h5-queue",
            "99",
            "--addr",
            "10.0.0.2:5076",
            "--prefix",
            "SIM:",
            "--pipeline",
            "512",
            "--pos-queue",
            "100",
            "--model",
            "ptycho",
            "--n",
            "128",
            "--path",
            "raster",
            "--points",
            "150",
            "--cols",
            "13",
            "--step-px",
            "12.5",
            "--photons",
            "1e5",
            "--seed",
            "7",
            "--wavelength-m",
            "1e-10",
            "--distance-m",
            "3",
            "--det-pixel-m",
            "55e-6",
            "--crop",
            "128",
            "--cx",
            "500",
            "--cy",
            "510",
            "--crop-decode",
            "0",
            "--workers",
            "4",
            "--queue",
            "64",
            "--verify-every",
            "10",
            "--pos-wait-ms",
            "50",
            "--batch",
            "8",
            "--flush-us",
            "250",
            "--idle-us",
            "20",
            "--ring-slots",
            "256",
            "--store-frames",
            "2000",
            "--store-chunk-mib",
            "16",
            "--graph-batch",
            "8",
            "--rt-threads",
            "2",
            "--cpus",
            "2-5",
            "--util-min",
            "0",
            "--idle-ms",
            "1000",
            "--connect-timeout-s",
            "5",
            "--recon",
            "ap",
            "--recon-nf-max",
            "8",
            "--recon-inertia",
            "0.05",
            "--recon-chunk-mib",
            "4",
            "--recon-snap-every",
            "3",
            "--recon-snaps",
            "50",
            "--recon-settle-max",
            "900",
        ]);
        let t = toml(
            r#"
            frames = 1234
            gpu = 1
            addr = "10.0.0.2:5076"
            prefix = "SIM:"
            pipeline = 512
            pos-queue = 100
            crop = 128
            cx = 500
            cy = 510
            crop-decode = false
            workers = 4
            queue = 64
            verify-every = 10
            pos-wait-ms = 50
            batch = 8
            flush-us = 250
            idle-us = 20
            ring-slots = 256
            store-frames = 2000
            store-chunk-mib = 16
            graph-batch = 8
            rt-threads = 2
            cpus = "2-5"
            util-min = 0
            idle-ms = 1000
            connect-timeout-s = 5

            [source.h5]
            path = "/x/scan.h5"
            readers = 2
            poll-us = 300
            queue = 99

            [model.ptycho]
            n = 128
            path = "raster"
            points = 150
            cols = 13
            step-px = 12.5
            photons = 1e5
            seed = 7
            wavelength-m = 1e-10
            distance-m = 3.0
            det-pixel-m = 55e-6

            [recon]
            algo = "ap"
            nf-max = 8
            inertia = 0.05
            chunk-mib = 4
            snap-every = 3
            snaps = 50
            settle-max = 900
            "#,
        );
        assert_eq!(f, want);
        assert_eq!(t, want);
    }

    #[test]
    fn the_zmq_source_and_synth_model_read_the_same_from_flags_and_toml() {
        let want = RecvCfg {
            source: SourceCfg::Zmq(ZmqCfg {
                endpoint: "tcp://10.0.0.3:9999".into(),
                rcvhwm: 50,
            }),
            model: ModelCfg::Synth(Synth {
                width: 300,
                height: 280,
                seed: 3,
                flux: 0.5,
                pool: 4,
            }),
            ..RecvCfg::default()
        };
        let f = flags(&[
            "--source",
            "zmq",
            "--zmq",
            "tcp://10.0.0.3:9999",
            "--rcvhwm",
            "50",
            "--width",
            "300",
            "--height",
            "280",
            "--seed",
            "3",
            "--flux",
            "0.5",
            "--pool",
            "4",
        ]);
        let t = toml(
            r#"
            [source.zmq]
            endpoint = "tcp://10.0.0.3:9999"
            rcvhwm = 50
            [model.synth]
            width = 300
            height = 280
            seed = 3
            flux = 0.5
            pool = 4
            "#,
        );
        assert_eq!(f, want);
        assert_eq!(t, want);
        // Unset keys of a source or model table keep their defaults.
        assert_eq!(
            toml("[source.zmq]\n[model.ptycho]\n"),
            RecvCfg {
                source: SourceCfg::Zmq(ZmqCfg::default()),
                model: ModelCfg::Ptycho(PtychoCfg::default()),
                ..RecvCfg::default()
            }
        );
    }

    #[test]
    fn a_file_with_a_wrong_key_or_value_is_refused() {
        for (t, needle) in [
            ("framez = 3", "framez"),
            ("[source.zmq]\nendpont = \"tcp://x:1\"", "endpont"),
            ("[source.h5]\nreaders = 2", "path"),
            ("[model.synth]\nn = 128", "`n`"),
            ("[model.ptycho]\npath = \"spiral\"", "spiral"),
            ("source = \"udp\"", "udp"),
            ("workers = 0", "workers"),
            ("batch = 0", "batch"),
            ("graph-batch = 0", "graph-batch"),
            ("[source.zmq]\n[source.h5]\npath = \"a.h5\"", "source"),
            ("[recon]\nalgo = \"dm\"", "dm"),
            ("[recon]\nsnap-every = 0", "snap-every"),
            ("[recon]\nsnapz = 3", "snapz"),
            ("[recon]\nsettle-every = 100", "settle-every"),
        ] {
            let e = ::toml::from_str::<RecvCfg>(t).unwrap_err().to_string();
            assert!(e.contains(needle), "{t:?}: {e}");
        }
    }

    /// The live run starts from the simulator's probe and planned scan and
    /// reconstructs whole frames: another model, a crop smaller or off the
    /// centre, or an empty scan is refused before any frame comes.
    #[test]
    fn recon_needs_the_ptycho_model_whole_frames_and_a_scan() {
        let rc = ReconCfg::default();
        let ptycho = ModelCfg::Ptycho(PtychoCfg {
            n: 64,
            points: 20,
            ..PtychoCfg::default()
        });
        let ok = RecvCfg {
            frames: 20,
            crop: 64,
            model: ptycho.clone(),
            ..RecvCfg::default()
        };
        let model = ok.model.build();
        let f = follow_cfg(&ok, &model, &rc).unwrap();
        assert_eq!((f.ap.shape.nframe, f.ap.shape.frame), (20, [64, 64]));
        let centred = RecvCfg {
            cx: Some(32),
            cy: Some(32),
            ..ok.clone()
        };
        assert!(follow_cfg(&centred, &model, &rc).is_ok());
        for bad in [
            RecvCfg {
                crop: 32,
                ..ok.clone()
            },
            RecvCfg {
                cx: Some(30),
                ..ok.clone()
            },
            RecvCfg {
                frames: 0,
                ..ok.clone()
            },
        ] {
            assert!(follow_cfg(&bad, &model, &rc).is_err(), "{bad:?}");
        }
        let synth = ModelCfg::default().build();
        assert!(follow_cfg(&ok, &synth, &rc).is_err());
    }
}
