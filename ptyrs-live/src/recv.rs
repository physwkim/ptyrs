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
//! frame, `graph-batch` frames per replay.

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
use crate::consumer::{checksum, run_checksum_consumer};
use crate::eiger::{EigerFrames, Progress, PullReport, run_pull};
use crate::handoff::work_queue;
use crate::model::ModelCfg;
use crate::pipeline::{Geometry, Placement, SourceStats, Truth, WorkerCfg, WorkerReport, worker};
use crate::pos::PosTable;
use crate::pva_source::{NtndFrames, PosStats, monitor_positions};
use crate::ring::{DeviceStore, Ring, UploaderConfig, run_uploader};
use crate::stats::{mono_ns, process_cpu_ns, wall_ns};
use crate::swmr::{SwmrCfg, SwmrReport, run_swmr};

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
    /// --h5-poll-us --h5-queue`, `--crop-decode 0|1` and [`ModelCfg::from_args`].
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
    /// Triggers that never reached the device store.
    pub lost: u64,
    /// Frames, positions and checksums that reached the GPU wrong, all checks.
    pub bad: usize,
    /// The `recv ...` lines, one per check group.
    pub lines: String,
}

impl RecvReport {
    /// 2 lost, 3 wrong, else 0.
    pub fn exit_code(&self) -> u8 {
        if self.lost > 0 {
            2
        } else if self.bad > 0 {
            3
        } else {
            0
        }
    }
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
    // From here on every exit, failed or not, goes through the one shutdown
    // below: the source and its Sender go, the workers drain, the uploader
    // lands what they committed and the consumer sees its last batch.
    let (btx, brx) = crossbeam_channel::unbounded();
    let consumer = {
        let (ctx, store, place) = (ctx.clone(), store.clone(), place.clone());
        let batch = cfg.graph_batch;
        std::thread::Builder::new()
            .name("graph".into())
            .spawn(move || {
                place.enter(place.rt.as_deref());
                run_checksum_consumer(ctx, store, batch, brx)
            })
            .unwrap()
    };
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
    let cons = consumer.join().unwrap();
    let t_end = wall_ns();
    let cpu_end = process_cpu_ns();
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
    // Every stored frame as the graph saw it, against its synthesis.
    let per = up.uids.len().div_ceil(threads).max(1);
    let sum_bad: usize = std::thread::scope(|sc| {
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
    });
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
    let mut s = String::new();
    // Writing to a String cannot fail.
    let mut line = |args: fmt::Arguments| {
        s.write_fmt(args).unwrap();
        s.push('\n');
    };
    line(format_args!(
        "recv expected={frames} received_unique={} gpu_frames={} lost={lost} dup={} beyond={} \
         gap_events={} gap_frames={} out_of_order={} queue_full={} rejected={} placeholders={} \
         decode_err={} pos_missing={} pos_mismatch={} pos_waits={} pos_updates={} pos_rejected={}",
        st.unique,
        up.frames,
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
         max_inflight={} h2d_MBps={}",
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
    line(format_args!(
        "recv graph launches={} consumed={} checksummed={} checksum_bad={sum_bad} out_mapped_MiB={}",
        cons.launches,
        cons.consumed,
        cons.checksums.len(),
        cons.out_mapped_bytes >> 20,
    ));
    let consumed_bad =
        usize::from(cons.consumed != up.frames || cons.checksums.len() != up.uids.len());
    let bad = (wr.verify_bad + wr.pos_mismatch) as usize
        + h2d_bad
        + e2e_bad
        + store_pos_bad
        + store_dup_uids
        + sum_bad
        + consumed_bad;
    Ok(RecvReport {
        lost,
        bad,
        lines: s,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
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
        ] {
            let e = ::toml::from_str::<RecvCfg>(t).unwrap_err().to_string();
            assert!(e.contains(needle), "{t:?}: {e}");
        }
    }
}
