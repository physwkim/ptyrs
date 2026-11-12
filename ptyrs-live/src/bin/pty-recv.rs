//! Receiver: PVA monitor of `<prefix>FRAME` and `<prefix>POS` → decode
//! workers → uniqueId position match → crop → pinned ring → async H2D on
//! GPU `--gpu`.
//!
//! The monitor callback only parses the NTNDArray head and hands a
//! zero-copy slice of the body to the workers. Workers reuse one image-sized
//! scratch each; the ring, its device mirror and the events are allocated
//! before the first frame.

use std::net::SocketAddr;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use crossbeam_channel::{Receiver, TrySendError};
use cuda_core::Context;
use epics_pva_rs::client_native::PvaClient;
use epics_pva_rs::pv_request::PvRequestBuilder;
use epics_pva_rs::pvdata::FieldDesc;
use ptyrs_live::args::Args;
use ptyrs_live::codec::{
    Bslz4Scratch, Codec, bslz4_decompress_into_u16, bslz4_decompress_range_u16, bytemuck_u16,
    bytemuck_u16_mut, lz4_decompress_into,
};
use ptyrs_live::ntnd::{FrameLayout, check_position_layout, parse_frame_head, parse_position};
use ptyrs_live::pos::{Pos, PosTable};
use ptyrs_live::ring::{Ring, UploaderConfig, run_uploader};
use ptyrs_live::sched::{parse_cpu_list, pin_current, set_util_min};
use ptyrs_live::stats::{Histogram, mono_ns, process_cpu_ns, thread_cpu_ns, wall_ns};
use ptyrs_live::synth::{Region, Synth, SynthScratch, position};

#[derive(Clone, Copy)]
struct Cfg {
    width: usize,
    height: usize,
    crop: usize,
    x0: usize,
    y0: usize,
    crop_decode: bool,
    verify_every: i32,
    pos_wait_ns: i64,
    synth: Synth,
}

impl Cfg {
    fn crop_region(&self) -> Region {
        Region {
            x0: self.x0,
            y0: self.y0,
            w: self.crop,
            h: self.crop,
        }
    }
}

struct Work {
    payload: Bytes,
    codec: Codec,
    uid: i32,
    t_rx: i64,
    t_pub: i64,
}

/// Monitor-callback bookkeeping; owned by the callback, read after shutdown.
#[derive(Default)]
struct RxState {
    layout: Option<(usize, FrameLayout)>,
    codec: Option<Codec>,
    codec_param: Option<i32>,
    uncompressed: i64,
    ts: (i64, i32),
    seen: Vec<u64>,
    unique: u64,
    dup: u64,
    beyond: u64,
    last_uid: i64,
    gap_events: u64,
    gap_frames: u64,
    out_of_order: u64,
    queue_full: u64,
    rejected: u64,
    placeholders: u64,
    first_rx: Option<i64>,
    last_rx: i64,
    cpu_first: i64,
    cpu_sample: (i64, i64),
    first_error: Option<String>,
}

impl RxState {
    fn reject(&mut self, why: String) {
        self.rejected += 1;
        self.first_error.get_or_insert(why);
    }
}

#[derive(Default)]
struct WorkerReport {
    frames: u64,
    decode_err: u64,
    pos_missing: u64,
    pos_waited: u64,
    pos_mismatch: u64,
    verified: u64,
    verify_bad: u64,
    cpu_ns: i64,
    decode: Option<Histogram>,
    /// Frames whose position had already arrived when the frame did.
    pos_first: u64,
    /// Frame arrival → position arrival, for the rest.
    pos_after_frame: Option<Histogram>,
}

/// Where each receive-path thread runs: the runtime threads share `rt`, the
/// uploader and every worker get a CPU of their own. Empty without `--cpus`.
struct Placement {
    rt: Option<Vec<usize>>,
    h2d: Option<usize>,
    workers: Vec<Option<usize>>,
    util_min: u32,
}

impl Placement {
    fn new(cpus: &str, rt_threads: usize, workers: usize, util_min: u32) -> Self {
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
    fn enter(&self, cpus: Option<&[usize]>) {
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

fn worker(rx: Receiver<Work>, ring: Arc<Ring>, pos: Arc<PosTable>, cfg: Cfg) -> WorkerReport {
    let mut r = WorkerReport {
        decode: Some(Histogram::new(100_000)),
        pos_after_frame: Some(Histogram::new(1_000_000)),
        ..Default::default()
    };
    let n = cfg.width * cfg.height;
    let mut img = vec![0u16; n];
    let mut bs = Bslz4Scratch::default();
    let mut ss = SynthScratch::default();
    let mut vbuf = vec![0u16; cfg.crop * cfg.crop];
    let rows = cfg.y0 * cfg.width..(cfg.y0 + cfg.crop) * cfg.width;
    let cpu0 = thread_cpu_ns();
    'frames: for w in rx.iter() {
        let t = mono_ns();
        let res = match w.codec {
            Codec::None => Ok(()),
            Codec::Lz4 => lz4_decompress_into(&w.payload, bytemuck_u16_mut(&mut img)),
            Codec::Bslz4 if cfg.crop_decode => {
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
        r.pos_mismatch += ((x, y) != position(w.uid)) as u64;
        let mut slot = ring.reserve();
        let dst = bytemuck_u16_mut(slot.frame());
        let row_bytes = cfg.crop * 2;
        for (i, d) in dst.chunks_exact_mut(row_bytes).enumerate() {
            let off = ((cfg.y0 + i) * cfg.width + cfg.x0) * 2;
            d.copy_from_slice(&src[off..off + row_bytes]);
        }
        slot.commit(w.uid, x, y, w.t_rx, w.t_pub);
        r.frames += 1;
        if cfg.verify_every > 0 && w.uid % cfg.verify_every == 0 {
            let s = cfg.synth;
            s.fill_region(s.pool_index(w.uid), cfg.crop_region(), &mut vbuf, &mut ss);
            r.verified += 1;
            let crop_ok = (0..cfg.crop).all(|i| {
                let off = ((cfg.y0 + i) * cfg.width + cfg.x0) * 2;
                src[off..off + row_bytes] == *bytemuck_u16(&vbuf[i * cfg.crop..(i + 1) * cfg.crop])
            });
            r.verify_bad += (!crop_ok) as u64;
        }
    }
    r.cpu_ns = thread_cpu_ns() - cpu0;
    r
}

fn on_frame(
    st: &mut RxState,
    desc: &FieldDesc,
    body: Bytes,
    order: epics_pva_rs::proto::ByteOrder,
    tx: &crossbeam_channel::Sender<Work>,
    cfg: &Cfg,
) {
    let t_rx = wall_ns();
    let key = desc as *const FieldDesc as usize;
    if st.layout.as_ref().map(|l| l.0) != Some(key) {
        match FrameLayout::from_desc(desc) {
            Ok(l) => st.layout = Some((key, l)),
            Err(e) => return st.reject(e.to_string()),
        }
    }
    let layout = &st.layout.as_ref().unwrap().1;
    let h = match parse_frame_head(layout, &body, order) {
        Ok(h) => h,
        Err(e) => return st.reject(e.to_string()),
    };
    let (ubyte, ushort) = (layout.ubyte_sel, layout.ushort_sel);
    if let Some(c) = h.codec {
        st.codec = c;
    }
    if let Some(p) = h.codec_param {
        st.codec_param = p;
    }
    if let Some(u) = h.uncompressed_size {
        st.uncompressed = u;
    }
    if let Some((s, ns)) = h.data_time_stamp {
        st.ts = (s.unwrap_or(st.ts.0), ns.unwrap_or(st.ts.1));
    }
    let Some(uid) = h.unique_id else {
        return st.reject("update without uniqueId".into());
    };
    if uid < 0 {
        st.placeholders += 1;
        return;
    }
    let Some((sel, n, range)) = h.value else {
        return st.reject("update without value".into());
    };
    let Some(codec) = st.codec else {
        return st.reject("unknown codec.name".into());
    };
    let image = (cfg.width * cfg.height) as i64;
    let ok = match codec {
        Codec::None => sel == ushort && n as i64 == image,
        _ => sel == ubyte && st.codec_param == Some(6) && st.uncompressed == 2 * image,
    };
    if !ok {
        return st.reject(format!(
            "payload does not match a {}x{} u16 image",
            cfg.width, cfg.height
        ));
    }
    if st.first_rx.is_none() {
        st.first_rx = Some(t_rx);
        st.cpu_first = process_cpu_ns();
    }
    st.last_rx = t_rx;
    let u = uid as usize;
    if u / 64 >= st.seen.len() {
        st.beyond += 1;
        return;
    }
    let (word, b) = (&mut st.seen[u / 64], 1u64 << (u % 64));
    if *word & b != 0 {
        st.dup += 1;
        return;
    }
    *word |= b;
    st.unique += 1;
    if st.unique.is_multiple_of(1024) {
        st.cpu_sample = (t_rx, process_cpu_ns());
    }
    let next = st.last_uid + 1;
    match (uid as i64).cmp(&next) {
        std::cmp::Ordering::Greater => {
            st.gap_events += 1;
            st.gap_frames += uid as u64 - next as u64;
        }
        std::cmp::Ordering::Less => st.out_of_order += 1,
        std::cmp::Ordering::Equal => {}
    }
    st.last_uid = st.last_uid.max(uid as i64);
    let t_pub = st.ts.0 * 1_000_000_000 + st.ts.1 as i64;
    let payload = body.slice(range as Range<usize>);
    match tx.try_send(Work {
        payload,
        codec,
        uid,
        t_rx,
        t_pub,
    }) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => st.queue_full += 1,
        Err(TrySendError::Disconnected(_)) => {}
    }
}

fn main() {
    let a = Args::parse();
    let addr: SocketAddr = a.str("addr", "127.0.0.1:15075").parse().expect("--addr");
    let frames: u64 = a.get("frames", 30_000);
    let workers: usize = a.get("workers", 16);
    let synth = Synth {
        width: a.get("width", 1028),
        height: a.get("height", 1062),
        seed: a.get("seed", 1),
        flux: a.get("flux", 1.0),
        pool: a.get("pool", 64),
    };
    let crop: usize = a.get("crop", 256);
    let (cx, cy) = synth.centre();
    let (cx, cy): (usize, usize) = (a.get("cx", cx), a.get("cy", cy));
    let cfg = Cfg {
        width: synth.width,
        height: synth.height,
        crop,
        x0: cx - crop / 2,
        y0: cy - crop / 2,
        // Only the crop rows reach the GPU; bslz4 blocks outside them are skipped.
        crop_decode: a.get::<u8>("crop-decode", 1) != 0,
        verify_every: a.get("verify-every", 0),
        // Only a position that never comes waits this long, holding one
        // worker throughout; arrival lag on its own connection is < 1 ms.
        pos_wait_ns: a.get::<i64>("pos-wait-ms", 1000) * 1_000_000,
        synth,
    };
    assert!(
        cfg.x0 + crop <= cfg.width && cfg.y0 + crop <= cfg.height,
        "crop outside the frame"
    );
    let ucfg = UploaderConfig {
        batch: a.get("batch", 16),
        flush_ns: a.get::<i64>("flush-us", 500) * 1000,
        idle: Duration::from_micros(a.get("idle-us", 50)),
        events: 64,
    };
    let prefix = a.str("prefix", "PTY:");
    let idle_ms: i64 = a.get("idle-ms", 3000);
    let connect_s: i64 = a.get("connect-timeout-s", 120);
    let rt_threads: usize = a.get("rt-threads", 1);
    let place = Arc::new(Placement::new(
        &a.str("cpus", ""),
        rt_threads,
        workers,
        a.get("util-min", 1024),
    ));
    // The main thread only polls for completion; it shares the runtime CPUs.
    place.enter(place.rt.as_deref());

    let ctx = Context::new(a.get("gpu", 0)).expect("CUDA context");
    let ring =
        Arc::new(Ring::new(&ctx, a.get("ring-slots", 1024), crop * crop).expect("pinned ring"));
    let stop = Arc::new(AtomicBool::new(false));
    let uploader = {
        let (ring, ctx, stop, place) = (ring.clone(), ctx.clone(), stop.clone(), place.clone());
        std::thread::Builder::new()
            .name("h2d".into())
            .spawn(move || {
                place.enter(place.h2d.as_ref().map(std::slice::from_ref));
                run_uploader(ring, ctx, ucfg, stop)
            })
            .unwrap()
    };
    let (tx, rx) = crossbeam_channel::bounded::<Work>(a.get("queue", 256));
    let pos = Arc::new(PosTable::new(1 << 16));
    let handles: Vec<_> = (0..workers)
        .map(|i| {
            let (rx, ring, pos, place) = (rx.clone(), ring.clone(), pos.clone(), place.clone());
            std::thread::Builder::new()
                .name(format!("decode{i}"))
                .spawn(move || {
                    place.enter(place.workers[i].as_ref().map(std::slice::from_ref));
                    worker(rx, ring, pos, cfg)
                })
                .unwrap()
        })
        .collect();
    drop(rx);

    let state = Arc::new(Mutex::new(RxState {
        seen: vec![0; (frames as usize).div_ceil(64)],
        last_uid: -1,
        ..Default::default()
    }));
    let unique = Arc::new(AtomicU64::new(0));
    let last_rx = Arc::new(AtomicI64::new(0));
    let pos_stats = Arc::new(Mutex::new((0u64, 0u64, None::<String>)));
    let rt = {
        let place = place.clone();
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(rt_threads)
            .on_thread_start(move || place.enter(place.rt.as_deref()))
            .enable_all()
            .build()
            .unwrap()
    };
    // POS gets a client, hence a TCP connection, of its own, as it would
    // from a separate motion IOC: on FRAME's connection it queued behind
    // the frame backlog and trailed frames by up to 0.9 s. The deep
    // pipelined queue keeps the server from squashing it during stalls.
    let pos_request = PvRequestBuilder::new()
        .record("pipeline", true)
        .record("queueSize", a.get::<u32>("pos-queue", 4096))
        .build()
        .to_pv_field();
    let t_start = mono_ns();
    rt.block_on(async {
        let client = PvaClient::builder()
            .server_addr(addr)
            .timeout(Duration::from_secs(10))
            // The server squashes FRAME updates once this many are unacked;
            // lz4 at 3 kHz runs the reader near its ceiling, and 256 (85 ms)
            // did not always cover a reader stall.
            .pipeline_size(a.get("pipeline", 1024))
            .build();
        let pos_client = PvaClient::builder()
            .server_addr(addr)
            .timeout(Duration::from_secs(10))
            .build();
        let pt = pos.clone();
        let ps = pos_stats.clone();
        let pos_sub = pos_client
            .pvmonitor_raw_frames_handle_with_request(
                &format!("{prefix}POS"),
                pos_request,
                move |desc, body, order| {
                    let mut s = ps.lock().unwrap();
                    let r = check_position_layout(desc).and_then(|()| parse_position(&body, order));
                    match r {
                        Ok((Some(uid), Some(x), Some(y))) if uid >= 0 => {
                            pt.insert(
                                uid,
                                Pos {
                                    x,
                                    y,
                                    t_rx: wall_ns(),
                                },
                            );
                            s.0 += 1;
                        }
                        Ok((Some(uid), ..)) if uid < 0 => {}
                        Ok(_) => {
                            s.1 += 1;
                            s.2.get_or_insert("partial position update".into());
                        }
                        Err(e) => {
                            s.1 += 1;
                            s.2.get_or_insert(e.to_string());
                        }
                    }
                },
                |ev| eprintln!("recv POS {ev:?}"),
            )
            .await
            .expect("POS monitor");
        let (st, un, lr) = (state.clone(), unique.clone(), last_rx.clone());
        let frame_sub = client
            .pvmonitor_raw_frames_handle(
                &format!("{prefix}FRAME"),
                move |desc, body, order| {
                    let mut s = st.lock().unwrap();
                    on_frame(&mut s, desc, body, order, &tx, &cfg);
                    un.store(s.unique, Ordering::Relaxed);
                    lr.store(s.last_rx, Ordering::Relaxed);
                },
                |ev| eprintln!("recv FRAME {ev:?}"),
            )
            .await
            .expect("FRAME monitor");
        loop {
            tokio::time::sleep(Duration::from_millis(20)).await;
            let lr = last_rx.load(Ordering::Relaxed);
            if unique.load(Ordering::Relaxed) >= frames {
                break;
            }
            if lr != 0 && wall_ns() - lr > idle_ms * 1_000_000 {
                eprintln!("recv idle for {idle_ms} ms, stopping");
                break;
            }
            if lr == 0 && mono_ns() - t_start > connect_s * 1_000_000_000 {
                eprintln!("recv no frame within {connect_s} s, stopping");
                break;
            }
        }
        drop(frame_sub);
        drop(pos_sub);
    });
    // Dropping the runtime drops the monitor callbacks and with them the
    // only Sender, so the workers drain the queue and return.
    rt.shutdown_timeout(Duration::from_secs(2));
    let reports: Vec<WorkerReport> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    stop.store(true, Ordering::Release);
    let up = uploader.join().unwrap().expect("uploader");
    let t_end = wall_ns();
    let cpu_end = process_cpu_ns();

    let st = state.lock().unwrap();
    let ps = pos_stats.lock().unwrap();
    let mut wr = WorkerReport::default();
    let mut decode = Histogram::new(100_000);
    let mut pos_after = Histogram::new(1_000_000);
    for r in &reports {
        wr.pos_first += r.pos_first;
        wr.frames += r.frames;
        wr.decode_err += r.decode_err;
        wr.pos_missing += r.pos_missing;
        wr.pos_waited += r.pos_waited;
        wr.pos_mismatch += r.pos_mismatch;
        wr.verified += r.verified;
        wr.verify_bad += r.verify_bad;
        wr.cpu_ns += r.cpu_ns;
    }
    for r in reports {
        if let Some(h) = r.decode {
            decode.merge(&h);
        }
        if let Some(h) = r.pos_after_frame {
            pos_after.merge(&h);
        }
    }

    // End-to-end check of whatever the ring holds now: device copy against
    // the host ring, and against a fresh synthesis of the same crop.
    let se = crop * crop;
    let slots = ring.slots();
    let host = ring.host_frames();
    let h2d_bad = (0..slots)
        .filter(|&i| host[i * se..(i + 1) * se] != up.device_frames[i * se..(i + 1) * se])
        .count();
    let filled: Vec<(usize, i32)> = (0..slots)
        .filter_map(|i| ring.slot_uid(i).map(|u| (i, u)))
        .collect();
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let e2e_bad: usize = std::thread::scope(|sc| {
        filled
            .chunks(filled.len().div_ceil(threads).max(1))
            .map(|chunk| {
                let dev = &up.device_frames;
                let dpos = &up.device_positions;
                sc.spawn(move || {
                    let mut buf = vec![0u16; se];
                    let mut ss = SynthScratch::default();
                    let s = cfg.synth;
                    chunk
                        .iter()
                        .filter(|&&(i, uid)| {
                            s.fill_region(s.pool_index(uid), cfg.crop_region(), &mut buf, &mut ss);
                            let (x, y) = position(uid);
                            dev[i * se..(i + 1) * se] != buf[..]
                                || dpos[2 * i] != x
                                || dpos[2 * i + 1] != y
                        })
                        .count()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .sum()
    });

    let first = st.first_rx.unwrap_or(t_end);
    let rx_span = (st.last_rx - first).max(1) as f64 / 1e9;
    let (cs_t, cs_cpu) = st.cpu_sample;
    let steady = if cs_t > first {
        (cs_cpu - st.cpu_first) as f64 / (cs_t - first) as f64
    } else {
        f64::NAN
    };
    let lost = frames - up.frames.min(frames);
    println!(
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
        ps.0,
        ps.1,
    );
    if let Some(e) = st.first_error.as_ref().or(ps.2.as_ref()) {
        println!("recv first_error={e}");
    }
    println!(
        "recv rx_rate_hz={:.1} cpu_cores_steady={:.2} cpu_cores_total={:.2} workers={} \
         worker_cpu_cores={:.2} h2d_cpu_cores={:.2} ring_stalls={} batches={} mean_batch={:.2} \
         max_inflight={} h2d_MBps={:.1}",
        st.unique as f64 / rx_span,
        steady,
        (cpu_end - st.cpu_first) as f64 / (t_end - first).max(1) as f64,
        workers,
        wr.cpu_ns as f64 / (t_end - first).max(1) as f64,
        up.cpu_ns as f64 / (t_end - first).max(1) as f64,
        ring.stalls(),
        up.batches,
        up.frames as f64 / up.batches.max(1) as f64,
        up.max_inflight,
        up.bytes as f64 / rx_span / 1e6,
    );
    println!("recv rx_to_gpu {}", up.rx_to_gpu.summary());
    println!("recv pub_to_gpu {}", up.pub_to_gpu.summary());
    println!("recv commit_to_gpu {}", up.commit_to_gpu.summary());
    println!("recv decode {}", decode.summary());
    println!(
        "recv pos_before_frame={} pos_after_frame {}",
        wr.pos_first,
        pos_after.summary()
    );
    println!(
        "recv verify sampled={} sampled_bad={} ring_slots_checked={} h2d_mismatch_slots={h2d_bad} \
         e2e_mismatch_slots={e2e_bad}",
        wr.verified,
        wr.verify_bad,
        filled.len()
    );
    let code = if lost > 0 {
        2
    } else if wr.verify_bad + h2d_bad as u64 + e2e_bad as u64 + wr.pos_mismatch > 0 {
        3
    } else {
        0
    };
    std::process::exit(code);
}
