//! Receiver: frames from a PVA monitor of `<prefix>FRAME` or, with
//! `--source zmq`, an Eiger stream v1 PULL socket; positions from a PVA
//! monitor of `<prefix>POS` → decode workers → uniqueId position match →
//! crop → pinned ring → async H2D on GPU `--gpu` into a device store
//! indexed by sequence number.
//!
//! The source only parses the frame metadata and hands a zero-copy slice
//! of the received buffer to the workers. Workers reuse one image-sized
//! scratch each; the ring, the store's address range and the events are
//! allocated before the first frame, and the store maps device memory in
//! `--store-chunk-mib` steps as frames arrive. A graph captured once before
//! the first frame stands in for the engine and checksums every stored
//! frame, `--graph-batch` frames per replay.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cuda_core::Context;
use epics_pva_rs::client_native::PvaClient;
use ptyrs_live::args::Args;
use ptyrs_live::consumer::{checksum, run_checksum_consumer};
use ptyrs_live::eiger::{EigerFrames, Progress, PullReport, run_pull};
use ptyrs_live::model::Model;
use ptyrs_live::pipeline::{
    Geometry, Placement, SourceStats, Truth, Work, WorkerCfg, WorkerReport, worker,
};
use ptyrs_live::pos::PosTable;
use ptyrs_live::pva_source::{NtndFrames, PosStats, monitor_positions};
use ptyrs_live::ring::{DeviceStore, Ring, UploaderConfig, run_uploader};
use ptyrs_live::stats::{mono_ns, process_cpu_ns, wall_ns};

fn main() {
    let a = Args::parse();
    let addr: SocketAddr = a.str("addr", "127.0.0.1:15075").parse().expect("--addr");
    let frames: u64 = a.get("frames", 30_000);
    let workers: usize = a.get("workers", 16);
    // The same model arguments as pty-sim's.
    let model = Model::from_args(&a);
    let (width, height) = model.frame_size();
    let crop: usize = a.get("crop", 256);
    let mut geom = Geometry::centred(
        width,
        height,
        crop,
        a.get("cx", width / 2),
        a.get("cy", height / 2),
    );
    // Only the crop rows reach the GPU; bslz4 blocks outside them are skipped.
    geom.crop_decode = a.get::<u8>("crop-decode", 1) != 0;
    let wcfg = WorkerCfg {
        geom,
        verify_every: a.get("verify-every", 0),
        // Only a position that never comes waits this long, holding one
        // worker throughout; arrival lag on its own connection is < 1 ms.
        pos_wait_ns: a.get::<i64>("pos-wait-ms", 1000) * 1_000_000,
    };
    let truth: Arc<dyn Truth> = Arc::new(model);
    let ucfg = UploaderConfig {
        batch: a.get("batch", 16),
        flush_ns: a.get::<i64>("flush-us", 500) * 1000,
        idle: Duration::from_micros(a.get("idle-us", 50)),
        events: 64,
    };
    let zmq_ep = match a.str("source", "pva").as_str() {
        "pva" => None,
        "zmq" => Some(a.str(
            "zmq",
            &format!("tcp://127.0.0.1:{}", ptyrs_live::eiger::PORT),
        )),
        s => panic!("--source {s}: expected pva or zmq"),
    };
    let rcvhwm: i32 = a.get("rcvhwm", 10_000);
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
    let store = Arc::new(
        DeviceStore::new(
            &ctx,
            a.get("store-frames", frames as usize),
            crop * crop,
            a.get::<usize>("store-chunk-mib", 32) << 20,
        )
        .expect("device store"),
    );
    let (btx, brx) = crossbeam_channel::unbounded();
    let consumer = {
        let (ctx, store, place) = (ctx.clone(), store.clone(), place.clone());
        let batch = a.get("graph-batch", 32);
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
    let (tx, rx) = crossbeam_channel::bounded::<Work>(a.get("queue", 256));
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
            .worker_threads(rt_threads)
            .on_thread_start(move || place.enter(place.rt.as_deref()))
            .enable_all()
            .build()
            .unwrap()
    };
    let pull_stop = Arc::new(AtomicBool::new(false));
    let t_start = mono_ns();
    let pulled: Option<PullReport> = rt.block_on(async {
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
        let pos_sub = monitor_positions(
            &pos_client,
            &format!("{prefix}POS"),
            a.get("pos-queue", 4096),
            pos.clone(),
            pos_stats.clone(),
        )
        .await
        .expect("POS monitor");
        // Frames from exactly one source; the other holds no Sender.
        let (frame_sub, pull) = match &zmq_ep {
            None => {
                let (st, pr) = (state.clone(), progress.clone());
                let mut src = NtndFrames::new(geom);
                let sub = client
                    .pvmonitor_raw_frames_handle(
                        &format!("{prefix}FRAME"),
                        move |desc, body, order| {
                            let mut s = st.lock().unwrap();
                            src.on_update(&mut s, desc, body, order, &tx);
                            pr.unique.store(s.unique, Ordering::Relaxed);
                            pr.last_rx.store(s.last_rx, Ordering::Relaxed);
                        },
                        |ev| eprintln!("recv FRAME {ev:?}"),
                    )
                    .await
                    .expect("FRAME monitor");
                (Some(sub), None)
            }
            Some(ep) => {
                let (st, pr, ep) = (state.clone(), progress.clone(), ep.clone());
                let (stop, place) = (pull_stop.clone(), place.clone());
                let h = std::thread::Builder::new()
                    .name("zmq-pull".into())
                    .spawn(move || {
                        // libzmq's I/O thread starts with the first socket
                        // and inherits this thread's CPUs.
                        place.enter(place.rt.as_deref());
                        run_pull(&ep, rcvhwm, EigerFrames::new(geom), st, tx, &stop, &pr)
                    })
                    .unwrap();
                (None, Some(h))
            }
        };
        loop {
            tokio::time::sleep(Duration::from_millis(20)).await;
            let lr = progress.last_rx.load(Ordering::Relaxed);
            // A ZMQ series ends with dseries_end; PVA has no end marker.
            if zmq_ep.is_none() && progress.unique.load(Ordering::Relaxed) >= frames {
                break;
            }
            if progress.series_end.load(Ordering::Acquire) {
                eprintln!("recv series end");
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
        pull_stop.store(true, Ordering::Release);
        let pulled = pull.map(|h| h.join().unwrap().expect("ZMQ PULL"));
        drop(frame_sub);
        drop(pos_sub);
        pulled
    });
    // Dropping the runtime drops the monitor callbacks and with them the
    // only Sender, so the workers drain the queue and return.
    rt.shutdown_timeout(Duration::from_secs(2));
    let reports: Vec<WorkerReport> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    stop.store(true, Ordering::Release);
    let up = uploader.join().unwrap().expect("uploader");
    let cons = consumer.join().unwrap().expect("graph consumer");
    let t_end = wall_ns();
    let cpu_end = process_cpu_ns();

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
            .expect("store readback");
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
    let dpos = store.read_positions(up.frames).expect("store readback");
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
        ps.updates,
        ps.rejected,
    );
    if let Some(p) = &pulled {
        println!(
            "recv zmq messages={} headers={} series_end={} rejected={}",
            p.messages, p.headers, p.ends, p.rejected
        );
    }
    if let Some(e) = st.first_error.as_ref().or(ps.first_error.as_ref()) {
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
         e2e_mismatch_slots={e2e_bad} store_frames={} store_mapped_MiB={} store_pos_bad={store_pos_bad} \
         store_dup_uids={store_dup_uids}",
        wr.verified,
        wr.verify_bad,
        filled.len(),
        up.uids.len(),
        store.mapped_bytes() >> 20,
    );
    println!(
        "recv graph launches={} consumed={} checksummed={} checksum_bad={sum_bad} out_mapped_MiB={}",
        cons.launches,
        cons.consumed,
        cons.checksums.len(),
        cons.out_mapped_bytes >> 20,
    );
    let bad = wr.verify_bad + wr.pos_mismatch;
    let consumed_bad =
        usize::from(cons.consumed != up.frames || cons.checksums.len() != up.uids.len());
    let code = if lost > 0 {
        2
    } else if bad as usize
        + h2d_bad
        + e2e_bad
        + store_pos_bad
        + store_dup_uids
        + sum_bad
        + consumed_bad
        > 0
    {
        3
    } else {
        0
    };
    std::process::exit(code);
}
