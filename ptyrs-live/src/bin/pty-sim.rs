//! Detector simulator: publishes `<prefix>POS` (`{uniqueId, x, y}`) over PVA
//! and frames at a fixed rate, either as `<prefix>FRAME` (NTNDArray,
//! NDPluginCodec payload) or, with `--out zmq`, as an Eiger stream v1
//! series on a ZMQ PUSH socket bound to `--zmq-port`.
//!
//! Frames come from the `--model` (synth or the M5 ptycho scan, see
//! `model.rs`), each distinct frame compressed once up front; frame `uid`
//! carries entry `uid % period`. The position of trigger `uid` is posted
//! just before its frame.

use std::sync::Arc;
use std::time::Duration;

use epics_pva_rs::nt::nd_array::{
    NdArrayBuffer, NdCodec, NdDimension, NdTimeStamp, NtNdArray, nt_nd_array_desc,
    nt_nd_array_value,
};
use epics_pva_rs::pvdata::{PvField, PvStructure, ScalarType, ScalarValue, VariantValue};
use epics_pva_rs::server_native::{PvaServer, PvaServerConfig, SharedPV, SharedSource};
use ptyrs_live::args::Args;
use ptyrs_live::codec::{Codec, bslz4_compress_u16, bytemuck_u16, lz4_compress};
use ptyrs_live::eiger::{
    ImageMeta, SeriesConfig, blob_prefix, end_message, header_message, image_parts,
};
use ptyrs_live::model::Model;
use ptyrs_live::ntnd::position_desc;
use ptyrs_live::pipeline::Truth;
use ptyrs_live::stats::{Histogram, mono_ns, process_cpu_ns, wall_ns};
use ptyrs_live::synth::Region;

#[derive(Clone)]
enum Payload {
    Raw(Arc<[u16]>),
    Compressed(Arc<[u8]>),
}

fn frame_value(p: &Payload, codec: Codec, w: usize, h: usize, uid: i32, wall: i64) -> PvField {
    let (value, csize) = match p {
        Payload::Raw(a) => (NdArrayBuffer::UShort(a.clone().into()), a.len() * 2),
        Payload::Compressed(b) => (NdArrayBuffer::UByte(b.clone().into()), b.len()),
    };
    let ts = NdTimeStamp {
        seconds_past_epoch: wall.div_euclid(1_000_000_000),
        nanoseconds: wall.rem_euclid(1_000_000_000) as i32,
        user_tag: 0,
    };
    let dim = |n: usize| NdDimension {
        size: n as i32,
        offset: 0,
        full_size: n as i32,
        binning: 1,
        reverse: false,
    };
    nt_nd_array_value(&NtNdArray {
        value,
        codec: NdCodec {
            name: codec.name().into(),
            // NDPluginCodec: the original NDDataType as a pvData ScalarType.
            parameters: (codec != Codec::None)
                .then(|| VariantValue::scalar(ScalarValue::Int(ScalarType::UShort as i32))),
        },
        compressed_size: csize as i64,
        uncompressed_size: (w * h * 2) as i64,
        unique_id: uid,
        data_time_stamp: ts.clone(),
        alarm: Default::default(),
        time_stamp: ts,
        dimension: vec![dim(w), dim(h)],
        attribute: vec![],
    })
}

fn pos_value(uid: i32, x: f64, y: f64) -> PvField {
    PvField::Structure(PvStructure {
        struct_id: "ptyrs:position:1.0".into(),
        fields: vec![
            ("uniqueId".into(), PvField::Scalar(ScalarValue::Int(uid))),
            ("x".into(), PvField::Scalar(ScalarValue::Double(x))),
            ("y".into(), PvField::Scalar(ScalarValue::Double(y))),
        ],
    })
}

impl Payload {
    fn bytes(&self) -> &[u8] {
        match self {
            Payload::Raw(a) => bytemuck_u16(a),
            Payload::Compressed(b) => b,
        }
    }
}

/// Frames go out as Eiger image messages; a full send queue drops the frame.
struct ZmqOut {
    sock: zmq::Socket,
    series: u64,
    t0: i64,
    period_ns: i64,
    /// Pool payloads with the Eiger blob prefix in front.
    blobs: Vec<Vec<u8>>,
}

impl ZmqOut {
    /// Binds, then blocks until a PULL peer takes the series header.
    fn start(
        port: u16,
        sndhwm: i32,
        pool: &[Payload],
        codec: Codec,
        (w, h): (usize, usize),
        frames: u64,
        rate: f64,
    ) -> Self {
        let ctx = zmq::Context::new();
        let sock = ctx.socket(zmq::PUSH).expect("PUSH socket");
        sock.set_sndhwm(sndhwm).expect("ZMQ_SNDHWM");
        sock.set_linger(5000).expect("ZMQ_LINGER");
        sock.bind(&format!("tcp://*:{port}")).expect("bind PUSH");
        let blobs = pool
            .iter()
            .map(|p| {
                let mut b = blob_prefix(codec, w, h);
                b.extend_from_slice(p.bytes());
                b
            })
            .collect();
        let t0 = wall_ns();
        let config = SeriesConfig {
            x_pixels_in_detector: w as u64,
            y_pixels_in_detector: h as u64,
            nimages: frames,
            ntrigger: 1,
            frame_time: 1.0 / rate,
            count_time: 1.0 / rate,
            ptyrs_series_start_ns: Some(t0),
        };
        let series = 1;
        sock.send_multipart(header_message(series, &config), 0)
            .expect("send dheader");
        Self {
            sock,
            series,
            t0,
            period_ns: (1e9 / rate) as i64,
            blobs,
        }
    }

    /// False if the frame was dropped at the high-water mark.
    fn send(&self, frame: u64, codec: Codec, w: usize, h: usize) -> bool {
        let blob = &self.blobs[(frame % self.blobs.len() as u64) as usize];
        let [head, desc, times] = image_parts(&ImageMeta {
            series: self.series,
            frame,
            width: w,
            height: h,
            codec,
            blob_len: blob.len(),
            start_ns: wall_ns() - self.t0,
            real_ns: self.period_ns,
        });
        let f = zmq::DONTWAIT;
        // libzmq checks the high-water mark on a message's first part and
        // rolls a partly queued message back, so any EAGAIN drops it whole.
        let r = self
            .sock
            .send(head, f | zmq::SNDMORE)
            .and_then(|()| self.sock.send(desc, f | zmq::SNDMORE))
            .and_then(|()| self.sock.send(&blob[..], f | zmq::SNDMORE))
            .and_then(|()| self.sock.send(times, f));
        match r {
            Ok(()) => true,
            Err(zmq::Error::EAGAIN) => false,
            Err(e) => panic!("send frame {frame}: {e}"),
        }
    }

    fn end(&self) {
        self.sock
            .send(end_message(self.series), 0)
            .expect("send dseries_end");
    }
}

fn build_pool(m: &Model, codec: Codec) -> Vec<Payload> {
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let (w, h) = m.frame_size();
    let mut pool: Vec<Option<Payload>> = vec![None; m.period()];
    let per = m.period().div_ceil(threads);
    std::thread::scope(|sc| {
        for (t, chunk) in pool.chunks_mut(per).enumerate() {
            sc.spawn(move || {
                for (k, slot) in chunk.iter_mut().enumerate() {
                    let mut raw = vec![0u16; w * h];
                    let all = Region { x0: 0, y0: 0, w, h };
                    m.fill((t * per + k) as i32, all, &mut raw);
                    *slot = Some(match codec {
                        Codec::None => Payload::Raw(raw.into()),
                        Codec::Lz4 => Payload::Compressed(lz4_compress(bytemuck_u16(&raw)).into()),
                        Codec::Bslz4 => Payload::Compressed(bslz4_compress_u16(&raw).into()),
                    });
                }
            });
        }
    });
    pool.into_iter()
        .map(|p| p.expect("every pool slot filled"))
        .collect()
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    let a = Args::parse();
    let codec = match a.str("codec", "bslz4").as_str() {
        "none" => Codec::None,
        "lz4" => Codec::Lz4,
        "bslz4" => Codec::Bslz4,
        c => panic!("--codec {c}: expected none, lz4 or bslz4"),
    };
    let rate: f64 = a.get("rate", 3000.0);
    let frames: u64 = a.get("frames", 30_000);
    let model = Arc::new(Model::from_args(&a));
    let port: u16 = a.get("port", 15075);
    let queue: usize = a.get("server-queue", 4);
    let prefix = a.str("prefix", "PTY:");
    let linger_ms: u64 = a.get("linger-ms", 3000);
    let zmq_out = match a.str("out", "pva").as_str() {
        "pva" => false,
        "zmq" => true,
        o => panic!("--out {o}: expected pva or zmq"),
    };
    let zmq_port: u16 = a.get("zmq-port", ptyrs_live::eiger::PORT);
    let sndhwm: i32 = a.get("sndhwm", 10_000);
    let (w, h) = model.frame_size();

    let t = mono_ns();
    let pool = build_pool(&model, codec);
    let raw_bytes = (w * h * 2) as f64;
    let sizes: Vec<usize> = pool
        .iter()
        .map(|p| match p {
            Payload::Raw(a) => a.len() * 2,
            Payload::Compressed(b) => b.len(),
        })
        .collect();
    let mean = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;
    let mut f0 = vec![0u16; w * h];
    model.fill(0, Region { x0: 0, y0: 0, w, h }, &mut f0);
    let zeros = f0.iter().filter(|&&v| v == 0).count() as f64 / f0.len() as f64;
    let photons: u64 = f0.iter().map(|&v| v as u64).sum();
    println!(
        "sim {} pool={} codec={} frame={}x{} gen_ms={} mean_bytes={:.0} ratio_mean={:.2} \
         ratio_min={:.2} ratio_max={:.2} zero_frac={:.3} photons_frame0={} max_count_frame0={}",
        model.describe(),
        pool.len(),
        codec.name(),
        w,
        h,
        (mono_ns() - t) / 1_000_000,
        mean,
        raw_bytes / mean,
        raw_bytes / *sizes.iter().max().unwrap() as f64,
        raw_bytes / *sizes.iter().min().unwrap() as f64,
        zeros,
        photons,
        f0.iter().max().unwrap()
    );

    let frame_pv = SharedPV::build_mailbox();
    let init_frame = frame_value(&pool[0], codec, w, h, -1, wall_ns());
    frame_pv
        .open(nt_nd_array_desc(), init_frame.clone())
        .expect("open FRAME");
    let pos_pv = SharedPV::build_mailbox();
    let init_pos = pos_value(-1, 0.0, 0.0);
    pos_pv
        .open(position_desc(), init_pos.clone())
        .expect("open POS");
    let source = SharedSource::new();
    if !zmq_out {
        source.add(format!("{prefix}FRAME"), frame_pv.clone());
    }
    source.add(format!("{prefix}POS"), pos_pv.clone());
    let mut cfg = PvaServerConfig::isolated();
    cfg.tcp_port = port;
    cfg.monitor_queue_depth = queue;
    let server = PvaServer::start(Arc::new(source), cfg).expect("start PVA server");
    println!(
        "sim listening={} server_queue={queue} rate={rate} frames={frames} out={}",
        server.tcp_addr(),
        if zmq_out { "zmq" } else { "pva" }
    );

    // A post reports the subscriber count; re-post the uid -1 placeholder
    // (ignored by the receiver) until both monitors are attached.
    loop {
        let f = frame_pv.try_post(init_frame.clone());
        let p = pos_pv.try_post(init_pos.clone());
        if (f >= 1 || zmq_out) && p >= 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let zmq = zmq_out.then(|| {
        let z = ZmqOut::start(zmq_port, sndhwm, &pool, codec, (w, h), frames, rate);
        println!(
            "sim zmq_port={zmq_port} sndhwm={sndhwm} series={}",
            z.series
        );
        z
    });
    tokio::time::sleep(Duration::from_millis(500)).await;

    let publisher = std::thread::spawn(move || {
        let period = 1e9 / rate;
        let mut post_ns = Histogram::new(100_000);
        let (mut max_lag, mut late, mut unsent) = (0i64, 0u64, 0u64);
        let t0 = mono_ns() + 1_000_000;
        let cpu0 = process_cpu_ns();
        for i in 0..frames {
            let target = t0 + (i as f64 * period) as i64;
            loop {
                let d = target - mono_ns();
                if d <= 0 {
                    break;
                }
                if d > 200_000 {
                    std::thread::sleep(Duration::from_nanos((d - 150_000) as u64));
                } else {
                    std::hint::spin_loop();
                }
            }
            let start = mono_ns();
            let lag = start - target;
            max_lag = max_lag.max(lag);
            late += (lag as f64 > period) as u64;
            let uid = i as i32;
            let (x, y) = model.position(uid);
            pos_pv.try_post(pos_value(uid, x, y));
            if let Some(z) = &zmq {
                unsent += !z.send(i, codec, w, h) as u64;
            } else {
                let v = frame_value(
                    &pool[(i % pool.len() as u64) as usize],
                    codec,
                    w,
                    h,
                    uid,
                    wall_ns(),
                );
                unsent += (frame_pv.try_post(v) == 0) as u64;
            }
            post_ns.record(mono_ns() - start);
        }
        if let Some(z) = &zmq {
            z.end();
        }
        let wall = mono_ns() - t0;
        let cpu = process_cpu_ns() - cpu0;
        println!(
            "sim published={frames} achieved_hz={:.1} max_lag_us={} late_frames={late} \
             unsent={unsent} cpu_cores={:.2} payload_MBps={:.1}",
            frames as f64 / (wall as f64 / 1e9),
            max_lag / 1000,
            cpu as f64 / wall as f64,
            mean * frames as f64 / (wall as f64 / 1e9) / 1e6
        );
        println!("sim post_latency {}", post_ns.summary());
    });
    tokio::task::spawn_blocking(move || publisher.join().expect("publisher panicked"))
        .await
        .expect("join publisher");
    tokio::time::sleep(Duration::from_millis(linger_ms)).await;
    drop(server);
}
