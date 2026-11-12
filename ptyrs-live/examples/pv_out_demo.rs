//! Rust half of the result-PV check, driven by `scripts/pv_out_check.py`:
//! writes a known pattern into rows `row0..row1` of a two-plane
//! `LiveObject` on its stream, publishes its snapshot preview, a two-mode
//! probe and an LLK history through `PvOut` as uid 0 until every PV has a
//! subscriber, then as uids `1..=posts`, the pattern moving with the uid.
//! The script computes the same patterns and compares what arrives.

use std::sync::Arc;
use std::time::{Duration, Instant};

use cuda_bindings as sys;
use cuda_core::{Context, IntoResult, PinnedHostBuffer};
use epics_pva_rs::server_native::{PvaServer, PvaServerConfig, SharedSource};
use ptyrs_live::args::Args;
use ptyrs_live::object::{LiveObject, Snapshot};
use ptyrs_live::pv_out::{AmpPhase, PvOut};

const FILL: [f32; 2] = [1.0, 0.5];

fn polar(a: f64, p: f64) -> [f32; 2] {
    [(a * p.cos()) as f32, (a * p.sin()) as f32]
}

fn object(z: usize, y: usize, x: usize, u: i32) -> [f32; 2] {
    let (z, y, x, u) = (z as f64, y as f64, x as f64, f64::from(u));
    let a = 0.6 + 0.3 * (0.011 * x + 0.017 * y + z).sin();
    polar(a, 0.05 * x - 0.03 * y + 0.9 * z + 0.4 * u)
}

fn probe(m: usize, y: usize, x: usize, np: usize, u: i32) -> [f32; 2] {
    let c = (np / 2) as f64;
    let r2 = (x as f64 - c).powi(2) + (y as f64 - c).powi(2);
    let w = 10.0 + 5.0 * m as f64;
    polar(
        (-r2 / (2.0 * w * w)).exp(),
        0.002 * r2 + m as f64 + 0.1 * f64::from(u),
    )
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() {
    let a = Args::parse();
    let port: u16 = a.get("port", 15175);
    let prefix = a.str("prefix", "PTYOUT:");
    let gpu: usize = a.get("gpu", 0);
    let n: usize = a.get("n", 1024);
    let (row0, row1): (usize, usize) = (a.get("row0", 300), a.get("row1", 460));
    let bin: usize = a.get("bin", 4);
    let np: usize = a.get("probe", 128);
    let posts: i32 = a.get("posts", 5);
    let period = Duration::from_millis(a.get("period-ms", 200));
    let wait_s: u64 = a.get("wait-s", 30);
    let linger = Duration::from_millis(a.get("linger-ms", 2000));

    let ctx = Context::new(gpu).expect("CUDA context");
    let s = ctx.new_stream().expect("stream");
    let mut o = LiveObject::new(&s, 2, [n, n], 4, FILL, 1).expect("LiveObject");
    // A tile backs its object rows across the whole width.
    o.ensure((row0 / 16..=(row1 - 1) / 16).map(|ty| [ty as u32, 0]))
        .expect("ensure");
    let mut snap = Snapshot::new(&o).expect("Snapshot");
    let band = (row1 - row0) * n;
    let mut stage = PinnedHostBuffer::<[f32; 2]>::zeroed(&ctx, 2 * band).expect("staging");

    let source = SharedSource::new();
    let out = PvOut::new(&source, &prefix).expect("open PVs");
    let mut cfg = PvaServerConfig::isolated();
    cfg.tcp_port = port;
    let server = PvaServer::start(Arc::new(source), cfg).expect("start PVA server");
    println!("pvout listening={} prefix={prefix}", server.tcp_addr());

    let mut publish = |u: i32| -> usize {
        let t = Instant::now();
        // Rows row0..row1 of both planes; the stream has finished reading
        // `stage` once the last snapshot was waited for.
        for (i, p) in stage.as_mut_slice().iter_mut().enumerate() {
            let (z, r) = (i / band, i % band);
            *p = object(z, row0 + r / n, r % n, u);
        }
        ctx.bind_to_thread().expect("bind");
        for z in 0..2 {
            let dst = o.ptr() + ((z * n + row0) * n * 8) as u64;
            // SAFETY: rows row0..row1 are backed; `stage` is not touched
            // again before the snapshot queued after this copy completes.
            unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    dst,
                    stage.as_ptr().add(z * band).cast(),
                    band * 8,
                    s.cu_stream(),
                )
            }
            .result()
            .expect("H2D");
        }
        let t_write = t.elapsed();
        o.snapshot(&mut snap).expect("snapshot");
        let t_queue = t.elapsed();
        let px = snap.wait().expect("snapshot wait");
        let t_copy = t.elapsed();
        let img = AmpPhase::new(px, 2, [n, n], bin);
        let t_preview = t.elapsed();
        let pr: Vec<[f32; 2]> = (0..2 * np * np)
            .map(|i| probe(i / (np * np), i / np % np, i % np, np, u))
            .collect();
        let llk: Vec<f64> = (0..=u).map(|k| 1e6 * 0.8f64.powi(k)).collect();
        let subs = [
            out.post_object(img, u).expect("post object"),
            out.post_probe(AmpPhase::new(&pr, 2, [np, np], 1), u)
                .expect("post probe"),
            out.post_llk(&llk, u).expect("post LLK"),
        ];
        if u > 0 {
            let ms = |d: Duration| d.as_secs_f64() * 1e3;
            println!(
                "pvout uid={u} write_ms={:.2} snapshot_queue_ms={:.3} copy_wait_ms={:.2} \
                 preview_ms={:.2} post_ms={:.2}",
                ms(t_write),
                ms(t_queue - t_write),
                ms(t_copy - t_queue),
                ms(t_preview - t_copy),
                ms(t.elapsed() - t_preview)
            );
        }
        subs.into_iter().min().unwrap()
    };

    let t0 = Instant::now();
    while publish(0) < 1 {
        assert!(
            t0.elapsed() < Duration::from_secs(wait_s),
            "no subscriber on every PV after {wait_s} s"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
    for u in 1..=posts {
        std::thread::sleep(period);
        publish(u);
    }
    println!("pvout done posts={posts}");
    std::thread::sleep(linger);
}
