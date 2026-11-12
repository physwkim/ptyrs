//! Decode cost of synthetic Eiger frames per codec, on 1 and on `--threads`
//! threads at once, each thread cycling through its own copy of the pool:
//! `cargo run --release -p ptyrs-live --example codec_bench -- --threads 12`.

use ptyrs_live::args::Args;
use ptyrs_live::codec::{
    Bslz4Scratch, Codec, bslz4_compress_u16, bslz4_decompress_into_u16, bslz4_decompress_range_u16,
    bytemuck_u16, bytemuck_u16_mut, lz4_compress, lz4_decompress_into,
};
use ptyrs_live::sched::{FairSched, set_fair};
use ptyrs_live::stats::{mono_ns, thread_cpu_ns};
use ptyrs_live::synth::Synth;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Full,
    CropRows,
}

/// Frames per second one thread decodes, and its CPU share.
fn run(
    s: &Synth,
    codec: Codec,
    mode: Mode,
    pool: &[Vec<u8>],
    secs: f64,
    gap_us: u64,
    util_min: u32,
) -> (f64, f64) {
    set_fair(FairSched {
        util_min,
        slice_ns: 0,
    })
    .expect("sched_setattr uclamp min");
    let n = s.width * s.height;
    let mut img = vec![0u16; n];
    let mut sc = Bslz4Scratch::default();
    let rows = (s.height / 2 - 128) * s.width..(s.height / 2 + 128) * s.width;
    let (t0, c0) = (mono_ns(), thread_cpu_ns());
    let mut frames = 0u64;
    let mut busy = 0i64;
    while ((mono_ns() - t0) as f64) < secs * 1e9 {
        if gap_us > 0 {
            std::thread::sleep(std::time::Duration::from_micros(gap_us));
        }
        let t = mono_ns();
        let p = &pool[frames as usize % pool.len()];
        match (codec, mode) {
            (Codec::Lz4, _) => lz4_decompress_into(p, bytemuck_u16_mut(&mut img)).unwrap(),
            (Codec::Bslz4, Mode::Full) => bslz4_decompress_into_u16(p, &mut img, &mut sc).unwrap(),
            (Codec::Bslz4, Mode::CropRows) => {
                bslz4_decompress_range_u16(p, &mut img, &mut sc, rows.clone()).unwrap()
            }
            (Codec::None, _) => unreachable!(),
        }
        busy += mono_ns() - t;
        frames += 1;
    }
    let (dt, dc) = ((mono_ns() - t0) as f64, (thread_cpu_ns() - c0) as f64);
    // Decode-only rate, so a sleep between frames does not count.
    (frames as f64 / (busy as f64 / 1e9), dc / dt)
}

fn main() {
    let a = Args::parse();
    let threads: usize = a.get("threads", 12);
    let secs: f64 = a.get("secs", 2.0);
    let gap_us: u64 = a.get("gap-us", 0);
    let util_min: u32 = a.get("util-min", 0);
    let s = Synth {
        width: 1028,
        height: 1062,
        seed: 1,
        flux: a.get("flux", 1.0),
        pool: 64,
    };
    let raw: Vec<Vec<u16>> = (0..s.pool as u64).map(|i| s.frame(i)).collect();
    let lz: Vec<Vec<u8>> = raw.iter().map(|r| lz4_compress(bytemuck_u16(r))).collect();
    let bs: Vec<Vec<u8>> = raw.iter().map(|r| bslz4_compress_u16(r)).collect();
    let raw_bytes = (s.width * s.height * 2) as f64;
    for (name, codec, mode, pool) in [
        ("lz4", Codec::Lz4, Mode::Full, &lz),
        ("bslz4", Codec::Bslz4, Mode::Full, &bs),
        ("bslz4-crop-rows", Codec::Bslz4, Mode::CropRows, &bs),
    ] {
        let ratio = raw_bytes * pool.len() as f64 / pool.iter().map(Vec::len).sum::<usize>() as f64;
        let (one, _) = std::thread::scope(|sc| {
            sc.spawn(|| run(&s, codec, mode, pool, secs, gap_us, util_min))
                .join()
                .unwrap()
        });
        let res: Vec<(f64, f64)> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads)
                .map(|_| {
                    let pool = pool.clone();
                    sc.spawn(move || run(&s, codec, mode, &pool, secs, gap_us, util_min))
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let agg: f64 = res.iter().map(|r| r.0).sum();
        println!(
            "{name:16} ratio={ratio:.2} 1thread_hz={one:.0} ({:.0} us/frame) \
             {threads}threads_hz={agg:.0} ({:.0} us/frame/thread)",
            1e6 / one,
            1e6 * threads as f64 / agg
        );
    }
}
