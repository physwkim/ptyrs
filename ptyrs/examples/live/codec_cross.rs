//! Rust half of the codec cross-check against C liblz4 and bitshuffle,
//! driven by `scripts/codec_cross.py`:
//!   `codec_cross --dump DIR` writes synthetic frames and their Rust payloads;
//!   `codec_cross --check DIR` decodes the payloads the C libraries made.

#![warn(clippy::disallowed_methods)]

use std::path::Path;

use ptyrs::live::args::Args;
use ptyrs::live::codec::{
    Bslz4Scratch, bslz4_compress_u16, bslz4_decompress_into_u16, bslz4_decompress_range_u16,
    bytemuck_u16, bytemuck_u16_mut, lz4_compress, lz4_decompress_into,
};
use ptyrs::live::synth::Synth;

/// `eiger` is pty-sim's frame at `--pool 16`; `bright` fills the high byte
/// planes; `odd` has a short last block and a raw `n % 8` tail.
fn cases() -> [(&'static str, Synth, u64); 3] {
    let s = |width, height, flux, pool| Synth {
        width,
        height,
        seed: 1,
        flux,
        pool,
    };
    [
        ("eiger", s(1028, 1062, 1.0, 16), 16),
        ("bright", s(1028, 1062, 30.0, 2), 2),
        ("odd", s(1027, 1061, 1.0, 2), 2),
    ]
}

fn read_u16(p: &Path) -> Vec<u16> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    b.as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect()
}

fn main() {
    let a = Args::parse();
    let (dump, check) = (a.str("dump", ""), a.str("check", ""));
    if !dump.is_empty() {
        let dir = Path::new(&dump);
        std::fs::create_dir_all(dir).unwrap();
        let mut list = String::new();
        for (name, s, n) in cases() {
            list += &format!("{name} {} {} {n}\n", s.width, s.height);
            for i in 0..n {
                let raw = s.frame(i);
                let f = |ext: &str| dir.join(format!("{name}_{i}.{ext}"));
                std::fs::write(f("raw"), bytemuck_u16(&raw)).unwrap();
                std::fs::write(f("rs.lz4"), lz4_compress(bytemuck_u16(&raw))).unwrap();
                std::fs::write(f("rs.bslz4"), bslz4_compress_u16(&raw)).unwrap();
            }
        }
        std::fs::write(dir.join("cases.txt"), list).unwrap();
        println!("codec_cross dumped {}", dir.display());
        return;
    }
    assert!(!check.is_empty(), "--dump DIR or --check DIR");
    let dir = Path::new(&check);
    let (mut ok, mut bad) = (0u32, 0u32);
    let mut sc = Bslz4Scratch::default();
    for (name, s, n) in cases() {
        let px = s.width * s.height;
        // The crop rows pty-recv decodes.
        let rows = (s.height / 2 - 128) * s.width..(s.height / 2 + 128) * s.width;
        for i in 0..n {
            let f = |ext: &str| dir.join(format!("{name}_{i}.{ext}"));
            let raw = read_u16(&f("raw"));
            assert_eq!(raw.len(), px);
            let mut out = vec![0u16; px];
            let lz = lz4_decompress_into(
                &std::fs::read(f("c.lz4")).unwrap(),
                bytemuck_u16_mut(&mut out),
            );
            let lz_ok = lz.is_ok() && out == raw;
            let bs = std::fs::read(f("c.bslz4")).unwrap();
            out.fill(0);
            let full_ok = bslz4_decompress_into_u16(&bs, &mut out, &mut sc).is_ok() && out == raw;
            out.fill(0);
            let range_ok = bslz4_decompress_range_u16(&bs, &mut out, &mut sc, rows.clone()).is_ok()
                && out[rows.clone()] == raw[rows.clone()];
            for (what, good) in [("lz4", lz_ok), ("bslz4", full_ok), ("bslz4-rows", range_ok)] {
                if good {
                    ok += 1;
                } else {
                    bad += 1;
                    println!("codec_cross MISMATCH {name}_{i} C {what} -> Rust decode");
                }
            }
        }
    }
    println!("codec_cross rust_decodes_c ok={ok} bad={bad}");
    std::process::exit((bad > 0) as i32);
}
