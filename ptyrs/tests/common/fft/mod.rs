#![allow(dead_code)]
use ptyrs::fft::C32;

/// splitmix64-based generator, uniform in [-1, 1); fixed seed gives identical input everywhere.
pub fn random_c32(len: usize, seed: u64) -> Vec<C32> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    };
    (0..len)
        .map(|_| C32 {
            re: next(),
            im: next(),
        })
        .collect()
}
