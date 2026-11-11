//! Synthetic far-field ptychography frames with Poisson counts.
//!
//! Every pixel is a pure function of `(seed, pool frame, x, y)`, so the
//! receiver can regenerate any crop and compare it bit for bit with what
//! came through compress → PVA → decode → crop → H2D. The model is a
//! bright-field disc (probe aperture) with speckle, and a dark field
//! falling as r^-3.5 under the same speckle, which leaves most of an
//! Eiger-sized frame at 0 or 1 count.

use std::f64::consts::TAU;

/// Plane waves summed into the speckle field.
const WAVES: usize = 24;
/// Speckle grain is about `2 pi / K_MAX` pixels.
const K_MAX: f64 = TAU / 10.0;
/// Bright-field disc radius in pixels.
const DISC_R: f64 = 48.0;
/// Mean counts per bright-field pixel at `flux = 1`.
const DISC_COUNTS: f64 = 300.0;
/// Dark-field counts just outside the disc at `flux = 1`.
const EDGE_COUNTS: f64 = 150.0;
const DARK_EXP: f64 = 3.5;

#[inline]
pub fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
pub(crate) fn hash3(seed: u64, a: u64, b: u64) -> u64 {
    splitmix64(seed ^ splitmix64(a ^ splitmix64(b ^ 0xD1B5_4A32_D192_ED03)))
}

/// Uniform in `(0, 1]`.
#[inline]
pub(crate) fn unit(h: u64) -> f64 {
    ((h >> 11) + 1) as f64 * (1.0 / (1u64 << 53) as f64)
}

#[inline]
pub(crate) fn poisson(lambda: f64, h: u64) -> u32 {
    if lambda <= 0.0 {
        return 0;
    }
    if lambda < 30.0 {
        let u = unit(h);
        let mut p = (-lambda).exp();
        let mut c = p;
        let mut k = 0u32;
        while u > c && k < 200 {
            k += 1;
            p *= lambda / k as f64;
            c += p;
        }
        k
    } else {
        let u1 = unit(h);
        let u2 = unit(splitmix64(h));
        let z = (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos();
        (lambda + lambda.sqrt() * z).round().max(0.0) as u32
    }
}

/// Scan position of trigger `uid`, metres: a 128 x 128 raster, 50 nm step.
pub fn position(uid: i32) -> (f64, f64) {
    let x = uid.rem_euclid(128) as f64 * 50e-9;
    let y = (uid / 128).rem_euclid(128) as f64 * 50e-9;
    (x, y)
}

#[derive(Debug, Clone, Copy)]
pub struct Synth {
    pub width: usize,
    pub height: usize,
    pub seed: u64,
    pub flux: f64,
    pub pool: usize,
}

/// `[x0, x0 + w) x [y0, y0 + h)` in frame pixels.
#[derive(Debug, Clone, Copy)]
pub struct Region {
    pub x0: usize,
    pub y0: usize,
    pub w: usize,
    pub h: usize,
}

/// Reused buffers for [`Synth::fill_region`].
#[derive(Default)]
pub struct SynthScratch {
    ex: Vec<(f64, f64)>,
}

impl Synth {
    /// Pool frame `uid` is published with.
    pub fn pool_index(&self, uid: i32) -> u64 {
        uid.rem_euclid(self.pool as i32) as u64
    }

    fn wave(&self, p: u64, k: usize) -> (f64, f64, f64) {
        let h = hash3(self.seed, p, 1 << 40 | k as u64);
        let r = K_MAX * unit(h).sqrt();
        let a = TAU * unit(splitmix64(h));
        let phi = TAU * unit(splitmix64(h ^ 0x5555));
        (r * a.cos(), r * a.sin(), phi)
    }

    /// Fill `out` (row stride `r.w`) with pool frame `p` over `r`.
    pub fn fill_region(&self, p: u64, r: Region, out: &mut [u16], s: &mut SynthScratch) {
        let Region { x0, y0, w, h } = r;
        assert!(out.len() >= w * h);
        let waves: [(f64, f64, f64); WAVES] = std::array::from_fn(|k| self.wave(p, k));
        let amp = 1.0 / (WAVES as f64).sqrt();
        s.ex.clear();
        for &(kx, _, _) in &waves {
            for x in x0..x0 + w {
                let (sn, cs) = (kx * x as f64).sin_cos();
                s.ex.push((cs, sn));
            }
        }
        let (cx, cy) = (self.width as f64 / 2.0, self.height as f64 / 2.0);
        for (row, y) in (y0..y0 + h).enumerate() {
            let ey: [(f64, f64); WAVES] = std::array::from_fn(|k| {
                let (_, ky, phi) = waves[k];
                let (sn, cs) = (ky * y as f64 + phi).sin_cos();
                (amp * cs, amp * sn)
            });
            let dy = y as f64 - cy;
            for col in 0..w {
                let x = x0 + col;
                let (mut re, mut im) = (0.0, 0.0);
                for (k, &(a, b)) in ey.iter().enumerate() {
                    let (c, d) = s.ex[k * w + col];
                    re += a * c - b * d;
                    im += a * d + b * c;
                }
                let speckle = re * re + im * im;
                let dx = x as f64 - cx;
                let r = (dx * dx + dy * dy).sqrt();
                let lambda = if r < DISC_R {
                    self.flux * DISC_COUNTS * (1.0 + 0.3 * (speckle - 1.0))
                } else {
                    self.flux * EDGE_COUNTS * (DISC_R / r).powf(DARK_EXP) * speckle
                };
                let idx = (y * self.width + x) as u64;
                out[row * w + col] = poisson(lambda, hash3(self.seed, p, idx)).min(65_535) as u16;
            }
        }
    }

    pub fn frame(&self, p: u64) -> Vec<u16> {
        let mut out = vec![0u16; self.width * self.height];
        let all = Region {
            x0: 0,
            y0: 0,
            w: self.width,
            h: self.height,
        };
        self.fill_region(p, all, &mut out, &mut SynthScratch::default());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_matches_the_full_frame() {
        let s = Synth {
            width: 300,
            height: 280,
            seed: 7,
            flux: 1.0,
            pool: 4,
        };
        let full = s.frame(3);
        let (x0, y0, w, h) = (90, 77, 64, 50);
        let mut crop = vec![0u16; w * h];
        s.fill_region(
            3,
            Region { x0, y0, w, h },
            &mut crop,
            &mut SynthScratch::default(),
        );
        for r in 0..h {
            assert_eq!(&crop[r * w..r * w + w], &full[(y0 + r) * 300 + x0..][..w]);
        }
    }

    #[test]
    fn frames_differ_across_the_pool_and_are_mostly_dark() {
        let s = Synth {
            width: 1028,
            height: 1062,
            seed: 1,
            flux: 1.0,
            pool: 2,
        };
        let a = s.frame(0);
        let b = s.frame(1);
        assert_ne!(a, b);
        let zeros = a.iter().filter(|&&v| v == 0).count() as f64 / a.len() as f64;
        let peak = *a.iter().max().unwrap();
        assert!(zeros > 0.5, "zero fraction {zeros}");
        assert!(peak > 200, "peak {peak}");
    }
}
