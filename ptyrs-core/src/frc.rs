//! Fourier ring correlation of two complex images (DESIGN §6, M5: a live result against the
//! offline one, and ptyrs against the reference packages).
//!
//! The second image is aligned to the first before the rings are compared: the peak of their
//! cross-correlation, refined on a grid of `1 / UPSAMPLE` pixel by a local DFT (Guizar-Sicairos,
//! Thurman & Fienup 2008), gives the sub-pixel shift, and its phase the global phase. Both
//! images are then Hann-windowed, so the cut edges of a cropped region, which are identical in
//! both and correlate perfectly, do not raise the high rings. Ring `r` holds the frequencies
//! whose radius times `min(ny, nx)` rounds to `r` (`r / min(ny, nx)` cycles per pixel), up to
//! Nyquist along the shorter side; its correlation is `Re Σ A conj(B) / sqrt(Σ|A|² Σ|B|²)`,
//! the real part because the global phase is already removed.

use std::f64::consts::PI;
use std::sync::Arc;

use rustfft::{Fft, FftPlanner};

use crate::{C64, CoreError};

/// Shifts are found to `1 / UPSAMPLE` pixel.
const UPSAMPLE: usize = 100;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ring {
    pub frc: f64,
    /// Half-bit threshold for this ring's pixel count ([`half_bit`]).
    pub half_bit: f64,
    pub npix: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Frc {
    /// Rings `0 ..= min(ny, nx) / 2`.
    pub rings: Vec<Ring>,
    /// `[dy, dx]` in pixels and `phase` in radians with `a(y, x) ≈ e^{i phase} b(y - dy, x - dx)`.
    pub shift: [f64; 2],
    pub phase: f64,
}

/// Half-bit information threshold for a ring of `npix` pixels (van Heel & Schatz 2005).
pub fn half_bit(npix: usize) -> f64 {
    let s = (npix as f64).sqrt();
    (0.2071 + 1.9102 / s) / (1.2071 + 0.9102 / s)
}

impl Frc {
    /// FRC of `b` aligned to `a`, both `[ny, nx]` row-major with sides of at least 2.
    pub fn compute(a: &[C64], b: &[C64], shape: [usize; 2]) -> Result<Self, CoreError> {
        let [ny, nx] = shape;
        let n = ny * nx;
        if ny < 2 || nx < 2 || a.len() != n || b.len() != n {
            return Err(CoreError::Shape(format!(
                "frc of {} and {} pixels as {ny}x{nx} (sides >= 2)",
                a.len(),
                b.len()
            )));
        }
        if let Some(i) = a.iter().chain(b).position(|z| !z.is_finite()) {
            let which = if i < n { "a" } else { "b" };
            return Err(CoreError::Invalid(format!(
                "frc: pixel {} of {which} is not finite",
                i % n
            )));
        }
        let fft = Fft2::new(shape);
        let mut fa = a.to_vec();
        let mut fb = b.to_vec();
        fft.run(&mut fa, false);
        fft.run(&mut fb, false);
        let (shift, phase) = register(&fft, &fa, &fb);

        let (fy, fx) = (freqs(ny), freqs(nx));
        for (row, &ky) in fb.chunks_exact_mut(nx).zip(&fy) {
            for (z, &kx) in row.iter_mut().zip(&fx) {
                let ramp = ky * shift[0] / ny as f64 + kx * shift[1] / nx as f64;
                *z *= C64::from_polar(1.0, phase - 2.0 * PI * ramp);
            }
        }
        // Aligned `b`, times `n`; the scale cancels in every ring.
        fft.run(&mut fb, true);
        fa.copy_from_slice(a);
        hann(&mut fa, shape);
        hann(&mut fb, shape);
        fft.run(&mut fa, false);
        fft.run(&mut fb, false);

        let m = ny.min(nx);
        let nring = m / 2 + 1;
        let (mut num, mut da, mut db) = (vec![0.0; nring], vec![0.0; nring], vec![0.0; nring]);
        let mut npix = vec![0usize; nring];
        for (y, &ky) in fy.iter().enumerate() {
            for (x, &kx) in fx.iter().enumerate() {
                let rho = (ky / ny as f64).hypot(kx / nx as f64) * m as f64;
                let r = rho.round() as usize;
                if r >= nring {
                    continue;
                }
                let (za, zb) = (fa[y * nx + x], fb[y * nx + x]);
                num[r] += (za * zb.conj()).re;
                da[r] += za.norm_sqr();
                db[r] += zb.norm_sqr();
                npix[r] += 1;
            }
        }
        let rings = (0..nring)
            .map(|r| {
                let den = (da[r] * db[r]).sqrt();
                Ring {
                    frc: if den > 0.0 { num[r] / den } else { 0.0 },
                    half_bit: half_bit(npix[r]),
                    npix: npix[r],
                }
            })
            .collect();
        Ok(Self {
            rings,
            shift,
            phase,
        })
    }

    /// First ring from 1 whose correlation is below its half-bit threshold (the resolution is
    /// `r / min(ny, nx)` cycles per pixel); `None` if every ring stays at or above it.
    pub fn crossing(&self) -> Option<usize> {
        (1..self.rings.len()).find(|&r| self.rings[r].frc < self.rings[r].half_bit)
    }
}

/// Shift and global phase taking `b` onto `a`, from their spectra.
fn register(fft: &Fft2, fa: &[C64], fb: &[C64]) -> ([f64; 2], f64) {
    let [ny, nx] = fft.shape;
    // cc(r) = Σ_k A conj(B) e^{+2πi k·r} = n Σ_y a(y + r) conj(b(y)), peaking at the shift.
    let x: Vec<C64> = fa.iter().zip(fb).map(|(a, b)| a * b.conj()).collect();
    let mut cc = x.clone();
    fft.run(&mut cc, true);
    let peak = argmax(&cc);
    let signed = |p: usize, n: usize| {
        if p > n / 2 {
            p as f64 - n as f64
        } else {
            p as f64
        }
    };
    let (y0, x0) = (signed(peak / nx, ny), signed(peak % nx, nx));

    // cc on offsets -0.75 ..= 0.75 pixel around the peak, `m` per axis, separably:
    // up[jy, jx] = Σ_ky ey[ky, jy] Σ_kx x[ky, kx] ex[kx, jx].
    let m = 3 * UPSAMPLE / 2 + 1;
    let off = |j: usize| (j as f64 - (m / 2) as f64) / UPSAMPLE as f64;
    let kernel = |n: usize, r0: f64| -> Vec<C64> {
        freqs(n)
            .iter()
            .flat_map(|&k| {
                (0..m).map(move |j| C64::from_polar(1.0, 2.0 * PI * k * (r0 + off(j)) / n as f64))
            })
            .collect()
    };
    let (ey, ex) = (kernel(ny, y0), kernel(nx, x0));
    let mut t = vec![C64::default(); ny * m];
    for (xrow, trow) in x.chunks_exact(nx).zip(t.chunks_exact_mut(m)) {
        for (&v, erow) in xrow.iter().zip(ex.chunks_exact(m)) {
            for (tj, &e) in trow.iter_mut().zip(erow) {
                *tj += v * e;
            }
        }
    }
    let mut up = vec![C64::default(); m * m];
    for (trow, erow) in t.chunks_exact(m).zip(ey.chunks_exact(m)) {
        for (urow, &e) in up.chunks_exact_mut(m).zip(erow) {
            for (u, &tj) in urow.iter_mut().zip(trow) {
                *u += e * tj;
            }
        }
    }
    let best = argmax(&up);
    ([y0 + off(best / m), x0 + off(best % m)], up[best].arg())
}

/// First index of the largest magnitude.
fn argmax(z: &[C64]) -> usize {
    let mut best = 0;
    for (i, v) in z.iter().enumerate() {
        if v.norm_sqr() > z[best].norm_sqr() {
            best = i;
        }
    }
    best
}

/// Signed frequency of each FFT bin, Nyquist negative (numpy `fftfreq(n) * n`).
fn freqs(n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| {
            if k < n.div_ceil(2) {
                k as f64
            } else {
                k as f64 - n as f64
            }
        })
        .collect()
}

/// Multiplies by the separable periodic Hann window.
fn hann(img: &mut [C64], [ny, nx]: [usize; 2]) {
    let w = |j: usize, n: usize| 0.5 - 0.5 * (2.0 * PI * j as f64 / n as f64).cos();
    for (y, row) in img.chunks_exact_mut(nx).enumerate() {
        let wy = w(y, ny);
        for (x, z) in row.iter_mut().enumerate() {
            *z *= wy * w(x, nx);
        }
    }
}

/// Unnormalized 2D FFT of one `[ny, nx]` plane, forward `e^{-2πi k·r}`.
struct Fft2 {
    shape: [usize; 2],
    /// `[forward, inverse]`.
    x: [Arc<dyn Fft<f64>>; 2],
    y: [Arc<dyn Fft<f64>>; 2],
}

impl Fft2 {
    fn new(shape: [usize; 2]) -> Self {
        let mut p = FftPlanner::new();
        Self {
            shape,
            x: [p.plan_fft_forward(shape[1]), p.plan_fft_inverse(shape[1])],
            y: [p.plan_fft_forward(shape[0]), p.plan_fft_inverse(shape[0])],
        }
    }

    fn run(&self, plane: &mut [C64], inverse: bool) {
        let [ny, nx] = self.shape;
        let dir = usize::from(inverse);
        self.x[dir].process(plane);
        let mut t = vec![C64::default(); ny * nx];
        for y in 0..ny {
            for x in 0..nx {
                t[x * ny + y] = plane[y * nx + x];
            }
        }
        self.y[dir].process(&mut t);
        for y in 0..ny {
            for x in 0..nx {
                plane[y * nx + x] = t[x * ny + y];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, Normal, Uniform};

    use super::*;
    use crate::sim;

    /// `img(r - s)`, periodic, by a Fourier phase ramp.
    fn shifted(img: &[C64], [ny, nx]: [usize; 2], s: [f64; 2]) -> Vec<C64> {
        let fft = Fft2::new([ny, nx]);
        let mut f = img.to_vec();
        fft.run(&mut f, false);
        for (row, &ky) in f.chunks_exact_mut(nx).zip(&freqs(ny)) {
            for (z, &kx) in row.iter_mut().zip(&freqs(nx)) {
                let ramp = ky * s[0] / ny as f64 + kx * s[1] / nx as f64;
                *z *= C64::from_polar(1.0 / (ny * nx) as f64, -2.0 * PI * ramp);
            }
        }
        fft.run(&mut f, true);
        f
    }

    #[test]
    fn half_bit_runs_from_one_to_its_limit() {
        assert!((half_bit(1) - 1.0).abs() < 1e-4);
        assert!((half_bit(usize::MAX) - 0.2071 / 1.2071).abs() < 1e-6);
        assert!((1..2000).all(|n| half_bit(n + 1) < half_bit(n)));
    }

    /// `b = e^{-i phi} a(r + s)` gives back `s` to the upsampling step and `phi`, and once
    /// aligned every ring correlates fully.
    #[test]
    fn recovers_a_sub_pixel_shift_and_global_phase() {
        let shape = [128, 96];
        let obj = sim::object(shape);
        let a = obj.plane(0);
        let (s, phi) = ([0.373, -1.6246], 1.1);
        let b: Vec<C64> = shifted(a, shape, [-s[0], -s[1]])
            .into_iter()
            .map(|z| z * C64::from_polar(1.0, -phi))
            .collect();
        let f = Frc::compute(a, &b, shape).unwrap();
        eprintln!("shift {:?}, phase {}", f.shift, f.phase);
        let step = 0.5 / UPSAMPLE as f64;
        assert!(
            (f.shift[0] - s[0]).abs() <= step && (f.shift[1] - s[1]).abs() <= step,
            "shift {:?}, want {s:?}",
            f.shift
        );
        assert!((f.phase - phi).abs() < 1e-3, "phase {}", f.phase);
        assert_eq!(f.rings.len(), 96 / 2 + 1);
        let worst = f.rings.iter().map(|r| r.frc).fold(f64::INFINITY, f64::min);
        assert!(worst > 0.999, "lowest ring {worst}");
        assert_eq!(f.crossing(), None);
    }

    /// Signal with SNR `s0 / (1 + (rho / rho0)^8)` per frequency, measured twice with
    /// independent unit white noise: the rings follow `snr / (1 + snr)` and drop below the
    /// half-bit threshold where that curve does.
    #[test]
    fn crossing_follows_a_known_snr() {
        let (ny, nx) = (256, 256);
        let n = ny * nx;
        let (s0, rho0) = (10.0, 32.0);
        let snr = |ky: f64, kx: f64| s0 / (1.0 + (ky.hypot(kx) / rho0).powi(8));
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let turn = Uniform::new(0.0, 2.0 * PI);
        // |X(k)|² = snr n, so that unit-variance noise has spectral power n per bin.
        let mut x: Vec<C64> = Vec::with_capacity(n);
        for &ky in &freqs(ny) {
            for &kx in &freqs(nx) {
                let amp = (snr(ky, kx) * n as f64).sqrt();
                x.push(C64::from_polar(amp / n as f64, turn.sample(&mut rng)));
            }
        }
        Fft2::new([ny, nx]).run(&mut x, true);
        let g = Normal::new(0.0, 0.5f64.sqrt()).unwrap();
        let mut noisy = || -> Vec<C64> {
            x.iter()
                .map(|&v| v + C64::new(g.sample(&mut rng), g.sample(&mut rng)))
                .collect()
        };
        let (a, b) = (noisy(), noisy());
        let f = Frc::compute(&a, &b, [ny, nx]).unwrap();

        // Expected ring correlation from the mean SNR of its pixels.
        let nring = f.rings.len();
        let (mut ssum, mut cnt) = (vec![0.0; nring], vec![0usize; nring]);
        for &ky in &freqs(ny) {
            for &kx in &freqs(nx) {
                let r = ky.hypot(kx).round() as usize;
                if r < nring {
                    ssum[r] += snr(ky, kx);
                    cnt[r] += 1;
                }
            }
        }
        let want: Vec<f64> = (0..nring)
            .map(|r| {
                let s = ssum[r] / cnt[r] as f64;
                s / (1.0 + s)
            })
            .collect();
        let want_cross = (1..nring).find(|&r| want[r] < half_bit(cnt[r])).unwrap();
        let got_cross = f.crossing().unwrap();
        // A ring's correlation scatters by (1 - frc²) / sqrt(independent pixels); the Hann
        // window correlates each frequency with its neighbours, about 3 pixels per sample.
        let worst_z = (0..nring)
            .map(|r| {
                let sigma = (1.0 - want[r] * want[r]) * (3.0 / cnt[r] as f64).sqrt();
                (f.rings[r].frc - want[r]).abs() / sigma
            })
            .fold(0.0, f64::max);
        eprintln!(
            "crossing ring {got_cross}, predicted {want_cross}; worst ring {worst_z:.2} sigma; \
             shift {:?}",
            f.shift
        );
        assert!(got_cross.abs_diff(want_cross) <= 3);
        assert!(worst_z < 4.0);
        assert!(f.shift.iter().all(|d| d.abs() <= 0.05), "{:?}", f.shift);
    }

    #[test]
    fn rejects_mismatched_and_non_finite_input() {
        let a = vec![C64::new(1.0, 0.0); 16];
        assert!(matches!(
            Frc::compute(&a, &a[..15], [4, 4]),
            Err(CoreError::Shape(_))
        ));
        let mut b = a.clone();
        b[5].im = f64::NAN;
        assert!(matches!(
            Frc::compute(&a, &b, [4, 4]),
            Err(CoreError::Invalid(m)) if m.contains("pixel 5 of b")
        ));
    }
}
