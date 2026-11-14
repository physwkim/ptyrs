//! Fourier ring correlation of two complex images (DESIGN §6, M5: a live result against the
//! offline one, and ptyrs against the reference packages).
//!
//! The rings compare a region of both images, the second aligned to the first. The peak of the
//! regions' cross-correlation, refined on a grid of `1 / UPSAMPLE` pixel by a local DFT
//! (Guizar-Sicairos, Thurman & Fienup 2008), gives the sub-pixel shift. The whole second image
//! is shifted by it before the region is cut, so what drifts across the region's edge comes
//! from outside it rather than wrapping round from the opposite edge. The peak of the spectrum
//! of the regions' windowed product `a conj(b)`, refined the same way, then gives the linear
//! phase ramp and the global phase, far-field ambiguities like the shift, which are removed. A
//! wrapped edge or a ramp biases the correlation's peak by a fraction of a pixel, so a second
//! pass correlates again with the first's shift applied and ramp removed. Both regions are
//! Hann-windowed, so their cut edges, which are identical in both and correlate perfectly, do
//! not raise the high rings. Ring `r` holds the frequencies whose radius times `min(ny, nx)`
//! rounds to `r` (`r / min(ny, nx)` cycles per pixel), up to Nyquist along the region's shorter
//! side; its correlation is `Re Σ A conj(B) / sqrt(Σ|A|² Σ|B|²)`, the real part because the
//! global phase is removed.

use std::f64::consts::PI;
use std::ops::Range;
use std::sync::Arc;

use rustfft::{Fft, FftPlanner};

use crate::base::{C64, CoreError};

/// Shifts are found to `1 / UPSAMPLE` pixel, ramps to `1 / UPSAMPLE` cycle over the region.
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
    /// Rings `0 ..= min(ny, nx) / 2` of the region's `[ny, nx]`.
    pub rings: Vec<Ring>,
    /// `shift` `[dy, dx]` in pixels, `ramp` `[gy, gx]` in cycles per pixel and `phase` in
    /// radians with `a(y, x) ≈ e^{i (phase + 2π (gy (y - y0) + gx (x - x0)))} b(y - dy, x - dx)`
    /// over the region, whose first pixel is `[y0, x0]`.
    pub shift: [f64; 2],
    pub ramp: [f64; 2],
    pub phase: f64,
}

/// Half-bit information threshold for a ring of `npix` pixels (van Heel & Schatz 2005).
pub fn half_bit(npix: usize) -> f64 {
    let s = (npix as f64).sqrt();
    (0.2071 + 1.9102 / s) / (1.2071 + 0.9102 / s)
}

impl Frc {
    /// FRC over the rows and columns of `region` of `b` aligned to `a`, both `shape` row-major;
    /// the region's sides are at least 2.
    pub fn compute(
        a: &[C64],
        b: &[C64],
        shape: [usize; 2],
        region: &[Range<usize>; 2],
    ) -> Result<Self, CoreError> {
        let n = shape[0] * shape[1];
        if a.len() != n
            || b.len() != n
            || region
                .iter()
                .zip(shape)
                .any(|(r, s)| r.len() < 2 || r.end > s)
        {
            return Err(CoreError::Shape(format!(
                "frc of {} and {} pixels as {shape:?} over {region:?} (inside, sides >= 2)",
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
        let whole = Fft2::new(shape);
        let part = Fft2::new(region.clone().map(|r| r.len()));
        let [ny, nx] = part.shape;
        let ra = crop(a, shape, region);
        let mut fa = ra.clone();
        part.run(&mut fa, false);
        let mut fb = b.to_vec();
        whole.run(&mut fb, false);
        // The second pass measures the fraction of a pixel the first's wrap and ramp left.
        let at = |n: usize| (0..n).map(|i| i as f64).collect::<Vec<_>>();
        let (mut shift, mut ramp, mut phase) = ([0.0; 2], [0.0; 2], 0.0);
        let mut rb = crop(b, shape, region);
        for _ in 0..2 {
            let mut fr = rb;
            unramp(&mut fr, nx, ramp, phase);
            part.run(&mut fr, false);
            let cross: Vec<C64> = fa.iter().zip(&fr).map(|(a, b)| a * b.conj()).collect();
            let (d, _) = peak(&part, &cross, [&freqs(ny), &freqs(nx)]);
            shift = [shift[0] + d[0], shift[1] + d[1]];
            rb = crop(&translate(&whole, &fb, shift), shape, region);
            // Σ w conj(a) b e^{2πi (fy y / ny + fx x / nx)} peaks at the ramp `f / [ny, nx]`
            // with the phase `-phase`.
            let mut prod: Vec<C64> = ra.iter().zip(&rb).map(|(a, b)| a.conj() * b).collect();
            hann(&mut prod, part.shape);
            let (f, v) = peak(&part, &prod, [&at(ny), &at(nx)]);
            ramp = [f[0] / ny as f64, f[1] / nx as f64];
            phase = -v.arg();
        }
        unramp(&mut rb, nx, ramp, phase);
        let (mut fa, mut fb) = (ra, rb);
        hann(&mut fa, part.shape);
        hann(&mut fb, part.shape);
        part.run(&mut fa, false);
        part.run(&mut fb, false);

        let (fy, fx) = (freqs(ny), freqs(nx));
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
            ramp,
            phase,
        })
    }

    /// First ring from 1 whose correlation is below its half-bit threshold (the resolution is
    /// `r / min(ny, nx)` cycles per pixel); `None` if every ring stays at or above it.
    pub fn crossing(&self) -> Option<usize> {
        (1..self.rings.len()).find(|&r| self.rings[r].frc < self.rings[r].half_bit)
    }
}

/// The `r` (signed, to `1 / UPSAMPLE`) and value of the largest
/// `S(r) = Σ x[ky, kx] e^{2πi (at[0][ky] r[0] / ny + at[1][kx] r[1] / nx)}`, `x` of `fft`'s
/// shape and `at[d][k]` either `k` or `k - n`: the integer peak of `x`'s inverse FFT, refined
/// around it by a local DFT. With `x = A conj(B)` of two spectra and the signed frequencies
/// as `at`, `S(r) = n Σ_y a(y + r) conj(b(y))` peaks at the shift.
fn peak(fft: &Fft2, x: &[C64], at: [&[f64]; 2]) -> ([f64; 2], C64) {
    let [ny, nx] = fft.shape;
    let mut cc = x.to_vec();
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

    // S on offsets -0.75 ..= 0.75 around the peak, `m` per axis, separably:
    // up[jy, jx] = Σ_ky ey[ky, jy] Σ_kx x[ky, kx] ex[kx, jx].
    let m = 3 * UPSAMPLE / 2 + 1;
    let off = |j: usize| (j as f64 - (m / 2) as f64) / UPSAMPLE as f64;
    let kernel = |at: &[f64], r0: f64| -> Vec<C64> {
        let n = at.len() as f64;
        at.iter()
            .flat_map(|&k| {
                (0..m).map(move |j| C64::from_polar(1.0, 2.0 * PI * k * (r0 + off(j)) / n))
            })
            .collect()
    };
    let (ey, ex) = (kernel(at[0], y0), kernel(at[1], x0));
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
    ([y0 + off(best / m), x0 + off(best % m)], up[best])
}

/// `img(y - d)`, periodic, from the spectrum of `img`.
fn translate(fft: &Fft2, spectrum: &[C64], d: [f64; 2]) -> Vec<C64> {
    let [ny, nx] = fft.shape;
    let scale = 1.0 / (ny * nx) as f64;
    let mut out = spectrum.to_vec();
    for (row, &ky) in out.chunks_exact_mut(nx).zip(&freqs(ny)) {
        for (z, &kx) in row.iter_mut().zip(&freqs(nx)) {
            let k = ky * d[0] / ny as f64 + kx * d[1] / nx as f64;
            *z *= C64::from_polar(scale, -2.0 * PI * k);
        }
    }
    fft.run(&mut out, true);
    out
}

/// Multiplies a region `nx` wide by `e^{i (phase + 2π ramp·r)}`, `r` from its first pixel.
fn unramp(img: &mut [C64], nx: usize, ramp: [f64; 2], phase: f64) {
    for (y, row) in img.chunks_exact_mut(nx).enumerate() {
        for (x, z) in row.iter_mut().enumerate() {
            let turn = ramp[0] * y as f64 + ramp[1] * x as f64;
            *z *= C64::from_polar(1.0, phase + 2.0 * PI * turn);
        }
    }
}

/// `region`'s pixels of an image of `shape`.
fn crop(img: &[C64], shape: [usize; 2], [rows, cols]: &[Range<usize>; 2]) -> Vec<C64> {
    rows.clone()
        .flat_map(|y| &img[y * shape[1]..][cols.clone()])
        .copied()
        .collect()
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
    use crate::base::sim;

    /// `img(r - s)`, periodic, by a Fourier phase ramp.
    fn shifted(img: &[C64], shape: [usize; 2], s: [f64; 2]) -> Vec<C64> {
        let fft = Fft2::new(shape);
        let mut f = img.to_vec();
        fft.run(&mut f, false);
        translate(&fft, &f, s)
    }

    #[test]
    fn half_bit_runs_from_one_to_its_limit() {
        assert!((half_bit(1) - 1.0).abs() < 1e-4);
        assert!((half_bit(usize::MAX) - 0.2071 / 1.2071).abs() < 1e-6);
        assert!((1..2000).all(|n| half_bit(n + 1) < half_bit(n)));
    }

    /// The whole of an image of `shape` as a region.
    fn all([ny, nx]: [usize; 2]) -> [Range<usize>; 2] {
        [0..ny, 0..nx]
    }

    /// The lowest ring's correlation.
    fn lowest(f: &Frc) -> f64 {
        f.rings.iter().map(|r| r.frc).fold(f64::INFINITY, f64::min)
    }

    /// `b = e^{-i phi} a(r + s)` gives back `s` to the upsampling step, `phi` and no ramp, and
    /// once aligned every ring correlates fully.
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
        let f = Frc::compute(a, &b, shape, &all(shape)).unwrap();
        eprintln!("shift {:?}, ramp {:?}, phase {}", f.shift, f.ramp, f.phase);
        let step = 0.5 / UPSAMPLE as f64;
        assert!(
            (f.shift[0] - s[0]).abs() <= step && (f.shift[1] - s[1]).abs() <= step,
            "shift {:?}, want {s:?}",
            f.shift
        );
        let bin = 1.0 / UPSAMPLE as f64;
        assert!(
            f.ramp
                .iter()
                .zip(shape)
                .all(|(g, n)| (g * n as f64).abs() <= bin),
            "ramp {:?}",
            f.ramp
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
        let f = Frc::compute(&a, &b, [ny, nx], &all([ny, nx])).unwrap();

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
             shift {:?}, ramp {:?}",
            f.shift, f.ramp
        );
        assert!(got_cross.abs_diff(want_cross) <= 3);
        assert!(worst_z < 4.0);
        assert!(f.shift.iter().all(|d| d.abs() <= 0.05), "{:?}", f.shift);
    }

    /// A copy drifted by `s` across the edges of the region 32..96: shifted whole, it matches
    /// in every ring; cut to the region first, as a crop-only FRC must, the rows and columns
    /// that left one edge wrap in at the other and the rings drop.
    #[test]
    fn a_drift_across_the_region_edge_comes_from_outside_it() {
        let shape = [128, 128];
        let obj = sim::object(shape);
        let a = obj.plane(0);
        let s = [9.37, -6.21];
        let b = shifted(a, shape, s);
        let region = [32..96, 32..96];
        let f = Frc::compute(a, &b, shape, &region).unwrap();
        let cut = |img: &[C64]| crop(img, shape, &region);
        let wrapped = Frc::compute(&cut(a), &cut(&b), [64, 64], &all([64, 64])).unwrap();
        eprintln!(
            "whole: shift {:?}, lowest {}; wrapped: shift {:?}, lowest {}, crossing {:?}",
            f.shift,
            lowest(&f),
            wrapped.shift,
            lowest(&wrapped),
            wrapped.crossing()
        );
        let step = 0.5 / UPSAMPLE as f64;
        assert!(
            (f.shift[0] + s[0]).abs() <= step && (f.shift[1] + s[1]).abs() <= step,
            "shift {:?}, want {:?}",
            f.shift,
            s.map(|d| -d)
        );
        assert!(lowest(&f) > 0.999, "lowest ring {}", lowest(&f));
        assert!(
            lowest(&wrapped) < 0.98,
            "wrapped lowest ring {}",
            lowest(&wrapped)
        );
    }

    /// `b = e^{-i (phi + 2π g·r)} a` over a 64x80 region gives back `g` to the upsampling step
    /// and `phi` at the region's first pixel to what that step leaves, and with the ramp out of
    /// the second correlation every ring correlates fully.
    #[test]
    fn removes_a_linear_phase_ramp() {
        let shape = [128, 128];
        let obj = sim::object(shape);
        let a = obj.plane(0);
        let (g, phi) = ([0.0055, -0.0031], 0.4);
        let b: Vec<C64> = a
            .iter()
            .enumerate()
            .map(|(i, z)| {
                let (y, x) = ((i / shape[1]) as f64, (i % shape[1]) as f64);
                z * C64::from_polar(1.0, -(phi + 2.0 * PI * (g[0] * y + g[1] * x)))
            })
            .collect();
        let region = [32..96, 24..104];
        let f = Frc::compute(a, &b, shape, &region).unwrap();
        eprintln!(
            "shift {:?}, ramp {:?}, phase {}, lowest {}",
            f.shift,
            f.ramp,
            f.phase,
            lowest(&f)
        );
        let bin = 0.5 / UPSAMPLE as f64;
        assert!(
            (f.ramp[0] - g[0]).abs() * 64.0 <= bin && (f.ramp[1] - g[1]).abs() * 80.0 <= bin,
            "ramp {:?}, want {g:?}",
            f.ramp
        );
        // The phase is fitted at the region's centre; a ramp off by `bin` cycles over each
        // side moves it at the first pixel by up to 2π bin.
        let want = phi + 2.0 * PI * (g[0] * 32.0 + g[1] * 24.0);
        let miss = C64::from_polar(1.0, f.phase - want).arg();
        assert!(
            miss.abs() <= 2.0 * PI * bin,
            "phase {}, want {want}",
            f.phase
        );
        assert!(lowest(&f) > 0.999, "lowest ring {}", lowest(&f));
    }

    #[test]
    fn rejects_mismatched_and_non_finite_input() {
        let a = vec![C64::new(1.0, 0.0); 16];
        let whole = all([4, 4]);
        assert!(matches!(
            Frc::compute(&a, &a[..15], [4, 4], &whole),
            Err(CoreError::Shape(_))
        ));
        for region in [[0..4, 1..5], [0..4, 2..3]] {
            assert!(matches!(
                Frc::compute(&a, &a, [4, 4], &region),
                Err(CoreError::Shape(_))
            ));
        }
        let mut b = a.clone();
        b[5].im = f64::NAN;
        assert!(matches!(
            Frc::compute(&a, &b, [4, 4], &whole),
            Err(CoreError::Invalid(m)) if m.contains("pixel 5 of b")
        ));
    }
}
