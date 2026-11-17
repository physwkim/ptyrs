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
//! global phase is removed. That is [`Registration::Blind`]; [`Frc::compute_with`] also
//! registers near an expected shift and leaves excluded frequencies out of the rings.

use std::f64::consts::PI;
use std::ops::Range;
use std::sync::Arc;

use rustfft::{Fft, FftPlanner};

use crate::base::{C64, CoreError};

/// Shifts are found to `1 / UPSAMPLE` pixel, ramps to `1 / UPSAMPLE` cycle over the region.
const UPSAMPLE: usize = 100;

/// The most passes of [`Registration::Near`]'s search.
pub const PASSES: usize = 8;

/// [`lattice`]'s radius in bins. The Hann window spreads each artifact peak over its main lobe
/// (2 bins) and sidelobes; at 2.5 bins the leak still made one-ring dips that set the crossing
/// (NS_180308067_256 at ring 48 instead of 75), from 4 bins on the crossings stay put.
pub const LATTICE_RADIUS: f64 = 4.0;

/// How [`Frc::compute_with`] registers the second image to the first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Registration {
    /// The regions' cross-correlation, twice, as the module describes.
    Blind,
    /// Near `shift`, the `[dy, dx]` with `a(y) ≈ b(y - shift)` expected. Each region first
    /// loses its own linear phase ramp and phase, the peak of its windowed spectrum: the
    /// object-probe ramp ambiguity, which a common ramp would leave in both and so move every
    /// ring off its spatial frequency. The relative ramp and phase are then fitted as in the
    /// blind registration. Without `radius` the shift is `shift`; with it, the shift within
    /// `radius` pixels of `shift` that maximizes the FRC below the crossing. Each pass moves to
    /// the peak, within the radius, of the cross-correlation normalized per ring over the rings
    /// below the last crossing, until the crossing and the shift settle or [`PASSES`] run out,
    /// and the pass with the highest crossing, then the highest FRC sum below it, is kept:
    /// along a ridge of the cross-correlation (an anisotropic object) the shift can drift
    /// without settling.
    Near {
        shift: [f64; 2],
        radius: Option<f64>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ring {
    pub frc: f64,
    /// Half-bit threshold for this ring's pixel count ([`half_bit`]).
    pub half_bit: f64,
    /// Frequencies in the ring, excluded ones aside.
    pub npix: usize,
}

/// One pass of [`Registration::Near`]'s search: the crossing at `shift`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pass {
    pub crossing: Option<usize>,
    pub shift: [f64; 2],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Frc {
    /// Rings `0 ..= min(ny, nx) / 2` of the region's `[ny, nx]`.
    pub rings: Vec<Ring>,
    /// `shift` `[dy, dx]` in pixels, `ramp` `[gy, gx]` in cycles per pixel and `phase` in
    /// radians with `a(y, x) ≈ e^{i (phase + 2π (gy (y - y0) + gx (x - x0)))} b(y - dy, x - dx)`
    /// over the region, whose first pixel is `[y0, x0]`, each region's own ramp and phase
    /// aside under [`Registration::Near`].
    pub shift: [f64; 2],
    pub ramp: [f64; 2],
    pub phase: f64,
    /// The search's passes in order, empty without one.
    pub passes: Vec<Pass>,
}

/// Half-bit information threshold for a ring of `npix` pixels (van Heel & Schatz 2005).
pub fn half_bit(npix: usize) -> f64 {
    let s = (npix as f64).sqrt();
    (0.2071 + 1.9102 / s) / (1.2071 + 0.9102 / s)
}

impl Frc {
    /// FRC over the rows and columns of `region` of `b` aligned to `a`, both `shape` row-major,
    /// registered blind; the region's sides are at least 2.
    pub fn compute(
        a: &[C64],
        b: &[C64],
        shape: [usize; 2],
        region: &[Range<usize>; 2],
    ) -> Result<Self, CoreError> {
        Self::compute_with(a, b, shape, region, Registration::Blind, None)
    }

    /// [`Self::compute`] registered by `registration`, with the `exclude` frequencies (the
    /// region's `[ny, nx]`, FFT order, as [`lattice`] gives them) left out of every ring and of
    /// its pixel count.
    pub fn compute_with(
        a: &[C64],
        b: &[C64],
        shape: [usize; 2],
        region: &[Range<usize>; 2],
        registration: Registration,
        exclude: Option<&[bool]>,
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
        finite(a, b)?;
        let part = Fft2::new(region.clone().map(|r| r.len()));
        let [ny, nx] = part.shape;
        let bins = Bins::new(part.shape, exclude)?;
        if let Registration::Near { shift, radius } = registration
            && (shift.iter().any(|d| !d.is_finite())
                || radius.is_some_and(|r| !(r.is_finite() && r >= 0.0)))
        {
            return Err(CoreError::Invalid(format!(
                "frc near {shift:?} within {radius:?}: a finite shift and radius >= 0"
            )));
        }
        let whole = Fft2::new(shape);
        let gauge = matches!(registration, Registration::Near { .. });
        let mut ra = crop(a, shape, region);
        if gauge {
            own(&part, &mut ra);
        }
        let mut fb = b.to_vec();
        whole.run(&mut fb, false);
        let ha = spectrum(&part, &ra);
        let (fy, fx) = (freqs(ny), freqs(nx));

        let mut passes = Vec::new();
        let (rb, shift, ramp, phase) = match registration {
            Registration::Blind => {
                let mut fa = ra.clone();
                part.run(&mut fa, false);
                // The second pass measures the fraction of a pixel the first's wrap and ramp
                // left.
                let (mut shift, mut ramp, mut phase) = ([0.0; 2], [0.0; 2], 0.0);
                let mut rb = crop(b, shape, region);
                for _ in 0..2 {
                    let mut fr = rb;
                    unramp(&mut fr, nx, ramp, phase);
                    part.run(&mut fr, false);
                    let cross: Vec<C64> = fa.iter().zip(&fr).map(|(a, b)| a * b.conj()).collect();
                    let (d, _) = peak(&part, &cross, [&fy, &fx], None);
                    shift = [shift[0] + d[0], shift[1] + d[1]];
                    rb = crop(&translate(&whole, &fb, shift), shape, region);
                    (ramp, phase) = fit(&part, ra.iter().zip(&rb).map(|(a, b)| a.conj() * b));
                }
                unramp(&mut rb, nx, ramp, phase);
                (rb, shift, ramp, phase)
            }
            Registration::Near { shift, radius } => {
                // `b` shifted by `s`, its own ramp and phase out, then the relative ones.
                let aligned = |s: [f64; 2]| {
                    let mut rb = crop(&translate(&whole, &fb, s), shape, region);
                    own(&part, &mut rb);
                    let (ramp, phase) = fit(&part, ra.iter().zip(&rb).map(|(a, b)| a.conj() * b));
                    unramp(&mut rb, nx, ramp, phase);
                    (rb, s, ramp, phase)
                };
                let mut at = aligned(shift);
                if let Some(radius) = radius {
                    let mut best: Option<((usize, f64), _)> = None;
                    let mut step: Option<[f64; 2]> = None;
                    for k in 0..=PASSES {
                        let hb = spectrum(&part, &at.0);
                        let (rings, den) = bins.correlate(&ha, &hb);
                        let crossing = first_below(&rings);
                        let c = crossing.unwrap_or(rings.len());
                        let score = (c, rings[1..c].iter().map(|r| r.frc).sum::<f64>());
                        if best
                            .as_ref()
                            .is_none_or(|(b, _)| score.0 > b.0 || (score.0 == b.0 && score.1 > b.1))
                        {
                            best = Some((score, at.clone()));
                        }
                        let settled = step.is_some_and(|d| d[0].hypot(d[1]) < 0.05)
                            && passes.last().is_some_and(|p: &Pass| p.crossing == crossing);
                        passes.push(Pass {
                            crossing,
                            shift: at.1,
                        });
                        // Each ring below the crossing weighs by its 1 / sqrt(Σ|A|² Σ|B|²).
                        let band = |r: &Option<usize>| r.filter(|r| (1..c).contains(r));
                        if settled || k == PASSES || !bins.ring.iter().any(|r| band(r).is_some()) {
                            break;
                        }
                        let cross: Vec<C64> = ha
                            .iter()
                            .zip(&hb)
                            .zip(&bins.ring)
                            .map(|((a, b), r)| {
                                let w = band(r)
                                    .map_or(0.0, |r| if den[r] > 0.0 { 1.0 / den[r] } else { 0.0 });
                                a * b.conj() * w
                            })
                            .collect();
                        let s = at.1;
                        let centre = [shift[0] - s[0], shift[1] - s[1]];
                        let (d, _) = peak(&part, &cross, [&fy, &fx], Some((centre, radius)));
                        step = Some(d);
                        at = aligned([s[0] + d[0], s[1] + d[1]]);
                    }
                    if let Some((_, kept)) = best {
                        at = kept;
                    }
                }
                at
            }
        };
        let (rings, _) = bins.correlate(&ha, &spectrum(&part, &rb));
        Ok(Self {
            rings,
            shift,
            ramp,
            phase,
            passes,
        })
    }

    /// First ring from 1 with pixels whose correlation is below its half-bit threshold (the
    /// resolution is `r / min(ny, nx)` cycles per pixel); `None` if every such ring stays at or
    /// above it.
    pub fn crossing(&self) -> Option<usize> {
        first_below(&self.rings)
    }
}

/// [`Frc::crossing`] of `rings`.
fn first_below(rings: &[Ring]) -> Option<usize> {
    (1..rings.len()).find(|&r| rings[r].npix > 0 && rings[r].frc < rings[r].half_bit)
}

/// The frequencies (FFT order) of an `[ny, nx]` region within `radius` bins of the reciprocal
/// lattice, origin aside, of a checkerboard half of a raster with steps `step` `[sy, sx]`
/// pixels: half lattice `(sy, sx)`, `(sy, -sx)`, reciprocal `{(p / 2sy, q / 2sx): p = q mod 2}`.
/// A raster's reconstruction carries a periodic artifact there (the probe-object ambiguity of
/// a periodic scan), on each half's own sublattice; [`LATTICE_RADIUS`] is the radius that
/// covers the windowed peaks.
pub fn lattice([ny, nx]: [usize; 2], step: [f64; 2], radius: f64) -> Result<Vec<bool>, CoreError> {
    if step.iter().any(|s| !(s.is_finite() && *s > 0.0)) || !(radius.is_finite() && radius >= 0.0) {
        return Err(CoreError::Invalid(format!(
            "lattice of steps {step:?} within {radius}: finite steps > 0 and radius >= 0"
        )));
    }
    let (fy, fx) = (freqs(ny), freqs(nx));
    let half = [ny as f64 / 2.0, nx as f64 / 2.0];
    // Signed distance from `c` on a periodic axis of `n` bins.
    let wrap = |k: f64, c: f64, h: f64, n: usize| (k - c + h).rem_euclid(n as f64) - h;
    let reach = step.map(|s| s as i64 + 1);
    let mut out = vec![false; ny * nx];
    for p in -reach[0]..=reach[0] {
        for q in -reach[1]..=reach[1] {
            let cy = p as f64 / (2.0 * step[0]) * ny as f64;
            let cx = q as f64 / (2.0 * step[1]) * nx as f64;
            if (p - q) % 2 != 0
                || (p == 0 && q == 0)
                || cy.abs() > half[0] + radius
                || cx.abs() > half[1] + radius
            {
                continue;
            }
            for (row, &ky) in out.chunks_exact_mut(nx).zip(&fy) {
                let dy = wrap(ky, cy, half[0], ny);
                for (o, &kx) in row.iter_mut().zip(&fx) {
                    let dx = wrap(kx, cx, half[1], nx);
                    *o |= dy * dy + dx * dx <= radius * radius;
                }
            }
        }
    }
    Ok(out)
}

/// The half-bit threshold of each ring of an `[ny, nx]` region without the `exclude`
/// frequencies, as [`Frc::compute_with`] counts them.
pub fn threshold(shape: [usize; 2], exclude: Option<&[bool]>) -> Result<Vec<f64>, CoreError> {
    Ok(Bins::new(shape, exclude)?
        .npix
        .into_iter()
        .map(half_bit)
        .collect())
}

/// The shift `s` `[dy, dx]`, to `1 / UPSAMPLE` pixel, with `a(y) ≈ b(y - s)` for two images of
/// `shape`: the peak of their whole, periodic cross-correlation, as [`Registration::Blind`]'s
/// first pass finds it.
pub fn shift(a: &[C64], b: &[C64], shape: [usize; 2]) -> Result<[f64; 2], CoreError> {
    let n = shape[0] * shape[1];
    if a.len() != n || b.len() != n || n == 0 {
        return Err(CoreError::Shape(format!(
            "shift of {} and {} pixels as {shape:?}",
            a.len(),
            b.len()
        )));
    }
    finite(a, b)?;
    let fft = Fft2::new(shape);
    let (mut fa, mut fb) = (a.to_vec(), b.to_vec());
    fft.run(&mut fa, false);
    fft.run(&mut fb, false);
    let cross: Vec<C64> = fa.iter().zip(&fb).map(|(a, b)| a * b.conj()).collect();
    Ok(peak(&fft, &cross, [&freqs(shape[0]), &freqs(shape[1])], None).0)
}

/// An error naming the first pixel of `a`, then `b`, that is not finite.
fn finite(a: &[C64], b: &[C64]) -> Result<(), CoreError> {
    match a.iter().chain(b).position(|z| !z.is_finite()) {
        Some(i) => {
            let which = if i < a.len() { "a" } else { "b" };
            Err(CoreError::Invalid(format!(
                "frc: pixel {} of {which} is not finite",
                i % a.len()
            )))
        }
        None => Ok(()),
    }
}

/// Which ring holds each frequency of a region's spectrum, `[ny, nx]` FFT order: `r` holds
/// those whose radius times `min(ny, nx)` rounds to `r`, rings `0 ..= min(ny, nx) / 2`;
/// frequencies past the last ring or excluded are in none.
struct Bins {
    ring: Vec<Option<usize>>,
    npix: Vec<usize>,
}

impl Bins {
    fn new([ny, nx]: [usize; 2], exclude: Option<&[bool]>) -> Result<Self, CoreError> {
        if let Some(e) = exclude
            && e.len() != ny * nx
        {
            return Err(CoreError::Shape(format!(
                "{} excluded flags for a [{ny}, {nx}] region",
                e.len()
            )));
        }
        let m = ny.min(nx);
        let mut npix = vec![0; m / 2 + 1];
        let mut ring = Vec::with_capacity(ny * nx);
        for &ky in &freqs(ny) {
            for &kx in &freqs(nx) {
                let rho = (ky / ny as f64).hypot(kx / nx as f64) * m as f64;
                let r = rho.round() as usize;
                let kept = r < npix.len() && !exclude.is_some_and(|e| e[ring.len()]);
                if kept {
                    npix[r] += 1;
                }
                ring.push(kept.then_some(r));
            }
        }
        Ok(Self { ring, npix })
    }

    /// Each ring's correlation of the spectra `fa` and `fb`, and its `sqrt(Σ|A|² Σ|B|²)`.
    fn correlate(&self, fa: &[C64], fb: &[C64]) -> (Vec<Ring>, Vec<f64>) {
        let nring = self.npix.len();
        let (mut num, mut da, mut db) = (vec![0.0; nring], vec![0.0; nring], vec![0.0; nring]);
        for ((r, za), zb) in self.ring.iter().zip(fa).zip(fb) {
            if let &Some(r) = r {
                num[r] += (za * zb.conj()).re;
                da[r] += za.norm_sqr();
                db[r] += zb.norm_sqr();
            }
        }
        let den: Vec<f64> = (0..nring).map(|r| (da[r] * db[r]).sqrt()).collect();
        let rings = (0..nring)
            .map(|r| Ring {
                frc: if den[r] > 0.0 { num[r] / den[r] } else { 0.0 },
                half_bit: half_bit(self.npix[r]),
                npix: self.npix[r],
            })
            .collect();
        (rings, den)
    }
}

/// The ramp `[gy, gx]` (cycles per pixel) and phase of a region's `prod`, windowed: the peak
/// `Σ w prod e^{2πi (fy y / ny + fx x / nx)}` is at the ramp `f / [ny, nx]` with the phase
/// `-phase`. With `prod = conj(a) b`, `b` times `e^{i (phase + 2π ramp·r)}` matches `a`.
fn fit(part: &Fft2, prod: impl Iterator<Item = C64>) -> ([f64; 2], f64) {
    let [ny, nx] = part.shape;
    let mut prod: Vec<C64> = prod.collect();
    hann(&mut prod, part.shape);
    let at = |n: usize| (0..n).map(|i| i as f64).collect::<Vec<_>>();
    let (f, v) = peak(part, &prod, [&at(ny), &at(nx)], None);
    ([f[0] / ny as f64, f[1] / nx as f64], -v.arg())
}

/// Removes a region's own linear phase ramp and phase, the peak of its windowed spectrum.
fn own(part: &Fft2, img: &mut [C64]) {
    let (ramp, phase) = fit(part, img.iter().copied());
    unramp(img, part.shape[1], ramp, phase);
}

/// The spectrum of a region, Hann-windowed.
fn spectrum(part: &Fft2, img: &[C64]) -> Vec<C64> {
    let mut f = img.to_vec();
    hann(&mut f, part.shape);
    part.run(&mut f, false);
    f
}

/// The `r` (signed, to `1 / UPSAMPLE`) and value of the largest
/// `S(r) = Σ x[ky, kx] e^{2πi (at[0][ky] r[0] / ny + at[1][kx] r[1] / nx)}`, `x` of `fft`'s
/// shape and `at[d][k]` either `k` or `k - n`: the integer peak of `x`'s inverse FFT, refined
/// around it by a local DFT. With `x = A conj(B)` of two spectra and the signed frequencies
/// as `at`, `S(r) = n Σ_y a(y + r) conj(b(y))` peaks at the shift. `window` `(centre, radius)`
/// keeps the integer peak to signed `r` within `radius` of `centre`.
fn peak(
    fft: &Fft2,
    x: &[C64],
    at: [&[f64]; 2],
    window: Option<([f64; 2], f64)>,
) -> ([f64; 2], C64) {
    let [ny, nx] = fft.shape;
    let mut cc = x.to_vec();
    fft.run(&mut cc, true);
    let signed = |p: usize, n: usize| {
        if p > n / 2 {
            p as f64 - n as f64
        } else {
            p as f64
        }
    };
    if let Some((c, radius)) = window {
        for (i, z) in cc.iter_mut().enumerate() {
            let (dy, dx) = (signed(i / nx, ny) - c[0], signed(i % nx, nx) - c[1]);
            if dy * dy + dx * dx > radius * radius {
                *z = C64::default();
            }
        }
    }
    let peak = argmax(&cc);
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
    use crate::worst::worst_of;

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
        let worst_z = worst_of((0..nring).map(|r| {
            let sigma = (1.0 - want[r] * want[r]) * (3.0 / cnt[r] as f64).sqrt();
            (f.rings[r].frc - want[r]).abs() / sigma
        }));
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

    /// `img` times `e^{-i 2π g·r}`.
    fn ramped(img: &[C64], shape: [usize; 2], g: [f64; 2]) -> Vec<C64> {
        img.iter()
            .enumerate()
            .map(|(i, z)| {
                let (y, x) = ((i / shape[1]) as f64, (i % shape[1]) as f64);
                z * C64::from_polar(1.0, -2.0 * PI * (g[0] * y + g[1] * x))
            })
            .collect()
    }

    /// Two copies with different own ramps, a whole number of pixels apart (a ramp is not
    /// periodic, so a sub-pixel Fourier shift would ring from the wrap's jump): blind, the
    /// relative ramp is their difference, 1.5 bins over the region, to the 0.1 bin its shift
    /// found under that ramp leaves; near the true shift, each loses its own first and what is
    /// left between them is the three fits' rounding. The shift is the one given.
    #[test]
    fn near_takes_each_regions_own_ramp_out_first() {
        let shape = [128, 128];
        let obj = sim::object(shape);
        let s = [3.0, -2.0];
        let (ga, gb) = ([0.0125, -0.0078], [-0.0047, 0.0109]);
        let a = ramped(obj.plane(0), shape, ga);
        let b = ramped(&shifted(obj.plane(0), shape, [-s[0], -s[1]]), shape, gb);
        let region = [16..112, 24..104];
        let blind = Frc::compute(&a, &b, shape, &region).unwrap();
        let near = Registration::Near {
            shift: s,
            radius: None,
        };
        let f = Frc::compute_with(&a, &b, shape, &region, near, None).unwrap();
        eprintln!(
            "blind ramp {:?}, near ramp {:?} shift {:?} lowest {}",
            blind.ramp,
            f.ramp,
            f.shift,
            lowest(&f)
        );
        // Cycles over the region within `bins`.
        let within = |g: f64, n: usize, bins: f64| (g * n as f64).abs() <= bins;
        let off = [
            blind.ramp[0] - (gb[0] - ga[0]),
            blind.ramp[1] - (gb[1] - ga[1]),
        ];
        assert!(
            within(off[0], 96, 0.15) && within(off[1], 80, 0.15),
            "blind off {off:?}"
        );
        let rounding = 1.5 / UPSAMPLE as f64;
        assert!(
            within(f.ramp[0], 96, rounding) && within(f.ramp[1], 80, rounding),
            "ramp {:?}",
            f.ramp
        );
        assert_eq!(f.shift, s);
        assert!(f.passes.is_empty());
        assert!(lowest(&f) > 0.999, "lowest ring {}", lowest(&f));
    }

    /// From a shift 0.72 pixel off the true one the search moves to it, and keeps it: its
    /// first pass is at the expected shift and crosses, the kept one does not.
    #[test]
    fn search_moves_to_the_true_shift_within_its_radius() {
        let shape = [128, 128];
        let obj = sim::object(shape);
        let a = obj.plane(0);
        let s = [2.37, -4.81];
        let b = shifted(a, shape, [-s[0], -s[1]]);
        let expect = [s[0] + 0.6, s[1] - 0.4];
        let near = Registration::Near {
            shift: expect,
            radius: Some(1.5),
        };
        let region = [16..112, 16..112];
        let f = Frc::compute_with(a, &b, shape, &region, near, None).unwrap();
        eprintln!(
            "shift {:?}, passes {:?}, lowest {}",
            f.shift,
            f.passes,
            lowest(&f)
        );
        let step = 0.5 / UPSAMPLE as f64 + 1e-12;
        assert!(
            (f.shift[0] - s[0]).abs() <= step && (f.shift[1] - s[1]).abs() <= step,
            "shift {:?}, want {s:?}",
            f.shift
        );
        assert_eq!(f.passes[0].shift, expect);
        assert!(f.passes[0].crossing.is_some());
        assert!(f.passes.len() <= PASSES + 1);
        assert_eq!(f.crossing(), None);
    }

    /// The search's integer peak stays within its radius: a copy 6 pixels off, searched within
    /// 2 of the start, ends no further than the radius plus the sub-pixel refinement.
    #[test]
    fn search_stays_within_its_radius() {
        let shape = [96, 96];
        let obj = sim::object(shape);
        let a = obj.plane(0);
        let b = shifted(a, shape, [-6.0, 0.0]);
        let near = Registration::Near {
            shift: [0.0, 0.0],
            radius: Some(2.0),
        };
        let f = Frc::compute_with(a, &b, shape, &all(shape), near, None).unwrap();
        eprintln!("shift {:?}, passes {:?}", f.shift, f.passes);
        // The integer peak within the radius, refined by up to 0.75 along each axis.
        let reach = 2.0 + 0.75 * 2f64.sqrt();
        assert!(f.shift[0].hypot(f.shift[1]) <= reach, "{:?}", f.shift);
    }

    /// Independent noise crosses at ring 1; with every frequency of ring 1 excluded, that ring
    /// is empty (no pixels, threshold NaN, as [`threshold`] gives it) and the crossing is 2.
    #[test]
    fn excluded_frequencies_leave_the_rings_and_an_empty_ring_never_crosses() {
        let shape = [64, 64];
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let g = Normal::new(0.0, 1.0).unwrap();
        let mut noise = || -> Vec<C64> {
            (0..64 * 64)
                .map(|_| C64::new(g.sample(&mut rng), g.sample(&mut rng)))
                .collect()
        };
        let (a, b) = (noise(), noise());
        let whole = all(shape);
        let open = Frc::compute(&a, &b, shape, &whole).unwrap();
        assert_eq!(open.crossing(), Some(1), "{:?}", &open.rings[..3]);
        let exclude = Bins::new(shape, None)
            .unwrap()
            .ring
            .iter()
            .map(|r| *r == Some(1))
            .collect::<Vec<_>>();
        let f =
            Frc::compute_with(&a, &b, shape, &whole, Registration::Blind, Some(&exclude)).unwrap();
        assert_eq!((f.rings[1].npix, f.rings[1].frc), (0, 0.0));
        assert!(f.rings[1].half_bit.is_nan());
        assert_eq!(f.crossing(), Some(2), "{:?}", &f.rings[..4]);
        let t = threshold(shape, Some(&exclude)).unwrap();
        assert!(
            t.iter()
                .zip(&f.rings)
                .all(|(t, r)| t.to_bits() == r.half_bit.to_bits())
        );
        assert_eq!(
            f.rings.iter().map(|r| r.npix).sum::<usize>() + exclude.iter().filter(|e| **e).count(),
            open.rings.iter().map(|r| r.npix).sum::<usize>()
        );
    }

    /// Steps of 4 pixels over 64: the checkerboard's reciprocal lattice is at multiples of 8
    /// bins with both indices of one parity, so (8, 8), (16, 0) and (8, -8) are in, (8, 0) and
    /// the origin out, each with the bins within the radius.
    #[test]
    fn lattice_marks_a_checkerboards_reciprocal_lattice() {
        let (n, radius) = (64, 1.0);
        let m = lattice([n, n], [4.0, 4.0], radius).unwrap();
        let at = |y: i64, x: i64| {
            m[(y.rem_euclid(n as i64) * n as i64 + x.rem_euclid(n as i64)) as usize]
        };
        for (y, x) in [
            (8, 8),
            (16, 0),
            (8, -8),
            (-8, 8),
            (0, 16),
            (32, 32),
            (9, 8),
            (8, 7),
        ] {
            assert!(at(y, x), "({y}, {x})");
        }
        for (y, x) in [(0, 0), (8, 0), (0, 8), (1, 0), (9, 9), (24, 0)] {
            assert!(!at(y, x), "({y}, {x})");
        }
        // 31 points (8 × 8 multiples of 8 mod 64, half of one parity, the origin aside), each
        // with its 4 neighbours.
        assert_eq!(m.iter().filter(|&&v| v).count(), 31 * 5);
        assert!(lattice([n, n], [0.0, 4.0], radius).is_err());
    }

    /// [`shift`] finds what the blind registration's first pass does.
    #[test]
    fn shift_recovers_a_sub_pixel_translation() {
        let shape = [80, 96];
        let obj = sim::object(shape);
        let a = obj.plane(0);
        let s = [-2.413, 5.188];
        let b = shifted(a, shape, [-s[0], -s[1]]);
        let got = shift(a, &b, shape).unwrap();
        let step = 0.5 / UPSAMPLE as f64 + 1e-12;
        assert!(
            (got[0] - s[0]).abs() <= step && (got[1] - s[1]).abs() <= step,
            "{got:?}"
        );
        assert!(matches!(shift(a, &b[1..], shape), Err(CoreError::Shape(_))));
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
        let with =
            |reg, exclude: Option<&[bool]>| Frc::compute_with(&a, &a, [4, 4], &whole, reg, exclude);
        assert!(matches!(
            with(Registration::Blind, Some(&[false; 15])),
            Err(CoreError::Shape(_))
        ));
        for (shift, radius) in [([f64::NAN, 0.0], None), ([0.0; 2], Some(-1.0))] {
            assert!(matches!(
                with(Registration::Near { shift, radius }, None),
                Err(CoreError::Invalid(_))
            ));
        }
    }
}
