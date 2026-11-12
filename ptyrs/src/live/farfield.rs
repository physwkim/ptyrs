//! M5 scan simulator (DESIGN.md §7 M5): far-field frames of a known object
//! under a two-mode probe at the points of a raster or Fermat-spiral scan,
//! streamed by pty-sim and dumped as ground truth for the engine.
//!
//! Model, also stated in the `meta.json` of [`Ptycho::dump`]:
//! - Point `k` sits at whole object pixels `(ix, iy)` from the scan centre,
//!   so frames are exact under any sub-pixel convention; its position is
//!   `(ix, iy) * dx` metres, `dx = λ z / (n p)`. Its window is
//!   `W = O[oy..oy + n, ox..ox + n]`, `(ox, oy)` being `(ix, iy)` less their
//!   minimum over the scan; the object is the windows' bounding box.
//! - Expected counts `I[v, u] = Σ_m |F[P_m W][v - n/2, u - n/2]|²`, `F` the
//!   unitary DFT `Σ_r f[r] exp(-2πi q·r / n) / n`: zero frequency on pixel
//!   `(n/2, n/2)`. Both modes are centred on pixel `(n/2, n/2)` of the
//!   window, and `Σ_m Σ |P_m|²` is the incident photon count.
//! - Counts are Poisson draws from `I` (normal approximation from 30 up, as
//!   in `synth`) clamped to 65535, a pure function of `(seed, k, pixel)`.
//! - The object is `exp((-β + iφ) t)` for a thickness `t` summed from
//!   seeded discs (one per 40² px, radius 4-30 px, thickness 0.2-1, edges
//!   antialiased over a pixel). The modes come from a pupil of radius
//!   `0.2 n` in the detector plane under a defocus phase `a |q|²` that
//!   spreads mode 0 over about `n / 4` pixels; mode 1 is that pupil times
//!   `q_x / R`, orthogonal to mode 0 by symmetry; powers 0.8 and 0.2.

use std::f64::consts::PI;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use rustfft::num_complex::{Complex32, Complex64};
use rustfft::{Fft, FftPlanner};

use crate::live::pipeline::Truth;
use crate::live::synth::{Region, hash3, poisson, splitmix64, unit};

const MODE_POWER: [f64; 2] = [0.8, 0.2];
/// Pupil radius over the frame side.
const PUPIL: f64 = 0.2;
const DISC_AREA: usize = 40 * 40;
const ABSORB: f64 = 0.15;
const PHASE: f64 = 0.8;
const OBJECT_TAG: u64 = 0x0B1E_C700;
const NOISE_TAG: u64 = 0x9015_E000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScanPath {
    /// Rows of `cols` points, x fastest, every row the same direction; the
    /// last row may be partial.
    Raster { cols: usize },
    /// `r = step √(k / π)`, `θ = k` golden angles: one point per `step²`, the
    /// density of a raster of the same step.
    Fermat,
}

#[derive(Debug, Clone, Copy)]
pub struct PtychoSpec {
    /// Frame side, even.
    pub n: usize,
    pub path: ScanPath,
    pub points: usize,
    /// Scan step, object pixels.
    pub step: f64,
    /// Incident photons per frame.
    pub photons: f64,
    pub seed: u64,
    pub wavelength: f64,
    pub distance: f64,
    pub det_pixel: f64,
}

impl PtychoSpec {
    /// Object pixel, m.
    pub fn dx(&self) -> f64 {
        self.wavelength * self.distance / (self.n as f64 * self.det_pixel)
    }
}

pub struct Ptycho {
    spec: PtychoSpec,
    /// `(ix, iy)` per point.
    scan: Vec<[i64; 2]>,
    /// `(oy, ox)` per point.
    offsets: Vec<[usize; 2]>,
    shape: [usize; 2],
    object: Vec<Complex32>,
    probe: Vec<Complex32>,
    fft: Arc<dyn Fft<f64>>,
}

impl Ptycho {
    pub fn new(spec: PtychoSpec) -> Self {
        let n = spec.n;
        assert!(n >= 4 && n.is_multiple_of(2), "frame side {n} must be even");
        assert!(
            (1..=i32::MAX as usize).contains(&spec.points),
            "{} points",
            spec.points
        );
        let scan = scan_points(spec.path, spec.points, spec.step);
        let lo = [0, 1].map(|a| scan.iter().map(|p| p[a]).min().unwrap());
        let hi = [0, 1].map(|a| scan.iter().map(|p| p[a]).max().unwrap());
        let shape = [1, 0].map(|a| (hi[a] - lo[a]) as usize + n);
        let offsets = scan
            .iter()
            .map(|p| [1, 0].map(|a| (p[a] - lo[a]) as usize))
            .collect();
        let mut planner = FftPlanner::new();
        let probe = probe_modes(n, spec.photons, &*planner.plan_fft_inverse(n));
        Self {
            spec,
            scan,
            offsets,
            shape,
            object: object(spec.seed, shape),
            probe,
            fft: planner.plan_fft_forward(n),
        }
    }

    pub fn spec(&self) -> &PtychoSpec {
        &self.spec
    }

    /// One line for the logs.
    pub fn describe(&self) -> String {
        let (s, [ny, nx]) = (&self.spec, self.shape);
        format!(
            "model=ptycho path={:?} points={} step_px={} photons={:.3e} object={ny}x{nx} \
             obj_pixel_nm={:.2}",
            s.path,
            self.points(),
            s.step,
            s.photons,
            s.dx() * 1e9
        )
    }

    pub fn points(&self) -> usize {
        self.scan.len()
    }

    /// `(x, y)` of point `k` from the scan centre, m.
    pub fn position(&self, k: usize) -> (f64, f64) {
        let [ix, iy] = self.scan[k];
        let dx = self.spec.dx();
        (ix as f64 * dx, iy as f64 * dx)
    }

    /// `(oy, ox)`: top-left object pixel of point `k`'s window.
    pub fn offset(&self, k: usize) -> [usize; 2] {
        self.offsets[k]
    }

    /// Row-major `[ny, nx]`.
    pub fn object(&self) -> (&[Complex32], [usize; 2]) {
        (&self.object, self.shape)
    }

    /// Two modes of `n x n`, row-major.
    pub fn probe(&self) -> &[Complex32] {
        &self.probe
    }

    /// Point that trigger `uid` sees; the scan repeats.
    pub fn point(&self, uid: i32) -> usize {
        uid.rem_euclid(self.points() as i32) as usize
    }

    /// Expected counts of point `k`, detector pixels row-major.
    pub fn intensity(&self, k: usize) -> Vec<f64> {
        let (n, h) = (self.spec.n, self.spec.n / 2);
        let [oy, ox] = self.offsets[k];
        let nxo = self.shape[1];
        let norm = 1.0 / (n * n) as f64;
        let mut out = vec![0f64; n * n];
        let mut buf = vec![Complex64::default(); n * n];
        for mode in self.probe.chunks_exact(n * n) {
            for (i, b) in buf.iter_mut().enumerate() {
                *b = wide(mode[i]) * wide(self.object[(oy + i / n) * nxo + ox + i % n]);
            }
            fft2(&*self.fft, &mut buf);
            for (i, v) in out.iter_mut().enumerate() {
                *v += buf[(i / n + h) % n * n + (i % n + h) % n].norm_sqr() * norm;
            }
        }
        out
    }

    pub fn frame(&self, k: usize) -> Vec<u16> {
        let seed = self.spec.seed ^ NOISE_TAG;
        self.intensity(k)
            .iter()
            .enumerate()
            .map(|(i, &l)| poisson(l, hash3(seed, k as u64, i as u64)).min(65_535) as u16)
            .collect()
    }

    /// Ground truth for the engine in `dir`: `object.npy` (complex64
    /// `[ny, nx]`), `probe.npy` (complex64 `[2, n, n]`), `positions.npy`
    /// (float64 `[points, 2]`, x then y, m), `offsets.npy` (int64
    /// `[points, 2]`, oy then ox), `frames.npy` (uint16 `[points, n, n]`)
    /// and `meta.json`.
    pub fn dump(&self, dir: &Path) -> io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let (n, np, s) = (self.spec.n, self.points(), &self.spec);
        let c8 = |a: &[Complex32]| -> Vec<u8> {
            a.iter()
                .flat_map(|c| [c.re.to_le_bytes(), c.im.to_le_bytes()])
                .flatten()
                .collect()
        };
        write_npy(
            &dir.join("object.npy"),
            "<c8",
            &self.shape,
            &c8(&self.object),
        )?;
        write_npy(&dir.join("probe.npy"), "<c8", &[2, n, n], &c8(&self.probe))?;
        let pos: Vec<u8> = (0..np)
            .flat_map(|k| {
                let (x, y) = self.position(k);
                [x.to_le_bytes(), y.to_le_bytes()]
            })
            .flatten()
            .collect();
        write_npy(&dir.join("positions.npy"), "<f8", &[np, 2], &pos)?;
        let off: Vec<u8> = self
            .offsets
            .iter()
            .flat_map(|o| o.map(|v| (v as i64).to_le_bytes()))
            .flatten()
            .collect();
        write_npy(&dir.join("offsets.npy"), "<i8", &[np, 2], &off)?;
        let mut w = npy_writer(&dir.join("frames.npy"), "<u2", &[np, n, n])?;
        let threads = std::thread::available_parallelism().map_or(8, |t| t.get());
        for ks in (0..np).collect::<Vec<_>>().chunks(threads) {
            let frames: Vec<Vec<u16>> = std::thread::scope(|sc| {
                let hs: Vec<_> = ks
                    .iter()
                    .map(|&k| sc.spawn(move || self.frame(k)))
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for f in frames {
                w.write_all(&f.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            }
        }
        w.flush()?;
        let (path, cols) = match s.path {
            ScanPath::Raster { cols } => ("raster", Some(cols)),
            ScanPath::Fermat => ("fermat", None),
        };
        let meta = serde_json::json!({
            "n": n,
            "points": np,
            "path": path,
            "cols": cols,
            "step_px": s.step,
            "photons": s.photons,
            "seed": s.seed,
            "wavelength_m": s.wavelength,
            "distance_m": s.distance,
            "det_pixel_m": s.det_pixel,
            "obj_pixel_m": s.dx(),
            "object_shape": self.shape,
            "mode_power": MODE_POWER,
            "pupil_radius_px": PUPIL * n as f64,
            "convention": "frames[k] ~ Poisson(sum_m |fftshift(fft2(probe[m] * \
                object[oy:oy+n, ox:ox+n], norm='ortho'))|^2) clipped to 65535, \
                (oy, ox) = offsets[k]; probe modes centred on pixel (n/2, n/2); \
                positions[k] = (x, y) m from the scan centre = (ox, oy) * obj_pixel_m \
                + a constant",
        });
        std::fs::write(dir.join("meta.json"), serde_json::to_vec_pretty(&meta)?)
    }
}

impl Truth for Ptycho {
    fn fill(&self, uid: i32, r: Region, out: &mut [u16]) {
        let n = self.spec.n;
        let f = self.frame(self.point(uid));
        for (row, o) in out.chunks_exact_mut(r.w).take(r.h).enumerate() {
            o.copy_from_slice(&f[(r.y0 + row) * n + r.x0..][..r.w]);
        }
    }

    fn position(&self, uid: i32) -> (f64, f64) {
        Ptycho::position(self, self.point(uid))
    }

    fn frame_size(&self) -> (usize, usize) {
        (self.spec.n, self.spec.n)
    }

    fn period(&self) -> usize {
        self.points()
    }
}

fn wide(c: Complex32) -> Complex64 {
    Complex64::new(c.re.into(), c.im.into())
}

/// `(ix, iy)` per point, whole pixels from the scan centre.
fn scan_points(path: ScanPath, points: usize, step: f64) -> Vec<[i64; 2]> {
    let golden = PI * (3.0 - 5f64.sqrt());
    (0..points)
        .map(|k| {
            let (x, y) = match path {
                ScanPath::Raster { cols } => {
                    assert!(cols > 0, "raster of 0 columns");
                    let c = |i: usize, len: usize| (i as f64 - (len - 1) as f64 / 2.0) * step;
                    (c(k % cols, cols), c(k / cols, points.div_ceil(cols)))
                }
                ScanPath::Fermat => {
                    let (r, t) = (step * (k as f64 / PI).sqrt(), k as f64 * golden);
                    (r * t.cos(), r * t.sin())
                }
            };
            [x.round() as i64, y.round() as i64]
        })
        .collect()
}

fn object(seed: u64, [ny, nx]: [usize; 2]) -> Vec<Complex32> {
    let mut t = vec![0f64; ny * nx];
    for j in 0..(ny * nx).div_ceil(DISC_AREA) {
        let mut h = hash3(seed, OBJECT_TAG, j as u64);
        let mut next = || {
            h = splitmix64(h);
            unit(h)
        };
        let (cy, cx) = (next() * ny as f64, next() * nx as f64);
        let (r, th) = (4.0 + 26.0 * next(), 0.2 + 0.8 * next());
        // Every pixel within r + 1/2 of the centre.
        let span = |c: f64, len: usize| {
            (
                (c - r - 1.0).max(0.0) as usize,
                ((c + r + 2.0) as usize).min(len),
            )
        };
        let ((y0, y1), (x0, x1)) = (span(cy, ny), span(cx, nx));
        for y in y0..y1 {
            for x in x0..x1 {
                let d = (y as f64 - cy).hypot(x as f64 - cx);
                t[y * nx + x] += th * (r - d + 0.5).clamp(0.0, 1.0);
            }
        }
    }
    t.into_iter()
        .map(|t| {
            let c = Complex64::from_polar((-ABSORB * t).exp(), PHASE * t);
            Complex32::new(c.re as f32, c.im as f32)
        })
        .collect()
}

fn probe_modes(n: usize, photons: f64, ifft: &dyn Fft<f64>) -> Vec<Complex32> {
    let (h, nf) = (n / 2, n as f64);
    let r = PUPIL * nf;
    // a |q|² moves ray q by n a |q| / π pixels: mode 0 spans 2 n a R / π = n / 4.
    let a = PI * (nf / 4.0) / (2.0 * nf * r);
    // DFT index → signed frequency.
    let q = |j: usize| ((j + h) % n) as f64 - h as f64;
    let mut out = Vec::with_capacity(2 * n * n);
    let mut buf = vec![Complex64::default(); n * n];
    for (m, power) in MODE_POWER.iter().enumerate() {
        for (i, b) in buf.iter_mut().enumerate() {
            let (qy, qx) = (q(i / n), q(i % n));
            let q2 = qx * qx + qy * qy;
            let w = if m == 0 { 1.0 } else { qx / r };
            *b = if q2 <= r * r {
                Complex64::from_polar(1.0, a * q2) * w
            } else {
                Complex64::default()
            };
        }
        fft2(ifft, &mut buf);
        let s = (power * photons / buf.iter().map(|c| c.norm_sqr()).sum::<f64>()).sqrt();
        // Pixel 0 of the transform onto pixel (n/2, n/2).
        out.extend((0..n * n).map(|i| {
            let c = buf[(i / n + h) % n * n + (i % n + h) % n] * s;
            Complex32::new(c.re as f32, c.im as f32)
        }));
    }
    out
}

/// In place over a row-major `n x n` buffer, `n = fft.len()`.
fn fft2(fft: &dyn Fft<f64>, buf: &mut [Complex64]) {
    let n = fft.len();
    let mut scratch = vec![Complex64::default(); fft.get_inplace_scratch_len()];
    for _ in 0..2 {
        fft.process_with_scratch(buf, &mut scratch);
        for y in 0..n {
            for x in y + 1..n {
                buf.swap(y * n + x, x * n + y);
            }
        }
    }
}

fn npy_writer(path: &Path, descr: &str, shape: &[usize]) -> io::Result<BufWriter<File>> {
    let dims: String = shape.iter().map(|d| format!("{d},")).collect();
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({dims}), }}");
    // Magic, version, length, header and newline end on a 64-byte boundary.
    header.push_str(&" ".repeat(63 - (10 + header.len()) % 64));
    header.push('\n');
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(b"\x93NUMPY\x01\x00")?;
    w.write_all(&(header.len() as u16).to_le_bytes())?;
    w.write_all(header.as_bytes())?;
    Ok(w)
}

fn write_npy(path: &Path, descr: &str, shape: &[usize], data: &[u8]) -> io::Result<()> {
    let mut w = npy_writer(path, descr, shape)?;
    w.write_all(data)?;
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(n: usize, path: ScanPath, points: usize, step: f64) -> PtychoSpec {
        PtychoSpec {
            n,
            path,
            points,
            step,
            photons: 1e6,
            seed: 3,
            wavelength: 1.5498e-10,
            distance: 5.0,
            det_pixel: 75e-6,
        }
    }

    /// Windows tile the object's bounding box exactly and the reported
    /// positions are the whole-pixel offsets times `dx`.
    fn check_windows(p: &Ptycho) {
        let n = p.spec().n;
        let (_, [ny, nx]) = p.object();
        let off: Vec<[usize; 2]> = (0..p.points()).map(|k| p.offset(k)).collect();
        assert_eq!(off.iter().map(|o| o[0]).min(), Some(0));
        assert_eq!(off.iter().map(|o| o[1]).min(), Some(0));
        assert_eq!(off.iter().map(|o| o[0] + n).max(), Some(ny));
        assert_eq!(off.iter().map(|o| o[1] + n).max(), Some(nx));
        let dx = p.spec().dx();
        let (x0, y0) = p.position(0);
        for (k, o) in off.iter().enumerate() {
            let (x, y) = p.position(k);
            assert_eq!(
                ((x - x0) / dx).round() as i64,
                o[1] as i64 - off[0][1] as i64
            );
            assert_eq!(
                ((y - y0) / dx).round() as i64,
                o[0] as i64 - off[0][0] as i64
            );
            assert_eq!((x / dx).round() * dx, x, "point {k} off the pixel grid");
        }
    }

    #[test]
    fn raster_rows_with_a_partial_last_row() {
        let p = Ptycho::new(spec(32, ScanPath::Raster { cols: 7 }, 45, 5.0));
        check_windows(&p);
        assert_eq!(p.object().1, [6 * 5 + 32, 6 * 5 + 32]);
        for k in 0..45 {
            assert_eq!(p.offset(k), [k / 7 * 5, k % 7 * 5], "point {k}");
        }
        // Centred: the full first row is symmetric about 0.
        assert_eq!(p.position(0).0, -p.position(6).0);
    }

    #[test]
    fn fermat_spiral_has_the_density_of_a_raster_of_the_same_step() {
        let (points, step) = (2000, 8.0);
        let p = Ptycho::new(spec(32, ScanPath::Fermat, points, step));
        check_windows(&p);
        let dx = p.spec().dx();
        assert_eq!(p.position(0), (0.0, 0.0));
        // Points inside radius rho: pi rho² / step², to within the rim.
        for rho in [60.0, 120.0, 200.0] {
            let inside = (0..points)
                .filter(|&k| {
                    let (x, y) = p.position(k);
                    (x / dx).hypot(y / dx) < rho
                })
                .count() as f64;
            let want = PI * rho * rho / (step * step);
            assert!(
                (inside - want).abs() < 2.0 * PI * rho / step,
                "rho {rho}: {inside} vs {want}"
            );
        }
    }

    #[test]
    fn probe_modes_are_orthogonal_centred_and_carry_the_photons() {
        let n = 64;
        let p = Ptycho::new(spec(n, ScanPath::Fermat, 1, 1.0));
        let modes: Vec<&[Complex32]> = p.probe().chunks_exact(n * n).collect();
        let dot = |a: &[Complex32], b: &[Complex32]| -> Complex64 {
            a.iter()
                .zip(b)
                .map(|(a, b)| wide(*a).conj() * wide(*b))
                .sum()
        };
        for (m, power) in MODE_POWER.iter().enumerate() {
            let got = dot(modes[m], modes[m]).re;
            assert!((got / (power * 1e6) - 1.0).abs() < 1e-6, "mode {m} {got}");
            // Centred on (n/2, n/2): mode 0 even about it, mode 1 odd.
            let sign = [1.0f32, -1.0][m];
            let at = |y: usize, x: usize| modes[m][y * n + x];
            for (y, x) in (1..n).flat_map(|y| (1..n).map(move |x| (y, x))) {
                let d = at(y, x) - at(n - y, n - x) * sign;
                assert!(
                    d.norm() <= 1e-6 * at(y, x).norm().max(1e-3),
                    "mode {m} ({y}, {x})"
                );
            }
        }
        let cross = dot(modes[0], modes[1]).norm() / (0.8f64 * 0.2).sqrt() / 1e6;
        assert!(cross < 1e-6, "overlap {cross}");
    }

    /// The model's intensity against a direct DFT of its own probe and
    /// object, written from the conventions in the module docs.
    #[test]
    fn intensity_is_the_shifted_unitary_dft_of_the_exit_waves() {
        let n = 16;
        let p = Ptycho::new(spec(n, ScanPath::Raster { cols: 3 }, 6, 3.0));
        let (obj, [_, nxo]) = p.object();
        let h = (n / 2) as f64;
        for k in [0, 4] {
            let [oy, ox] = p.offset(k);
            let got = p.intensity(k);
            for v in 0..n {
                for u in 0..n {
                    let mut want = 0.0;
                    for mode in p.probe().chunks_exact(n * n) {
                        let mut s = Complex64::default();
                        for y in 0..n {
                            for x in 0..n {
                                let e = -2.0
                                    * PI
                                    * ((v as f64 - h) * y as f64 + (u as f64 - h) * x as f64)
                                    / n as f64;
                                s += wide(mode[y * n + x])
                                    * wide(obj[(oy + y) * nxo + ox + x])
                                    * Complex64::from_polar(1.0, e);
                            }
                        }
                        want += (s / n as f64).norm_sqr();
                    }
                    let g = got[v * n + u];
                    assert!(
                        (g - want).abs() <= 1e-9 * want.max(1e-3),
                        "k {k} ({v}, {u}): {g} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn frames_are_poisson_about_the_intensity_and_crops_match() {
        let n = 64;
        let p = Ptycho::new(spec(n, ScanPath::Fermat, 5, 6.0));
        let (obj, [_, nxo]) = p.object();
        for k in 0..5 {
            let i = p.intensity(k);
            // Parseval: detected photons are the exit waves' power.
            let [oy, ox] = p.offset(k);
            let exit: f64 = p
                .probe()
                .chunks_exact(n * n)
                .flat_map(|m| {
                    (0..n * n).map(move |j| {
                        (wide(m[j]) * wide(obj[(oy + j / n) * nxo + ox + j % n])).norm_sqr()
                    })
                })
                .sum();
            let total: f64 = i.iter().sum();
            assert!(
                (total / exit - 1.0).abs() < 1e-9,
                "k {k}: {total} vs {exit}"
            );
            let f = p.frame(k);
            assert_eq!(f, p.frame(k));
            let counts: f64 = f.iter().map(|&c| f64::from(c)).sum();
            assert!(
                (counts - total).abs() < 5.0 * total.sqrt(),
                "k {k}: {counts} vs {total}"
            );
            let r = Region {
                x0: 9,
                y0: 20,
                w: 30,
                h: 17,
            };
            let mut crop = vec![0u16; r.w * r.h];
            Truth::fill(&p, k as i32 + 5, r, &mut crop);
            for row in 0..r.h {
                assert_eq!(
                    crop[row * r.w..][..r.w],
                    f[(r.y0 + row) * n + r.x0..][..r.w]
                );
            }
        }
        assert_ne!(p.frame(0), p.frame(1));
    }
}
