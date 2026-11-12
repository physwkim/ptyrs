//! Simulated reference scans for the M3 solvers (DESIGN §7 M3): raster and Fermat scans with
//! position errors, per-frame intensity variation, an additive background, fly-scan blur and a
//! varying probe (OPR), all through the PyNX far-field model the engine uses
//! (`icalc = Σ_modes |FT_ortho(fftshift(P·O·q))|²`, detector order), then Poisson-sampled.
//!
//! True positions sit on whole object pixels so PyNX, pty-chi and ptyrs see one exact forward
//! model; only the nominal positions written to the CXI carry the sub-pixel errors.

use std::f64::consts::PI;
use std::path::Path;
use std::sync::Arc;

use crate::base::{
    C64, Counts, Geometry, Object, Positions, Probe, Scan, Stack, required_obj_shape, sim,
    view_corner,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use rustfft::{Fft, FftPlanner};
use serde_json::{Value, json};

#[derive(Debug, thiserror::Error)]
pub enum RefgenError {
    #[error(transparent)]
    Core(#[from] crate::base::CoreError),
    #[error(transparent)]
    Io(#[from] crate::io::IoError),
    #[error(transparent)]
    File(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Invalid(String),
}

/// fftshift index map of an even `[ny, nx]` plane (its own inverse).
#[inline]
pub fn shift(j: usize, ny: usize, nx: usize) -> usize {
    let (y, x) = (j / nx, j % nx);
    ((y + ny / 2) % ny) * nx + (x + nx / 2) % nx
}

/// PyNX far-field quadratic phase `exp(iπ/(λz)·((x-nx/2)²px² + (y-ny/2)²py²))`, natural order.
pub fn quad_phase(geometry: &Geometry) -> Vec<C64> {
    let [ny, nx] = geometry.frame;
    let [py, px] = geometry.pixel_size_object();
    let f = PI / (geometry.wavelength * geometry.distance);
    let mut q = Vec::with_capacity(ny * nx);
    for y in 0..ny {
        let dy = y as f64 - (ny / 2) as f64;
        for x in 0..nx {
            let dx = x as f64 - (nx / 2) as f64;
            q.push(C64::from_polar(
                1.0,
                f * (dx * dx * px * px + dy * dy * py * py),
            ));
        }
    }
    q
}

/// The far-field intensity of one object window under a set of probe modes.
pub struct Far {
    frame: [usize; 2],
    quad: Vec<C64>,
    fx: Arc<dyn Fft<f64>>,
    fy: Arc<dyn Fft<f64>>,
    psi: Vec<C64>,
    col: Vec<C64>,
}

impl Far {
    pub fn new(geometry: &Geometry) -> Self {
        let [ny, nx] = geometry.frame;
        let mut planner = FftPlanner::new();
        Self {
            frame: geometry.frame,
            quad: quad_phase(geometry),
            fx: planner.plan_fft_forward(nx),
            fy: planner.plan_fft_forward(ny),
            psi: vec![C64::default(); ny * nx],
            col: vec![C64::default(); ny],
        }
    }

    /// `out[i] += w·Σ_k |FT_ortho(fftshift(P_k·O_win·q))|²` at detector pixel `i`, where
    /// `O_win` is the `frame`-sized window of `object` (row length `nxo`) at `corner`.
    pub fn accumulate(
        &mut self,
        probes: &[&[C64]],
        object: &[C64],
        nxo: usize,
        corner: [usize; 2],
        w: f64,
        out: &mut [f64],
    ) {
        let [ny, nx] = self.frame;
        let n = ny * nx;
        let scale = w / n as f64;
        for p in probes {
            for (j, v) in self.psi.iter_mut().enumerate() {
                let i = shift(j, ny, nx);
                let (y, x) = (i / nx, i % nx);
                *v = p[i] * object[(corner[0] + y) * nxo + corner[1] + x] * self.quad[i];
            }
            for row in self.psi.chunks_exact_mut(nx) {
                self.fx.process(row);
            }
            for x in 0..nx {
                for y in 0..ny {
                    self.col[y] = self.psi[y * nx + x];
                }
                self.fy.process(&mut self.col);
                for y in 0..ny {
                    self.psi[y * nx + x] = self.col[y];
                }
            }
            for (i, o) in out.iter_mut().enumerate() {
                *o += scale * self.psi[shift(i, ny, nx)].norm_sqr();
            }
        }
    }
}

/// PyNX log-likelihoods (CUDA `LLKAll`, masked pixels skipped) normalised by `nb_obs`
/// (`Ptycho.get_llk(norm=True)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Llk {
    pub poisson: f64,
    pub gaussian: f64,
    pub euclidian: f64,
    pub nb_obs: usize,
}

/// [`Llk`] of `icalc` (`[nframe, ny, nx]`, detector order) against `counts`:
/// Poisson `icalc - iobs + iobs·ln(iobs/icalc)` (`icalc` where `iobs = 0`), Gaussian
/// `(iobs - icalc)²/(iobs + 1)`, Euclidian `4(√iobs - √icalc)²`.
pub fn llk(icalc: &[f64], counts: &Counts) -> Llk {
    let (mut p, mut g, mut e, mut nb_obs) = (0.0, 0.0, 0.0, 0usize);
    for (&c, &v) in icalc.iter().zip(counts.as_slice()) {
        let Some(iobs) = Counts::intensity(v) else {
            continue;
        };
        nb_obs += 1;
        p += if iobs > 0.0 {
            c - iobs + iobs * (iobs / c).ln()
        } else {
            c
        };
        g += (iobs - c).powi(2) / (iobs + 1.0);
        e += 4.0 * (iobs.sqrt() - c.sqrt()).powi(2);
    }
    let n = nb_obs as f64;
    Llk {
        poisson: p / n,
        gaussian: g / n,
        euclidian: e / n,
        nb_obs,
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Pattern {
    Raster { ny: usize, nx: usize },
    Fermat { n: usize },
}

/// Expected background counts per pixel, `flat + peak·exp(-r²/(2·width²))` with `r` the
/// distance in detector pixels from the zero-frequency pixel `[ny/2, nx/2]`.
#[derive(Clone, Copy, Debug)]
pub struct Background {
    pub flat: f64,
    pub peak: f64,
    pub width_px: f64,
}

impl Background {
    pub fn plane(&self, frame: [usize; 2]) -> Vec<f64> {
        let [ny, nx] = frame;
        let mut b = Vec::with_capacity(ny * nx);
        for y in 0..ny {
            for x in 0..nx {
                let r2 =
                    (y as f64 - (ny / 2) as f64).powi(2) + (x as f64 - (nx / 2) as f64).powi(2);
                b.push(self.flat + self.peak * (-r2 / (2.0 * self.width_px.powi(2))).exp());
            }
        }
        b
    }
}

#[derive(Clone, Debug)]
pub struct Spec {
    pub name: String,
    pub geometry: Geometry,
    pub pattern: Pattern,
    /// Scan step in object pixels.
    pub step_px: f64,
    /// Probe modes of the truth ([`sim::probe`]).
    pub nmode: usize,
    /// Probe modes of the shared initial guess.
    pub init_nmode: usize,
    /// Mean counts of a frame before intensity variation and background.
    pub photons_per_frame: f64,
    pub seed: u64,
    /// σ (object pixels, per axis) of nominal − true positions; 0 for none.
    pub position_error_px: f64,
    /// σ of the log-normal per-frame flux scale (mean 1); 0 for none.
    pub intensity_sigma: f64,
    pub background: Option<Background>,
    /// x offsets (object pixels) the probe visits during one exposure, equally weighted;
    /// `[0]` is a step scan.
    pub fly_px: Vec<i64>,
    /// Peak weight `A` of the probe variation `P0 + a_n·V`, `a_n = A·sin(3πn/N)`,
    /// `V = i·(r/σ)²·P0` scaled to `‖P0‖`; 0 for none.
    pub opr_amplitude: f64,
}

pub struct Generated {
    /// Nominal positions and the Poisson counts.
    pub scan: Scan,
    /// True object and probe in the PyNX convention, the probe scaled to counts.
    pub truth: (Object, Probe),
    /// Initial guess shared by every code: unit object, Gaussian probe with `init_nmode` modes.
    pub init: (Object, Probe),
    pub meta: Value,
}

fn rng(seed: u64, stream: u64) -> ChaCha8Rng {
    let mut r = ChaCha8Rng::seed_from_u64(seed);
    r.set_stream(stream);
    r
}

fn normals(r: &mut ChaCha8Rng, n: usize) -> Vec<f64> {
    (0..n).map(|_| StandardNormal.sample(r)).collect()
}

/// Hermite-Gauss probe modes on an envelope of width `sigma_amp` with phase
/// `curvature·(r/sigma_phase)²`; mode `k` takes power `weights[k] / Σweights · total`.
fn hermite_modes(
    frame: [usize; 2],
    weights: &[f64],
    sigma_amp: f64,
    sigma_phase: f64,
    curvature: f64,
    total: f64,
) -> Result<Probe, RefgenError> {
    const ORDERS: [(i32, i32); 6] = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)];
    let hermite = |n: i32, t: f64| match n {
        0 => 1.0,
        1 => 2.0 * t,
        _ => 4.0 * t * t - 2.0,
    };
    if weights.is_empty() || weights.len() > ORDERS.len() {
        return Err(RefgenError::Invalid(format!(
            "probe modes must be 1..={}",
            ORDERS.len()
        )));
    }
    let [ny, nx] = frame;
    let wsum: f64 = weights.iter().sum();
    let mut data = Vec::with_capacity(weights.len() * ny * nx);
    for (&(a, b), &w) in ORDERS.iter().zip(weights) {
        let start = data.len();
        for iy in 0..ny {
            for ix in 0..nx {
                let (dy, dx) = (iy as f64 - (ny / 2) as f64, ix as f64 - (nx / 2) as f64);
                let (u, v) = (dx / sigma_amp, dy / sigma_amp);
                let amp = hermite(a, u) * hermite(b, v) * (-(u * u + v * v) / 2.0).exp();
                let phase = curvature * (dx * dx + dy * dy) / (sigma_phase * sigma_phase);
                data.push(C64::from_polar(amp, phase));
            }
        }
        let plane = &mut data[start..];
        let norm = plane.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        let s = (total * w / wsum).sqrt() / norm;
        plane.iter_mut().for_each(|c| *c *= s);
    }
    Ok(Probe::new(Stack::from_vec([weights.len(), ny, nx], data)?))
}

/// Side of the [`sim::object`] tile layered over the full-size one; its star resolves down to
/// about 1.6 px, so the scattering reaches the detector edge.
pub const TILE: usize = 64;

/// [`sim::object`] of `shape` times [`sim::object`] of `[TILE, TILE]` tiled over it (two thin
/// layers in contact). The tile border is within 3.4e-3 of 0.95, so seams stay below the texture.
pub fn textured_object(shape: [usize; 2]) -> Object {
    let coarse = sim::object(shape);
    let fine = sim::object([TILE, TILE]);
    let (c, f) = (coarse.plane(0), fine.plane(0));
    let data = c
        .iter()
        .enumerate()
        .map(|(i, &v)| v * f[(i / shape[1] % TILE) * TILE + i % shape[1] % TILE])
        .collect();
    Object::new(Stack::from_vec([1, shape[0], shape[1]], data).expect("shape"))
}

fn pairs(p: &Positions, scale: [f64; 2]) -> Value {
    json!(
        p.y.iter()
            .zip(&p.x)
            .map(|(y, x)| [y / scale[0], x / scale[1]])
            .collect::<Vec<_>>()
    )
}

/// Simulates `spec`. Counts above `MAX_COUNT` are an error (lower `photons_per_frame`).
pub fn generate(spec: &Spec) -> Result<Generated, RefgenError> {
    let g = spec.geometry;
    let [ny, nx] = g.frame;
    let plane = ny * nx;
    let pixel = g.pixel_size_object();
    if spec.fly_px.is_empty() {
        return Err(RefgenError::Invalid(
            "fly_px must hold at least one offset".into(),
        ));
    }

    // True positions on whole object pixels; nominal = true + zero-mean errors.
    let base = match spec.pattern {
        Pattern::Raster { ny, nx } => sim::raster(ny, nx, spec.step_px * pixel[1]),
        Pattern::Fermat { n } => sim::fermat(n, spec.step_px * pixel[1]),
    };
    let snap = |v: &[f64], p: f64| v.iter().map(|x| (x / p).round() * p).collect::<Vec<_>>();
    let truth_pos = Positions::new(snap(&base.y, pixel[0]), snap(&base.x, pixel[1]))?;
    let nframe = truth_pos.len();
    let nominal = if spec.position_error_px > 0.0 {
        let mut r = rng(spec.seed, 1);
        let mut axis = |v: &[f64], p: f64| {
            let e = normals(&mut r, nframe);
            let m = e.iter().sum::<f64>() / nframe as f64;
            v.iter()
                .zip(&e)
                .map(|(x, e)| x + (e - m) * spec.position_error_px * p)
                .collect::<Vec<_>>()
        };
        let y = axis(&truth_pos.y, pixel[0]);
        let x = axis(&truth_pos.x, pixel[1]);
        Positions::new(y, x)?
    } else {
        truth_pos.clone()
    };
    let truth_pp = truth_pos.pynx_preprocess(pixel);
    let nominal_pp = nominal.pynx_preprocess(pixel);
    // True positions in the frame a reconstruction from the nominal ones works in (PyNX
    // subtracts the mean; the snap of `truth_pp` may differ from this by a common sub-pixel).
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (my, mx) = (mean(&nominal.y), mean(&nominal.x));
    let in_nominal_frame = Positions::new(
        truth_pos.y.iter().map(|y| y - my).collect(),
        truth_pos.x.iter().map(|x| x - mx).collect(),
    )?;

    // One object covering the true windows (with fly offsets) and the nominal ones.
    let (fmin, fmax) = (
        *spec.fly_px.iter().min().unwrap() as f64,
        *spec.fly_px.iter().max().unwrap() as f64,
    );
    let mut cover_y = truth_pp.y.clone();
    let mut cover_x: Vec<f64> = truth_pp.x.iter().map(|x| x + fmin * pixel[1]).collect();
    cover_y.extend(&truth_pp.y);
    cover_x.extend(truth_pp.x.iter().map(|x| x + fmax * pixel[1]));
    cover_y.extend(&nominal_pp.y);
    cover_x.extend(&nominal_pp.x);
    let obj_shape = required_obj_shape(&Positions::new(cover_y, cover_x)?, pixel, g.frame, 16);
    let object = textured_object(obj_shape);
    let probe = sim::probe(g.frame, spec.nmode)?;

    // Probe variation of mode 0.
    let sigma = ny.min(nx) as f64 / 12.0;
    let p0 = probe.plane(0);
    let mut eig: Vec<C64> = p0
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let (dy, dx) = (
                (i / nx) as f64 - (ny / 2) as f64,
                (i % nx) as f64 - (nx / 2) as f64,
            );
            C64::i() * ((dx * dx + dy * dy) / (sigma * sigma)) * p
        })
        .collect();
    let norm = |v: &[C64]| v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    let s = norm(p0) / norm(&eig);
    eig.iter_mut().for_each(|c| *c *= s);
    let opr: Vec<f64> = (0..nframe)
        .map(|n| spec.opr_amplitude * (3.0 * PI * n as f64 / nframe as f64).sin())
        .collect();

    let mut far = Far::new(&g);
    let mut intensity = vec![0.0; nframe * plane];
    let w = 1.0 / spec.fly_px.len() as f64;
    let mut varied = p0.to_vec();
    for (f, out) in intensity.chunks_exact_mut(plane).enumerate() {
        let mut modes: Vec<&[C64]> = (0..spec.nmode).map(|k| probe.plane(k)).collect();
        if spec.opr_amplitude != 0.0 {
            for ((v, p), e) in varied.iter_mut().zip(p0).zip(&eig) {
                *v = p + opr[f] * e;
            }
            modes[0] = &varied;
        }
        // `obj_shape` was sized over these same windows, fly offsets included.
        let corner = |obj, n, d| view_corner(obj, n, d).expect("object covers every true window");
        let oy = corner(obj_shape[0], ny, truth_pp.y[f] / pixel[0]);
        for &d in &spec.fly_px {
            let ox = corner(obj_shape[1], nx, truth_pp.x[f] / pixel[1] + d as f64);
            far.accumulate(&modes, object.plane(0), obj_shape[1], [oy, ox], w, out);
        }
    }
    let scale = spec.photons_per_frame / (intensity.iter().sum::<f64>() / nframe as f64);
    let flux: Vec<f64> = if spec.intensity_sigma > 0.0 {
        let s = spec.intensity_sigma;
        normals(&mut rng(spec.seed, 2), nframe)
            .into_iter()
            .map(|z| (s * z - s * s / 2.0).exp())
            .collect()
    } else {
        vec![1.0; nframe]
    };
    let bg = spec.background.map(|b| b.plane(g.frame));
    for (out, &s) in intensity.chunks_exact_mut(plane).zip(&flux) {
        out.iter_mut().for_each(|v| *v *= scale * s);
        if let Some(bg) = &bg {
            out.iter_mut().zip(bg).for_each(|(v, b)| *v += b);
        }
    }
    let counts = sim::poisson([nframe, ny, nx], &intensity, None, spec.seed)?;
    let truth_llk = llk(&intensity, &counts);
    let total: f64 = counts.as_slice().iter().map(|&c| c as f64).sum();
    let max = counts.as_slice().iter().copied().max().unwrap_or(0);
    drop(intensity);

    let mut truth_probe = probe.0.clone();
    let root = scale.sqrt();
    truth_probe
        .as_mut_slice()
        .iter_mut()
        .for_each(|c| *c *= root);
    let mut weights = vec![1.0];
    weights.resize(spec.init_nmode, 0.1);
    let init_probe = hermite_modes(
        g.frame,
        &weights,
        1.2 * sigma,
        sigma,
        1.2,
        total / nframe as f64,
    )?;
    let init_object = Object::new(Stack::filled(
        [1, obj_shape[0], obj_shape[1]],
        C64::new(1.0, 0.0),
    ));

    let meta = json!({
        "name": spec.name,
        "geometry": {
            "wavelength_m": g.wavelength,
            "distance_m": g.distance,
            "detector_pixel_m": g.detector_pixel,
            "frame": g.frame,
            "pixel_size_object_m": pixel,
        },
        "pattern": match spec.pattern {
            Pattern::Raster { ny, nx } => json!({"raster": [ny, nx]}),
            Pattern::Fermat { n } => json!({"fermat": n}),
        },
        "step_px": spec.step_px,
        "nframe": nframe,
        "obj_shape": obj_shape,
        "nmode": spec.nmode,
        "init_nmode": spec.init_nmode,
        "seed": spec.seed,
        "photons_per_frame": spec.photons_per_frame,
        "scale": scale,
        "counts": {"max": max, "mean_per_frame": total / nframe as f64},
        "positions": {
            "true_m": pairs(&truth_pos, [1.0, 1.0]),
            "nominal_m": pairs(&nominal, [1.0, 1.0]),
            "true_pp_px": pairs(&truth_pp, pixel),
            "nominal_pp_px": pairs(&nominal_pp, pixel),
            "true_in_nominal_frame_px": pairs(&in_nominal_frame, pixel),
            "error_sigma_px": spec.position_error_px,
        },
        "intensity_scale": (spec.intensity_sigma > 0.0).then_some(&flux),
        "intensity_sigma": spec.intensity_sigma,
        "background": spec.background.map(|b| json!({
            "flat": b.flat, "peak": b.peak, "width_px": b.width_px,
            "formula": "flat + peak*exp(-r^2/(2 width^2)), r from detector pixel [ny/2, nx/2]",
        })),
        "fly_px": spec.fly_px,
        "opr": (spec.opr_amplitude != 0.0).then(|| json!({
            "amplitude": spec.opr_amplitude,
            "weights": opr,
            "model": "mode 0 = P0 + a_n*V, V = i*(r/sigma)^2*P0 scaled to |P0|, sigma = min(ny,nx)/12",
        })),
        "llk_truth": {
            "poisson": truth_llk.poisson,
            "gaussian": truth_llk.gaussian,
            "euclidian": truth_llk.euclidian,
            "nb_obs": truth_llk.nb_obs,
        },
        "files": {
            "cxi": format!("{}.cxi", spec.name),
            "truth": format!("{}_truth.h5", spec.name),
            "init": format!("{}_init.h5", spec.name),
        },
    });

    Ok(Generated {
        scan: Scan::new(g, nominal, counts)?,
        truth: (object, Probe::new(truth_probe)),
        init: (init_object, init_probe),
        meta,
    })
}

/// Writes `<name>.cxi`, `<name>_truth.h5`, `<name>_init.h5` and `<name>.json` into `dir`.
pub fn write(dir: impl AsRef<Path>, generated: &Generated) -> Result<(), RefgenError> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)?;
    let files = &generated.meta["files"];
    let file = |k: &str| dir.join(files[k].as_str().expect("file name"));
    crate::io::write_cxi(file("cxi"), &generated.scan)?;
    crate::io::write_modes(file("truth"), &generated.truth.0, &generated.truth.1)?;
    crate::io::write_modes(file("init"), &generated.init.0, &generated.init.1)?;
    let name = generated.meta["name"].as_str().expect("name");
    std::fs::write(
        dir.join(format!("{name}.json")),
        serde_json::to_vec_pretty(&generated.meta)?,
    )?;
    Ok(())
}
