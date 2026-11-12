//! Simulation inputs: a known complex object, a multi-mode probe, raster and Fermat scans, and
//! Poisson sampling. The far-field model that turns object and probe into intensities lives in
//! `ptyrs-engine` (`sim::simulate`), so the simulator and the reconstruction share one model.

use std::f64::consts::PI;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Poisson};

use crate::{C64, CoreError, Counts, MASKED, MAX_COUNT, Object, Positions, Probe, Stack};

/// Rectangular raster of `ny × nx` points with spacing `step` (metres), centred on 0, y-major.
pub fn raster(ny: usize, nx: usize, step: f64) -> Positions {
    let (mut y, mut x) = (Vec::with_capacity(ny * nx), Vec::with_capacity(ny * nx));
    for iy in 0..ny {
        for ix in 0..nx {
            y.push((iy as f64 - (ny as f64 - 1.0) / 2.0) * step);
            x.push((ix as f64 - (nx as f64 - 1.0) / 2.0) * step);
        }
    }
    Positions { y, x }
}

/// Fermat spiral of `n` points, `r_k = c·sqrt(k)`, `θ_k = k·golden angle`, with `c` chosen so
/// the mean nearest-neighbour spacing is about `step` (metres).
pub fn fermat(n: usize, step: f64) -> Positions {
    let golden = PI * (3.0 - 5f64.sqrt());
    let c = step / PI.sqrt();
    let (y, x) = (0..n)
        .map(|k| {
            let (r, t) = (c * (k as f64).sqrt(), k as f64 * golden);
            (r * t.sin(), r * t.cos())
        })
        .unzip();
    Positions { y, x }
}

fn smoothstep(e0: f64, e1: f64, v: f64) -> f64 {
    let t = ((v - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Known single-mode test object `[1, ny, nx]`: a Siemens star and three Gaussian features in
/// phase (about -0.4..1.6 rad) and amplitude (about 0.5..0.95), centred on the array.
pub fn object(shape: [usize; 2]) -> Object {
    let [ny, nx] = shape;
    let s = ny.min(nx) as f64 / 2.0;
    let blob = |u: f64, v: f64, cu: f64, cv: f64, w: f64| {
        (-((u - cu).powi(2) + (v - cv).powi(2)) / (2.0 * w * w)).exp()
    };
    let mut data = Vec::with_capacity(ny * nx);
    for iy in 0..ny {
        for ix in 0..nx {
            let (u, v) = (
                (ix as f64 - nx as f64 / 2.0) / s,
                (iy as f64 - ny as f64 / 2.0) / s,
            );
            let r = u.hypot(v);
            let star = 0.5
                * (1.0 + (3.0 * (12.0 * v.atan2(u)).cos()).tanh())
                * smoothstep(0.08, 0.12, r)
                * (1.0 - smoothstep(0.55, 0.6, r));
            let phase =
                star + 0.6 * blob(u, v, 0.55, -0.45, 0.12) - 0.4 * blob(u, v, -0.5, 0.5, 0.1);
            let amp = 0.95 - 0.3 * blob(u, v, -0.45, -0.55, 0.15) - 0.15 * star;
            data.push(C64::from_polar(amp, phase));
        }
    }
    Object::new(Stack::from_vec([1, ny, nx], data).expect("shape"))
}

/// Hermite polynomials H0..H2 (physicists').
fn hermite(n: usize, t: f64) -> f64 {
    match n {
        0 => 1.0,
        1 => 2.0 * t,
        _ => 4.0 * t * t - 2.0,
    }
}

/// `nmode` mutually orthogonal probe modes `[nmode, ny, nx]` (Hermite-Gauss orders (0,0),
/// (1,0), (0,1), (1,1), (2,0), (0,2)) on one Gaussian envelope (σ = n/12) with a defocus
/// curvature. Mode `k` carries power `4^-k` (normalised), total `Σ|P|² = 1`.
pub fn probe(frame: [usize; 2], nmode: usize) -> Result<Probe, CoreError> {
    const ORDERS: [(usize, usize); 6] = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)];
    if nmode == 0 || nmode > ORDERS.len() {
        return Err(CoreError::Invalid(format!(
            "probe modes must be 1..={}",
            ORDERS.len()
        )));
    }
    let [ny, nx] = frame;
    let sigma = ny.min(nx) as f64 / 12.0;
    let weights: Vec<f64> = (0..nmode).map(|k| 0.25f64.powi(k as i32)).collect();
    let wsum: f64 = weights.iter().sum();
    let mut data = Vec::with_capacity(nmode * ny * nx);
    for (k, &(a, b)) in ORDERS[..nmode].iter().enumerate() {
        let start = data.len();
        for iy in 0..ny {
            for ix in 0..nx {
                let (u, v) = (
                    (ix as f64 - nx as f64 / 2.0) / sigma,
                    (iy as f64 - ny as f64 / 2.0) / sigma,
                );
                let r2 = u * u + v * v;
                let amp = hermite(a, u) * hermite(b, v) * (-r2 / 2.0).exp();
                data.push(C64::from_polar(amp, 1.5 * r2 + 0.3 * k as f64));
            }
        }
        let plane = &mut data[start..];
        let norm = plane.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        let scale = (weights[k] / wsum).sqrt() / norm;
        plane.iter_mut().for_each(|c| *c *= scale);
    }
    Ok(Probe::new(Stack::from_vec([nmode, ny, nx], data)?))
}

/// Poisson-samples `intensity` (`[nframe, ny, nx]`, detector order) with a ChaCha8 stream
/// seeded by `seed`. `detector_mask` (`[ny, nx]`, true = masked) stores [`MASKED`] there.
pub fn poisson(
    dims: [usize; 3],
    intensity: &[f64],
    detector_mask: Option<&[bool]>,
    seed: u64,
) -> Result<Counts, CoreError> {
    let plane = dims[1] * dims[2];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(intensity.len());
    for (index, &lambda) in intensity.iter().enumerate() {
        if detector_mask.is_some_and(|m| m[index % plane]) {
            data.push(MASKED);
            continue;
        }
        let value = if lambda > 0.0 {
            Poisson::new(lambda)
                .map_err(|e| CoreError::Invalid(format!("Poisson({lambda}): {e}")))?
                .sample(&mut rng)
        } else {
            0.0
        };
        if value > f64::from(MAX_COUNT) {
            return Err(CoreError::CountOutOfRange {
                index,
                value,
                max: MAX_COUNT,
            });
        }
        data.push(value as u16);
    }
    Counts::from_raw(dims, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_modes_are_normalised_and_orthogonal() {
        let p = probe([32, 32], 3).unwrap();
        let power: Vec<f64> = (0..3)
            .map(|k| p.plane(k).iter().map(|c| c.norm_sqr()).sum())
            .collect();
        assert!((power.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((power[0] / power[1] - 4.0).abs() < 1e-9);
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            let dot: C64 = p
                .plane(a)
                .iter()
                .zip(p.plane(b))
                .map(|(x, y)| x * y.conj())
                .sum();
            assert!(dot.norm() < 1e-6, "modes {a},{b}: {dot}");
        }
    }

    #[test]
    fn fermat_spacing_and_raster_layout() {
        let f = fermat(200, 1.0);
        let mut nn = 0.0;
        for i in 0..f.len() {
            let d = (0..f.len())
                .filter(|&j| j != i)
                .map(|j| (f.x[i] - f.x[j]).hypot(f.y[i] - f.y[j]))
                .fold(f64::INFINITY, f64::min);
            nn += d;
        }
        let mean = nn / f.len() as f64;
        assert!((0.7..1.3).contains(&mean), "mean spacing {mean}");
        let r = raster(2, 3, 2.0);
        assert_eq!(r.y, vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]);
        assert_eq!(r.x, vec![-2.0, 0.0, 2.0, -2.0, 0.0, 2.0]);
    }

    #[test]
    fn poisson_is_seeded_and_masks() {
        let lam = vec![0.0, 5.0, 100.0, 1e3];
        let mask = [false, false, false, true];
        let a = poisson([1, 2, 2], &lam, Some(&mask), 7).unwrap();
        let b = poisson([1, 2, 2], &lam, Some(&mask), 7).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.as_slice()[0], 0);
        assert_eq!(a.as_slice()[3], MASKED);
        assert!(poisson([1, 1, 1], &[1e6], None, 1).is_err());
    }

    #[test]
    fn object_amplitude_and_phase_ranges() {
        let o = object([64, 64]);
        for c in o.as_slice() {
            assert!((0.45..=0.96).contains(&c.norm()), "{c}");
        }
    }
}
