//! Seeded varied objects for training and testing the predictor.

use std::f64::consts::PI;

use ptyrs_core::{C64, Object, Stack};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// One soft-edged shape of a random object, evaluated inside its bounding box.
enum Shape {
    Ellipse { a: f64, b: f64, edge: f64 },
    Rect { a: f64, b: f64, edge: f64 },
    Blob { sigma: f64 },
}

impl Shape {
    /// Profile in `0..=1` at `(u, v)`, the offset from the centre in the shape's rotated frame.
    fn at(&self, u: f64, v: f64) -> f64 {
        match *self {
            Shape::Ellipse { a, b, edge } => {
                let r = ((u / a).powi(2) + (v / b).powi(2)).sqrt();
                soft(a.min(b) * (1.0 - r), edge)
            }
            Shape::Rect { a, b, edge } => soft((a - u.abs()).min(b - v.abs()), edge),
            Shape::Blob { sigma } => (-(u * u + v * v) / (2.0 * sigma * sigma)).exp(),
        }
    }

    fn reach(&self) -> f64 {
        match *self {
            Shape::Ellipse { a, b, edge } | Shape::Rect { a, b, edge } => a.hypot(b) + 2.0 * edge,
            Shape::Blob { sigma } => 4.0 * sigma,
        }
    }
}

/// Edge ramp: 0 outside, 1 inside, linear over `edge` pixels across the boundary `d = 0`.
fn soft(d: f64, edge: f64) -> f64 {
    (d / edge + 0.5).clamp(0.0, 1.0)
}

/// Adds `count` random shapes with signed heights into `t`.
fn scatter(t: &mut [f64], shape: [usize; 2], count: usize, rng: &mut ChaCha8Rng) {
    let [ny, nx] = shape;
    for _ in 0..count {
        let size = 2.0 * (12.0f64).powf(rng.r#gen::<f64>()); // 2..24 px, log-uniform
        let aspect = rng.gen_range(0.3..1.0);
        let edge = rng.gen_range(0.5..2.0);
        let s = match rng.gen_range(0..3) {
            0 => Shape::Ellipse {
                a: size,
                b: size * aspect,
                edge,
            },
            1 => Shape::Rect {
                a: size,
                b: size * aspect,
                edge,
            },
            _ => Shape::Blob { sigma: size / 2.0 },
        };
        let height = rng.gen_range(0.2..1.0) * if rng.gen_bool(0.25) { -1.0 } else { 1.0 };
        let (cy, cx) = (rng.gen_range(0.0..ny as f64), rng.gen_range(0.0..nx as f64));
        let (sin, cos) = rng.gen_range(0.0..PI).sin_cos();
        let r = s.reach();
        let y0 = (cy - r).floor().max(0.0) as usize;
        let y1 = ((cy + r).ceil() as usize).min(ny);
        let x0 = (cx - r).floor().max(0.0) as usize;
        let x1 = ((cx + r).ceil() as usize).min(nx);
        for y in y0..y1 {
            for x in x0..x1 {
                let (dy, dx) = (y as f64 - cy, x as f64 - cx);
                t[y * nx + x] += height * s.at(cos * dx + sin * dy, -sin * dx + cos * dy);
            }
        }
    }
}

/// A seeded random single-mode object `[1, ny, nx]`: soft ellipses, rectangles and Gaussian
/// blobs (2–24 px) on a flat background `1 + 0i`. Phase is `φmax·tanh(t_p)` and amplitude
/// `exp(-μ·|tanh(t_a)|)`, where `t_a` blends `t_p` with an independent layout, and `φmax`
/// (0.5–2 rad) and `μ` (0–0.7) are drawn per object, so phase lies in about ±2 rad and
/// amplitude in about 0.5..1.
pub fn random_object(shape: [usize; 2], seed: u64) -> Object {
    let [ny, nx] = shape;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let count = (ny * nx).div_ceil(300);
    let mut tp = vec![0.0; ny * nx];
    let mut ti = vec![0.0; ny * nx];
    scatter(&mut tp, shape, count, &mut rng);
    scatter(&mut ti, shape, count, &mut rng);
    let phi = rng.gen_range(0.5..2.0);
    let mu = rng.gen_range(0.0..0.7);
    let same = rng.gen_range(0.0..1.0);
    let data = tp
        .iter()
        .zip(&ti)
        .map(|(&p, &i)| {
            let ta = same * p + (1.0 - same) * i;
            C64::from_polar((-mu * ta.tanh().abs()).exp(), phi * p.tanh())
        })
        .collect();
    Object::new(Stack::from_vec([1, ny, nx], data).expect("shape"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_objects_vary_and_stay_in_range() {
        let a = random_object([96, 80], 1);
        let b = random_object([96, 80], 2);
        assert_eq!(a, random_object([96, 80], 1));
        assert_ne!(a, b);
        for o in [&a, &b] {
            let (mut ph_min, mut ph_max) = (0.0f64, 0.0f64);
            for c in o.as_slice() {
                ph_min = ph_min.min(c.arg());
                ph_max = ph_max.max(c.arg());
                assert!(c.norm() <= 1.0 + 1e-12 && c.norm() > 0.45, "{c}");
                assert!(c.arg().abs() <= 2.0 + 1e-12, "{c}");
            }
            // Features exist: the object is not flat.
            assert!(ph_max - ph_min > 0.3, "phase span {}", ph_max - ph_min);
        }
    }
}
