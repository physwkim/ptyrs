//! Seeded varied objects for training and testing the predictor.

use std::f64::consts::PI;

use crate::base::{C64, Object, Stack};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// One shape of an object, evaluated inside its bounding box: soft ones for [`random_object`],
/// hard-edged polygons, gratings and spoke stars for [`structured_object`].
enum Shape {
    Ellipse {
        a: f64,
        b: f64,
        edge: f64,
    },
    Rect {
        a: f64,
        b: f64,
        edge: f64,
    },
    Blob {
        sigma: f64,
    },
    /// The convex polygon inside the half-planes `nu·u + nv·v <= d` given as `(nu, nv, d)`,
    /// within `reach` of the centre.
    Polygon {
        sides: Vec<(f64, f64, f64)>,
        reach: f64,
        edge: f64,
    },
    /// Lines `duty·period` wide every `period` pixels across `u`, cut to the rectangle
    /// `|u| <= a`, `|v| <= b`.
    Grating {
        a: f64,
        b: f64,
        period: f64,
        duty: f64,
        edge: f64,
    },
    /// `spokes` wedges, each `duty` of its angular period wide, between the radii `inner` and
    /// `outer`.
    Star {
        spokes: usize,
        inner: f64,
        outer: f64,
        duty: f64,
        edge: f64,
    },
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
            Shape::Polygon {
                ref sides, edge, ..
            } => soft(
                sides
                    .iter()
                    .map(|&(nu, nv, d)| d - nu * u - nv * v)
                    .fold(f64::INFINITY, f64::min),
                edge,
            ),
            Shape::Grating {
                a,
                b,
                period,
                duty,
                edge,
            } => soft(
                stripe(u, period, duty).min(a - u.abs()).min(b - v.abs()),
                edge,
            ),
            Shape::Star {
                spokes,
                inner,
                outer,
                duty,
                edge,
            } => {
                let r = u.hypot(v);
                let wedge = r * stripe(v.atan2(u), 2.0 * PI / spokes as f64, duty);
                soft(wedge.min(r - inner).min(outer - r), edge)
            }
        }
    }

    fn reach(&self) -> f64 {
        match *self {
            Shape::Ellipse { a, b, edge }
            | Shape::Rect { a, b, edge }
            | Shape::Grating { a, b, edge, .. } => a.hypot(b) + 2.0 * edge,
            Shape::Blob { sigma } => 4.0 * sigma,
            Shape::Polygon { reach, edge, .. } => reach + 2.0 * edge,
            Shape::Star { outer, edge, .. } => outer + 2.0 * edge,
        }
    }

    /// The polygon with vertices at `angles` (ascending, radians) on a circle of radius `r`.
    fn polygon(r: f64, angles: &[f64], edge: f64) -> Shape {
        let vertex = |k: usize| {
            let (sin, cos) = angles[k % angles.len()].sin_cos();
            (r * cos, r * sin)
        };
        // Counter-clockwise vertices put the inside left of each side: its outward normal is
        // the side turned clockwise.
        let sides = (0..angles.len())
            .filter_map(|k| {
                let ((u0, v0), (u1, v1)) = (vertex(k), vertex(k + 1));
                let len = (u1 - u0).hypot(v1 - v0);
                (len > 1e-9).then(|| {
                    let (nu, nv) = ((v1 - v0) / len, (u0 - u1) / len);
                    (nu, nv, nu * u0 + nv * v0)
                })
            })
            .collect();
        Shape::Polygon {
            sides,
            reach: r,
            edge,
        }
    }

    /// Adds `height` times the profile into `t`, the shape centred at `centre` (y, x) and
    /// turned by the angle whose sine and cosine are `turn`.
    fn add_to(
        &self,
        t: &mut [f64],
        shape: [usize; 2],
        height: f64,
        centre: (f64, f64),
        turn: (f64, f64),
    ) {
        let [ny, nx] = shape;
        let ((cy, cx), (sin, cos)) = (centre, turn);
        let r = self.reach();
        let y0 = (cy - r).floor().max(0.0) as usize;
        let y1 = ((cy + r).ceil() as usize).min(ny);
        let x0 = (cx - r).floor().max(0.0) as usize;
        let x1 = ((cx + r).ceil() as usize).min(nx);
        for y in y0..y1 {
            for x in x0..x1 {
                let (dy, dx) = (y as f64 - cy, x as f64 - cx);
                t[y * nx + x] += height * self.at(cos * dx + sin * dy, -sin * dx + cos * dy);
            }
        }
    }
}

/// Edge ramp: 0 outside, 1 inside, linear over `edge` pixels across the boundary `d = 0`.
fn soft(d: f64, edge: f64) -> f64 {
    (d / edge + 0.5).clamp(0.0, 1.0)
}

/// Signed distance from `x` to the nearest edge of stripes `duty·period` wide starting at every
/// multiple of `period`: positive inside a stripe.
fn stripe(x: f64, period: f64, duty: f64) -> f64 {
    let x = x.rem_euclid(period);
    let w = duty * period;
    if x < w {
        x.min(w - x)
    } else {
        -(x - w).min(period - x)
    }
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
        let centre = (rng.gen_range(0.0..ny as f64), rng.gen_range(0.0..nx as f64));
        s.add_to(t, shape, height, centre, rng.gen_range(0.0..PI).sin_cos());
    }
}

/// Adds `count` random hard-edged shapes into the phase and amplitude layouts `tp` and `ta`, with
/// a signed phase height and an amplitude height of 0–1.5 drawn apart for each: convex polygons
/// (radius 3–96 px, 3–8 vertices) and gratings of line pairs (period 4–32 px, cut to rectangles
/// of 20–120 px), each edge rising over 0.3–1 px, and 1–2 spoke stars (4–40 spokes, outer
/// radius 16–200 px, hole up to 0.3 of it, edges over 0.3–6 px) centred in the middle half of
/// the object.
fn scatter_hard(
    tp: &mut [f64],
    ta: &mut [f64],
    shape: [usize; 2],
    count: usize,
    rng: &mut ChaCha8Rng,
) {
    let [ny, nx] = shape;
    let mut add = |s: Shape, centre: (f64, f64), rng: &mut ChaCha8Rng| {
        let hp = rng.gen_range(0.3..1.5) * if rng.gen_bool(0.25) { -1.0 } else { 1.0 };
        let ha = rng.gen_range(0.0..1.5);
        let turn = rng.gen_range(0.0..2.0 * PI).sin_cos();
        s.add_to(tp, shape, hp, centre, turn);
        s.add_to(ta, shape, ha, centre, turn);
    };
    for _ in 0..rng.gen_range(1..=2) {
        let outer = 16.0 * 12.5f64.powf(rng.r#gen::<f64>()); // log-uniform
        let s = Shape::Star {
            spokes: rng.gen_range(4..=40),
            inner: outer * rng.gen_range(0.0..0.3),
            outer,
            duty: rng.gen_range(0.3..0.7),
            edge: 0.3 * 20f64.powf(rng.r#gen::<f64>()),
        };
        let centre = (
            rng.gen_range(0.25..0.75) * ny as f64,
            rng.gen_range(0.25..0.75) * nx as f64,
        );
        add(s, centre, rng);
    }
    for _ in 0..count {
        let edge = rng.gen_range(0.3..1.0);
        let s = if rng.gen_bool(0.5) {
            let r = 3.0 * 32f64.powf(rng.r#gen::<f64>()); // log-uniform
            let mut angles: Vec<f64> = (0..rng.gen_range(3..=8))
                .map(|_| rng.gen_range(0.0..2.0 * PI))
                .collect();
            angles.sort_by(f64::total_cmp);
            Shape::polygon(r, &angles, edge)
        } else {
            let a = rng.gen_range(10.0..60.0);
            Shape::Grating {
                a,
                b: a * rng.gen_range(0.3..1.0),
                period: 4.0 * 8f64.powf(rng.r#gen::<f64>()), // log-uniform
                duty: rng.gen_range(0.3..0.7),
                edge,
            }
        };
        let centre = (rng.gen_range(0.0..ny as f64), rng.gen_range(0.0..nx as f64));
        add(s, centre, rng);
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

/// A seeded structured single-mode object `[1, ny, nx]`: hard-edged polygons, line-pair
/// gratings and spoke stars on a flat background `1 + 0i`, each with its own phase and amplitude
/// contrast. Phase is `φmax·tanh(t_p)` and amplitude `exp(-μ·|tanh(t_a)|)` of the phase and
/// amplitude layouts, with `φmax` (0.5–2 rad) and `μ` (0–0.7) drawn per object as in
/// [`random_object`].
pub fn structured_object(shape: [usize; 2], seed: u64) -> Object {
    let [ny, nx] = shape;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut tp = vec![0.0; ny * nx];
    let mut ta = vec![0.0; ny * nx];
    scatter_hard(&mut tp, &mut ta, shape, (ny * nx).div_ceil(2500), &mut rng);
    let phi = rng.gen_range(0.5..2.0);
    let mu = rng.gen_range(0.0..0.7);
    let data = tp
        .iter()
        .zip(&ta)
        .map(|(&p, &a)| C64::from_polar((-mu * a.tanh().abs()).exp(), phi * p.tanh()))
        .collect();
    Object::new(Stack::from_vec([1, ny, nx], data).expect("shape"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_objects_vary_and_stay_in_range() {
        in_range(random_object, [96, 80]);
    }

    #[test]
    fn structured_objects_vary_and_stay_in_range() {
        in_range(structured_object, [160, 128]);
    }

    #[test]
    fn polygons_gratings_and_stars_rise_over_their_edge() {
        let near = |got: f64, want: f64| assert!((got - want).abs() < 1e-12, "{got} != {want}");
        // The square |u|, |v| <= 10.
        let q = PI / 4.0;
        let square = Shape::polygon(10.0 * 2f64.sqrt(), &[q, 3.0 * q, 5.0 * q, 7.0 * q], 0.5);
        for (u, v, want) in [
            (0.0, 0.0, 1.0),
            (7.0, -7.0, 1.0),
            (9.75, 0.0, 1.0),
            (10.0, 0.0, 0.5),
            (0.0, -10.0, 0.5),
            (-10.25, 0.0, 0.0),
            (0.0, 10.25, 0.0),
        ] {
            near(square.at(u, v), want);
        }
        // Lines on u in [0, 4) + 8k, cut to |u| <= 20, |v| <= 10.
        let grating = Shape::Grating {
            a: 20.0,
            b: 10.0,
            period: 8.0,
            duty: 0.5,
            edge: 0.5,
        };
        for (u, v, want) in [
            (2.0, 0.0, 1.0),
            (-6.0, 9.0, 1.0),
            (6.0, 0.0, 0.0),
            (4.0, 0.0, 0.5),
            (8.0, 0.0, 0.5),
            (18.0, 0.0, 1.0),
            (26.0, 0.0, 0.0),
            (2.0, 10.5, 0.0),
        ] {
            near(grating.at(u, v), want);
        }
        // Wedges on angles [0, π/4) + kπ/2 between radii 5 and 20.
        let star = Shape::Star {
            spokes: 4,
            inner: 5.0,
            outer: 20.0,
            duty: 0.5,
            edge: 0.5,
        };
        let at = |r: f64, t: f64| star.at(r * t.cos(), r * t.sin());
        let p = PI / 8.0;
        for (r, t, want) in [
            (10.0, p, 1.0),
            (10.0, 5.0 * p, 1.0),
            (10.0, 3.0 * p, 0.0),
            (10.0, -p, 0.0),
            (10.0, 2.0 * p, 0.5),
            (5.0, p, 0.5),
            (20.0, p, 0.5),
            (21.0, p, 0.0),
            (0.0, 0.0, 0.0),
        ] {
            near(at(r, t), want);
        }
    }

    /// `object` gives one object per seed, others for other seeds, with a feature somewhere and
    /// phase within ±2 rad and amplitude within 0.45..1.
    fn in_range(object: fn([usize; 2], u64) -> Object, shape: [usize; 2]) {
        let a = object(shape, 1);
        let b = object(shape, 2);
        assert_eq!(a, object(shape, 1));
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
