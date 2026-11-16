//! The probe variations the GPU LSQML tests run OPR from, and their changes against the CPU's.

use ptyrs::base::{C64, Probe, Stack};
use ptyrs::engine::{Feature, ProbeVariation};

/// Which weights a [`variation`] refines; the eigenmodes always are.
#[derive(Clone, Copy, Debug)]
pub enum Refined {
    /// The eigenmodes', without main-mode weights.
    Weights,
    /// The eigenmodes' and the main mode's.
    Both,
    /// The main mode's, the eigenmodes' held.
    Intensity,
}

/// `k` eigenmodes of probe mode 0, it times `x`, `i y` and `x² − y²` in turn (the plane's
/// centre at 0, its edges at ±2), weights near ±0.05 at each of `nframe` frames and, unless
/// `refined` is [`Refined::Weights`], main-mode weights near 1.
pub fn variation(probe: &Probe, nframe: usize, k: usize, refined: Refined) -> ProbeVariation {
    let [ny, nx] = probe.plane_shape();
    let at = |j: usize| {
        let y = 4.0 * (j / nx) as f64 / ny as f64 - 2.0;
        let x = 4.0 * (j % nx) as f64 / nx as f64 - 2.0;
        [y, x]
    };
    let modes = (0..k)
        .flat_map(|i| {
            probe.plane(0).iter().enumerate().map(move |(j, &p)| {
                let [y, x] = at(j);
                p * match i % 3 {
                    0 => C64::new(x, 0.0),
                    1 => C64::new(0.0, y),
                    _ => C64::new(x * x - y * y, 0.0),
                }
            })
        })
        .collect();
    let modes = Stack::from_vec([k, ny, nx], modes).unwrap();
    // At ±0.2 the CPU alone, from the start rounded to f32, moves its steps 3e-5 with Poisson
    // in batches of 12 that sample the object, where here it moves them 1e-7.
    let w = (0..nframe * k)
        .map(|j| 0.05 * (1.3 * (j / k) as f64 + 0.7 * (j % k) as f64 + 0.4).sin())
        .collect();
    let c = (0..nframe)
        .map(|f| 1.0 + 0.05 * (0.9 * f as f64).cos())
        .collect();
    let refine_w = !matches!(refined, Refined::Intensity);
    let var = ProbeVariation::new(
        modes,
        Feature {
            values: w,
            refine: refine_w,
        },
    );
    match refined {
        Refined::Weights => var,
        Refined::Both | Refined::Intensity => var.with_intensity(Feature::refined(c)),
    }
}

/// `max |Δa − Δb| / max |Δb|` of the changes from `start` of refined values; for held ones,
/// 0 when `a` is `start` in `f32`, as the device holds them, and 1 otherwise.
fn changes_rel(a: &Feature<Vec<f64>>, b: &Feature<Vec<f64>>, start: &Feature<Vec<f64>>) -> f64 {
    assert_eq!(a.refine, start.refine);
    if !start.refine {
        let f32s = |v: &[f64]| v.iter().map(|&v| v as f32 as f64).collect::<Vec<_>>();
        return if a.values == f32s(&start.values) {
            0.0
        } else {
            1.0
        };
    }
    let delta =
        |v: &[f64]| -> Vec<f64> { v.iter().zip(&start.values).map(|(v, s)| v - s).collect() };
    let (a, b) = (delta(&a.values), delta(&b.values));
    let d = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    d / b.iter().map(|b| b.abs()).fold(0.0, f64::max)
}

/// The changes of `a`'s eigenmodes, weights and main-mode weights from `start`'s against `b`'s,
/// by [`changes_rel`]; all 0 without eigenmodes.
pub fn variation_rel(a: &ProbeVariation, b: &ProbeVariation, start: &ProbeVariation) -> [f64; 3] {
    if start.eigenmodes().is_empty() {
        return [0.0; 3];
    }
    let re_im = |v: &ProbeVariation| {
        Feature::refined(
            v.eigenmodes()
                .as_slice()
                .iter()
                .flat_map(|c| [c.re, c.im])
                .collect(),
        )
    };
    let intensity = match (a.intensity(), b.intensity(), start.intensity()) {
        (Some(a), Some(b), Some(s)) => changes_rel(a, b, s),
        (None, None, None) => 0.0,
        _ => 1.0,
    };
    [
        changes_rel(&re_im(a), &re_im(b), &re_im(start)),
        changes_rel(a.weights(), b.weights(), start.weights()),
        intensity,
    ]
}
