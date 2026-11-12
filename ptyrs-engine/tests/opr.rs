//! OPR against central differences and the float64 reference `refs/f64/opr_step.py`, run on the
//! state it reads: `cargo run --release -p ptyrs-engine --example llk_trace -- <dir> dm 1 2 32
//! 40 4`, i.e. `common::dm_scenario(32, 40, 2, 2e5, 4.0)` after `scale_obj_probe`, with the
//! reference's eigenmode and weights. The reference writes
//! `<dir>/opr_{eigen,weights,values}_f64.txt`, saved to `tests/data`.

use std::ops::Range;

use ptyrs_core::{C64, Stack, View};
use ptyrs_engine::{
    Cost, Fft2, Llk, ProbeVariation, Ptycho, llk, obj_probe_to_psi, opr_step, opr_weight_grad,
};

mod common;

fn state() -> Ptycho {
    let (scan, object, probe) = common::dm_scenario(32, 40, 2, 2e5, 4.0);
    let mut p = Ptycho::new(&scan, object, probe).unwrap();
    p.scale_obj_probe();
    p
}

fn table(text: &str) -> Vec<Vec<f64>> {
    text.lines()
        .map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect())
        .collect()
}

/// `max |a − b| / max |b|`.
fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let d = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    d / b.iter().map(|b| b.abs()).fold(0.0, f64::max)
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

fn re_im(s: &Stack<C64>) -> Vec<f64> {
    s.as_slice().iter().flat_map(|c| [c.re, c.im]).collect()
}

/// The views of `frames`.
fn batch(p: &Ptycho, frames: Range<usize>) -> &[View] {
    let nmode = p.addr().nmode();
    &p.addr().views()[frames.start * nmode..frames.end * nmode]
}

const EIGEN: &str = include_str!("data/opr_eigen_f64.txt");
const WEIGHTS: &str = include_str!("data/opr_weights_f64.txt");
const BATCH: Range<usize> = 0..20;

/// Columns `re, im` of `opr_eigen_f64.txt` (one row per probe pixel) as one plane.
fn eigenmode(re: usize) -> Stack<C64> {
    let data = table(EIGEN)
        .iter()
        .map(|r| C64::new(r[re], r[re + 1]))
        .collect();
    Stack::from_vec([1, 32, 32], data).unwrap()
}

/// Column `c` of `opr_weights_f64.txt`, one row per frame.
fn weights(c: usize) -> Vec<f64> {
    table(WEIGHTS).iter().map(|r| r[c]).collect()
}

/// The reference's eigenmode `U` and weights `w`.
fn variation() -> ProbeVariation {
    ProbeVariation::new(eigenmode(2), weights(1))
}

/// [`llk`] of each frame of `frames` with the per-frame probes of `var`.
fn frame_llk(p: &Ptycho, frames: Range<usize>, var: &ProbeVariation) -> Vec<Llk> {
    let (views, probe) = var.unique(batch(p, frames.clone()), p.probe());
    let (n, nmode) = (probe.plane_len(), p.addr().nmode());
    let mut psi = vec![C64::default(); views.len() * n];
    obj_probe_to_psi(&views, p.object(), &probe, p.quad(), &mut psi);
    Fft2::new(probe.plane_shape()).forward(&mut psi);
    frames
        .zip(psi.chunks_exact(nmode * n))
        .map(|(f, block)| llk(p.observed(), f..f + 1, nmode, block))
        .collect()
}

/// The Poisson and amplitude costs over every frame against the reference's `L` and `F`.
#[test]
fn costs_match_opr_ref() {
    let p = state();
    let ref_ = table(include_str!("data/opr_values_f64.txt"))[0].clone();
    let nb_obs = p.observed().frames(0..40).iter().flatten().count();
    assert_eq!(nb_obs as f64, ref_[0]);
    let all = frame_llk(&p, 0..40, &variation());
    let l: f64 = all.iter().map(|l| l.poisson).sum();
    let f: f64 = all.iter().map(|l| l.euclidian).sum::<f64>() / 4.0;
    let (el, ef) = (rel(l, ref_[1]), rel(f, ref_[2]));
    assert!(el < 1e-13 && ef < 1e-13, "L {el:.3e} F {ef:.3e}");
}

/// [`opr_weight_grad`] against the reference's `dL/dw` and `dF/dw` and against central
/// differences of each frame's cost in its own weight.
#[test]
fn weight_grad_matches_ref_and_central_differences() {
    let p = state();
    let var = variation();
    let views = batch(&p, 0..40);
    for (cost, col) in [(Cost::Poisson, 2), (Cost::Amplitude, 3)] {
        let g = opr_weight_grad(
            cost,
            views,
            p.observed(),
            p.quad(),
            p.object(),
            p.probe(),
            &var,
        );
        let err = max_rel(&g, &weights(col));
        assert!(
            err < 1e-13,
            "{cost:?} against the reference: max rel {err:.3e}"
        );
        let h = 1e-6;
        let shifted = |t: f64| {
            let w = var.weights().iter().map(|w| w + t).collect();
            frame_llk(&p, 0..40, &ProbeVariation::new(var.eigenmode().clone(), w))
        };
        let value = |l: &Llk| match cost {
            Cost::Poisson => l.poisson,
            Cost::Amplitude => l.euclidian / 4.0,
        };
        let fd: Vec<f64> = shifted(h)
            .iter()
            .zip(shifted(-h))
            .map(|(a, b)| (value(a) - value(&b)) / (2.0 * h))
            .collect();
        let err = max_rel(&g, &fd);
        assert!(
            err < 1e-8,
            "{cost:?} against differences: max rel {err:.3e}"
        );
    }
}

/// One [`opr_step`] on frames 0–19 against the reference's `U''` (eigen columns `eig`) and
/// `w'` (weight column `col`); weights outside the batch stay as they are.
fn assert_ref(cost: Cost, eig: usize, col: usize) {
    let p = state();
    let mut var = variation();
    let step = opr_step(
        cost,
        batch(&p, BATCH),
        p.observed(),
        p.quad(),
        p.object(),
        p.probe(),
        &mut var,
    );
    assert_eq!((step.first, step.count), (0, 20));
    if cost == Cost::Poisson {
        let err = max_rel(&step.a_far, &weights(6)[BATCH]);
        assert!(err < 1e-13, "a_far max rel {err:.3e}");
    }
    let err = max_rel(&re_im(var.eigenmode()), &re_im(&eigenmode(eig)));
    assert!(err < 1e-13, "eigenmode max rel {err:.3e}");
    let (want, w0) = (weights(col), weights(1));
    let moved: Vec<f64> = var.weights()[BATCH]
        .iter()
        .zip(&w0[BATCH])
        .map(|(a, b)| a - b)
        .collect();
    let want: Vec<f64> = want[BATCH]
        .iter()
        .zip(&w0[BATCH])
        .map(|(a, b)| a - b)
        .collect();
    let err = max_rel(&moved, &want);
    assert!(err < 1e-13, "weight change max rel {err:.3e}");
    assert_eq!(var.weights()[BATCH.end..], w0[BATCH.end..]);
}

#[test]
fn gaussian_step_matches_opr_ref() {
    assert_ref(Cost::Amplitude, 4, 4);
}

#[test]
fn poisson_step_matches_opr_ref() {
    assert_ref(Cost::Poisson, 6, 5);
}

/// A one-frame batch leaves the eigenmode and every weight unchanged.
#[test]
fn single_frame_batch_moves_nothing() {
    let p = state();
    let mut var = variation();
    let step = opr_step(
        Cost::Poisson,
        batch(&p, 3..4),
        p.observed(),
        p.quad(),
        p.object(),
        p.probe(),
        &mut var,
    );
    assert_eq!((step.first, step.count, step.a_far.len()), (3, 1, 1));
    assert_eq!(var, variation());
}
