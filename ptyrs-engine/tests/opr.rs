//! OPR against central differences and the float64 references `refs/f64/opr_step.py` (one
//! eigenmode) and `refs/f64/lsqml_opr_step.py` (three, with the whole LSQML step), run on the
//! state they read: `cargo run --release -p ptyrs-engine --example llk_trace -- <dir> dm 1 2 32
//! 40 4`, i.e. `common::dm_scenario(32, 40, 2, 2e5, 4.0)` after `scale_obj_probe`, with the
//! references' eigenmodes and weights. They write `<dir>/opr_{eigen,weights,values}_f64.txt` and
//! `<dir>/lsqml_opr_{eigen,weights,obj,probe,steps,values}_f64.txt`, saved to `tests/data`.

use std::ops::Range;
use std::slice;

use ptyrs_core::{C64, Stack, View};
use ptyrs_engine::{
    Cost, Fft2, Llk, Lsqml, LsqmlParams, LsqmlStep, Optics, ProbeVariation, Ptycho, illumination,
    llk, obj_probe_to_psi, opr_weight_grad,
};

mod common;

fn state() -> (Ptycho, Optics) {
    let (scan, object, probe) = common::dm_scenario(32, 40, 2, 2e5, 4.0);
    let mut p = Ptycho::new(&scan, object, probe).unwrap();
    p.scale_obj_probe();
    (p, Optics::far_field(&scan.geometry))
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

/// `re_im(a − b)`.
fn change(a: &Stack<C64>, b: &Stack<C64>) -> Vec<f64> {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .flat_map(|(a, b)| [a.re - b.re, a.im - b.im])
        .collect()
}

/// The views of `frames`, in order.
fn views(p: &Ptycho, frames: &[usize]) -> Vec<View> {
    let nmode = p.addr().nmode();
    frames
        .iter()
        .flat_map(|&f| &p.addr().views()[f * nmode..(f + 1) * nmode])
        .copied()
        .collect()
}

/// A reference's eigenmode and weight tables, one row per eigenmode pixel and per frame.
struct Ref {
    k: usize,
    eigen: &'static str,
    weights: &'static str,
    values: &'static str,
    /// The `re` column of `U`, `U''` Gaussian and `U''` Poisson.
    u: [usize; 3],
    /// The first of the `k` columns of `w`, `dL/dw`, `dF/dw`, `w'` Gaussian and `w'` Poisson.
    w: [usize; 5],
}

const OPR: Ref = Ref {
    k: 1,
    eigen: include_str!("data/opr_eigen_f64.txt"),
    weights: include_str!("data/opr_weights_f64.txt"),
    values: include_str!("data/opr_values_f64.txt"),
    u: [2, 4, 6],
    w: [1, 2, 3, 4, 5],
};

const LSQML_OPR: Ref = Ref {
    k: 3,
    eigen: include_str!("data/lsqml_opr_eigen_f64.txt"),
    weights: include_str!("data/lsqml_opr_weights_f64.txt"),
    values: include_str!("data/lsqml_opr_values_f64.txt"),
    u: [3, 5, 7],
    w: [1, 4, 7, 10, 13],
};

impl Ref {
    /// Columns `re, im` of the eigenmode table as `k` planes.
    fn eigenmodes(&self, re: usize) -> Stack<C64> {
        let data = table(self.eigen)
            .iter()
            .map(|r| C64::new(r[re], r[re + 1]))
            .collect();
        Stack::from_vec([self.k, 32, 32], data).unwrap()
    }

    /// Columns `c..c + k` of the weight table, frame-major.
    fn weights(&self, c: usize) -> Vec<f64> {
        table(self.weights)
            .iter()
            .flat_map(|r| r[c..c + self.k].to_vec())
            .collect()
    }

    /// The reference's eigenmodes `U` and weights `w`.
    fn variation(&self) -> ProbeVariation {
        ProbeVariation::new(self.eigenmodes(self.u[0]), self.weights(self.w[0]))
    }
}

/// [`llk`] of each frame of `frames` with the per-frame probes of `var`.
fn frame_llk(p: &Ptycho, frames: Range<usize>, var: &ProbeVariation) -> Vec<Llk> {
    let frames: Vec<usize> = frames.collect();
    let (views, probe) = var.unique(&views(p, &frames), p.probe());
    let (n, nmode) = (probe.plane_len(), p.addr().nmode());
    let mut psi = vec![C64::default(); views.len() * n];
    obj_probe_to_psi(&views, p.object(), &probe, p.quad(), &mut psi);
    Fft2::new(probe.plane_shape()).forward(&mut psi);
    frames
        .iter()
        .zip(psi.chunks_exact(nmode * n))
        .map(|(&f, block)| llk(p.observed(), f..f + 1, nmode, block))
        .collect()
}

/// The Poisson and amplitude costs over every frame against each reference's `L` and `F`.
#[test]
fn costs_match_opr_refs() {
    let (p, _) = state();
    for r in [&OPR, &LSQML_OPR] {
        let ref_ = table(r.values)[0].clone();
        let nb_obs = p.observed().frames(0..40).iter().flatten().count();
        assert_eq!(nb_obs as f64, ref_[0]);
        let all = frame_llk(&p, 0..40, &r.variation());
        let l: f64 = all.iter().map(|l| l.poisson).sum();
        let f: f64 = all.iter().map(|l| l.euclidian).sum::<f64>() / 4.0;
        let (el, ef) = (rel(l, ref_[1]), rel(f, ref_[2]));
        assert!(el < 1e-13 && ef < 1e-13, "k {}: L {el:.3e} F {ef:.3e}", r.k);
    }
}

/// [`opr_weight_grad`] against each reference's `dL/dw` and `dF/dw` and against central
/// differences of each frame's cost in its own weights, one eigenmode at a time.
#[test]
fn weight_grad_matches_refs_and_central_differences() {
    let (p, _) = state();
    let all: Vec<usize> = (0..40).collect();
    let views = views(&p, &all);
    for r in [&OPR, &LSQML_OPR] {
        let var = r.variation();
        for (cost, col) in [(Cost::Poisson, r.w[1]), (Cost::Amplitude, r.w[2])] {
            let g = opr_weight_grad(
                cost,
                &views,
                p.observed(),
                p.quad(),
                p.object(),
                p.probe(),
                &var,
            );
            let err = max_rel(&g, &r.weights(col));
            assert!(
                err < 1e-13,
                "k {} {cost:?} against the reference: max rel {err:.3e}",
                r.k
            );
            let h = 1e-6;
            let value = |l: &Llk| match cost {
                Cost::Poisson => l.poisson,
                Cost::Amplitude => l.euclidian / 4.0,
            };
            // each frame's cost moves with its own weights only
            let shifted = |i: usize, t: f64| {
                let w = var
                    .weights()
                    .iter()
                    .enumerate()
                    .map(|(j, w)| if j % r.k == i { w + t } else { *w })
                    .collect();
                let var = ProbeVariation::new(var.eigenmodes().clone(), w);
                frame_llk(&p, 0..40, &var)
            };
            let mut fd = vec![0.0; g.len()];
            for i in 0..r.k {
                for (f, (a, b)) in shifted(i, h).iter().zip(shifted(i, -h)).enumerate() {
                    fd[f * r.k + i] = (value(a) - value(&b)) / (2.0 * h);
                }
            }
            let err = max_rel(&g, &fd);
            eprintln!("k {} {cost:?} against differences: max rel {err:.3e}", r.k);
            assert!(
                err < 1e-8,
                "k {} {cost:?} against differences: max rel {err:.3e}",
                r.k
            );
        }
    }
}

/// One [`Lsqml::step`] on `frames` from `var`, with the illumination of every view at the shared
/// probe as preconditioner: the step and the object and probe changes.
fn step(
    p: &Ptycho,
    optics: &Optics,
    cost: Cost,
    frames: &[usize],
    var: &mut ProbeVariation,
) -> (LsqmlStep, Vec<f64>, Vec<f64>) {
    let (mut object, mut probe) = (p.object().clone(), p.probe().clone());
    let pre = illumination(p.addr().views(), object.dims(), &probe);
    let views = views(p, frames);
    let params = LsqmlParams {
        batch_size: frames.len(),
        cost,
        seed: 0,
    };
    let s = Lsqml::new(params).with_preconditioner(pre).step(
        &views,
        p.observed(),
        optics,
        slice::from_mut(&mut object),
        &mut probe,
        var,
    );
    assert_eq!(s.frames, frames);
    let d_obj = change(&object, p.object());
    let d_probe = change(&probe, p.probe());
    (s, d_obj, d_probe)
}

/// `var`'s eigenmodes against `r`'s eigen columns from `u` and its weight changes on `frames`
/// against `r`'s weight columns from `w`; weights outside `frames` stay as they are.
fn assert_variation(r: &Ref, var: &ProbeVariation, frames: &[usize], u: usize, w: usize) {
    let err = max_rel(&re_im(var.eigenmodes()), &re_im(&r.eigenmodes(u)));
    eprintln!("k {} eigenmodes: max rel {err:.3e}", r.k);
    assert!(err < 1e-13, "k {} eigenmodes: max rel {err:.3e}", r.k);
    let (w0, want, k) = (r.weights(r.w[0]), r.weights(w), r.k);
    let (mut moved, mut expected) = (Vec::new(), Vec::new());
    for f in 0..w0.len() / k {
        let ws = f * k..(f + 1) * k;
        if frames.contains(&f) {
            for j in ws {
                moved.push(var.weights()[j] - w0[j]);
                expected.push(want[j] - w0[j]);
            }
        } else {
            assert_eq!(var.weights()[ws.clone()], w0[ws], "frame {f}");
        }
    }
    let err = max_rel(&moved, &expected);
    eprintln!("k {k} weight change: max rel {err:.3e}");
    assert!(err < 1e-13, "k {k} weight change: max rel {err:.3e}");
}

/// The one-eigenmode reference's step on frames 0–19 (`a_far` in weight column 6).
fn assert_opr_ref(cost: Cost, noise: usize) {
    let (p, optics) = state();
    let mut var = OPR.variation();
    let frames: Vec<usize> = (0..20).collect();
    let (s, _, _) = step(&p, &optics, cost, &frames, &mut var);
    if cost == Cost::Poisson {
        let err = max_rel(&s.a_far, &OPR.weights(6)[..20]);
        assert!(err < 1e-13, "a_far max rel {err:.3e}");
    }
    assert_variation(&OPR, &var, &frames, OPR.u[1 + noise], OPR.w[3 + noise]);
}

#[test]
fn gaussian_step_matches_opr_ref() {
    assert_opr_ref(Cost::Amplitude, 0);
}

#[test]
fn poisson_step_matches_opr_ref() {
    assert_opr_ref(Cost::Poisson, 1);
}

/// The three-eigenmode reference's LSQML step on its batch: per-frame `a_far a_o a_p`, the
/// object and probe changes and the eigenmodes and weights after it.
fn assert_lsqml_opr_ref(cost: Cost, noise: usize) {
    let (p, optics) = state();
    let mut var = LSQML_OPR.variation();
    let rows = table(include_str!("data/lsqml_opr_steps_f64.txt"));
    let frames: Vec<usize> = rows.iter().map(|r| r[0] as usize).collect();
    let (s, d_obj, d_probe) = step(&p, &optics, cost, &frames, &mut var);
    for (name, got, col, tol) in [
        ("a_far", &s.a_far, 1, 1e-13),
        ("a_o", &s.a_obj[0], 2, 1e-12),
        ("a_p", &s.a_probe, 3, 1e-12),
    ] {
        let want: Vec<f64> = rows.iter().map(|r| r[col + 3 * noise]).collect();
        let err = max_rel(got, &want);
        eprintln!("{cost:?} {name}: max rel {err:.3e}");
        assert!(err < tol, "{cost:?} {name}: max rel {err:.3e}");
    }
    for (name, text, lead, got) in [
        (
            "object",
            include_str!("data/lsqml_opr_obj_f64.txt"),
            2,
            d_obj,
        ),
        (
            "probe",
            include_str!("data/lsqml_opr_probe_f64.txt"),
            3,
            d_probe,
        ),
    ] {
        let c = lead + 2 * noise;
        let want: Vec<f64> = table(text).iter().flat_map(|r| [r[c], r[c + 1]]).collect();
        let err = max_rel(&got, &want);
        eprintln!("{cost:?} {name}: max rel {err:.3e}");
        assert!(err < 1e-12, "{cost:?} {name}: max rel {err:.3e}");
    }
    let r = &LSQML_OPR;
    assert_variation(r, &var, &frames, r.u[1 + noise], r.w[3 + noise]);
}

#[test]
fn gaussian_step_matches_lsqml_opr_ref() {
    assert_lsqml_opr_ref(Cost::Amplitude, 0);
}

#[test]
fn poisson_step_matches_lsqml_opr_ref() {
    assert_lsqml_opr_ref(Cost::Poisson, 1);
}

/// A one-frame batch leaves every eigenmode and weight unchanged.
#[test]
fn single_frame_batch_moves_nothing() {
    let (p, optics) = state();
    for r in [&OPR, &LSQML_OPR] {
        let mut var = r.variation();
        let (s, _, _) = step(&p, &optics, Cost::Poisson, &[3], &mut var);
        assert_eq!(s.a_far.len(), 1);
        assert_eq!(var, r.variation(), "k {}", r.k);
    }
}
