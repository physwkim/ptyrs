//! OPR against central differences and the float64 references `refs/f64/opr_step.py` (one
//! eigenmode), `refs/f64/lsqml_opr_step.py` (three, with the whole LSQML step) and
//! `refs/f64/lsqml_opr_int_step.py` (that, with the main mode's weights refined), run on the
//! state they read: `cargo run --release -p ptyrs --example engine_llk_trace -- <dir> dm 1 2 32
//! 40 4`, i.e. `common::dm_scenario(32, 40, 2, 2e5, 4.0)` after `scale_obj_probe`, with the
//! references' eigenmodes and weights. They write `<dir>/opr_{eigen,weights,values}_f64.txt`,
//! `<dir>/lsqml_opr_{eigen,weights,obj,probe,steps,values}_f64.txt` and the same
//! `lsqml_opr_int_` files, saved to `tests/engine/data`. [`Lsqml::llk`] is also held against
//! [`Ptycho::llk`] at a per-frame intensity scale.

use std::ops::Range;
use std::slice;

use ptyrs::base::{C64, Stack, View};
use ptyrs::engine::{
    Cost, Feature, Features, Fft2, Llk, Lsqml, LsqmlParams, LsqmlStep, Optics, ProbeVariation,
    Ptycho, Sampling, illumination, llk, obj_probe_to_psi, opr_weight_grad,
};

#[path = "common/engine/mod.rs"]
mod common;
#[path = "common/worst.rs"]
mod worst;
use worst::worst_of;

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
    let d = worst_of(a.iter().zip(b).map(|(a, b)| (a - b).abs()));
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
    name: &'static str,
    k: usize,
    eigen: &'static str,
    weights: &'static str,
    values: &'static str,
    /// The `re` column of `U`, `U''` Gaussian and `U''` Poisson.
    u: [usize; 3],
    /// The first of the `k` columns of `w`, `dL/dw`, `dF/dw`, `w'` Gaussian and `w'` Poisson.
    w: [usize; 5],
    /// The main mode's weight columns `c`, `c'` Gaussian and `c'` Poisson, when refined.
    c: Option<[usize; 3]>,
}

const OPR: Ref = Ref {
    name: "opr",
    k: 1,
    eigen: include_str!("engine/data/opr_eigen_f64.txt"),
    weights: include_str!("engine/data/opr_weights_f64.txt"),
    values: include_str!("engine/data/opr_values_f64.txt"),
    u: [2, 4, 6],
    w: [1, 2, 3, 4, 5],
    c: None,
};

const LSQML_OPR: Ref = Ref {
    name: "lsqml_opr",
    k: 3,
    eigen: include_str!("engine/data/lsqml_opr_eigen_f64.txt"),
    weights: include_str!("engine/data/lsqml_opr_weights_f64.txt"),
    values: include_str!("engine/data/lsqml_opr_values_f64.txt"),
    u: [3, 5, 7],
    w: [1, 4, 7, 10, 13],
    c: None,
};

const LSQML_OPR_INT: Ref = Ref {
    name: "lsqml_opr_int",
    k: 3,
    eigen: include_str!("engine/data/lsqml_opr_int_eigen_f64.txt"),
    weights: include_str!("engine/data/lsqml_opr_int_weights_f64.txt"),
    values: include_str!("engine/data/lsqml_opr_int_values_f64.txt"),
    u: [3, 5, 7],
    w: [2, 5, 8, 12, 16],
    c: Some([1, 11, 15]),
};

/// The LSQML references' `steps`, `obj` and `probe` tables.
const LSQML_OPR_STEP: [&str; 3] = [
    include_str!("engine/data/lsqml_opr_steps_f64.txt"),
    include_str!("engine/data/lsqml_opr_obj_f64.txt"),
    include_str!("engine/data/lsqml_opr_probe_f64.txt"),
];
const LSQML_OPR_INT_STEP: [&str; 3] = [
    include_str!("engine/data/lsqml_opr_int_steps_f64.txt"),
    include_str!("engine/data/lsqml_opr_int_obj_f64.txt"),
    include_str!("engine/data/lsqml_opr_int_probe_f64.txt"),
];

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

    /// Column `c` of the weight table.
    fn column(&self, c: usize) -> Vec<f64> {
        table(self.weights).iter().map(|r| r[c]).collect()
    }

    /// The reference's eigenmodes `U` and main-mode weights `c` with eigenmode weights `w`.
    fn variation_with(&self, w: Vec<f64>) -> ProbeVariation {
        let var = ProbeVariation::new(self.eigenmodes(self.u[0]), Feature::refined(w));
        match self.c {
            Some([c, ..]) => var.with_intensity(Feature::refined(self.column(c))),
            None => var,
        }
    }

    /// The reference's eigenmodes `U` and weights `c` and `w`.
    fn variation(&self) -> ProbeVariation {
        self.variation_with(self.weights(self.w[0]))
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
    for r in [&OPR, &LSQML_OPR, &LSQML_OPR_INT] {
        let ref_ = table(r.values)[0].clone();
        let nb_obs = p.observed().frames(0..40).iter().flatten().count();
        assert_eq!(nb_obs as f64, ref_[0]);
        let all = frame_llk(&p, 0..40, &r.variation());
        let l: f64 = all.iter().map(|l| l.poisson).sum();
        let f: f64 = all.iter().map(|l| l.euclidian).sum::<f64>() / 4.0;
        let (el, ef) = (rel(l, ref_[1]), rel(f, ref_[2]));
        assert!(
            el < 1e-13 && ef < 1e-13,
            "{}: L {el:.3e} F {ef:.3e}",
            r.name
        );
    }
}

/// [`Lsqml::llk`] of each reference's variation against its `L` and `F` over every frame: the
/// LLK of LSQML's model reads each frame's own probe.
#[test]
fn lsqml_llk_matches_opr_refs() {
    let (p, optics) = state();
    let run = Lsqml::new(LsqmlParams {
        batch_size: 40,
        cost: Cost::Poisson,
        seed: 0,
    })
    .unwrap();
    let objects = slice::from_ref(p.object());
    for r in [&OPR, &LSQML_OPR, &LSQML_OPR_INT] {
        let ref_ = table(r.values)[0].clone();
        let var = r.variation();
        let got = run.llk(p.addr(), p.observed(), &optics, objects, p.probe(), &var);
        let nb_obs = ref_[0];
        let el = rel(got.poisson * nb_obs, ref_[1]);
        let ef = rel(got.euclidian * nb_obs / 4.0, ref_[2]);
        assert!(
            el < 1e-13 && ef < 1e-13,
            "{}: L {el:.3e} F {ef:.3e}",
            r.name
        );
    }
}

/// [`Lsqml::llk`] of held offsets under each sampling against [`Ptycho::llk`]. With one probe
/// mode and one eigenmode `U = P`, frame `f`'s own probe `c_f·P + w_f·U` is `P` at the
/// intensity scale `(c_f + w_f)²`, which `Ptycho` models; without a variation it is `Ptycho`'s
/// LLK as it is.
#[test]
fn lsqml_llk_matches_ptycho_at_each_frames_scale() {
    let (scan, object, probe) = common::dm_scenario(32, 40, 1, 2e5, 4.0);
    let mut p = Ptycho::new(&scan, object, probe).unwrap();
    p.scale_obj_probe();
    let (object, probe) = (p.object().clone(), p.probe().clone());
    let nframe = p.addr().nframe();
    let d: Vec<[f64; 2]> = (0..nframe)
        .map(|f| [0.4 * (0.7 * f as f64).sin(), 0.4 * (1.3 * f as f64).cos()])
        .collect();
    let [_, ny, nx] = object.dims();
    let inside = p.addr().views().iter().all(|v| {
        let at = d[v.frame as usize];
        let at = [v.oy as f64 + at[0], v.ox as f64 + at[1]];
        (0.0..=(ny - 32) as f64).contains(&at[0]) && (0.0..=(nx - 32) as f64).contains(&at[1])
    });
    assert!(inside, "a view reads past the object's edge");
    let c: Vec<f64> = (0..nframe).map(|f| 1.0 + 0.05 * (f % 3) as f64).collect();
    let w: Vec<f64> = (0..nframe).map(|f| 0.1 * (f % 5) as f64 - 0.2).collect();
    let scale = c.iter().zip(&w).map(|(c, w)| (c + w).powi(2)).collect();
    let u = Stack::from_vec([1, 32, 32], probe.plane(0).to_vec()).unwrap();
    let var = ProbeVariation::new(u, Feature::held(w)).with_intensity(Feature::held(c));
    let none = ProbeVariation::none([32, 32]);
    let runs = [
        ("none", &none, None),
        ("scaled", &var, Some(Feature::held(scale))),
    ];
    let position = Some(Feature::held(d));
    for sampling in [Sampling::Patch, Sampling::FullObject] {
        let run = Lsqml::new(LsqmlParams {
            batch_size: nframe,
            cost: Cost::Poisson,
            seed: 0,
        })
        .unwrap()
        .with_sampling(sampling)
        .with_features(Features {
            position: position.clone(),
            ..Features::default()
        })
        .unwrap();
        p.set_sampling(sampling);
        for (named, var, scale) in runs.clone() {
            let objects = slice::from_ref(&object);
            let got = run.llk(p.addr(), p.observed(), p.optics(), objects, &probe, var);
            p.set_features(Features {
                position: position.clone(),
                scale,
                ..Features::default()
            })
            .unwrap();
            let want = p.llk();
            let err = worst_of(
                [
                    (got.poisson, want.poisson),
                    (got.gaussian, want.gaussian),
                    (got.euclidian, want.euclidian),
                    (got.nb_photons_calc, want.nb_photons_calc),
                ]
                .map(|(g, w)| rel(g, w)),
            );
            eprintln!("{sampling:?} {named}: max rel {err:.3e}");
            assert!(err < 1e-12, "{sampling:?} {named}: max rel {err:.3e}");
        }
    }
}

/// [`opr_weight_grad`] against each reference's `dL/dw` and `dF/dw` and against central
/// differences of each frame's cost in its own weights, one eigenmode at a time.
#[test]
fn weight_grad_matches_refs_and_central_differences() {
    let (p, _) = state();
    let all: Vec<usize> = (0..40).collect();
    let views = views(&p, &all);
    for r in [&OPR, &LSQML_OPR, &LSQML_OPR_INT] {
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
                "{} {cost:?} against the reference: max rel {err:.3e}",
                r.name
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
                    .values
                    .iter()
                    .enumerate()
                    .map(|(j, w)| if j % r.k == i { w + t } else { *w })
                    .collect();
                frame_llk(&p, 0..40, &r.variation_with(w))
            };
            let mut fd = vec![0.0; g.len()];
            for i in 0..r.k {
                for (f, (a, b)) in shifted(i, h).iter().zip(shifted(i, -h)).enumerate() {
                    fd[f * r.k + i] = (value(a) - value(&b)) / (2.0 * h);
                }
            }
            let err = max_rel(&g, &fd);
            eprintln!("{} {cost:?} against differences: max rel {err:.3e}", r.name);
            assert!(
                err < 1e-8,
                "{} {cost:?} against differences: max rel {err:.3e}",
                r.name
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
    let s = Lsqml::new(params).unwrap().with_preconditioner(pre).step(
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

/// `got`'s change from `start` on `frames` against `want`'s, `per` values a frame; values
/// outside `frames` stay as they are.
fn assert_change(
    name: &str,
    got: &[f64],
    start: &[f64],
    want: &[f64],
    per: usize,
    frames: &[usize],
) {
    let (mut moved, mut expected) = (Vec::new(), Vec::new());
    for f in 0..start.len() / per {
        let ws = f * per..(f + 1) * per;
        if frames.contains(&f) {
            for j in ws {
                moved.push(got[j] - start[j]);
                expected.push(want[j] - start[j]);
            }
        } else {
            assert_eq!(got[ws.clone()], start[ws], "{name} frame {f}");
        }
    }
    let err = max_rel(&moved, &expected);
    eprintln!("{name} change: max rel {err:.3e}");
    assert!(err < 1e-13, "{name} change: max rel {err:.3e}");
}

/// `var`'s eigenmodes against `r`'s eigen columns and its weight changes on `frames` against
/// `r`'s weight columns after the `noise` (0 Gaussian, 1 Poisson) step.
fn assert_variation(r: &Ref, var: &ProbeVariation, frames: &[usize], noise: usize) {
    let err = max_rel(
        &re_im(var.eigenmodes()),
        &re_im(&r.eigenmodes(r.u[1 + noise])),
    );
    eprintln!("{} eigenmodes: max rel {err:.3e}", r.name);
    assert!(err < 1e-13, "{} eigenmodes: max rel {err:.3e}", r.name);
    let (start, want) = (r.weights(r.w[0]), r.weights(r.w[3 + noise]));
    let name = format!("{} w", r.name);
    assert_change(&name, &var.weights().values, &start, &want, r.k, frames);
    match (r.c, var.intensity()) {
        (Some(c), Some(got)) => {
            let (start, want) = (r.column(c[0]), r.column(c[1 + noise]));
            assert_change(
                &format!("{} c", r.name),
                &got.values,
                &start,
                &want,
                1,
                frames,
            );
        }
        (c, got) => assert_eq!(c.is_some(), got.is_some(), "{} c", r.name),
    }
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
    assert_variation(&OPR, &var, &frames, noise);
}

#[test]
fn gaussian_step_matches_opr_ref() {
    assert_opr_ref(Cost::Amplitude, 0);
}

#[test]
fn poisson_step_matches_opr_ref() {
    assert_opr_ref(Cost::Poisson, 1);
}

/// A three-eigenmode reference's LSQML step on its batch, from its `steps`, `obj` and `probe`
/// tables: per-frame `a_far a_o a_p`, the object and probe changes and the eigenmodes and
/// weights after it.
fn assert_lsqml_opr_ref(r: &Ref, [steps, obj, probe]: [&str; 3], cost: Cost, noise: usize) {
    let (p, optics) = state();
    let mut var = r.variation();
    let rows = table(steps);
    let frames: Vec<usize> = rows.iter().map(|r| r[0] as usize).collect();
    let (s, d_obj, d_probe) = step(&p, &optics, cost, &frames, &mut var);
    for (name, got, col, tol) in [
        ("a_far", &s.a_far, 1, 1e-13),
        ("a_o", &s.a_obj[0], 2, 1e-12),
        ("a_p", &s.a_probe, 3, 1e-12),
    ] {
        let want: Vec<f64> = rows.iter().map(|r| r[col + 3 * noise]).collect();
        let err = max_rel(got, &want);
        eprintln!("{} {cost:?} {name}: max rel {err:.3e}", r.name);
        assert!(err < tol, "{} {cost:?} {name}: max rel {err:.3e}", r.name);
    }
    for (name, text, lead, got) in [("object", obj, 2, d_obj), ("probe", probe, 3, d_probe)] {
        let c = lead + 2 * noise;
        let want: Vec<f64> = table(text).iter().flat_map(|r| [r[c], r[c + 1]]).collect();
        let err = max_rel(&got, &want);
        eprintln!("{} {cost:?} {name}: max rel {err:.3e}", r.name);
        assert!(err < 1e-12, "{} {cost:?} {name}: max rel {err:.3e}", r.name);
    }
    assert_variation(r, &var, &frames, noise);
}

#[test]
fn gaussian_step_matches_lsqml_opr_ref() {
    assert_lsqml_opr_ref(&LSQML_OPR, LSQML_OPR_STEP, Cost::Amplitude, 0);
}

#[test]
fn poisson_step_matches_lsqml_opr_ref() {
    assert_lsqml_opr_ref(&LSQML_OPR, LSQML_OPR_STEP, Cost::Poisson, 1);
}

#[test]
fn gaussian_step_matches_lsqml_opr_int_ref() {
    assert_lsqml_opr_ref(&LSQML_OPR_INT, LSQML_OPR_INT_STEP, Cost::Amplitude, 0);
}

#[test]
fn poisson_step_matches_lsqml_opr_int_ref() {
    assert_lsqml_opr_ref(&LSQML_OPR_INT, LSQML_OPR_INT_STEP, Cost::Poisson, 1);
}

/// A one-frame batch leaves every eigenmode and `w` unchanged; refined main-mode weights move
/// at that frame as in the reference's batch, a frame's `chi` being its own under the Gaussian
/// model (Poisson's `a_far` takes the batch mean).
#[test]
fn single_frame_batch_moves_only_main_mode_weights() {
    let (p, optics) = state();
    for r in [&OPR, &LSQML_OPR, &LSQML_OPR_INT] {
        let mut var = r.variation();
        let (s, _, _) = step(&p, &optics, Cost::Amplitude, &[3], &mut var);
        assert_eq!(s.a_far.len(), 1);
        let start = r.variation();
        assert_eq!(var.eigenmodes(), start.eigenmodes(), "{}", r.name);
        assert_eq!(var.weights(), start.weights(), "{}", r.name);
        match r.c {
            Some([c, cg, _]) => {
                let got = var.intensity().unwrap();
                let name = format!("{} c", r.name);
                assert_change(&name, &got.values, &r.column(c), &r.column(cg), 1, &[3]);
            }
            None => assert_eq!(var, start, "{}", r.name),
        }
    }
}

/// Held weights or main-mode weights keep their values through a step; the rest of it, the
/// eigenmodes' move from the held weights among it, is the step with them refined, bit for
/// bit.
#[test]
fn held_weights_and_intensity_keep_their_values() {
    let (p, optics) = state();
    let r = &LSQML_OPR_INT;
    let rows = table(LSQML_OPR_INT_STEP[0]);
    let frames: Vec<usize> = rows.iter().map(|r| r[0] as usize).collect();
    let start = r.variation();
    let mut refined = start.clone();
    let want = step(&p, &optics, Cost::Poisson, &frames, &mut refined);
    let (w, c) = (start.weights(), start.intensity().unwrap());
    assert_ne!(refined.weights().values, w.values);
    assert_ne!(refined.intensity().unwrap().values, c.values);
    assert_ne!(refined.eigenmodes(), start.eigenmodes());
    for [hold_w, hold_c] in [[true, false], [false, true], [true, true]] {
        let name = format!("hold weights {hold_w}, intensity {hold_c}");
        let feature = |f: &Feature<Vec<f64>>, hold: bool| Feature {
            values: f.values.clone(),
            refine: !hold,
        };
        let var = ProbeVariation::new(start.eigenmodes().clone(), feature(w, hold_w));
        let mut var = var.with_intensity(feature(c, hold_c));
        let got = step(&p, &optics, Cost::Poisson, &frames, &mut var);
        assert_eq!(got, want, "{name}: step, object and probe");
        assert_eq!(var.eigenmodes(), refined.eigenmodes(), "{name}: eigenmodes");
        let kept = |hold: bool, start: &Feature<Vec<f64>>, moved: &Feature<Vec<f64>>| {
            feature(if hold { start } else { moved }, hold)
        };
        let want_w = kept(hold_w, w, refined.weights());
        assert_eq!(*var.weights(), want_w, "{name}: weights");
        let want_c = kept(hold_c, c, refined.intensity().unwrap());
        assert_eq!(var.intensity(), Some(&want_c), "{name}: main-mode weights");
    }
}
