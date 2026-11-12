//! LSQML against central differences and the float64 reference `refs/f64/lsqml_step.py`, run on
//! the state it reads: `cargo run --release -p ptyrs-engine --example llk_trace -- <dir> dm 1 2
//! 32 40 4`, i.e. `common::dm_scenario(32, 40, 2, 2e5, 4.0)` after `scale_obj_probe`. The
//! reference writes `<dir>/lsqml_{gaussian,poisson}_{obj,probe,steps}_f64.txt`, and
//! `refs/f64/lsqml_epochs.py` on the same state `<dir>/lsqml_epochs_*_f64.txt`, saved to
//! `tests/data`. Without eigenmodes ([`ProbeVariation::none`]) every frame sees the shared probe;
//! `tests/opr.rs` checks the step with them.

use std::ops::Range;

use ptyrs_core::{C64, Object, Probe, Stack, View};
use ptyrs_engine::{
    Cost, Fft2, Lsqml, LsqmlBatch, LsqmlParams, Optics, ProbeVariation, Ptycho, illumination,
    obj_probe_to_psi, random_batches,
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

/// A complex array shaped as `like`, uniform in −1/2..1/2 per part, scaled to `like`'s largest
/// modulus.
fn direction(like: &Stack<C64>, seed: &mut u64) -> Stack<C64> {
    let mut uniform = || {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let scale = like.as_slice().iter().map(|c| c.norm()).fold(0.0, f64::max);
    let data = (0..like.as_slice().len())
        .map(|_| C64::new(uniform(), uniform()) * scale)
        .collect();
    Stack::from_vec(like.dims(), data).unwrap()
}

fn dot(a: &Stack<C64>, b: &Stack<C64>) -> f64 {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(a, b)| (a.conj() * b).re)
        .sum()
}

/// `x + t·d`.
fn moved(x: &Stack<C64>, t: f64, d: &Stack<C64>) -> Stack<C64> {
    let data = x
        .as_slice()
        .iter()
        .zip(d.as_slice())
        .map(|(x, d)| x + t * d)
        .collect();
    Stack::from_vec(x.dims(), data).unwrap()
}

fn re_im(s: &Stack<C64>) -> Vec<f64> {
    s.as_slice().iter().flat_map(|c| [c.re, c.im]).collect()
}

/// The views of `frames`.
fn batch(p: &Ptycho, frames: Range<usize>) -> &[View] {
    let nmode = p.addr().nmode();
    &p.addr().views()[frames.start * nmode..frames.end * nmode]
}

/// The cost whose `icalc` derivative is LSQML's residual with pty-chi's offsets:
/// `(icalc + 1e-6) − iobs·ln(icalc + 1e-6)` (Poisson) and
/// `icalc − 2√(iobs·icalc) + 2e-9·√iobs·ln(√icalc + 1e-9)` (Gaussian), up to constants.
fn offset_cost(cost: Cost, iobs: f64, icalc: f64) -> f64 {
    match cost {
        Cost::Poisson => icalc + 1e-6 - iobs * (icalc + 1e-6).ln(),
        Cost::Amplitude => {
            let (d, a) = (iobs.max(0.0).sqrt(), icalc.sqrt());
            icalc - 2.0 * d * a + 2e-9 * d * (a + 1e-9).ln()
        }
    }
}

/// `Σ_f w_f Σ_pixels offset_cost(iobs, icalc)` over `frames` at `(object, probe)`.
fn weighted_cost(
    p: &Ptycho,
    cost: Cost,
    frames: Range<usize>,
    w: &[f64],
    object: &Object,
    probe: &Probe,
) -> f64 {
    let views = batch(p, frames.clone());
    let (n, nmode) = (probe.plane_len(), p.addr().nmode());
    let mut psi = vec![C64::default(); views.len() * n];
    obj_probe_to_psi(views, object, probe, p.quad(), &mut psi);
    Fft2::new(probe.plane_shape()).forward(&mut psi);
    frames
        .zip(psi.chunks_exact(nmode * n))
        .zip(w)
        .map(|((f, block), w)| {
            let iobs = p.observed().frames(f..f + 1);
            let sum: f64 = (0..n)
                .filter_map(|j| {
                    let icalc = (0..nmode).map(|m| block[m * n + j].norm_sqr()).sum();
                    iobs[j].map(|iobs| offset_cost(cost, iobs, icalc))
                })
                .sum();
            w * sum
        })
        .sum()
}

const FRAMES: Range<usize> = 0..20;

/// `−2·obj_direction` and `−2·count·probe_direction` against central differences of
/// `Σ_f w_f L_f` along random object and probe directions, `L_f` the frame's [`offset_cost`]
/// and `w_f` its `a_far` (Poisson) or `2·a_far` (Gaussian), held at the state's values.
#[test]
fn directions_match_central_differences() {
    let (p, optics) = state();
    let (o, pr) = (p.object(), p.probe());
    for (cost, scale) in [(Cost::Amplitude, 2.0), (Cost::Poisson, 1.0)] {
        let none = ProbeVariation::none(pr.plane_shape());
        let run = Lsqml::new(LsqmlParams {
            batch_size: FRAMES.len(),
            cost,
            seed: 0,
        });
        let b = LsqmlBatch::new(&run, batch(&p, FRAMES), p.observed(), &optics, o, pr, &none);
        let w: Vec<f64> = b.a_far().iter().map(|a| scale * a).collect();
        let mut g_o = b.obj_direction();
        let mut g_p = b.probe_direction();
        g_o.as_mut_slice().iter_mut().for_each(|c| *c *= -2.0);
        let count = FRAMES.len() as f64;
        g_p.as_mut_slice()
            .iter_mut()
            .for_each(|c| *c *= -2.0 * count);
        let mut seed = 0x2545_f491_4f6c_dd1d;
        let (d_o, d_p) = (direction(o, &mut seed), direction(pr, &mut seed));
        let f = |o: &Object, pr: &Probe| weighted_cost(&p, cost, FRAMES, &w, o, pr);
        // the probe differences' h² truncation is 1e-7 at h 1e-5, so they take h 1e-6
        let (ho, hp) = (1e-5, 1e-6);
        let fo = (f(&Object::new(moved(o, ho, &d_o)), pr)
            - f(&Object::new(moved(o, -ho, &d_o)), pr))
            / (2.0 * ho);
        let fp = (f(o, &Probe::new(moved(pr, hp, &d_p))) - f(o, &Probe::new(moved(pr, -hp, &d_p))))
            / (2.0 * hp);
        let (eo, ep) = (rel(dot(&g_o, &d_o), fo), rel(dot(&g_p, &d_p), fp));
        eprintln!("{cost:?}: object {eo:.3e} probe {ep:.3e}");
        assert!(
            eo < 1e-7 && ep < 1e-7,
            "{cost:?}: object {eo:.3e} probe {ep:.3e}"
        );
    }
}

/// One [`Lsqml::step`] on frames 0..20 with the illumination of every view as preconditioner
/// against the reference's object (`iy ix re im`) and probe (`m iy ix re im`) changes and
/// per-frame `frame a_far a_o a_p`.
fn assert_ref(cost: Cost, obj: &str, probe: &str, steps: &str) {
    let (p, optics) = state();
    let (mut object, mut pr) = (p.object().clone(), p.probe().clone());
    let pre = illumination(p.addr().views(), object.dims(), &pr);
    let mut var = ProbeVariation::none(pr.plane_shape());
    let params = LsqmlParams {
        batch_size: 20,
        cost,
        seed: 0,
    };
    let s = Lsqml::new(params).with_preconditioner(pre).step(
        batch(&p, FRAMES),
        p.observed(),
        &optics,
        &mut object,
        &mut pr,
        &mut var,
    );
    assert_eq!(var, ProbeVariation::none(pr.plane_shape()));
    assert_eq!(s.frames, FRAMES.collect::<Vec<_>>());
    let rows = table(steps);
    assert_eq!(rows.len(), FRAMES.len());
    for (k, r) in rows.iter().enumerate() {
        assert_eq!(r[0] as usize, k);
    }
    for (name, got, col, tol) in [
        ("a_far", &s.a_far, 1, 1e-13),
        ("a_o", &s.a_obj, 2, 1e-12),
        ("a_p", &s.a_probe, 3, 1e-12),
    ] {
        let want: Vec<f64> = rows.iter().map(|r| r[col]).collect();
        let err = max_rel(got, &want);
        eprintln!("{cost:?} {name}: max rel {err:.3e}");
        assert!(err < tol, "{cost:?} {name}: max rel {err:.3e}");
    }
    assert_changes(cost, &p, &object, &pr, obj, probe, 1e-12);
}

/// The changes of `object` and `pr` from `p`'s against the reference's object (`iy ix re im`)
/// and probe (`m iy ix re im`) ones, each to `tol` of its largest.
fn assert_changes(
    cost: Cost,
    p: &Ptycho,
    object: &Object,
    pr: &Probe,
    obj: &str,
    probe: &str,
    tol: f64,
) {
    for (name, text, lead, got, before) in [
        ("object", obj, 2, &**object, &**p.object()),
        ("probe", probe, 3, &**pr, &**p.probe()),
    ] {
        let got = re_im(&moved(got, -1.0, before));
        let want: Vec<f64> = table(text)
            .iter()
            .flat_map(|r| [r[lead], r[lead + 1]])
            .collect();
        let err = max_rel(&got, &want);
        eprintln!("{cost:?} {name}: max rel {err:.3e}");
        assert!(err < tol, "{cost:?} {name}: max rel {err:.3e}");
    }
}

#[test]
fn gaussian_step_matches_lsqml_ref() {
    assert_ref(
        Cost::Amplitude,
        include_str!("data/lsqml_gaussian_obj_f64.txt"),
        include_str!("data/lsqml_gaussian_probe_f64.txt"),
        include_str!("data/lsqml_gaussian_steps_f64.txt"),
    );
}

#[test]
fn poisson_step_matches_lsqml_ref() {
    assert_ref(
        Cost::Poisson,
        include_str!("data/lsqml_poisson_obj_f64.txt"),
        include_str!("data/lsqml_poisson_probe_f64.txt"),
        include_str!("data/lsqml_poisson_steps_f64.txt"),
    );
}

/// Three [`Lsqml::epoch`]s of batches of 16 (seed 7) against pty-chi's own run of them,
/// `refs/f64/lsqml_epochs.py`: every step's `epoch frame a_far a_o a_p`, then the object and
/// probe changes over the run. Epoch 0 is each frame's first visit, a Poisson `a_far` solved
/// from 1/2; epochs 1–2 revisit every frame, from its `a_far` of the epoch before.
fn assert_epochs_ref(cost: Cost, obj: &str, probe: &str, steps: &str) {
    let (p, optics) = state();
    let (addr, obs) = (p.addr(), p.observed());
    let (mut object, mut pr) = (p.object().clone(), p.probe().clone());
    let mut var = ProbeVariation::none(pr.plane_shape());
    let mut lsqml = Lsqml::new(LsqmlParams {
        batch_size: 16,
        cost,
        seed: 7,
    });
    let rows = table(steps);
    let mut next = 0;
    for e in 0..3 {
        for s in lsqml.epoch(addr, obs, &optics, &mut object, &mut pr, &mut var) {
            let want = &rows[next..next + s.frames.len()];
            next += s.frames.len();
            let frames: Vec<usize> = want.iter().map(|r| r[1] as usize).collect();
            assert!(want.iter().all(|r| r[0] == e as f64), "epoch {e}");
            assert_eq!(s.frames, frames, "epoch {e}");
            for (name, got, col, tol) in [
                ("a_far", &s.a_far, 2, 1e-13),
                ("a_o", &s.a_obj, 3, 1e-12),
                ("a_p", &s.a_probe, 4, 1e-12),
            ] {
                let want: Vec<f64> = want.iter().map(|r| r[col]).collect();
                let err = max_rel(got, &want);
                eprintln!("{cost:?} epoch {e} {name}: max rel {err:.3e}");
                assert!(err < tol, "{cost:?} epoch {e} {name}: max rel {err:.3e}");
            }
        }
    }
    assert_eq!(next, rows.len());
    assert_changes(cost, &p, &object, &pr, obj, probe, 1e-12);
}

#[test]
fn gaussian_epochs_match_lsqml_ref() {
    assert_epochs_ref(
        Cost::Amplitude,
        include_str!("data/lsqml_epochs_gaussian_obj_f64.txt"),
        include_str!("data/lsqml_epochs_gaussian_probe_f64.txt"),
        include_str!("data/lsqml_epochs_gaussian_steps_f64.txt"),
    );
}

#[test]
fn poisson_epochs_match_lsqml_ref() {
    assert_epochs_ref(
        Cost::Poisson,
        include_str!("data/lsqml_epochs_poisson_obj_f64.txt"),
        include_str!("data/lsqml_epochs_poisson_probe_f64.txt"),
        include_str!("data/lsqml_epochs_poisson_steps_f64.txt"),
    );
}

/// [`Lsqml::epoch`] over 12 epochs of two batches with three eigenmodes of probe mode 0: the
/// preconditioner starts from the renormalised shared probe's illumination, averages in the
/// epoch's illumination at epochs 0–10 and stays at epoch 11, and the epoch's steps are
/// [`Lsqml::step`] of the run as it entered the epoch, with that preconditioner, on the epoch's
/// [`random_batches`].
#[test]
fn epoch_follows_ptychi_preconditioner_schedule() {
    let (p, optics) = state();
    let (addr, obs) = (p.addr(), p.observed());
    let mut lsqml = Lsqml::new(LsqmlParams {
        batch_size: 20,
        cost: Cost::Poisson,
        seed: 7,
    });
    assert!(lsqml.preconditioner().is_none());
    let (mut object, mut pr) = (p.object().clone(), p.probe().clone());
    let mut seed = 0x9e37_79b9_7f4a_7c15;
    let eigenmodes = direction(&Stack::filled([3, 32, 32], C64::new(1.0, 0.0)), &mut seed);
    let weights = direction(&Stack::filled([1, 40, 3], C64::new(1e-3, 0.0)), &mut seed);
    let weights = weights.as_slice().iter().map(|w| w.re).collect();
    let mut var = ProbeVariation::new(eigenmodes, weights);
    let start = var.clone();
    let n = pr.plane_len() as f64;
    let power = (0..addr.nframe())
        .map(|f| obs.frames(f..f + 1).iter().flatten().sum::<f64>())
        .fold(0.0, f64::max);
    let mut want: Option<Vec<f64>> = None;
    for e in 0..12 {
        let illum = illumination(addr.views(), object.dims(), &pr);
        let prev = want.clone().unwrap_or_else(|| {
            illum
                .as_slice()
                .iter()
                .map(|i| i * power / (4.0 * n * n))
                .collect()
        });
        let next: Vec<f64> = if e <= 10 {
            prev.iter()
                .zip(illum.as_slice())
                .map(|(p, i)| (p + i) / 2.0)
                .collect()
        } else {
            prev
        };
        let (mut o2, mut p2, mut v2) = (object.clone(), pr.clone(), var.clone());
        let replay = lsqml.clone();
        let steps = lsqml.epoch(addr, obs, &optics, &mut object, &mut pr, &mut var);
        let pre = lsqml.preconditioner().unwrap();
        assert_eq!(pre.as_slice(), &next[..], "epoch {e}");
        let mut replay = replay.with_preconditioner(pre.clone());
        let batches = random_batches(40, 20, 7, e);
        assert_eq!(
            steps.iter().map(|s| &s.frames).collect::<Vec<_>>(),
            batches.iter().collect::<Vec<_>>(),
            "epoch {e}"
        );
        for (s, frames) in steps.iter().zip(&batches) {
            let views: Vec<View> = frames
                .iter()
                .flat_map(|&f| batch(&p, f..f + 1))
                .copied()
                .collect();
            let r = replay.step(&views, obs, &optics, &mut o2, &mut p2, &mut v2);
            assert_eq!(&r, s, "epoch {e}");
        }
        assert_eq!(o2.as_slice(), object.as_slice(), "epoch {e}");
        assert_eq!(p2.as_slice(), pr.as_slice(), "epoch {e}");
        assert_eq!(v2, var, "epoch {e}");
        assert_ne!(var, start, "epoch {e}");
        want = Some(next);
    }
}
