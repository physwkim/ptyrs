//! BH against central differences and the float64 reference `refs/f64/bh_step.py`, run on the
//! state it reads: `cargo run --release -p ptyrs-engine --example llk_trace -- <dir> dm 1 2 32
//! 40 4`, i.e. `common::dm_scenario(32, 40, 2, 2e5, 4.0)` after `scale_obj_probe`. The
//! reference writes `<dir>/bh_{full,mini}_{obj,probe,steps}_f64.txt`, saved to `tests/data`.

use std::ops::Range;

use ptyrs_core::{C64, Object, Probe, Stack};
use ptyrs_engine::{Bh, BhBatch, BhParams, Fft2, Ptycho, llk, obj_probe_to_psi};

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

/// A complex normal-ish array shaped as `like`, scaled to `like`'s largest modulus.
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

/// The views of `frames`.
fn batch(p: &Ptycho, frames: Range<usize>) -> &[ptyrs_core::View] {
    let nmode = p.addr().nmode();
    &p.addr().views()[frames.start * nmode..frames.end * nmode]
}

/// `F` of `frames` at `(object, probe)` without the amplitude offset: a quarter of the
/// engine's Euclidian sum.
fn cost(p: &Ptycho, frames: Range<usize>, object: &Object, probe: &Probe) -> f64 {
    let views = batch(p, frames.clone());
    let mut psi = vec![C64::default(); views.len() * probe.plane_len()];
    obj_probe_to_psi(views, object, probe, p.quad(), &mut psi);
    Fft2::new(probe.plane_shape()).forward(&mut psi);
    llk(p.observed(), frames, p.addr().nmode(), &psi).euclidian / 4.0
}

const FRAMES: Range<usize> = 0..16;

/// `⟨g, d⟩` against `(F(x + h·d) − F(x − h·d)) / 2h` along random object and probe directions.
#[test]
fn grads_match_central_differences() {
    let p = state();
    let (o, pr) = (p.object(), p.probe());
    let b = BhBatch::new(batch(&p, FRAMES), p.observed(), p.quad(), o, pr);
    let [g_o, g_p] = b.grads();
    let mut seed = 0x2545_f491_4f6c_dd1d;
    let (d_o, d_p) = (direction(o, &mut seed), direction(pr, &mut seed));
    let h = 1e-5;
    let fo = (cost(&p, FRAMES, &Object::new(moved(o, h, &d_o)), pr)
        - cost(&p, FRAMES, &Object::new(moved(o, -h, &d_o)), pr))
        / (2.0 * h);
    let fp = (cost(&p, FRAMES, o, &Probe::new(moved(pr, h, &d_p)))
        - cost(&p, FRAMES, o, &Probe::new(moved(pr, -h, &d_p))))
        / (2.0 * h);
    // the gradient carries the 1e-8 amplitude offset, the differences do not
    let (eo, ep) = (rel(dot(&g_o, &d_o), fo), rel(dot(&g_p, &d_p), fp));
    assert!(eo < 1e-7 && ep < 1e-7, "object {eo:.3e} probe {ep:.3e}");
}

/// `H(u, v)` against `(⟨g(x + h·v), u⟩ − ⟨g(x − h·v), u⟩) / 2h` for random displacements
/// `u ≠ v` over both the object and the probe.
#[test]
fn hessian_matches_central_differences() {
    let p = state();
    let (o, pr) = (p.object(), p.probe());
    let views = batch(&p, FRAMES);
    let mut seed = 0x9e37_79b9_7f4a_7c15;
    let u = [direction(o, &mut seed), direction(pr, &mut seed)];
    let v = [direction(o, &mut seed), direction(pr, &mut seed)];
    let h_an =
        BhBatch::new(views, p.observed(), p.quad(), o, pr).hessian([&u[0], &u[1]], [&v[0], &v[1]]);
    let h_sym =
        BhBatch::new(views, p.observed(), p.quad(), o, pr).hessian([&v[0], &v[1]], [&u[0], &u[1]]);
    assert!(rel(h_sym, h_an) < 1e-13, "H(u, v) {h_an} H(v, u) {h_sym}");
    let du = |t: f64| {
        let (o, pr) = (
            Object::new(moved(o, t, &v[0])),
            Probe::new(moved(pr, t, &v[1])),
        );
        let [a, b] = BhBatch::new(views, p.observed(), p.quad(), &o, &pr).grads();
        dot(&a, &u[0]) + dot(&b, &u[1])
    };
    let h = 1e-6;
    let fd = (du(h) - du(-h)) / (2.0 * h);
    assert!(rel(h_an, fd) < 1e-7, "H {h_an} differences {fd}");
}

/// Two epochs of [`Bh::epoch`] against a reference's per-step `step epoch first count beta
/// alpha top bottom` and per-epoch object (`iy ix re0 im0 re1 im1`) and probe (`m iy ix ...`)
/// changes.
fn assert_ref(batch_size: usize, steps: &str, obj: &str, probe: &str) {
    let p = state();
    let mut bh = Bh::new(BhParams {
        batch_size,
        probe_rho: 0.01,
    });
    let (mut object, mut pr) = (p.object().clone(), p.probe().clone());
    let mut got = Vec::new();
    let mut changes = Vec::new();
    for _ in 0..2 {
        let (o0, p0) = (object.clone(), pr.clone());
        got.push(bh.epoch(p.addr(), p.observed(), p.quad(), &mut object, &mut pr));
        changes.push([moved(&object, -1.0, &o0), moved(&pr, -1.0, &p0)]);
    }
    let rows = table(steps);
    let got: Vec<_> = got
        .iter()
        .enumerate()
        .flat_map(|(e, s)| s.iter().map(move |s| (e, s)))
        .collect();
    assert_eq!(got.len(), rows.len());
    for (k, ((epoch, s), r)) in got.iter().zip(&rows).enumerate() {
        assert_eq!(
            [k, *epoch, s.first, s.count],
            [r[0], r[1], r[2], r[3]].map(|v| v as usize)
        );
        if r[4] == 0.0 {
            assert_eq!(s.beta, 0.0, "step {k}");
        } else {
            assert!(
                rel(s.beta, r[4]) < 1e-13,
                "step {k}: beta {} ref {}",
                s.beta,
                r[4]
            );
        }
        for (name, got, want) in [
            ("alpha", s.alpha, r[5]),
            ("top", s.top, r[6]),
            ("bottom", s.bottom, r[7]),
        ] {
            assert!(rel(got, want) < 1e-13, "step {k}: {name} {got} ref {want}");
        }
    }
    // the probe change sums the probe gradient over every frame of a batch
    for (name, text, which, lead, tol) in
        [("object", obj, 0, 2, 1e-13), ("probe", probe, 1, 3, 5e-12)]
    {
        let rows = table(text);
        for (e, c) in changes.iter().enumerate() {
            let got: Vec<f64> = c[which]
                .as_slice()
                .iter()
                .flat_map(|c| [c.re, c.im])
                .collect();
            let want: Vec<f64> = rows
                .iter()
                .flat_map(|r| [r[lead + 2 * e], r[lead + 2 * e + 1]])
                .collect();
            let err = max_rel(&got, &want);
            assert!(err < tol, "{name} epoch {e}: max rel {err:.3e}");
        }
    }
}

#[test]
fn full_batch_matches_bh_ref() {
    assert_ref(
        40,
        include_str!("data/bh_full_steps_f64.txt"),
        include_str!("data/bh_full_obj_f64.txt"),
        include_str!("data/bh_full_probe_f64.txt"),
    );
}

#[test]
fn mini_batch_matches_bh_ref() {
    assert_ref(
        16,
        include_str!("data/bh_mini_steps_f64.txt"),
        include_str!("data/bh_mini_obj_f64.txt"),
        include_str!("data/bh_mini_probe_f64.txt"),
    );
}
