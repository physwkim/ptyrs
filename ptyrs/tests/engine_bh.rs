//! BH against central differences and the float64 reference `refs/f64/bh_step.py`, run on the
//! state it reads: `cargo run --release -p ptyrs --example engine_llk_trace -- <dir> dm 1 2 32
//! 40 4`, i.e. `common::dm_scenario(32, 40, 2, 2e5, 4.0)` after `scale_obj_probe`. The
//! reference writes `<dir>/bh_{full,mini}_{obj,probe,steps}_f64.txt`, saved to `tests/data`.

use std::ops::Range;
use std::slice;

use ptyrs::base::{C64, Object, Probe, Stack};
use ptyrs::engine::{Bh, BhBatch, BhParams, BhVector, Optics, Ptycho, llk, obj_probe_to_psi};

#[path = "common/engine/mod.rs"]
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

fn dot(a: &[C64], b: &[C64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| (a.conj() * b).re).sum()
}

/// The displacement of an object and a probe change, [`BhBatch`]'s layout.
fn flat(o: &Stack<C64>, p: &Stack<C64>) -> BhVector {
    BhVector {
        x: [o.as_slice(), p.as_slice()].concat(),
        d: Vec::new(),
    }
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
fn batch(p: &Ptycho, frames: Range<usize>) -> &[ptyrs::base::View] {
    let nmode = p.addr().nmode();
    &p.addr().views()[frames.start * nmode..frames.end * nmode]
}

/// `F` of `frames` at `(object, probe)` without the amplitude offset: a quarter of the
/// engine's Euclidian sum.
fn cost(p: &Ptycho, optics: &Optics, frames: Range<usize>, object: &Object, probe: &Probe) -> f64 {
    let views = batch(p, frames.clone());
    let mut psi = vec![C64::default(); views.len() * probe.plane_len()];
    obj_probe_to_psi(views, object, probe, optics.quad(), &mut psi);
    optics.propagator().forward(&mut psi);
    llk(p.observed(), frames, p.addr().nmode(), &psi).euclidian / 4.0
}

const FRAMES: Range<usize> = 0..16;

/// `⟨g, d⟩` against `(F(x + h·d) − F(x − h·d)) / 2h` along random object and probe directions.
#[test]
fn grads_match_central_differences() {
    let (p, optics) = state();
    let (o, pr) = (p.object(), p.probe());
    let b = BhBatch::new(
        batch(&p, FRAMES),
        p.observed(),
        &optics,
        slice::from_ref(o),
        pr,
        None,
    );
    let g = b.grads();
    let (g_o, g_p) = g.x.split_at(o.as_slice().len());
    let mut seed = 0x2545_f491_4f6c_dd1d;
    let (d_o, d_p) = (direction(o, &mut seed), direction(pr, &mut seed));
    let h = 1e-5;
    let f = |o: &Object, pr: &Probe| cost(&p, &optics, FRAMES, o, pr);
    let fo = (f(&Object::new(moved(o, h, &d_o)), pr) - f(&Object::new(moved(o, -h, &d_o)), pr))
        / (2.0 * h);
    let fp = (f(o, &Probe::new(moved(pr, h, &d_p))) - f(o, &Probe::new(moved(pr, -h, &d_p))))
        / (2.0 * h);
    // the gradient carries the 1e-8 amplitude offset, the differences do not
    let (eo, ep) = (
        rel(dot(g_o, d_o.as_slice()), fo),
        rel(dot(g_p, d_p.as_slice()), fp),
    );
    assert!(eo < 1e-7 && ep < 1e-7, "object {eo:.3e} probe {ep:.3e}");
}

/// `H(u, v)` against `(⟨g(x + h·v), u⟩ − ⟨g(x − h·v), u⟩) / 2h` for random displacements
/// `u ≠ v` over both the object and the probe.
#[test]
fn hessian_matches_central_differences() {
    let (p, optics) = state();
    let (o, pr) = (p.object(), p.probe());
    let views = batch(&p, FRAMES);
    let mut seed = 0x9e37_79b9_7f4a_7c15;
    let (u_o, u_p) = (direction(o, &mut seed), direction(pr, &mut seed));
    let (v_o, v_p) = (direction(o, &mut seed), direction(pr, &mut seed));
    let (u, v) = (flat(&u_o, &u_p), flat(&v_o, &v_p));
    let b = BhBatch::new(views, p.observed(), &optics, slice::from_ref(o), pr, None);
    let (h_an, h_sym) = (b.hessian(&u, &v), b.hessian(&v, &u));
    assert!(rel(h_sym, h_an) < 1e-13, "H(u, v) {h_an} H(v, u) {h_sym}");
    let du = |t: f64| {
        let (o, pr) = (Object::new(moved(o, t, &v_o)), moved(pr, t, &v_p));
        let g = BhBatch::new(views, p.observed(), &optics, &[o], &Probe::new(pr), None).grads();
        g.dot(&u)
    };
    let h = 1e-6;
    let fd = (du(h) - du(-h)) / (2.0 * h);
    assert!(rel(h_an, fd) < 1e-7, "H {h_an} differences {fd}");
}

/// Two epochs of [`Bh::epoch`] with seed 7 against a reference's per-step `step epoch count
/// beta alpha top bottom frames..` and per-epoch object (`iy ix re0 im0 re1 im1`) and probe
/// (`m iy ix ...`) changes.
fn assert_ref(batch_size: usize, steps: &str, obj: &str, probe: &str) {
    let (p, optics) = state();
    let mut bh = Bh::new(BhParams {
        batch_size,
        probe_rho: 0.01,
        seed: 7,
    });
    let (mut object, mut pr) = (p.object().clone(), p.probe().clone());
    let mut got = Vec::new();
    let mut changes = Vec::new();
    for _ in 0..2 {
        let (o0, p0) = (object.clone(), pr.clone());
        let objects = slice::from_mut(&mut object);
        got.push(bh.epoch(p.addr(), p.observed(), &optics, objects, &mut pr));
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
        let count = r[2] as usize;
        assert_eq!([k, *epoch, count], [r[0], r[1], r[2]].map(|v| v as usize));
        let frames: Vec<usize> = r[7..].iter().map(|&f| f as usize).collect();
        assert_eq!((frames.len(), &s.frames), (count, &frames), "step {k}");
        if r[3] == 0.0 {
            assert_eq!(s.beta, 0.0, "step {k}");
        } else {
            assert!(
                rel(s.beta, r[3]) < 1e-13,
                "step {k}: beta {} ref {}",
                s.beta,
                r[3]
            );
        }
        for (name, got, want) in [
            ("alpha", s.alpha, r[4]),
            ("top", s.top, r[5]),
            ("bottom", s.bottom, r[6]),
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
        include_str!("engine/data/bh_full_steps_f64.txt"),
        include_str!("engine/data/bh_full_obj_f64.txt"),
        include_str!("engine/data/bh_full_probe_f64.txt"),
    );
}

#[test]
fn mini_batch_matches_bh_ref() {
    assert_ref(
        16,
        include_str!("engine/data/bh_mini_steps_f64.txt"),
        include_str!("engine/data/bh_mini_obj_f64.txt"),
        include_str!("engine/data/bh_mini_probe_f64.txt"),
    );
}
