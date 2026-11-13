//! BH with [`Features::position`] against the f64 model of its steps, `refs/f64/bh_pos.py`
//! (no package refines positions on this model; pty-chi's BH moves the object instead), run on
//! the state it reads: `cargo run --release -p ptyrs --example engine_llk_trace -- <dir> dm 1
//! 2 32 40 4`, i.e. `common::dm_scenario(32, 40, 2, 2e5, 4.0)` after `scale_obj_probe`, with
//! the model's offsets. It writes `<dir>/bh_pos_{full,mini}_*_f64.txt`, saved to
//! `tests/engine/data`.

use std::slice;

use ptyrs::base::C64;
use ptyrs::engine::{Bh, BhParams, Features, Optics, Ptycho};

#[path = "common/engine/mod.rs"]
mod common;

fn table(text: &str) -> Vec<Vec<f64>> {
    text.lines()
        .map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect())
        .collect()
}

fn max_abs(a: &[f64]) -> f64 {
    a.iter().map(|a| a.abs()).fold(0.0, f64::max)
}

/// `max |a − b| / scale`.
fn max_rel(a: &[f64], b: &[f64], scale: f64) -> f64 {
    assert_eq!(a.len(), b.len());
    let d = a.iter().zip(b).map(|(a, b)| (a - b).abs());
    d.fold(0.0, f64::max) / scale
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

/// `re_im(a − b)`.
fn change(a: &[C64], b: &[C64]) -> Vec<f64> {
    (a.iter().zip(b))
        .flat_map(|(a, b)| [a.re - b.re, a.im - b.im])
        .collect()
}

/// Columns `c, c + 1` of an offsets table, `[dy, dx]` per frame.
fn offsets(rows: &[Vec<f64>], c: usize) -> Vec<[f64; 2]> {
    rows.iter().map(|r| [r[c], r[c + 1]]).collect()
}

fn assert_close(what: &str, got: &[f64], want: &[f64], scale: f64, tol: f64) {
    let err = max_rel(got, want, scale);
    eprintln!("{what}: max rel {err:.3e}");
    assert!(err < tol, "{what}: max rel {err:.3e}");
}

/// Two epochs of [`Bh::epoch`] with seed 7, probe rho 0.01 and position rho 2 from the
/// model's offsets: its per-step `step epoch count beta alpha top bottom frames..`, the
/// object (`iy ix re im`) and probe (`m iy ix re im`) changes and the offsets after.
fn assert_model(batch_size: usize, steps: &str, obj: &str, probe: &str, table_offsets: &str) {
    let (scan, object, probe0) = common::dm_scenario(32, 40, 2, 2e5, 4.0);
    let mut p = Ptycho::new(&scan, object, probe0).unwrap();
    p.scale_obj_probe();
    let optics = Optics::far_field(&scan.geometry);
    let rows = table(table_offsets);
    let start = offsets(&rows, 1);
    let features = Features {
        position: Some(start.clone()),
        ..Features::default()
    };
    let params = BhParams {
        batch_size,
        probe_rho: 0.01,
        position_rho: 2.0,
        seed: 7,
    };
    let mut bh = Bh::new(params).with_features(features).unwrap();
    let (o0, p0) = (p.object().clone(), p.probe().clone());
    let (mut object, mut pr) = (o0.clone(), p0.clone());
    let mut got = Vec::new();
    for epoch in 0..2 {
        let objects = slice::from_mut(&mut object);
        let s = bh.epoch(p.addr(), p.observed(), &optics, objects, &mut pr);
        got.extend(s.into_iter().map(|s| (epoch, s)));
    }
    let want = table(steps);
    assert_eq!(got.len(), want.len());
    for (k, ((epoch, s), r)) in got.iter().zip(&want).enumerate() {
        let count = r[2] as usize;
        assert_eq!([k, *epoch, count], [r[0], r[1], r[2]].map(|v| v as usize));
        let frames: Vec<usize> = r[7..].iter().map(|&f| f as usize).collect();
        assert_eq!(s.frames, frames, "step {k}");
        if r[3] == 0.0 {
            assert_eq!(s.beta, 0.0, "step {k}");
        }
        for (name, got, want) in [
            ("beta", s.beta, r[3]),
            ("alpha", s.alpha, r[4]),
            ("top", s.top, r[5]),
            ("bottom", s.bottom, r[6]),
        ] {
            if want != 0.0 {
                let err = rel(got, want);
                eprintln!("step {k} {name}: rel {err:.3e}");
                assert!(err < 1e-12, "step {k}: {name} {got} model {want}");
            }
        }
    }
    for (name, text, lead, x0, x) in [
        ("object", obj, 2, o0.as_slice(), object.as_slice()),
        ("probe", probe, 3, p0.as_slice(), pr.as_slice()),
    ] {
        let want: Vec<f64> = table(text)
            .iter()
            .flat_map(|r| [r[lead], r[lead + 1]])
            .collect();
        let start: Vec<f64> = x0.iter().flat_map(|c| [c.re, c.im]).collect();
        assert_close(name, &change(x, x0), &want, max_abs(&start), 1e-12);
    }
    let delta = |d: &[[f64; 2]]| -> Vec<f64> {
        d.iter()
            .zip(&start)
            .flat_map(|(d, s)| [d[0] - s[0], d[1] - s[1]])
            .collect()
    };
    let got = bh.features().position.as_ref().unwrap();
    let want = delta(&offsets(&rows, 3));
    assert_close("offsets", &delta(got), &want, max_abs(&want), 1e-12);
}

/// One batch of every frame: CG across the two epochs, the offsets' part of the direction
/// carried per frame.
#[test]
fn full_batch_matches_the_model() {
    assert_model(
        40,
        include_str!("engine/data/bh_pos_full_steps_f64.txt"),
        include_str!("engine/data/bh_pos_full_obj_f64.txt"),
        include_str!("engine/data/bh_pos_full_probe_f64.txt"),
        include_str!("engine/data/bh_pos_full_offsets_f64.txt"),
    );
}

/// Batches of 16: steepest descent, each step moving its batch's offsets alone.
#[test]
fn mini_batch_matches_the_model() {
    assert_model(
        16,
        include_str!("engine/data/bh_pos_mini_steps_f64.txt"),
        include_str!("engine/data/bh_pos_mini_obj_f64.txt"),
        include_str!("engine/data/bh_pos_mini_probe_f64.txt"),
        include_str!("engine/data/bh_pos_mini_offsets_f64.txt"),
    );
}
