//! LSQML with [`Features::position`] against the float64 reference `refs/f64/lsqml_pos_step.py`,
//! run on the state it reads: `cargo run --release -p ptyrs --example engine_llk_trace -- <dir>
//! dm 1 2 32 40 4`, i.e. `common::dm_scenario(32, 40, 2, 2e5, 4.0)` after `scale_obj_probe`, with
//! the reference's offsets. Its step files are its step, which matches pty-chi's own; its epoch
//! files are pty-chi's run. It writes `<dir>/lsqml_pos_*_f64.txt`, saved to `tests/data`.

use std::slice;

use ptyrs::base::{C64, Stack, View};
use ptyrs::engine::{
    Cost, Features, Lsqml, LsqmlParams, LsqmlStep, Optics, ProbeVariation, Ptycho, illumination,
};

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

/// Columns `c, c + 1` of an offsets table, `[dy, dx]` per frame.
fn offsets(rows: &[Vec<f64>], c: usize) -> Vec<[f64; 2]> {
    rows.iter().map(|r| [r[c], r[c + 1]]).collect()
}

/// `run` refining the positions from `start`.
fn with_positions(run: Lsqml, start: &[[f64; 2]]) -> Lsqml {
    run.with_features(Features {
        position: Some(start.to_vec()),
        ..Features::default()
    })
    .unwrap()
}

fn assert_close(what: &str, got: &[f64], want: &[f64], tol: f64) {
    let err = max_rel(got, want);
    eprintln!("{what}: max rel {err:.3e}");
    assert!(err < tol, "{what}: max rel {err:.3e}");
}

/// The offset changes `run` made from `start` against the reference's columns `c, c + 1`.
fn assert_offsets(what: &str, run: &Lsqml, rows: &[Vec<f64>], c: usize, tol: f64) {
    let (start, want) = (offsets(rows, 1), offsets(rows, c));
    let got = run.features().position.as_ref().unwrap();
    let delta = |d: &[[f64; 2]]| -> Vec<f64> {
        d.iter()
            .zip(&start)
            .flat_map(|(d, s)| [d[0] - s[0], d[1] - s[1]])
            .collect()
    };
    assert_close(&format!("{what} offsets"), &delta(got), &delta(&want), tol);
}

/// The object and probe changes against the reference's tables, `noise` 0 Gaussian, 1 Poisson.
fn assert_obj_probe(what: &str, d_obj: &[f64], d_probe: &[f64], files: [&str; 2], noise: usize) {
    for (name, text, lead, got) in [
        ("object", files[0], 2, d_obj),
        ("probe", files[1], 3, d_probe),
    ] {
        let c = lead + 2 * noise;
        let want: Vec<f64> = table(text).iter().flat_map(|r| [r[c], r[c + 1]]).collect();
        assert_close(&format!("{what} {name}"), got, &want, 1e-12);
    }
}

/// `a_far a_o a_p` of `s` against the reference's rows from column `col`.
fn assert_steps(what: &str, s: &LsqmlStep, rows: &[Vec<f64>], col: usize) {
    for (name, got, c, tol) in [
        ("a_far", &s.a_far, col, 1e-13),
        ("a_o", &s.a_obj[0], col + 1, 1e-12),
        ("a_p", &s.a_probe, col + 2, 1e-12),
    ] {
        let want: Vec<f64> = rows.iter().map(|r| r[c]).collect();
        assert_close(&format!("{what} {name}"), got, &want, tol);
    }
}

/// The reference's eigenmodes `U` and weights `w`, columns `u` and `w` of its tables.
fn variation(u: usize, w: usize) -> (Stack<C64>, Vec<f64>) {
    let eigen = table(include_str!("engine/data/lsqml_pos_opr_eigen_f64.txt"));
    let u = eigen.iter().map(|r| C64::new(r[u], r[u + 1])).collect();
    let weights = table(include_str!("engine/data/lsqml_pos_opr_weights_f64.txt"));
    let w = weights.iter().flat_map(|r| r[w..w + 3].to_vec()).collect();
    (Stack::from_vec([3, 32, 32], u).unwrap(), w)
}

/// One [`Lsqml::step`] from the reference's offsets on its batch, with the illumination of
/// every view at the shared probe as preconditioner, with three eigenmodes of probe mode 0 when
/// `opr`: the step sizes, object and probe changes, offsets and eigenmodes and weights after it.
fn assert_step_ref(cost: Cost, noise: usize, opr: bool) {
    let (p, optics) = state();
    let (steps, offset_rows, files) = if opr {
        (
            include_str!("engine/data/lsqml_pos_opr_steps_f64.txt"),
            include_str!("engine/data/lsqml_pos_opr_offsets_f64.txt"),
            [
                include_str!("engine/data/lsqml_pos_opr_obj_f64.txt"),
                include_str!("engine/data/lsqml_pos_opr_probe_f64.txt"),
            ],
        )
    } else {
        (
            include_str!("engine/data/lsqml_pos_steps_f64.txt"),
            include_str!("engine/data/lsqml_pos_offsets_f64.txt"),
            [
                include_str!("engine/data/lsqml_pos_obj_f64.txt"),
                include_str!("engine/data/lsqml_pos_probe_f64.txt"),
            ],
        )
    };
    let (rows, offset_rows) = (table(steps), table(offset_rows));
    let frames: Vec<usize> = rows.iter().map(|r| r[0] as usize).collect();
    let mut var = if opr {
        let (u, w) = variation(3, 1);
        ProbeVariation::new(u, w)
    } else {
        ProbeVariation::none([32, 32])
    };
    let (mut object, mut probe) = (p.object().clone(), p.probe().clone());
    let pre = illumination(p.addr().views(), object.dims(), &probe);
    let params = LsqmlParams {
        batch_size: frames.len(),
        cost,
        seed: 0,
    };
    let run = Lsqml::new(params).with_preconditioner(pre);
    let mut run = with_positions(run, &offsets(&offset_rows, 1));
    let s = run.step(
        &views(&p, &frames),
        p.observed(),
        &optics,
        slice::from_mut(&mut object),
        &mut probe,
        &mut var,
    );
    assert_eq!(s.frames, frames);
    let what = format!("{cost:?}{}", if opr { " OPR" } else { "" });
    assert_steps(&what, &s, &rows, 1 + 3 * noise);
    let (d_obj, d_probe) = (change(&object, p.object()), change(&probe, p.probe()));
    assert_obj_probe(&what, &d_obj, &d_probe, files, noise);
    assert_offsets(&what, &run, &offset_rows, 3 + 2 * noise, 1e-12);
    if opr {
        let (_, w0) = variation(3, 1);
        let (u, w) = variation(5 + 2 * noise, 4 + 3 * noise);
        let eigen = (re_im(var.eigenmodes()), re_im(&u));
        assert_close(&format!("{what} eigenmodes"), &eigen.0, &eigen.1, 1e-13);
        let moved = |x: &[f64]| x.iter().zip(&w0).map(|(x, w)| x - w).collect::<Vec<_>>();
        assert_close(
            &format!("{what} weight change"),
            &moved(var.weights()),
            &moved(&w),
            1e-13,
        );
    }
}

#[test]
fn gaussian_step_matches_lsqml_pos_ref() {
    assert_step_ref(Cost::Amplitude, 0, false);
}

#[test]
fn poisson_step_matches_lsqml_pos_ref() {
    assert_step_ref(Cost::Poisson, 1, false);
}

#[test]
fn gaussian_opr_step_matches_lsqml_pos_ref() {
    assert_step_ref(Cost::Amplitude, 0, true);
}

#[test]
fn poisson_opr_step_matches_lsqml_pos_ref() {
    assert_step_ref(Cost::Poisson, 1, true);
}

/// Three [`Lsqml::epoch`]s of batches of 16 (seed 7) from the reference's offsets against
/// pty-chi's own run: every step's `epoch frame a_far a_o a_p`, then the object, probe and
/// offset changes over the run.
fn assert_epochs_ref(cost: Cost, noise: usize) {
    let (p, optics) = state();
    let (addr, obs) = (p.addr(), p.observed());
    let (mut object, mut pr) = (p.object().clone(), p.probe().clone());
    let mut var = ProbeVariation::none(pr.plane_shape());
    let offset_rows = table(include_str!("engine/data/lsqml_pos_epochs_offsets_f64.txt"));
    let run = Lsqml::new(LsqmlParams {
        batch_size: 16,
        cost,
        seed: 7,
    });
    let mut run = with_positions(run, &offsets(&offset_rows, 1));
    let rows = table(include_str!("engine/data/lsqml_pos_epochs_steps_f64.txt"));
    let mut next = 0;
    for e in 0..3 {
        for s in run.epoch(
            addr,
            obs,
            &optics,
            slice::from_mut(&mut object),
            &mut pr,
            &mut var,
        ) {
            let want = &rows[next..next + s.frames.len()];
            next += s.frames.len();
            let frames: Vec<usize> = want.iter().map(|r| r[1] as usize).collect();
            assert!(want.iter().all(|r| r[0] == e as f64), "epoch {e}");
            assert_eq!(s.frames, frames, "epoch {e}");
            assert_steps(&format!("{cost:?} epoch {e}"), &s, want, 2 + 3 * noise);
        }
    }
    assert_eq!(next, rows.len());
    let files = [
        include_str!("engine/data/lsqml_pos_epochs_obj_f64.txt"),
        include_str!("engine/data/lsqml_pos_epochs_probe_f64.txt"),
    ];
    let (d_obj, d_probe) = (change(&object, p.object()), change(&pr, p.probe()));
    let what = format!("{cost:?} epochs");
    assert_obj_probe(&what, &d_obj, &d_probe, files, noise);
    assert_offsets(&what, &run, &offset_rows, 3 + 2 * noise, 1e-11);
}

#[test]
fn gaussian_epochs_match_ptychi_with_positions() {
    assert_epochs_ref(Cost::Amplitude, 0);
}

#[test]
fn poisson_epochs_match_ptychi_with_positions() {
    assert_epochs_ref(Cost::Poisson, 1);
}

/// pty-chi's LSQML has no per-frame intensity scale or detector background; offsets must be
/// finite.
#[test]
fn with_features_refuses_what_ptychi_lsqml_lacks() {
    let run = || {
        Lsqml::new(LsqmlParams {
            batch_size: 4,
            cost: Cost::Poisson,
            seed: 0,
        })
    };
    let refused = [
        Features {
            scale: Some(vec![1.0; 40]),
            ..Features::default()
        },
        Features {
            background: Some(vec![0.0; 32 * 32]),
            ..Features::default()
        },
        Features {
            position: Some(vec![[0.0, f64::NAN]; 40]),
            ..Features::default()
        },
    ];
    for f in refused {
        assert!(run().with_features(f.clone()).is_err(), "{f:?}");
    }
    let f = Features {
        position: Some(vec![[0.25, -0.25]; 40]),
        ..Features::default()
    };
    assert_eq!(run().with_features(f.clone()).unwrap().features(), &f);
}
