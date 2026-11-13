//! rPIE with [`Features::position`] against pty-chi's `RPIEReconstructor` in f64 with the
//! positions optimizable, `refs/f64/rpie_pos.py`, run on the state it reads: `cargo run
//! --release -p ptyrs --example engine_llk_trace -- <dir> dm 1 2 32 40 4`, i.e.
//! `common::dm_scenario(32, 40, 2, 2e5, 4.0)` after `scale_obj_probe`, with the reference's
//! offsets. It writes `<dir>/rpie_pos_*_f64.txt`, saved to `tests/data`.

use ptyrs::base::{C64, Stack};
use ptyrs::engine::{Features, Ptycho, RpieParams};

#[path = "common/engine/mod.rs"]
mod common;

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

/// `re_im(a − b)`.
fn change(a: &Stack<C64>, b: &Stack<C64>) -> Vec<f64> {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .flat_map(|(a, b)| [a.re - b.re, a.im - b.im])
        .collect()
}

/// Columns `c, c + 1` of an offsets table, `[dy, dx]` per frame.
fn offsets(rows: &[Vec<f64>], c: usize) -> Vec<[f64; 2]> {
    rows.iter().map(|r| [r[c], r[c + 1]]).collect()
}

fn assert_close(what: &str, got: &[f64], want: &[f64], tol: f64) {
    let err = max_rel(got, want);
    eprintln!("{what}: max rel {err:.3e}");
    assert!(err < tol, "{what}: max rel {err:.3e}");
}

/// `nepoch` epochs of `params` from the reference's starting offsets against its object,
/// probe and offset changes, `files` its obj, probe and offsets tables.
fn assert_run(what: &str, params: RpieParams, nepoch: usize, files: [&str; 3]) {
    let (scan, object, probe) = common::dm_scenario(32, 40, 2, 2e5, 4.0);
    let mut p = Ptycho::new(&scan, object, probe).unwrap();
    p.scale_obj_probe();
    let rows = table(files[2]);
    let start = offsets(&rows, 1);
    p.set_features(Features {
        position: Some(start.clone()),
        ..Features::default()
    })
    .unwrap();
    let (o0, p0) = (p.object().clone(), p.probe().clone());
    p.rpie(&params, nepoch).unwrap();

    for (name, text, lead, got) in [
        ("object", files[0], 2, change(p.object(), &o0)),
        ("probe", files[1], 3, change(p.probe(), &p0)),
    ] {
        let want: Vec<f64> = table(text)
            .iter()
            .flat_map(|r| [r[lead], r[lead + 1]])
            .collect();
        assert_close(&format!("{what} {name}"), &got, &want, 1e-12);
    }
    let delta = |d: &[[f64; 2]]| -> Vec<f64> {
        d.iter()
            .zip(&start)
            .flat_map(|(d, s)| [d[0] - s[0], d[1] - s[1]])
            .collect()
    };
    let got = p.features().position.as_ref().unwrap();
    let want = offsets(&rows, 3);
    assert_close(
        &format!("{what} offsets"),
        &delta(got),
        &delta(&want),
        1e-12,
    );
}

/// One epoch of two batches of 20 from offsets within 0.4.
#[test]
fn rpie_positions_step_matches_ptychi() {
    let params = RpieParams {
        obj_step: Some(0.9),
        probe_step: Some(0.8),
        obj_alpha: 0.3,
        probe_alpha: 0.2,
        batch_size: 20,
        calc_llk: 1,
    };
    assert_run(
        "step",
        params,
        1,
        [
            include_str!("engine/data/rpie_pos_step_obj_f64.txt"),
            include_str!("engine/data/rpie_pos_step_probe_f64.txt"),
            include_str!("engine/data/rpie_pos_step_offsets_f64.txt"),
        ],
    );
}

/// Three epochs of ten batches of 4 from offsets within 0.15.
#[test]
fn rpie_positions_epochs_match_ptychi() {
    let params = RpieParams {
        obj_step: Some(1.0),
        probe_step: Some(1.0),
        obj_alpha: 0.5,
        probe_alpha: 0.5,
        batch_size: 4,
        calc_llk: 1,
    };
    assert_run(
        "epochs",
        params,
        3,
        [
            include_str!("engine/data/rpie_pos_epochs_obj_f64.txt"),
            include_str!("engine/data/rpie_pos_epochs_probe_f64.txt"),
            include_str!("engine/data/rpie_pos_epochs_offsets_f64.txt"),
        ],
    );
}
