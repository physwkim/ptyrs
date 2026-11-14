//! rPIE with [`Features::position`] against pty-chi's `RPIEReconstructor` in f64 with the
//! positions optimizable, `refs/f64/rpie_pos.py`, run on the state it reads: `cargo run
//! --release -p ptyrs --example engine_llk_trace -- <dir> dm 1 2 32 40 4`, i.e.
//! `common::dm_scenario(32, 40, 2, 2e5, 4.0)` after `scale_obj_probe`, with the reference's
//! offsets. It writes `<dir>/rpie_pos_*_f64.txt`, saved to `tests/data`. The same cases under
//! pty-chi's FULL_OBJECT shift operator, [`Sampling::FullObject`] here, are
//! `refs/py/ptychi_full_object.py`'s `<out>/rpie_full_object_*_f64.txt`, saved to
//! `tests/engine/data`, with one on an odd-sided object plane.

use ptyrs::base::{C64, Object, Stack};
use ptyrs::engine::{Features, Ptycho, RpieParams, Sampling};

#[path = "common/engine/mod.rs"]
mod common;
#[path = "common/worst.rs"]
mod worst;
use worst::worst_of;

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

/// The scenario after `scale_obj_probe`; with `odd`, on its 76 x 80 object plane less the first
/// row and column, both sides odd, every window one pixel nearer the origin and so on the same
/// pixels.
fn scenario(odd: bool) -> Ptycho {
    let (scan, object, probe) = common::dm_scenario(32, 40, 2, 2e5, 4.0);
    let mut p = Ptycho::new(&scan, object, probe).unwrap();
    p.scale_obj_probe();
    if !odd {
        return p;
    }
    let [_, ny, nx] = p.object().dims();
    let plane = (p.object().plane(0).chunks(nx).skip(1))
        .flat_map(|row| &row[1..])
        .copied()
        .collect();
    let object = Object::new(Stack::from_vec([1, ny - 1, nx - 1], plane).unwrap());
    Ptycho::new(&scan, object, p.probe().clone()).unwrap()
}

/// `nepoch` epochs of `params` on `p` under `sampling` from the reference's starting offsets
/// against its object, probe and offset changes, `files` its obj, probe and offsets tables.
fn assert_run(
    what: &str,
    mut p: Ptycho,
    sampling: Sampling,
    params: RpieParams,
    nepoch: usize,
    files: [&str; 3],
) {
    p.set_sampling(sampling);
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
        // 5x the worst measured, the PATCH step's (5.1e-13): pty-chi's absolute positions cost
        // the change digits
        3e-12,
    );
}

/// One epoch of two batches of 20 from offsets within 0.4.
const STEP: RpieParams = RpieParams {
    obj_step: Some(0.9),
    probe_step: Some(0.8),
    obj_alpha: 0.3,
    probe_alpha: 0.2,
    batch_size: 20,
    calc_llk: 1,
};

/// Three epochs of ten batches of 4 from offsets within 0.15.
const EPOCHS: RpieParams = RpieParams {
    obj_step: Some(1.0),
    probe_step: Some(1.0),
    obj_alpha: 0.5,
    probe_alpha: 0.5,
    batch_size: 4,
    calc_llk: 1,
};

/// [`STEP`] under PATCH.
#[test]
fn rpie_positions_step_matches_ptychi() {
    assert_run(
        "step",
        scenario(false),
        Sampling::Patch,
        STEP,
        1,
        [
            include_str!("engine/data/rpie_pos_step_obj_f64.txt"),
            include_str!("engine/data/rpie_pos_step_probe_f64.txt"),
            include_str!("engine/data/rpie_pos_step_offsets_f64.txt"),
        ],
    );
}

/// [`EPOCHS`] under PATCH.
#[test]
fn rpie_positions_epochs_match_ptychi() {
    assert_run(
        "epochs",
        scenario(false),
        Sampling::Patch,
        EPOCHS,
        3,
        [
            include_str!("engine/data/rpie_pos_epochs_obj_f64.txt"),
            include_str!("engine/data/rpie_pos_epochs_probe_f64.txt"),
            include_str!("engine/data/rpie_pos_epochs_offsets_f64.txt"),
        ],
    );
}

/// [`STEP`] under FULL_OBJECT.
#[test]
fn rpie_full_object_positions_step_matches_ptychi() {
    assert_run(
        "full-object step",
        scenario(false),
        Sampling::FullObject,
        STEP,
        1,
        [
            include_str!("engine/data/rpie_full_object_step_obj_f64.txt"),
            include_str!("engine/data/rpie_full_object_step_probe_f64.txt"),
            include_str!("engine/data/rpie_full_object_step_offsets_f64.txt"),
        ],
    );
}

/// [`EPOCHS`] under FULL_OBJECT.
#[test]
fn rpie_full_object_positions_epochs_match_ptychi() {
    assert_run(
        "full-object epochs",
        scenario(false),
        Sampling::FullObject,
        EPOCHS,
        3,
        [
            include_str!("engine/data/rpie_full_object_epochs_obj_f64.txt"),
            include_str!("engine/data/rpie_full_object_epochs_probe_f64.txt"),
            include_str!("engine/data/rpie_full_object_epochs_offsets_f64.txt"),
        ],
    );
}

/// [`STEP`] under FULL_OBJECT on the 75 x 79 plane, whose middle frequencies are positive.
#[test]
fn rpie_full_object_odd_plane_matches_ptychi() {
    assert_run(
        "full-object odd plane",
        scenario(true),
        Sampling::FullObject,
        STEP,
        1,
        [
            include_str!("engine/data/rpie_full_object_odd_obj_f64.txt"),
            include_str!("engine/data/rpie_full_object_odd_probe_f64.txt"),
            include_str!("engine/data/rpie_full_object_odd_offsets_f64.txt"),
        ],
    );
}
