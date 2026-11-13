//! ML with [`Features::position`] against the f64 model of its step, `refs/f64/ml_pos.py`
//! (no package refines positions this way), run on the state it reads: `cargo run --release
//! -p ptyrs --example engine_llk_trace -- <dir> dm 1 2 32 40 4`, i.e.
//! `common::dm_scenario(32, 40, 2, 2e5, 4.0)` after `scale_obj_probe`, with the model's
//! offsets. It writes `<dir>/ml_pos_*_f64.txt`, saved to `tests/engine/data`.

use ptyrs::base::C64;
use ptyrs::engine::{Features, MlParams, Ptycho};

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

/// Four cycles updating object and probe from offsets within 0.4.
#[test]
fn ml_positions_match_the_model() {
    let (scan, object, probe) = common::dm_scenario(32, 40, 2, 2e5, 4.0);
    let mut p = Ptycho::new(&scan, object, probe).unwrap();
    p.scale_obj_probe();
    let rows = table(include_str!("engine/data/ml_pos_offsets_f64.txt"));
    let start = offsets(&rows, 1);
    p.set_features(Features {
        position: Some(start.clone()),
        ..Features::default()
    })
    .unwrap();
    let (o0, p0) = (p.object().clone(), p.probe().clone());
    let params = MlParams {
        update_object: true,
        update_probe: true,
        calc_llk: 1,
        zero_phase_ramp: false,
    };
    p.ml(&params, 4);

    // Against the start arrays, not the changes: ML's opening rescale sums in sequence, a
    // relative roundoff near 1e-14 of the whole probe, while the probe moves by 0.2% a cycle.
    for (name, text, lead, x0, x) in [
        (
            "object",
            include_str!("engine/data/ml_pos_obj_f64.txt"),
            2,
            o0.as_slice(),
            p.object().as_slice(),
        ),
        (
            "probe",
            include_str!("engine/data/ml_pos_probe_f64.txt"),
            3,
            p0.as_slice(),
            p.probe().as_slice(),
        ),
    ] {
        let want: Vec<f64> = table(text)
            .iter()
            .flat_map(|r| [r[lead], r[lead + 1]])
            .collect();
        let start: Vec<f64> = x0.iter().flat_map(|c| [c.re, c.im]).collect();
        assert_close(name, &change(x, x0), &want, max_abs(&start), 1e-13);
    }
    let delta = |d: &[[f64; 2]]| -> Vec<f64> {
        d.iter()
            .zip(&start)
            .flat_map(|(d, s)| [d[0] - s[0], d[1] - s[1]])
            .collect()
    };
    let (got, want) = (p.features().position.as_ref().unwrap(), offsets(&rows, 3));
    let want = delta(&want);
    assert_close("offsets", &delta(got), &want, max_abs(&want), 1e-12);
}
