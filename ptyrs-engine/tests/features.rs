//! The feature operators against the float64 references in `refs/f64`, run on the state they
//! read: `cargo run --release -p ptyrs-engine --example llk_trace -- <dir> dm 1 2 32 40 4`,
//! i.e. `common::dm_scenario(32, 40, 2, 2e5, 4.0)` after `scale_obj_probe`. Each reference
//! writes `<dir>/<name>_f64.txt` and `<dir>/<name>_values_f64.txt` (`nb_obs L F`), saved to
//! `tests/data`.

use ptyrs_core::{C64, Scan};
use ptyrs_engine::{
    Cost, Fft2, Ptycho, intensity_grad, llk, obj_probe_shift_to_psi, obj_probe_to_psi,
    position_grad, quad_phase, scale_psi,
};

pub mod common;

fn state() -> (Scan, Ptycho) {
    let (scan, object, probe) = common::dm_scenario(32, 40, 2, 2e5, 4.0);
    let mut p = Ptycho::new(&scan, object, probe).unwrap();
    p.scale_obj_probe();
    (scan, p)
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

/// `nb_obs L F` of a values file against the state's count and the engine's Poisson and
/// Euclidian sums of the far field `psi` (`F` is a quarter of the Euclidian sum).
fn assert_values(p: &Ptycho, psi: &[C64], values: &str) {
    let v = &table(values)[0];
    assert_eq!(p.observed().nb_obs() as f64, v[0]);
    let l = llk(p.observed(), 0..p.addr().nframe(), p.addr().nmode(), psi);
    let rel = |a: f64, b: f64| (a - b).abs() / b.abs();
    assert!(rel(l.poisson, v[1]) < 1e-12, "L {} ref {}", l.poisson, v[1]);
    assert!(
        rel(l.euclidian / 4.0, v[2]) < 1e-12,
        "F {} ref {}",
        l.euclidian / 4.0,
        v[2]
    );
}

/// `refs/f64/pos_grad.py`: rows `k s_y s_x dL/ds_y dL/ds_x dF/ds_y dF/ds_x`, `s` the window
/// start in object pixels, whose rounding is the view corner.
#[test]
fn position_matches_pos_grad_ref() {
    let (scan, p) = state();
    let rows = table(include_str!("data/pos_grad_f64.txt"));
    let offsets: Vec<[f64; 2]> = rows
        .iter()
        .enumerate()
        .map(|(f, r)| {
            let v = p.addr().frame_views(f)[0];
            let d = [r[1] - v.oy as f64, r[2] - v.ox as f64];
            assert!(
                d[0].abs() < 0.5 && d[1].abs() < 0.5,
                "frame {f} offset {d:?}"
            );
            d
        })
        .collect();
    let views = p.addr().views();
    let quad = quad_phase(&scan.geometry);
    let fft = Fft2::new(scan.geometry.frame);
    let mut psi = vec![C64::default(); views.len() * p.probe().plane_len()];

    // zero offsets: the unshifted operator
    let mut plain = psi.clone();
    obj_probe_to_psi(views, p.object(), p.probe(), &quad, &mut plain);
    let zero = vec![[0.0; 2]; offsets.len()];
    obj_probe_shift_to_psi(views, p.object(), p.probe(), &quad, &zero, &mut psi);
    let d = psi
        .iter()
        .zip(&plain)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    let m = plain.iter().map(|b| b.norm()).fold(0.0, f64::max);
    assert!(d / m < 1e-13, "zero offsets: max rel {}", d / m);

    obj_probe_shift_to_psi(views, p.object(), p.probe(), &quad, &offsets, &mut psi);
    fft.forward(&mut psi);
    assert_values(&p, &psi, include_str!("data/pos_grad_values_f64.txt"));

    for (cost, col) in [(Cost::Poisson, 3), (Cost::Amplitude, 5)] {
        let g = position_grad(
            views,
            p.object(),
            p.probe(),
            &quad,
            p.observed(),
            &offsets,
            cost,
        );
        let got: Vec<f64> = g.iter().flatten().copied().collect();
        let want: Vec<f64> = rows.iter().flat_map(|r| [r[col], r[col + 1]]).collect();
        let e = max_rel(&got, &want);
        assert!(e < 1e-12, "{cost:?}: max rel {e:.3e}");
    }
}

/// Far field of the state's object and probe at the view corners.
fn far_field(scan: &Scan, p: &Ptycho) -> Vec<C64> {
    let views = p.addr().views();
    let mut psi = vec![C64::default(); views.len() * p.probe().plane_len()];
    obj_probe_to_psi(
        views,
        p.object(),
        p.probe(),
        &quad_phase(&scan.geometry),
        &mut psi,
    );
    Fft2::new(scan.geometry.frame).forward(&mut psi);
    psi
}

/// `refs/f64/intensity_step.py`: rows `k s dL/ds dF/ds s'_gaussian s'_poisson a_far_poisson`,
/// `s` scaling every mode's intensity.
#[test]
fn intensity_matches_intensity_step_ref() {
    let (scan, p) = state();
    let rows = table(include_str!("data/intensity_f64.txt"));
    let scale: Vec<f64> = rows.iter().map(|r| r[1]).collect();
    let (nframe, nmode) = (p.addr().nframe(), p.addr().nmode());
    let phi = far_field(&scan, &p);
    let mut psi = phi.clone();
    scale_psi(p.observed(), 0..nframe, nmode, &scale, &mut psi);
    assert_values(&p, &psi, include_str!("data/intensity_values_f64.txt"));

    for (cost, col) in [(Cost::Poisson, 2), (Cost::Amplitude, 3)] {
        let g = intensity_grad(p.observed(), 0..nframe, nmode, &scale, &phi, cost);
        let want: Vec<f64> = rows.iter().map(|r| r[col]).collect();
        let e = max_rel(&g, &want);
        assert!(e < 1e-12, "{cost:?}: max rel {e:.3e}");
    }
    // a batch leaves the other frames at zero
    let n = nmode * p.probe().plane_len();
    let g = intensity_grad(
        p.observed(),
        20..30,
        nmode,
        &scale,
        &phi[20 * n..30 * n],
        Cost::Poisson,
    );
    let full = intensity_grad(p.observed(), 0..nframe, nmode, &scale, &phi, Cost::Poisson);
    assert!(g[..20].iter().chain(&g[30..]).all(|&v| v == 0.0));
    assert_eq!(g[20..30], full[20..30]);
}
