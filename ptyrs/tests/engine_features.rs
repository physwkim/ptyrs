//! The feature operators against the float64 references in `refs/f64`, run on the state they
//! read: `cargo run --release -p ptyrs --example engine_llk_trace -- <dir> dm 1 2 32 40 4`,
//! i.e. `common::dm_scenario(32, 40, 2, 2e5, 4.0)` after `scale_obj_probe`. Each reference
//! writes `<dir>/<name>_f64.txt` and `<dir>/<name>_values_f64.txt` (`nb_obs L F`), saved to
//! `tests/data`.

use ptyrs::base::{C64, Scan};
use ptyrs::engine::{
    ApParams, Cost, DmParams, Features, Fft2, Ptycho, background_cost, background_grad,
    intensity_grad, llk, obj_probe_shift_to_psi, obj_probe_to_psi, position_grad, scale_psi, shift,
};

#[path = "common/engine/mod.rs"]
mod common;

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
    let rows = table(include_str!("engine/data/pos_grad_f64.txt"));
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
    let fft = Fft2::new(scan.geometry.frame);
    let mut psi = vec![C64::default(); views.len() * p.probe().plane_len()];

    // zero offsets: the unshifted operator
    let mut plain = psi.clone();
    obj_probe_to_psi(views, p.object(), p.probe(), p.quad(), &mut plain);
    let zero = vec![[0.0; 2]; offsets.len()];
    obj_probe_shift_to_psi(&p, views, &zero, &mut psi);
    let d = psi
        .iter()
        .zip(&plain)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    let m = plain.iter().map(|b| b.norm()).fold(0.0, f64::max);
    assert!(d / m < 1e-13, "zero offsets: max rel {}", d / m);

    obj_probe_shift_to_psi(&p, views, &offsets, &mut psi);
    fft.forward(&mut psi);
    assert_values(
        &p,
        &psi,
        include_str!("engine/data/pos_grad_values_f64.txt"),
    );

    for (cost, col) in [(Cost::Poisson, 3), (Cost::Amplitude, 5)] {
        let g = position_grad(&p, views, &offsets, cost);
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
    obj_probe_to_psi(views, p.object(), p.probe(), p.quad(), &mut psi);
    Fft2::new(scan.geometry.frame).forward(&mut psi);
    psi
}

/// `refs/f64/intensity_step.py`: rows `k s dL/ds dF/ds s'_gaussian s'_poisson a_far_poisson`,
/// `s` scaling every mode's intensity.
#[test]
fn intensity_matches_intensity_step_ref() {
    let (scan, p) = state();
    let rows = table(include_str!("engine/data/intensity_f64.txt"));
    let scale: Vec<f64> = rows.iter().map(|r| r[1]).collect();
    let (nframe, nmode) = (p.addr().nframe(), p.addr().nmode());
    let phi = far_field(&scan, &p);
    let mut psi = phi.clone();
    scale_psi(p.observed(), 0..nframe, nmode, &scale, &mut psi);
    assert_values(
        &p,
        &psi,
        include_str!("engine/data/intensity_values_f64.txt"),
    );

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

/// `refs/f64/background_step.py`: rows `iy ix b dL/db dF/db d2L/db2 d2F/db2 b'_ap` in detector
/// order (DC at `[ny/2, nx/2]`), one plane added to every frame.
#[test]
fn background_matches_background_step_ref() {
    let (scan, p) = state();
    let rows = table(include_str!("engine/data/background_f64.txt"));
    let [ny, nx] = scan.geometry.frame;
    assert_eq!(rows.len(), ny * nx);
    assert!(
        rows.iter()
            .enumerate()
            .all(|(i, r)| r[0] == (i / nx) as f64 && r[1] == (i % nx) as f64)
    );
    let shifted =
        |col: usize| -> Vec<f64> { (0..ny * nx).map(|j| rows[shift(j, ny, nx)][col]).collect() };
    let b = shifted(2);
    let (nframe, nmode) = (p.addr().nframe(), p.addr().nmode());
    let obs = p.observed();
    let psi = far_field(&scan, &p);

    // no background: the engine's sums
    let zero = vec![0.0; ny * nx];
    let l = llk(obs, 0..nframe, nmode, &psi);
    let rel = |a: f64, b: f64| (a - b).abs() / b.abs();
    let l0 = background_cost(obs, 0..nframe, nmode, &psi, &zero, Cost::Poisson);
    let f0 = background_cost(obs, 0..nframe, nmode, &psi, &zero, Cost::Amplitude);
    assert!(rel(l0, l.poisson) < 1e-13, "L {l0} llk {}", l.poisson);
    assert!(
        rel(f0, l.euclidian / 4.0) < 1e-13,
        "F {f0} llk {}",
        l.euclidian / 4.0
    );

    let v = &table(include_str!("engine/data/background_values_f64.txt"))[0];
    assert_eq!(obs.nb_obs() as f64, v[0]);
    for (cost, value, col) in [(Cost::Poisson, v[1], 3), (Cost::Amplitude, v[2], 4)] {
        let c = background_cost(obs, 0..nframe, nmode, &psi, &b, cost);
        assert!(rel(c, value) < 1e-12, "{cost:?}: {c} ref {value}");
        let g = background_grad(obs, 0..nframe, nmode, &psi, &b, cost);
        let e = max_rel(&g, &shifted(col));
        assert!(e < 1e-12, "{cost:?}: max rel {e:.3e}");
    }
}

/// AP and DM from the truth's object and probe on counts drawn at the view corners, the probe
/// moved by zero-mean offsets of up to 0.4 pixels: over 100 cycles the offsets' rms falls at
/// every tenth cycle and below 0.65 of the start's (0.60 for AP and 0.40 for DM when written),
/// their mean stays 0, and the LLK falls.
#[test]
fn position_refines_toward_the_drawn_positions() {
    let (sim, object, probe) = common::scenario::truth(32, 40, 1, 1e6, 4.0);
    let nframe = sim.scan.positions.len();
    let mut d0: Vec<[f64; 2]> = (0..nframe)
        .map(|f| {
            let f = f as f64;
            [0.4 * (1.7 * f + 0.3).sin(), 0.4 * (2.3 * f).cos()]
        })
        .collect();
    for axis in 0..2 {
        let mean = d0.iter().map(|d| d[axis]).sum::<f64>() / nframe as f64;
        d0.iter_mut().for_each(|d| d[axis] -= mean);
    }
    let rms = |d: &[[f64; 2]]| {
        (d.iter().flatten().map(|v| v * v).sum::<f64>() / (2 * d.len()) as f64).sqrt()
    };
    for algo in ["ap", "dm"] {
        let mut p = Ptycho::new(&sim.scan, object.clone(), probe.clone()).unwrap();
        p.set_features(Features {
            position: Some(d0.clone()),
            ..Features::default()
        })
        .unwrap();
        p.scale_obj_probe();
        let l0 = p.llk().poisson;
        let mut last = rms(&d0);
        for _ in 0..10 {
            match algo {
                "ap" => p.ap(&ApParams::default(), 10),
                _ => p.dm(&DmParams::default(), 10),
            }
            let r = rms(p.features().position.as_deref().unwrap());
            println!("{algo} cycle {}: rms offset {r:.4}", p.cycle());
            assert!(
                r < last,
                "{algo} cycle {}: rms offset {r} from {last}",
                p.cycle()
            );
            last = r;
        }
        let d = p.features().position.as_deref().unwrap();
        let mean = [0, 1].map(|a| d.iter().map(|d| d[a]).sum::<f64>() / nframe as f64);
        assert!(
            mean.iter().all(|m| m.abs() < 1e-12),
            "{algo}: mean offset {mean:?}"
        );
        assert!(
            rms(d) < 0.65 * rms(&d0),
            "{algo}: rms offset {} from {}",
            rms(d),
            rms(&d0)
        );
        assert!(
            p.llk().poisson < l0,
            "{algo}: llk {} from {l0}",
            p.llk().poisson
        );
    }
}
