//! pty-chi parity of the free-space propagators: `refs/f64/propagators.py` (pty-chi
//! `propagate.py` in float64) writes `propagator_{select,cases,fields}_f64.txt`, saved to
//! `tests/data`.
//!
//! pty-chi's propagator is `√k·U` and its backward `Uᴴ/√k` over natural (centred) planes; a
//! [`Propagator`] is `U` over shifted ones. So with `i = shift(j)` on the object plane,
//! `adjoint(y)[j] = √k·bwd[i]`, and on the detector plane `forward(x)[j] = fwd[i]/√k`. The
//! exception is Fourier: pty-chi transforms the natural plane into the corner-origin layout,
//! which is already the engine's, so there the detector index is `j` itself and the engine's
//! transform of the shifted plane carries an extra `(−1)^(ky+kx)`.

use ptyrs::base::C64;
use ptyrs::engine::{Propagator, Wavefield, shift};

fn table(text: &str) -> Vec<Vec<f64>> {
    text.lines()
        .map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect())
        .collect()
}

fn kind(p: &Propagator) -> usize {
    match p {
        Propagator::Fourier(_) => 0,
        Propagator::AngularSpectrum(_) => 1,
        Propagator::Fresnel(_) => 2,
    }
}

fn wavefield(r: &[f64]) -> Wavefield {
    Wavefield::new([r[0] as usize, r[1] as usize], r[2], [r[3], r[4]], r[5]).unwrap()
}

fn rel(a: f64, b: f64) -> f64 {
    if a == b { 0.0 } else { (a - b).abs() / b.abs() }
}

#[test]
fn free_space_selection_matches_ptychi() {
    for r in table(include_str!("engine/data/propagator_select_f64.txt")) {
        let w = wavefield(&r);
        let p = Propagator::free_space(&w);
        assert_eq!(kind(&p), r[6] as usize, "kind at {r:?}");
        assert!(
            rel(w.fresnel_number(), r[7]) <= 1e-15,
            "Fresnel number at {r:?}"
        );
        assert!(rel(p.intensity_scale(), r[8]) <= 1e-15, "scale at {r:?}");
    }
}

#[test]
fn propagators_match_ptychi() {
    let fields = table(include_str!("engine/data/propagator_fields_f64.txt"));
    for c in table(include_str!("engine/data/propagator_cases_f64.txt")) {
        let w = wavefield(&c[2..8]);
        let p = if c[1] == 1.0 {
            Propagator::angular_spectrum(&w).unwrap()
        } else {
            Propagator::free_space(&w)
        };
        assert_eq!(kind(&p), c[8] as usize, "kind of case {}", c[0]);
        let k = p.intensity_scale();
        assert!(rel(k, c[9]) <= 1e-15, "scale of case {}", c[0]);

        let [ny, nx] = w.shape();
        let rows: Vec<_> = fields.iter().filter(|r| r[0] == c[0]).collect();
        assert_eq!(rows.len(), ny * nx);
        let at = |i: usize, col: usize| C64::new(rows[i][col], rows[i][col + 1]);
        // (index into pty-chi's detector plane, sign) of the engine's detector index j.
        let det = |j: usize| match p {
            Propagator::Fourier(_) => {
                let parity = j / nx + j % nx;
                (j, if parity.is_multiple_of(2) { 1.0 } else { -1.0 })
            }
            _ => (shift(j, ny, nx), 1.0),
        };
        let mut x: Vec<_> = (0..ny * nx).map(|j| at(shift(j, ny, nx), 3)).collect();
        let mut y: Vec<_> = (0..ny * nx)
            .map(|j| {
                let (i, s) = det(j);
                at(i, 5) * s
            })
            .collect();
        p.forward(&mut x);
        p.adjoint(&mut y);
        let fwd: Vec<_> = (0..ny * nx)
            .map(|j| {
                let (i, s) = det(j);
                at(i, 7) * s / k.sqrt()
            })
            .collect();
        let bwd: Vec<_> = (0..ny * nx)
            .map(|j| at(shift(j, ny, nx), 9) * k.sqrt())
            .collect();
        for (name, got, want) in [("forward", &x, &fwd), ("adjoint", &y, &bwd)] {
            let d = got
                .iter()
                .zip(want)
                .map(|(a, b)| (a - b).norm())
                .fold(0.0, f64::max);
            let m = want.iter().map(|b| b.norm()).fold(0.0, f64::max);
            assert!(d / m <= 1e-14, "case {} {name}: {:.3e}", c[0], d / m);
        }
    }
}
