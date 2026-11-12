//! BH in pty-chi's planar near-field model against `refs/f64/bh_near.py`, which runs pty-chi's
//! BH on `refs/f64/lsqml_near.py`'s state and counts, writing `bh_near_*_f64.txt`, saved to
//! `tests/data`. Cases: 0 an angular-spectrum propagator, 1 a Fresnel transform (pty-chi run
//! with the intensity scale on its residual), 2 an angular spectrum with an evanescent band.

use std::slice;

use ptyrs_core::{AddrTable, C64, Counts, Object, Positions, Probe, Stack};
use ptyrs_engine::{Bh, BhParams, Observed, Optics, Propagator, Wavefield};

const SHAPE: [usize; 2] = [24, 24];
const OBJ: [usize; 2] = [40, 40];
const NPROBE: usize = 2;
const EPOCHS: usize = 2;
/// Method 0: CG over one batch of all 16 frames; 1: steepest descent over batches of 6.
const BATCH: [usize; 2] = [16, 6];

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

fn stack(dims: [usize; 3], rows: &[Vec<f64>], lead: usize) -> Stack<C64> {
    let data = rows
        .iter()
        .map(|r| C64::new(r[lead], r[lead + 1]))
        .collect();
    Stack::from_vec(dims, data).unwrap()
}

/// The start object and probe, the frames' table and per case its optics and counts.
fn state() -> (AddrTable, Object, Probe, Vec<(Optics, Observed)>) {
    // corner = (OBJ − SHAPE)/2 + position at a pixel of 1
    let corners = table(include_str!("data/lsqml_near_corners_f64.txt"));
    let at = |axis: usize| -> Vec<f64> {
        let centre = ((OBJ[axis] - SHAPE[axis]) / 2) as f64;
        corners.iter().map(|r| r[1 + axis] - centre).collect()
    };
    let positions = Positions::new(at(0), at(1)).unwrap();
    let addr = AddrTable::new(&positions, [1.0, 1.0], SHAPE, OBJ, NPROBE, 1).unwrap();
    for (f, r) in corners.iter().enumerate() {
        let v = addr.frame_views(f)[0];
        assert_eq!([v.oy as f64, v.ox as f64], [r[1], r[2]], "frame {f}");
    }
    let object = Object::new(stack(
        [1, OBJ[0], OBJ[1]],
        &table(include_str!("data/lsqml_near_obj_f64.txt")),
        2,
    ));
    let probe = Probe::new(stack(
        [NPROBE, SHAPE[0], SHAPE[1]],
        &table(include_str!("data/lsqml_near_probe_f64.txt")),
        3,
    ));
    let counts = table(include_str!("data/lsqml_near_counts_f64.txt"));
    let cases = table(include_str!("data/lsqml_near_cases_f64.txt"))
        .iter()
        .map(|c| {
            let w = Wavefield::new(SHAPE, c[1], [c[2], c[2]], c[3]).unwrap();
            let optics = Optics::free_space(&w);
            let kind = match optics.propagator() {
                Propagator::Fourier(_) => 0,
                Propagator::AngularSpectrum(_) => 1,
                Propagator::Fresnel(_) => 2,
            };
            assert_eq!(kind, c[4] as usize, "kind of case {}", c[0]);
            let values: Vec<f64> = counts
                .iter()
                .filter(|r| r[0] == c[0])
                .flat_map(|r| r[2..].iter().copied())
                .collect();
            let dims = [corners.len(), SHAPE[0], SHAPE[1]];
            let obs = Observed::from_counts(&Counts::from_f64(dims, &values, None).unwrap());
            (optics, obs)
        })
        .collect();
    (addr, object, probe, cases)
}

/// Every case and method: each step's frames, `beta`, `alpha`, `top` and `bottom`, and the
/// object and probe changes over the run.
#[test]
fn bh_matches_ptychi_in_near_field() {
    let (addr, object0, probe0, cases) = state();
    let steps = table(include_str!("data/bh_near_steps_f64.txt"));
    let objs = table(include_str!("data/bh_near_obj_f64.txt"));
    let probes = table(include_str!("data/bh_near_probe_f64.txt"));
    for (case, (optics, obs)) in cases.iter().enumerate() {
        for (method, &batch_size) in BATCH.iter().enumerate() {
            let at = |r: &&Vec<f64>| r[0] as usize == case && r[1] as usize == method;
            let mut bh = Bh::new(BhParams {
                batch_size,
                probe_rho: 1.0,
                seed: 7,
            });
            let (mut object, mut probe) = (object0.clone(), probe0.clone());
            let got: Vec<_> = (0..EPOCHS)
                .flat_map(|e| {
                    let objects = slice::from_mut(&mut object);
                    let s = bh.epoch(&addr, obs, optics, objects, &mut probe);
                    s.into_iter().map(move |s| (e, s))
                })
                .collect();
            let rows: Vec<_> = steps.iter().filter(at).collect();
            assert_eq!(got.len(), rows.len(), "case {case} method {method}");
            for (k, ((epoch, s), r)) in got.iter().zip(&rows).enumerate() {
                let at = format!("case {case} method {method} step {k}");
                let count = r[4] as usize;
                assert_eq!([k, *epoch, count], [r[2], r[3], r[4]].map(|v| v as usize));
                let frames: Vec<usize> = r[9..].iter().map(|&f| f as usize).collect();
                assert_eq!(s.frames, frames, "{at}");
                if r[5] == 0.0 {
                    assert_eq!(s.beta, 0.0, "{at}");
                } else {
                    assert!(
                        rel(s.beta, r[5]) < 1e-11,
                        "{at}: beta {} ref {}",
                        s.beta,
                        r[5]
                    );
                }
                for (name, got, want) in [
                    ("alpha", s.alpha, r[6]),
                    ("top", s.top, r[7]),
                    ("bottom", s.bottom, r[8]),
                ] {
                    assert!(rel(got, want) < 1e-13, "{at}: {name} {got} ref {want}");
                }
            }
            for (name, now, start, rows, lead) in [
                ("object", object.as_slice(), object0.as_slice(), &objs, 4),
                ("probe", probe.as_slice(), probe0.as_slice(), &probes, 5),
            ] {
                let got: Vec<f64> = now
                    .iter()
                    .zip(start)
                    .flat_map(|(a, b)| [(a - b).re, (a - b).im])
                    .collect();
                let want: Vec<f64> = rows
                    .iter()
                    .filter(at)
                    .flat_map(|r| [r[lead], r[lead + 1]])
                    .collect();
                let err = max_rel(&got, &want);
                assert!(err < 1e-12, "case {case} method {method} {name}: {err:.3e}");
            }
        }
    }
}
