//! BH through a multislice object against `refs/f64/bh_multislice.py`, which runs the step rule
//! of [`Bh::epoch`] by torch autograd on `refs/f64/lsqml_multislice.py`'s forward model,
//! state and counts, writing `bh_multislice_*_f64.txt`, saved to `tests/engine/data`. Three
//! slices with unequal gaps; cases 0 an angular-spectrum exit, 1 a Fresnel transform, 2 the far
//! field. The reference's Hessian is the autograd Hessian of the cost whose gradient is the
//! engine's, plus the closed-form difference to [`BhBatch::hessian`]'s, whose `c_u·c_v` term
//! is over `(A + ε)³` where that cost's is over `A·(A + ε)²`: each step saves that difference
//! of `H(D, D)` (`eps`), printed here as its share of `bottom`.
//!
//! [`BhBatch::hessian`]: ptyrs::engine::BhBatch::hessian

use ptyrs::base::{AddrTable, C64, Counts, Geometry, Object, Positions, Probe, Stack};
use ptyrs::engine::{Bh, BhParams, Observed, Optics, Wavefield};

#[path = "common/worst.rs"]
mod worst;
use worst::worst_of;

const SHAPE: [usize; 2] = [24, 24];
const OBJ: [usize; 2] = [40, 40];
const NPROBE: usize = 2;
const NSLICE: usize = 3;
const EPOCHS: usize = 3;
/// Method 0: CG over one batch of all 16 frames; 1: steepest descent over batches of 6.
const BATCH: [usize; 2] = [16, 6];
/// At least 5x the worst measured: `beta`, `alpha`, `top`, `bottom` 2.4e-14 (far field, CG);
/// the slices' and the probe's change 2.4e-13 (Fresnel, the probe).
const STEP_BOUND: f64 = 2e-13;
const CHANGE_BOUND: f64 = 2e-12;

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

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

fn stack(dims: [usize; 3], rows: &[&Vec<f64>], lead: usize) -> Stack<C64> {
    let data = rows
        .iter()
        .map(|r| C64::new(r[lead], r[lead + 1]))
        .collect();
    Stack::from_vec(dims, data).unwrap()
}

/// The start slices and probe, the frames' table and per case its optics and counts.
fn state() -> (AddrTable, Vec<Object>, Probe, Vec<(Optics, Observed)>) {
    // corner = (OBJ − SHAPE)/2 + position at a pixel of 1
    let corners = table(include_str!("engine/data/lsqml_multislice_corners_f64.txt"));
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
    let obj = table(include_str!("engine/data/lsqml_multislice_obj_f64.txt"));
    let objects = (0..NSLICE)
        .map(|s| {
            let rows: Vec<_> = obj.iter().filter(|r| r[0] == s as f64).collect();
            Object::new(stack([1, OBJ[0], OBJ[1]], &rows, 3))
        })
        .collect();
    let probe = table(include_str!("engine/data/lsqml_multislice_probe_f64.txt"));
    let probe = Probe::new(stack(
        [NPROBE, SHAPE[0], SHAPE[1]],
        &probe.iter().collect::<Vec<_>>(),
        3,
    ));
    let counts = table(include_str!("engine/data/lsqml_multislice_counts_f64.txt"));
    let cases = table(include_str!("engine/data/lsqml_multislice_cases_f64.txt"))
        .iter()
        .map(|c| {
            // kind 0 is the far field at detector distance c[3], detector pixel λz/(ny·c[2])
            let (optics, pixel) = if c[4] == 0.0 {
                let lz = c[1] * c[3];
                let g = Geometry::new(c[1], c[3], lz / (SHAPE[0] as f64 * c[2]), SHAPE).unwrap();
                (Optics::far_field(&g), g.pixel_size_object())
            } else {
                let pixel = [c[2], c[2]];
                let w = Wavefield::new(SHAPE, c[1], pixel, c[3]).unwrap();
                (Optics::free_space(&w), pixel)
            };
            let gaps: Vec<_> = c[7..]
                .iter()
                .map(|&d| Wavefield::new(SHAPE, c[1], pixel, d).unwrap())
                .collect();
            let optics = optics.with_slices(&gaps).unwrap();
            assert!(rel(optics.exit().intensity_scale(), c[5]) <= 1e-15);
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
    (addr, objects, probe, cases)
}

/// Every case and method: each step's frames, `beta`, `alpha`, `top` and `bottom`, and every
/// slice's and the probe's change over the run.
#[test]
fn bh_matches_autograd_through_slices() {
    let (addr, objects0, probe0, cases) = state();
    let steps = table(include_str!("engine/data/bh_multislice_steps_f64.txt"));
    let objs = table(include_str!("engine/data/bh_multislice_obj_f64.txt"));
    let probes = table(include_str!("engine/data/bh_multislice_probe_f64.txt"));
    for (case, (optics, obs)) in cases.iter().enumerate() {
        for (method, &batch_size) in BATCH.iter().enumerate() {
            let run = format!("case {case} method {method}");
            let of_run = |r: &&Vec<f64>| r[0] as usize == case && r[1] as usize == method;
            let mut bh = Bh::new(BhParams {
                batch_size,
                probe_rho: 1.0,
                position_rho: 0.1,
                seed: 7,
            })
            .unwrap();
            let (mut objects, mut probe) = (objects0.clone(), probe0.clone());
            let got: Vec<_> = (0..EPOCHS)
                .flat_map(|e| {
                    let s = bh.epoch(&addr, obs, optics, &mut objects, &mut probe);
                    s.into_iter().map(move |s| (e, s))
                })
                .collect();
            let rows: Vec<_> = steps.iter().filter(of_run).collect();
            assert_eq!(got.len(), rows.len(), "{run}");
            let (mut betas, mut sizes, mut eps) = (Vec::new(), Vec::new(), Vec::new());
            for (k, ((epoch, s), r)) in got.iter().zip(&rows).enumerate() {
                let at = format!("{run} step {k}");
                assert_eq!([k, *epoch], [r[2], r[3]].map(|v| v as usize), "{at}");
                let frames: Vec<usize> = r[10..].iter().map(|&f| f as usize).collect();
                assert_eq!(s.frames, frames, "{at}");
                if r[5] == 0.0 {
                    assert_eq!(s.beta, 0.0, "{at}");
                } else {
                    betas.push(rel(s.beta, r[5]));
                }
                for (got, want) in [(s.alpha, r[6]), (s.top, r[7]), (s.bottom, r[8])] {
                    sizes.push(rel(got, want));
                }
                eps.push((r[9] / r[8]).abs());
            }
            let (betas, sizes) = (worst_of(betas), worst_of(sizes));
            eprintln!(
                "{run}: beta {betas:.3e} alpha, top, bottom {sizes:.3e}; EPS term up to {:.1e} \
                 of bottom",
                worst_of(eps)
            );
            for (name, got, rows) in [
                ("slices", change(&flat(&objects), &flat(&objects0)), &objs),
                (
                    "probe",
                    change(probe.as_slice(), probe0.as_slice()),
                    &probes,
                ),
            ] {
                let want: Vec<f64> = rows
                    .iter()
                    .filter(of_run)
                    .flat_map(|r| [r[5], r[6]])
                    .collect();
                let err = max_rel(&got, &want);
                eprintln!("{run} {name}: {err:.3e}");
                assert!(err < CHANGE_BOUND, "{run} {name}: {err:.3e}");
            }
            for (name, e) in [("beta", betas), ("alpha, top, bottom", sizes)] {
                assert!(e < STEP_BOUND, "{run} {name}: {e:.3e}");
            }
        }
    }
}

/// Every slice's values, slice-major.
fn flat(objects: &[Object]) -> Vec<C64> {
    objects.iter().flat_map(|o| o.as_slice().to_vec()).collect()
}

/// `re im` of `now − start`.
fn change(now: &[C64], start: &[C64]) -> Vec<f64> {
    let d = now.iter().zip(start).map(|(a, b)| a - b);
    d.flat_map(|d| [d.re, d.im]).collect()
}
