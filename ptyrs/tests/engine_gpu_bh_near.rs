//! GPU BH under the propagators with a plane stage against the CPU's `f64` [`Bh`] from the
//! same start, on the cases of `common/near.rs` and on angular spectrum with an evanescent
//! band, which BH carries as the CPU's does.

use std::slice;

use ptyrs::base::{C64, Object, Probe, Stack};
use ptyrs::engine::{Bh, BhParams, BhStep, Feature, Features, Sampling, SliceWaves, llk};

#[path = "common/near.rs"]
mod near;
use near::{Case, NPOS, PIXEL, cases, drawn_offsets};
#[path = "common/worst.rs"]
mod worst;
use worst::worst_of;

/// The far-field GPU-against-CPU BH tests' bound on steps, arrays and LLK.
const TOL: f64 = 1e-4;
/// A wavelength whose angular spectrum over [`PIXEL`] drops the corners of the frequency
/// grid, `λ / pixel` the CPU's near case's 5/3.
const BAND_WAVELENGTH: f64 = PIXEL / 0.6;

/// [`cases`] and angular spectrum with an evanescent band.
fn bh_cases() -> impl Iterator<Item = Case> {
    let band = near::case("evanescent band", BAND_WAVELENGTH, 6.0 * BAND_WAVELENGTH);
    cases().into_iter().chain([band])
}

/// The CPU's normalised Poisson LLK of `object` and `probe` under `Sampling::FullObject`, the
/// object read at each view's corner plus its frame's entry of `offsets`.
fn full_object_llk(c: &Case, object: &Object, probe: &Probe, offsets: &[[f64; 2]]) -> f64 {
    let (optics, views) = (c.optics(), c.addr.views());
    let offsets = (!offsets.is_empty()).then_some(offsets);
    let objects = slice::from_ref(object);
    let waves = SliceWaves::with_sampling(
        &optics,
        views,
        objects,
        probe,
        offsets,
        Sampling::FullObject,
    );
    let mut psi = waves.psi().to_vec();
    optics.exit().forward(&mut psi);
    llk(&c.obs, 0..c.addr.nframe(), c.addr.nmode(), &psi)
        .normalised(c.obs.nb_obs())
        .poisson
}

fn params(batch_size: usize) -> BhParams {
    BhParams {
        batch_size,
        probe_rho: 0.01,
        position_rho: 0.1,
        seed: 7,
    }
}

fn rel_l2(a: &Stack<C64>, b: &Stack<C64>) -> f64 {
    let (mut e, mut t) = (0.0, 0.0);
    for (x, y) in a.as_slice().iter().zip(b.as_slice()) {
        e += (x - y).norm_sqr();
        t += y.norm_sqr();
    }
    (e / t).sqrt()
}

/// The worst relative error of the GPU's steps; a CPU 0 (`β` in steepest descent) must be an
/// exact GPU 0.
fn step_err(got: &[BhStep], want: &[BhStep]) -> f64 {
    assert_eq!(got.len(), want.len());
    let mut worst = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g.frames, w.frames);
        for (a, b) in [
            (g.beta, w.beta),
            (g.alpha, w.alpha),
            (g.top, w.top),
            (g.bottom, w.bottom),
        ] {
            let e = if b == 0.0 {
                a.abs()
            } else {
                (a - b).abs() / b.abs()
            };
            worst = worst_of([worst, e]);
        }
    }
    worst
}

fn check(what: &str, names: &[&str], errs: &[f64]) {
    for (name, e) in names.iter().zip(errs) {
        assert!(*e < TOL, "{what} {name}: {e:e}");
    }
}

/// Under each propagator, eagerly and from graphs, after a fresh GPU run of each of the
/// first epochs: the last epoch's steps, the arrays and the LLK the GPU records after it.
/// One batch of every frame (CG from the second epoch) and batches of 4, the last of 2.
#[test]
fn bh_steps_match_cpu_under_every_propagator() {
    const NEPOCH: usize = 3;
    for c in bh_cases() {
        for batch_size in [NPOS, 4] {
            let p = params(batch_size);
            let mut bh = Bh::new(p).unwrap();
            let (mut o, mut pr) = (c.object.clone(), c.probe.clone());
            let start = c.llk(&o, &pr);
            let mut worst = [[0.0f64; 4]; 2];
            let mut last = start;
            for nepoch in 1..=NEPOCH {
                let want = bh.epoch(
                    &c.addr,
                    &c.obs,
                    &c.optics(),
                    slice::from_mut(&mut o),
                    &mut pr,
                );
                let llk = c.llk(&o, &pr);
                last = llk;
                for (graph, worst) in [false, true].into_iter().zip(&mut worst) {
                    let mut gpu = c.gpu(graph);
                    gpu.bh(&p, nepoch, 1).unwrap();
                    let got = gpu.history().last().unwrap().llk.poisson;
                    let errs = [
                        step_err(&gpu.bh_steps().unwrap(), &want),
                        rel_l2(&gpu.object().unwrap(), &o),
                        rel_l2(&gpu.probe().unwrap(), &pr),
                        (got - llk).abs() / llk,
                    ];
                    for (w, e) in worst.iter_mut().zip(errs) {
                        *w = worst_of([*w, e]);
                    }
                }
            }
            for (graph, w) in [false, true].into_iter().zip(worst) {
                let what = format!("{} batch {batch_size} graph {graph}", c.name);
                eprintln!(
                    "{what}: step {:.2e} object {:.2e} probe {:.2e} llk {:.2e}, llk {start:.4e} -> {last:.4e}",
                    w[0], w[1], w[2], w[3]
                );
                check(&what, &["step", "object", "probe", "llk"], &w);
            }
        }
    }
}

/// BH refining and holding positions from drawn offsets under each propagator, eagerly and
/// from graphs: after four epochs the last epoch's steps, the arrays and the offsets, which
/// held keep the bits they were set with.
#[test]
fn bh_position_matches_cpu_under_every_propagator() {
    const NEPOCH: usize = 4;
    let d0 = drawn_offsets();
    for c in bh_cases() {
        for position in [Feature::refined(d0.clone()), Feature::held(d0.clone())] {
            let named = if position.refine { "refined" } else { "held" };
            let moved = Features {
                position: Some(position),
                ..Features::default()
            };
            for batch_size in [NPOS, 4] {
                let p = params(batch_size);
                let mut bh = Bh::new(p).unwrap().with_features(moved.clone()).unwrap();
                let (mut o, mut pr) = (c.object.clone(), c.probe.clone());
                let mut steps = Vec::new();
                for _ in 0..NEPOCH {
                    steps = bh.epoch(
                        &c.addr,
                        &c.obs,
                        &c.optics(),
                        slice::from_mut(&mut o),
                        &mut pr,
                    );
                }
                let want = bh.features().position.clone().unwrap().values;
                let scale = want.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
                for graph in [false, true] {
                    let mut gpu = c.gpu(graph);
                    gpu.set_features(&moved).unwrap();
                    let set = gpu.features().unwrap().position.unwrap();
                    gpu.bh(&p, NEPOCH, 0).unwrap();
                    let got = gpu.features().unwrap().position.unwrap();
                    let what = format!("{} {named} batch {batch_size} graph {graph}", c.name);
                    if !set.refine {
                        assert!(got.values == set.values, "{what}: held offsets moved");
                    }
                    let offset = worst_of(
                        (got.values.iter().flatten().zip(want.iter().flatten()))
                            .map(|(g, w)| (g - w).abs()),
                    ) / scale;
                    let errs = [
                        step_err(&gpu.bh_steps().unwrap(), &steps),
                        rel_l2(&gpu.object().unwrap(), &o),
                        rel_l2(&gpu.probe().unwrap(), &pr),
                        offset,
                    ];
                    eprintln!(
                        "{what}: step {:.2e} object {:.2e} probe {:.2e} offset {:.2e}",
                        errs[0], errs[1], errs[2], errs[3]
                    );
                    check(&what, &["step", "object", "probe", "offset"], &errs);
                }
            }
        }
    }
}

/// BH under `Sampling::FullObject` under each propagator, eagerly and from graphs, still and
/// refining or holding drawn offsets that sample the object: after four epochs the last
/// epoch's steps, the arrays, the offsets and the LLK the GPU records after it, against the
/// CPU's under the same sampling.
#[test]
fn bh_full_object_matches_cpu_under_every_propagator() {
    const NEPOCH: usize = 4;
    let moved = |position| Features {
        position: Some(position),
        ..Features::default()
    };
    let d0 = drawn_offsets();
    let runs = [
        ("still", Features::default()),
        ("positions", moved(Feature::refined(d0.clone()))),
        ("held", moved(Feature::held(d0))),
    ];
    for c in bh_cases() {
        for (named, features) in runs.clone() {
            for batch_size in [NPOS, 4] {
                let p = params(batch_size);
                let mut bh = Bh::new(p)
                    .unwrap()
                    .with_sampling(Sampling::FullObject)
                    .with_features(features.clone())
                    .unwrap();
                let (mut o, mut pr) = (c.object.clone(), c.probe.clone());
                let mut steps = Vec::new();
                for _ in 0..NEPOCH {
                    steps = bh.epoch(
                        &c.addr,
                        &c.obs,
                        &c.optics(),
                        slice::from_mut(&mut o),
                        &mut pr,
                    );
                }
                let want = bh.features().position.clone();
                let want = want.map(|d| d.values).unwrap_or_default();
                let llk = full_object_llk(&c, &o, &pr, &want);
                let scale = want.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
                for graph in [false, true] {
                    let mut gpu = c.gpu(graph);
                    gpu.set_sampling(Sampling::FullObject).unwrap();
                    gpu.set_features(&features).unwrap();
                    let set = gpu.features().unwrap().position;
                    gpu.bh(&p, NEPOCH, 1).unwrap();
                    let got = gpu.features().unwrap().position;
                    let what = format!("{} {named} batch {batch_size} graph {graph}", c.name);
                    if let Some(set) = set.filter(|d| !d.refine) {
                        let kept = got.as_ref().is_some_and(|d| d.values == set.values);
                        assert!(kept, "{what}: held offsets moved");
                    }
                    let got = got.map(|d| d.values).unwrap_or_default();
                    assert_eq!(got.len(), want.len());
                    let offset = worst_of(
                        (got.iter().flatten().zip(want.iter().flatten()))
                            .map(|(g, w)| (g - w).abs() / scale),
                    );
                    let recorded = gpu.history().last().unwrap().llk.poisson;
                    let errs = [
                        step_err(&gpu.bh_steps().unwrap(), &steps),
                        rel_l2(&gpu.object().unwrap(), &o),
                        rel_l2(&gpu.probe().unwrap(), &pr),
                        offset,
                        (recorded - llk).abs() / llk,
                    ];
                    eprintln!(
                        "{what}: step {:.2e} object {:.2e} probe {:.2e} offset {:.2e} llk {:.2e}",
                        errs[0], errs[1], errs[2], errs[3], errs[4]
                    );
                    check(&what, &["step", "object", "probe", "offset", "llk"], &errs);
                }
            }
        }
    }
}
