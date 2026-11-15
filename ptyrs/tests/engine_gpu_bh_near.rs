//! GPU BH under the propagators with a plane stage against the CPU's `f64` [`Bh`] from the
//! same start, on the cases of `common/near.rs`.

use std::slice;

use ptyrs::base::{C64, Stack};
use ptyrs::engine::{Bh, BhParams, BhStep, Feature, Features, Ptycho, Sampling};

#[path = "common/near.rs"]
mod near;
use near::{NPOS, cases, drawn_offsets};
#[path = "common/worst.rs"]
mod worst;
use worst::worst_of;

/// The far-field GPU-against-CPU BH tests' bound on steps, arrays and LLK.
const TOL: f64 = 1e-4;

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
    for c in cases() {
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

/// BH refining positions from drawn offsets under each propagator, eagerly and from graphs:
/// after four epochs the last epoch's steps, the arrays and the offsets.
#[test]
fn bh_position_matches_cpu_under_every_propagator() {
    const NEPOCH: usize = 4;
    let moved = Features {
        position: Some(Feature::refined(drawn_offsets())),
        ..Features::default()
    };
    for c in cases() {
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
                gpu.bh(&p, NEPOCH, 0).unwrap();
                let got = gpu.features().unwrap().position.unwrap().values;
                let offset = worst_of(
                    (got.iter().flatten().zip(want.iter().flatten())).map(|(g, w)| (g - w).abs()),
                ) / scale;
                let errs = [
                    step_err(&gpu.bh_steps().unwrap(), &steps),
                    rel_l2(&gpu.object().unwrap(), &o),
                    rel_l2(&gpu.probe().unwrap(), &pr),
                    offset,
                ];
                let what = format!("{} positions batch {batch_size} graph {graph}", c.name);
                eprintln!(
                    "{what}: step {:.2e} object {:.2e} probe {:.2e} offset {:.2e}",
                    errs[0], errs[1], errs[2], errs[3]
                );
                check(&what, &["step", "object", "probe", "offset"], &errs);
            }
        }
    }
}

/// BH under `Sampling::FullObject` under each propagator, eagerly and from graphs, still and
/// refining drawn offsets that sample the object: after four epochs the last epoch's steps,
/// the arrays, the offsets and the LLK the GPU records after it, against the CPU's under the
/// same sampling.
#[test]
fn bh_full_object_matches_cpu_under_every_propagator() {
    const NEPOCH: usize = 4;
    let moved = Features {
        position: Some(Feature::refined(drawn_offsets())),
        ..Features::default()
    };
    for c in cases() {
        for (named, features) in [("still", Features::default()), ("positions", moved.clone())] {
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
                let want = bh.features().clone();
                let mut cpu =
                    Ptycho::with_optics(&c.scan, c.optics(), o.clone(), pr.clone()).unwrap();
                cpu.set_sampling(Sampling::FullObject);
                cpu.set_features(want.clone()).unwrap();
                let llk = cpu.llk().poisson;
                let want = want.position.map(|d| d.values).unwrap_or_default();
                let scale = want.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
                for graph in [false, true] {
                    let mut gpu = c.gpu(graph);
                    gpu.set_sampling(Sampling::FullObject).unwrap();
                    gpu.set_features(&features).unwrap();
                    gpu.bh(&p, NEPOCH, 1).unwrap();
                    let got = gpu
                        .features()
                        .unwrap()
                        .position
                        .map(|d| d.values)
                        .unwrap_or_default();
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
                    let what = format!("{} {named} batch {batch_size} graph {graph}", c.name);
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
