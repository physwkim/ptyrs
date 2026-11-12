//! GPU LSQML against the CPU's `f64` [`Lsqml`] from the same scaled start, with no probe
//! variation: each epoch's steps and the arrays after the first epochs, eagerly and from
//! graphs, and the LLK after each of enough epochs to reach a kept preconditioner; Gaussian
//! and Poisson costs, one batch of every frame, batches of 12 (two stacks each but the last)
//! and of 4 (the last of 2). Frames are on the fused passes' smallest side, two incoherent
//! probe modes, one masked row and column.

use std::slice;
use std::sync::Arc;

use cuda_core::CudaContext;
use ptyrs::base::{C64, Counts, MASKED, Object, Probe, Scan, Stack};
use ptyrs::engine::{
    Algorithm, BhParams, Cost, Features, GpuOptions, GpuPtycho, LlkRecord, Lsqml, LsqmlParams,
    LsqmlStep, Optics, ProbeVariation, Ptycho,
};
use ptyrs::gpu::FUSED_SIDES;

#[path = "common/engine/mod.rs"]
mod common;

/// 30 positions: stacks of 7 and batches of 12 or 4 leave a shorter last one.
const NPOS: usize = 30;

const OPTIONS: GpuOptions = GpuOptions {
    stack_size: 7,
    graph: true,
    llk_slots: 3,
    fused: true,
};

fn ctx() -> Arc<CudaContext> {
    CudaContext::new(0).expect("CUDA device 0")
}

/// The parity scenario on `frame`-pixel frames, scaled on the CPU, with row 3 and column 20
/// of every frame masked.
fn start(frame: usize) -> (Scan, Object, Probe) {
    let (mut scan, object, probe) = common::dm_scenario(frame, NPOS, 2, 2e5, frame as f64 / 8.0);
    let dims = scan.counts.dims();
    let [_, ny, nx] = dims;
    let data = scan
        .counts
        .as_slice()
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let (y, x) = ((i / nx) % ny, i % nx);
            if y == 3 || x == 20 { MASKED } else { c }
        })
        .collect();
    scan.counts = Counts::from_raw(dims, data).unwrap();
    let mut p = Ptycho::new(&scan, object, probe).unwrap();
    p.scale_obj_probe();
    let (object, probe) = p.into_modes();
    (scan, object, probe)
}

/// `|a − b| / |b|` over the whole array.
fn rel_l2(a: &Stack<C64>, b: &Stack<C64>) -> f64 {
    let (mut e, mut t) = (0.0, 0.0);
    for (x, y) in a.as_slice().iter().zip(b.as_slice()) {
        e += (x - y).norm_sqr();
        t += y.norm_sqr();
    }
    (e / t).sqrt()
}

/// `|a − b| / |b|`, or `|a|` where `b` is 0 (a clamped `a_far`).
fn rel(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        a.abs()
    } else {
        (a - b).abs() / b.abs()
    }
}

fn runs() -> impl Iterator<Item = (Cost, usize)> {
    [Cost::Amplitude, Cost::Poisson]
        .into_iter()
        .flat_map(|cost| [NPOS, 12, 4].map(|b| (cost, b)))
}

/// After a fresh GPU run of each of the first epochs (the preconditioner built, then
/// updated), the last epoch's steps (frames, `a_far`, `a_obj`, `a_probe`) and the arrays.
#[test]
fn lsqml_steps_match_cpu() {
    const NEPOCH: usize = 3;
    let frame = FUSED_SIDES[0] as usize;
    for (graph, (cost, batch_size)) in [false, true]
        .into_iter()
        .flat_map(|g| runs().map(move |r| (g, r)))
    {
        let options = GpuOptions { graph, ..OPTIONS };
        let params = LsqmlParams {
            batch_size,
            cost,
            seed: 7,
        };
        let (scan, object, probe) = start(frame);
        let at = Ptycho::new(&scan, object.clone(), probe.clone()).unwrap();
        let optics = Optics::far_field(&scan.geometry);
        let (mut o, mut p) = (object.clone(), probe.clone());
        let mut var = ProbeVariation::none(probe.plane_shape());
        let mut cpu = Lsqml::new(params);
        let mut worst = [0.0f64; 5];
        for nepoch in 1..=NEPOCH {
            let want = cpu.epoch(
                at.addr(),
                at.observed(),
                &optics,
                slice::from_mut(&mut o),
                &mut p,
                &mut var,
            );
            let mut gpu =
                GpuPtycho::new(&ctx(), &scan, object.clone(), probe.clone(), options).unwrap();
            gpu.lsqml(&params, nepoch, 0).unwrap();
            let got = gpu.lsqml_steps().unwrap();
            let frames = |s: &[LsqmlStep]| -> Vec<Vec<usize>> {
                s.iter().map(|s| s.frames.clone()).collect()
            };
            assert_eq!(frames(&got), frames(&want), "epoch {nepoch}");
            for (g, w) in got.iter().zip(&want) {
                for (i, (a, b)) in [
                    (&g.a_far, &w.a_far),
                    (&g.a_obj[0], &w.a_obj[0]),
                    (&g.a_probe, &w.a_probe),
                ]
                .into_iter()
                .enumerate()
                {
                    for (a, b) in a.iter().zip(b) {
                        worst[i] = worst[i].max(rel(*a, *b));
                    }
                }
            }
            worst[3] = worst[3].max(rel_l2(&gpu.object().unwrap(), &o));
            worst[4] = worst[4].max(rel_l2(&gpu.probe().unwrap(), &p));
        }
        eprintln!(
            "lsqml steps {cost:?} graph {graph} batch {batch_size}: a_far {:.2e} a_obj {:.2e} \
             a_probe {:.2e} object {:.2e} probe {:.2e}",
            worst[0], worst[1], worst[2], worst[3], worst[4]
        );
        let names = ["a_far", "a_obj", "a_probe", "object", "probe"];
        for (what, e) in names.into_iter().zip(worst) {
            assert!(
                e < 1e-4,
                "{cost:?} graph {graph} batch {batch_size} {what}: {e:e}"
            );
        }
    }
}

/// The LLK after each of 12 epochs, tagged LSQML (epoch 11 keeps the preconditioner of
/// epoch 10), and the final arrays.
#[test]
fn lsqml_matches_cpu() {
    const NEPOCH: usize = 12;
    let frame = FUSED_SIDES[0] as usize;
    for (cost, batch_size) in runs() {
        let params = LsqmlParams {
            batch_size,
            cost,
            seed: 7,
        };
        let (scan, object, probe) = start(frame);
        let at = Ptycho::new(&scan, object.clone(), probe.clone()).unwrap();
        let optics = Optics::far_field(&scan.geometry);
        let (mut o, mut p) = (object.clone(), probe.clone());
        let mut var = ProbeVariation::none(probe.plane_shape());
        let mut cpu = Lsqml::new(params);
        let want: Vec<_> = (0..NEPOCH)
            .map(|cycle| {
                cpu.epoch(
                    at.addr(),
                    at.observed(),
                    &optics,
                    slice::from_mut(&mut o),
                    &mut p,
                    &mut var,
                );
                let llk = Ptycho::new(&scan, o.clone(), p.clone()).unwrap().llk();
                LlkRecord {
                    cycle,
                    algorithm: Algorithm::Lsqml,
                    llk,
                }
            })
            .collect();
        let mut gpu = GpuPtycho::new(&ctx(), &scan, object, probe, OPTIONS).unwrap();
        gpu.lsqml(&params, NEPOCH, 1).unwrap();
        let got = gpu.history();
        assert_eq!(got.len(), want.len());
        let mut llk = 0.0f64;
        for (g, w) in got.iter().zip(&want) {
            assert_eq!((g.cycle, g.algorithm), (w.cycle, w.algorithm));
            for (a, b) in [
                (g.llk.poisson, w.llk.poisson),
                (g.llk.gaussian, w.llk.gaussian),
                (g.llk.euclidian, w.llk.euclidian),
                (g.llk.nb_photons_calc, w.llk.nb_photons_calc),
            ] {
                llk = llk.max(rel(a, b));
            }
        }
        let errs = [
            llk,
            rel_l2(&gpu.object().unwrap(), &o),
            rel_l2(&gpu.probe().unwrap(), &p),
        ];
        eprintln!(
            "lsqml {cost:?} batch {batch_size}: llk {:.2e} object {:.2e} probe {:.2e}, \
             poisson llk {:.4e} -> {:.4e}",
            errs[0],
            errs[1],
            errs[2],
            want[0].llk.poisson,
            want[NEPOCH - 1].llk.poisson
        );
        for (what, e) in ["llk", "object", "probe"].into_iter().zip(errs) {
            assert!(e < 1e-4, "{cost:?} batch {batch_size} {what}: {e:e}");
        }
    }
}

/// The worst relative difference of `got`'s steps from `want`'s, the frames equal.
fn steps_rel(got: &[LsqmlStep], want: &[LsqmlStep]) -> f64 {
    assert_eq!(got.len(), want.len());
    let mut worst = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g.frames, w.frames);
        for (a, b) in [
            (&g.a_far, &w.a_far),
            (&g.a_obj[0], &w.a_obj[0]),
            (&g.a_probe, &w.a_probe),
        ] {
            for (a, b) in a.iter().zip(b) {
                worst = worst.max(rel(*a, *b));
            }
        }
    }
    worst
}

/// Every LSQML call starts a fresh [`Lsqml`], as on the CPU: its first epoch's steps match a
/// fresh CPU run from the arrays it starts at, after another LSQML call (whose graphs it
/// replays) and after a BH call (which drops them). A zero batch size, modelled features and
/// frames off the fused sides fail.
#[test]
fn lsqml_restarts_and_rejects() {
    let frame = FUSED_SIDES[0] as usize;
    let params = LsqmlParams {
        batch_size: 4,
        cost: Cost::Poisson,
        seed: 7,
    };
    let bh = BhParams {
        batch_size: 4,
        probe_rho: 0.01,
        seed: 7,
    };
    let (scan, object, probe) = start(frame);
    let optics = Optics::far_field(&scan.geometry);
    let mut gpu = GpuPtycho::new(&ctx(), &scan, object, probe, OPTIONS).unwrap();
    gpu.lsqml(&params, 2, 0).unwrap();
    for between_bh in [false, true] {
        if between_bh {
            gpu.bh(&bh, 1, 0).unwrap();
        }
        let (mut o, mut p) = (gpu.object().unwrap(), gpu.probe().unwrap());
        let at = Ptycho::new(&scan, o.clone(), p.clone()).unwrap();
        let mut var = ProbeVariation::none(p.plane_shape());
        let want = Lsqml::new(params).epoch(
            at.addr(),
            at.observed(),
            &optics,
            slice::from_mut(&mut o),
            &mut p,
            &mut var,
        );
        gpu.lsqml(&params, 1, 0).unwrap();
        let errs = [
            steps_rel(&gpu.lsqml_steps().unwrap(), &want),
            rel_l2(&gpu.object().unwrap(), &o),
            rel_l2(&gpu.probe().unwrap(), &p),
        ];
        eprintln!(
            "lsqml restart after bh {between_bh}: steps {:.2e} object {:.2e} probe {:.2e}",
            errs[0], errs[1], errs[2]
        );
        for (what, e) in ["steps", "object", "probe"].into_iter().zip(errs) {
            assert!(e < 1e-4, "after bh {between_bh} {what}: {e:e}");
        }
    }

    let zero = LsqmlParams {
        batch_size: 0,
        ..params
    };
    assert!(gpu.lsqml(&zero, 1, 1).is_err());
    for features in [
        Features {
            scale: Some(vec![1.0; NPOS]),
            background: None,
        },
        Features {
            scale: None,
            background: Some(vec![1.0; frame * frame]),
        },
    ] {
        gpu.set_features(&features).unwrap();
        assert!(gpu.lsqml(&params, 1, 0).is_err(), "{features:?}");
    }

    let (scan, object, probe) = start(32);
    let m1 = GpuOptions {
        fused: false,
        ..OPTIONS
    };
    let mut off = GpuPtycho::new(&ctx(), &scan, object, probe, m1).unwrap();
    assert!(off.lsqml(&params, 1, 0).is_err());
}
