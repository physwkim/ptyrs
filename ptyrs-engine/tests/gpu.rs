//! The GPU backend against the CPU oracle: per-record LLK and the final object and probe of AP,
//! DM, rPIE and ML updating the object, the probe or both, and of BH in one or more batches, from the
//! same scaled start; AP and DM by M1's launches and by the fused passes; the ML gradients;
//! and the LLK ring's record of a scan's first frames.

use std::sync::Arc;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, DeviceBuffer};
use ptyrs_core::{C64, Counts, MASKED, Object, Probe, Scan, Stack};
use ptyrs_engine::{
    Algorithm, ApParams, Backend, Bh, BhParams, BhStep, DmParams, Features, GpuOptions, GpuPtycho,
    Llk, LlkRecord, MlParams, Observed, Params, Ptycho, RpieParams, Step, llk,
};
use ptyrs_gpu::{C32, FUSED_SIDES, Kernels, LlkPartial, LlkRing, LlkScratch, StackTable, ViewDims};

mod common;
mod ptychi;

/// 30 positions: stacks of 7 and batches of 4 both leave a shorter last one. Three LLK
/// slots: the ring wraps within every LLK-taking call.
const NPOS: usize = 30;

/// A frame side and the options run on it.
#[derive(Clone, Copy, Debug)]
struct Setup {
    frame: usize,
    options: GpuOptions,
}

/// M1's launches on 32-pixel frames.
const M1: Setup = Setup {
    frame: 32,
    options: GpuOptions {
        stack_size: 7,
        graph: true,
        llk_slots: 3,
        fused: false,
    },
};

/// The fused passes on the smallest side they run.
const FUSED: Setup = Setup {
    frame: FUSED_SIDES[0] as usize,
    options: GpuOptions {
        fused: true,
        ..M1.options
    },
};

fn ctx() -> Arc<CudaContext> {
    CudaContext::new(0).expect("CUDA device 0")
}

/// The parity scenario on `frame`-pixel frames, `frame / 8` object pixels apart, scaled on
/// the CPU; `masked` masks row 3 and column 20 of every frame.
fn start(frame: usize, nprobe: usize, masked: bool) -> (Scan, Object, Probe) {
    let (mut scan, object, probe) =
        common::dm_scenario(frame, NPOS, nprobe, 2e5, frame as f64 / 8.0);
    if masked {
        scan = mask(scan, |y, x| y == 3 || x == 20);
    }
    let mut p = Ptycho::new(&scan, object, probe).unwrap();
    p.scale_obj_probe();
    let (object, probe) = p.into_modes();
    (scan, object, probe)
}

/// `scan` with pixel `(y, x)` of every frame masked where `masked(y, x)`.
fn mask(mut scan: Scan, masked: impl Fn(usize, usize) -> bool) -> Scan {
    let dims = scan.counts.dims();
    let [_, ny, nx] = dims;
    let data = scan
        .counts
        .as_slice()
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            if masked((i / nx) % ny, i % nx) {
                MASKED
            } else {
                c
            }
        })
        .collect();
    scan.counts = Counts::from_raw(dims, data).unwrap();
    scan
}

/// `|a - b| / |b|` over the whole array.
fn rel_l2(a: &Stack<C64>, b: &Stack<C64>) -> f64 {
    let (mut e, mut t) = (0.0, 0.0);
    for (x, y) in a.as_slice().iter().zip(b.as_slice()) {
        e += (x - y).norm_sqr();
        t += y.norm_sqr();
    }
    (e / t).sqrt()
}

fn llk_rel(g: &[LlkRecord], c: &[LlkRecord]) -> f64 {
    assert_eq!(g.len(), c.len());
    let mut worst = 0.0f64;
    for (g, c) in g.iter().zip(c) {
        assert_eq!((g.cycle, g.algorithm), (c.cycle, c.algorithm));
        for (a, b) in [
            (g.llk.poisson, c.llk.poisson),
            (g.llk.gaussian, c.llk.gaussian),
            (g.llk.euclidian, c.llk.euclidian),
            (g.llk.nb_photons_calc, c.llk.nb_photons_calc),
        ] {
            worst = worst.max((a - b).abs() / b.abs());
        }
    }
    worst
}

/// Runs `params` for `ncycle` on both backends and returns the worst LLK and the object and
/// probe relative errors.
fn compare(setup: Setup, nprobe: usize, masked: bool, params: Params, ncycle: usize) -> [f64; 3] {
    let (scan, object, probe) = start(setup.frame, nprobe, masked);
    let mut cpu = Ptycho::new(&scan, object.clone(), probe.clone()).unwrap();
    let mut gpu = GpuPtycho::new(&ctx(), &scan, object, probe, setup.options).unwrap();
    let step = Step {
        params,
        count: ncycle,
    };
    cpu.run(&step).unwrap();
    gpu.run(&step).unwrap();
    assert_eq!(gpu.cycle(), cpu.cycle());
    let errs = [
        llk_rel(gpu.history(), cpu.history()),
        rel_l2(&gpu.object().unwrap(), cpu.object()),
        rel_l2(&gpu.probe().unwrap(), cpu.probe()),
    ];
    eprintln!(
        "fused {} {params:?}: llk {:.2e} object {:.2e} probe {:.2e}",
        setup.options.fused, errs[0], errs[1], errs[2]
    );
    errs
}

/// Object only, probe only, both.
const UPDATES: [(Option<f64>, Option<f64>); 3] = [
    (Some(0.01), None),
    (None, Some(0.001)),
    (Some(0.01), Some(0.001)),
];

fn check(errs: [f64; 3], tol: [f64; 3]) {
    for (what, (e, t)) in ["llk", "object", "probe"].iter().zip(errs.iter().zip(tol)) {
        assert!(*e < t, "{what}: {e:e} >= {t:e}");
    }
}

#[test]
fn ap_matches_cpu() {
    for setup in [M1, FUSED] {
        for (o, p) in UPDATES {
            let params = ApParams {
                obj_inertia: o,
                probe_inertia: p,
                calc_llk: 1,
                zero_phase_ramp: true,
            };
            let errs = compare(setup, 2, true, Params::Ap(params), 10);
            check(errs, [1e-4, 1e-4, 1e-4]);
        }
    }
}

#[test]
fn dm_matches_cpu() {
    for setup in [M1, FUSED] {
        for (o, p) in UPDATES {
            let params = DmParams {
                obj_inertia: o,
                probe_inertia: p,
                calc_llk: 1,
                ..DmParams::default()
            };
            let errs = compare(setup, 2, true, Params::Dm(params), 10);
            check(errs, [1e-4, 1e-4, 1e-4]);
        }
        // two object-then-probe rounds per cycle, one mode, no mask
        let params = DmParams {
            loop_obj_probe: 2,
            calc_llk: 3,
            ..DmParams::default()
        };
        check(
            compare(setup, 1, false, Params::Dm(params), 7),
            [1e-4, 1e-4, 1e-4],
        );
    }
}

#[test]
fn rpie_matches_cpu() {
    for (o, p) in UPDATES {
        let params = RpieParams {
            obj_step: o.map(|_| 1.0),
            probe_step: p.map(|_| 1.0),
            ..ptychi::RPIE
        };
        check(
            compare(M1, 2, true, Params::Rpie(params), 10),
            [1e-4, 1e-4, 1e-4],
        );
    }
}

/// ML's gradients of the same estimate, in `f32` against `f64`, by M1's launches and by the
/// fused passes.
#[test]
fn ml_grads_match_cpu() {
    for setup in [M1, FUSED] {
        let (scan, object, probe) = start(setup.frame, 2, true);
        let cpu = Ptycho::new(&scan, object.clone(), probe.clone()).unwrap();
        let mut gpu = GpuPtycho::new(&ctx(), &scan, object, probe, setup.options).unwrap();
        let [go, gp] = gpu.ml_grads().unwrap();
        let errs = [rel_l2(&go, &cpu.obj_grad()), rel_l2(&gp, &cpu.probe_grad())];
        eprintln!(
            "ml grads {setup:?}: object {:.2e} probe {:.2e}",
            errs[0], errs[1]
        );
        assert!(errs[0] < 1e-5 && errs[1] < 1e-5, "{errs:?}");
        // the sums are consumed: a second call gives the same gradients
        let [go2, gp2] = gpu.ml_grads().unwrap();
        assert_eq!((go2, gp2), (go, gp));
    }
}

/// ML's starting scale counts the calculated photons over the unmasked pixels only, as the
/// observed total does: with the bright centre of every frame masked, holding over a tenth of
/// the calculated photons, the first LLK record's photon count is the observed total.
#[test]
fn ml_scale_leaves_masked_pixels_out() {
    let (frame, c) = (M1.frame, M1.frame / 2);
    let (scan, object, probe) = common::dm_scenario(frame, NPOS, 2, 2e5, frame as f64 / 8.0);
    let open = Ptycho::new(&scan, object.clone(), probe.clone()).unwrap();
    let scan = mask(scan, |y, x| y.abs_diff(c) < 3 && x.abs_diff(c) < 3);
    let cpu = Ptycho::new(&scan, object.clone(), probe.clone()).unwrap();
    let share = 1.0 - cpu.llk().nb_photons_calc / open.llk().nb_photons_calc;
    assert!(share > 0.1, "masked share {share}");
    let total = cpu.observed().total();
    let mut gpu = GpuPtycho::new(&ctx(), &scan, object, probe, M1.options).unwrap();
    let params = MlParams {
        calc_llk: 1,
        ..MlParams::default()
    };
    gpu.ml(&params, 1).unwrap();
    let got = gpu.history()[0].llk.nb_photons_calc;
    assert!((got - total).abs() < 1e-5 * total, "{got} vs {total}");
}

/// ML's direction (β from the last gradient across both arrays) and Poisson quadratic step
/// against the CPU, for each pair of updated arrays, with the gradients by M1's launches and
/// by the fused passes.
#[test]
fn ml_matches_cpu() {
    for (setup, (o, p)) in [M1, FUSED]
        .into_iter()
        .flat_map(|s| UPDATES.map(|u| (s, u)))
    {
        let params = MlParams {
            update_object: o.is_some(),
            update_probe: p.is_some(),
            calc_llk: 1,
            zero_phase_ramp: true,
        };
        let (scan, object, probe) = start(setup.frame, 2, true);
        let mut cpu = Ptycho::new(&scan, object.clone(), probe.clone()).unwrap();
        let mut gpu = GpuPtycho::new(&ctx(), &scan, object, probe, setup.options).unwrap();
        cpu.ml(&params, 10);
        gpu.ml(&params, 10).unwrap();
        assert_eq!(gpu.cycle(), cpu.cycle());
        let errs = [
            llk_rel(gpu.history(), cpu.history()),
            rel_l2(&gpu.object().unwrap(), cpu.object()),
            rel_l2(&gpu.probe().unwrap(), cpu.probe()),
        ];
        eprintln!(
            "ml fused {} {params:?}: llk {:.2e} object {:.2e} probe {:.2e}",
            setup.options.fused, errs[0], errs[1], errs[2]
        );
        check(errs, [1e-4, 1e-4, 1e-4]);
    }
}

/// BH against the CPU's `f64` BH from the same start: the LLK after each of 20 epochs and the
/// final arrays, by M1's launches and on the fused passes' frame side. One batch of every
/// frame (CG after the first epoch); batches of 12, two stacks each in M1 but the last; and
/// batches of 4, the last of 2.
#[test]
fn bh_matches_cpu() {
    const NEPOCH: usize = 20;
    for (setup, batch_size) in [M1, FUSED]
        .into_iter()
        .flat_map(|s| [NPOS, 12, 4].map(|b| (s, b)))
    {
        let params = BhParams {
            batch_size,
            probe_rho: 0.01,
            seed: 7,
        };
        let (scan, object, probe) = start(setup.frame, 2, true);
        let at = Ptycho::new(&scan, object.clone(), probe.clone()).unwrap();
        let (mut o, mut p) = (object.clone(), probe.clone());
        let mut bh = Bh::new(params);
        let want: Vec<_> = (0..NEPOCH)
            .map(|cycle| {
                bh.epoch(at.addr(), at.observed(), at.quad(), &mut o, &mut p);
                let llk = Ptycho::new(&scan, o.clone(), p.clone()).unwrap().llk();
                LlkRecord {
                    cycle,
                    algorithm: Algorithm::Bh,
                    llk,
                }
            })
            .collect();
        let mut gpu = GpuPtycho::new(&ctx(), &scan, object, probe, setup.options).unwrap();
        gpu.bh(&params, NEPOCH, 1).unwrap();
        let errs = [
            llk_rel(gpu.history(), &want),
            rel_l2(&gpu.object().unwrap(), &o),
            rel_l2(&gpu.probe().unwrap(), &p),
        ];
        eprintln!(
            "bh fused {} batch {batch_size}: llk {:.2e} object {:.2e} probe {:.2e}, llk {:.4e} -> {:.4e}",
            setup.options.fused,
            errs[0],
            errs[1],
            errs[2],
            want[0].llk.poisson,
            want[NEPOCH - 1].llk.poisson
        );
        check(errs, [1e-4, 1e-4, 1e-4]);
    }
}

/// BH step by step against the CPU's `f64` BH, eagerly and from graphs: after a fresh GPU
/// run of each of the first epochs, the last epoch's steps (frames, `β`, `α`, `top`,
/// `bottom`) and the arrays. One batch of every frame (CG from the second epoch) and batches
/// of 4, the last of 2.
#[test]
fn bh_steps_match_cpu() {
    const NEPOCH: usize = 3;
    let runs = [M1, FUSED]
        .into_iter()
        .flat_map(|s| [false, true].map(|graph| (s, graph)))
        .flat_map(|(s, graph)| [NPOS, 4].map(|b| (s, graph, b)));
    for (setup, graph, batch_size) in runs {
        let options = GpuOptions {
            graph,
            ..setup.options
        };
        let params = BhParams {
            batch_size,
            probe_rho: 0.01,
            seed: 7,
        };
        let (scan, object, probe) = start(setup.frame, 2, true);
        let at = Ptycho::new(&scan, object.clone(), probe.clone()).unwrap();
        let (mut o, mut p) = (object.clone(), probe.clone());
        let mut bh = Bh::new(params);
        let mut worst = [0.0f64; 3];
        for nepoch in 1..=NEPOCH {
            let want = bh.epoch(at.addr(), at.observed(), at.quad(), &mut o, &mut p);
            let mut gpu =
                GpuPtycho::new(&ctx(), &scan, object.clone(), probe.clone(), options).unwrap();
            gpu.bh(&params, nepoch, 0).unwrap();
            let got = gpu.bh_steps().unwrap();
            let frames =
                |s: &[BhStep]| -> Vec<Vec<usize>> { s.iter().map(|s| s.frames.clone()).collect() };
            assert_eq!(frames(&got), frames(&want), "epoch {nepoch}");
            for (g, w) in got.iter().zip(&want) {
                for (a, b) in [
                    (g.beta, w.beta),
                    (g.alpha, w.alpha),
                    (g.top, w.top),
                    (g.bottom, w.bottom),
                ] {
                    // a CPU 0 (β in steepest descent) must be an exact GPU 0
                    worst[0] = worst[0].max(if b == 0.0 {
                        a.abs()
                    } else {
                        (a - b).abs() / b.abs()
                    });
                }
            }
            worst[1] = worst[1].max(rel_l2(&gpu.object().unwrap(), &o));
            worst[2] = worst[2].max(rel_l2(&gpu.probe().unwrap(), &p));
        }
        eprintln!(
            "bh steps fused {} graph {graph} batch {batch_size}: step {:.2e} object {:.2e} probe {:.2e}",
            setup.options.fused, worst[0], worst[1], worst[2]
        );
        for (what, e) in ["step", "object", "probe"].into_iter().zip(worst) {
            assert!(e < 1e-4, "{what}: {e:e}");
        }
    }
}

/// [`GpuPtycho::new`] compiles BH's Tile dot products, so graphed BH epochs, one batch of
/// every frame (CG from the second epoch) and batches of 4, compile no Tile kernel in their
/// captures or replays. cutile keeps a compile for the process, so a later setup whose
/// kernels match an earlier one's compiles nothing.
#[test]
fn bh_graphs_compile_no_tile_kernel() {
    let before = cutile::tile_kernel::jit_compile_count();
    for setup in [M1, FUSED] {
        let (scan, object, probe) = start(setup.frame, 2, true);
        let mut g = GpuPtycho::new(&ctx(), &scan, object, probe, setup.options).unwrap();
        let jit = cutile::tile_kernel::jit_compile_count();
        assert!(
            jit > before,
            "{setup:?}: no GpuPtycho::new compiled a Tile kernel"
        );
        for batch_size in [NPOS, 4] {
            let params = BhParams {
                batch_size,
                probe_rho: 0.01,
                seed: 7,
            };
            g.bh(&params, 3, 1).unwrap();
        }
        assert_eq!(
            cutile::tile_kernel::jit_compile_count(),
            jit,
            "{setup:?}: BH compiled a Tile kernel"
        );
    }
}

/// Every update consumes its sums, so each algorithm starts from zero sums whatever ran
/// before: rPIE's steps, then AP's and DM's updates, against the CPU. The last rPIE batch
/// size differs, so its epochs get their own batches and graph.
#[test]
fn algorithm_switches_match_cpu() {
    for setup in [M1, FUSED] {
        switches_match_cpu(setup);
    }
}

fn switches_match_cpu(setup: Setup) {
    let (scan, object, probe) = start(setup.frame, 2, true);
    let mut cpu = Ptycho::new(&scan, object.clone(), probe.clone()).unwrap();
    let mut gpu = GpuPtycho::new(&ctx(), &scan, object, probe, setup.options).unwrap();
    let (o, p) = UPDATES[2];
    let steps = [
        Params::Rpie(ptychi::RPIE),
        Params::Ap(ApParams {
            obj_inertia: o,
            probe_inertia: p,
            calc_llk: 1,
            zero_phase_ramp: false,
        }),
        Params::Dm(DmParams {
            obj_inertia: o,
            probe_inertia: p,
            calc_llk: 1,
            ..DmParams::default()
        }),
        Params::Rpie(ptychi::RPIE),
        Params::Rpie(RpieParams {
            batch_size: 3,
            ..ptychi::RPIE
        }),
    ];
    for params in steps {
        let step = Step { params, count: 3 };
        cpu.run(&step).unwrap();
        gpu.run(&step).unwrap();
    }
    // ML's state outlives the AP between two ML calls, the second of which starts its
    // direction afresh with the probe no longer updated
    let ml = MlParams {
        update_object: true,
        update_probe: true,
        calc_llk: 1,
        zero_phase_ramp: false,
    };
    let only_obj = MlParams {
        update_probe: false,
        ..ml
    };
    cpu.ml(&ml, 3);
    gpu.ml(&ml, 3).unwrap();
    let step = Step {
        params: steps[1],
        count: 3,
    };
    cpu.run(&step).unwrap();
    gpu.run(&step).unwrap();
    cpu.ml(&only_obj, 3);
    gpu.ml(&only_obj, 3).unwrap();
    check(
        [
            llk_rel(gpu.history(), cpu.history()),
            rel_l2(&gpu.object().unwrap(), cpu.object()),
            rel_l2(&gpu.probe().unwrap(), cpu.probe()),
        ],
        [1e-4, 1e-4, 1e-4],
    );
}

/// An object set between AP calls, after graphs were captured, gives the bits and the LLK of
/// a fresh reconstruction built with that object and the probe so far; one of other dims is
/// refused.
#[test]
fn set_object_matches_a_fresh_start() {
    let ap = ApParams {
        obj_inertia: Some(0.01),
        probe_inertia: Some(0.001),
        calc_llk: 1,
        zero_phase_ramp: false,
    };
    for setup in [M1, FUSED] {
        let (scan, object, probe) = start(setup.frame, 2, true);
        let mut g = GpuPtycho::new(&ctx(), &scan, object.clone(), probe, setup.options).unwrap();
        g.ap(&ap, 2).unwrap();
        let turn = C64::from_polar(0.9, 0.3);
        let values = object.as_slice().iter().map(|&c| c * turn).collect();
        let other = Object::new(Stack::from_vec(object.dims(), values).unwrap());
        let wrong = Object::new(Stack::filled([1, 4, 4], C64::new(1.0, 0.0)));
        assert!(g.set_object(&wrong).is_err());
        g.set_object(&other).unwrap();
        let mut fresh =
            GpuPtycho::new(&ctx(), &scan, other, g.probe().unwrap(), setup.options).unwrap();
        g.ap(&ap, 1).unwrap();
        fresh.ap(&ap, 1).unwrap();
        assert_eq!(g.history()[2].llk, fresh.history()[0].llk, "{setup:?}");
        assert_eq!(g.object().unwrap(), fresh.object().unwrap(), "{setup:?}");
        assert_eq!(g.probe().unwrap(), fresh.probe().unwrap(), "{setup:?}");
    }
}

#[test]
fn gpu_rejects_what_it_cannot_run() {
    let (scan, object, probe) = start(M1.frame, 1, false);
    let two = Object::new(Stack::filled(
        [2, object.dims()[1], object.dims()[2]],
        C64::new(1.0, 0.0),
    ));
    let mut g = GpuPtycho::new(&ctx(), &scan, two, probe.clone(), M1.options).unwrap();
    assert!(g.rpie(&ptychi::RPIE, 1).is_err());
    let zero = GpuOptions {
        stack_size: 0,
        ..M1.options
    };
    assert!(GpuPtycho::new(&ctx(), &scan, object.clone(), probe.clone(), zero).is_err());
    let no_ring = GpuOptions {
        llk_slots: 0,
        ..M1.options
    };
    assert!(GpuPtycho::new(&ctx(), &scan, object.clone(), probe.clone(), no_ring).is_err());
    // the fused passes have no 32-point FFT; the error says which sides they take and how
    // to run without them
    assert!(!FUSED_SIDES.contains(&(M1.frame as u32)));
    let err = GpuPtycho::new(&ctx(), &scan, object, probe, FUSED.options)
        .err()
        .unwrap()
        .to_string();
    assert!(
        err.contains(&format!("{FUSED_SIDES:?}")) && err.contains("fused: false"),
        "{err}"
    );
}

/// AP, DM and ML cycles (AP's and ML's with an LLK, DM's and ML's first included), rPIE
/// epochs and LLK passes replayed from captured graphs give the eager bits: the object, the
/// probe and every LLK record read from the ring, across the parameter and algorithm switches
/// that make a new capture and calls that reuse the last one, with a short last stack and
/// batch, and the public LLK query between calls.
#[test]
fn graph_replays_bitwise_like_eager() {
    for setup in [M1, FUSED] {
        graph_like_eager(setup);
    }
}

fn graph_like_eager(setup: Setup) {
    let (scan, object, probe) = start(setup.frame, 2, true);
    let run = |graph: bool| {
        let options = GpuOptions {
            graph,
            ..setup.options
        };
        let mut g = GpuPtycho::new(&ctx(), &scan, object.clone(), probe.clone(), options).unwrap();
        let dm = DmParams {
            calc_llk: 3,
            zero_phase_ramp: false,
            ..DmParams::default()
        };
        let ap = ApParams {
            obj_inertia: Some(0.01),
            probe_inertia: Some(0.001),
            calc_llk: 3,
            zero_phase_ramp: false,
        };
        let rpie = RpieParams {
            calc_llk: 3,
            ..ptychi::RPIE
        };
        // new capture, new parameters, new algorithm, the same graph in a second call
        g.dm(&dm, 7).unwrap();
        g.dm(&DmParams { alpha: 0.1, ..dm }, 4).unwrap();
        g.ap(&ap, 7).unwrap();
        let query = g.llk().unwrap();
        g.ap(&ap, 4).unwrap();
        // batches of 4 leave a short last one; batches of 3 are new buffers and a new graph
        g.rpie(&rpie, 7).unwrap();
        g.rpie(&rpie, 4).unwrap();
        g.rpie(
            &RpieParams {
                batch_size: 3,
                ..rpie
            },
            4,
        )
        .unwrap();
        let ml = MlParams {
            update_object: true,
            update_probe: true,
            calc_llk: 3,
            zero_phase_ramp: false,
        };
        g.ml(&ml, 7).unwrap();
        g.ml(&ml, 4).unwrap();
        let bits = |s: &Stack<C64>| -> Vec<[u64; 2]> {
            s.as_slice()
                .iter()
                .map(|v| [v.re.to_bits(), v.im.to_bits()])
                .collect()
        };
        let llk_bits =
            |l: Llk| [l.poisson, l.gaussian, l.euclidian, l.nb_photons_calc].map(f64::to_bits);
        let mut llk: Vec<_> = g
            .history()
            .iter()
            .map(|r| (Some((r.cycle, r.algorithm)), llk_bits(r.llk)))
            .collect();
        llk.push((None, llk_bits(query)));
        (bits(&g.object().unwrap()), bits(&g.probe().unwrap()), llk)
    };
    let (eager, graph) = (run(false), run(true));
    assert_eq!(eager.2.len(), 3 + 3 + 2 + 2 + 3 + 2 + 2 + 3 + 2 + 1);
    assert!(eager.2 == graph.2, "{setup:?}: LLK records differ");
    assert!(eager.0 == graph.0, "{setup:?}: object differs");
    assert!(eager.1 == graph.1, "{setup:?}: probe differs");
}

/// A ring record of the frames a live scan has landed, `0 .. n`, against the CPU LLK over the
/// same frames: the GPU's LLK partials stack by stack, the frames counted in one range as their
/// stacks land and pushed by every stack that holds one below `n`, read back with `n` and the frames'
/// unmasked pixels; `n` inside a stack (10) and the whole scan. The mask differs from frame to
/// frame, so the count is the frames' own.
#[test]
fn ring_record_of_the_first_frames_matches_the_cpu_llk() {
    let (scan, _, _) = start(M1.frame, 2, false);
    let [nframe, ny, nx] = scan.counts.dims();
    let (nmode, npix) = (2, ny * nx);
    let data = scan.counts.as_slice().iter().enumerate();
    let data = data.map(|(i, &c)| if (i + i / npix) % 9 == 0 { MASKED } else { c });
    let counts = Counts::from_raw([nframe, ny, nx], data.collect()).unwrap();
    // Far fields in the GPU's scaling, `sum |psi|^2 / npix` per pixel, and the same for the CPU.
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        0.5 + (seed >> 40) as f32 / (1u64 << 23) as f32
    };
    let amp = (npix as f32 / nmode as f32).sqrt();
    let psi: Vec<C32> = (0..nframe * nmode * npix)
        .map(|_| C32 {
            re: amp * next(),
            im: amp * next(),
        })
        .collect();
    let root = (npix as f64).sqrt();
    let psi64: Vec<C64> = psi
        .iter()
        .map(|c| C64::new(f64::from(c.re) / root, f64::from(c.im) / root))
        .collect();

    let cuda = ctx();
    let kernels = Kernels::load(&cuda).unwrap();
    let stream = cuda.new_stream().unwrap();
    let ex = ExecutionContext::new(stream.clone());
    let side = [ny as u32, nx as u32];
    let dims = ViewDims {
        nframe: nframe as u32,
        nmode: nmode as u32,
        nprobe: nmode as u32,
        nobj: 1,
        frame: side,
        obj: side,
        quad_phase: 0.0,
    };
    let stacks: Vec<_> = (0..nframe as u32)
        .step_by(7)
        .map(|f| f..(f + 7).min(nframe as u32))
        .collect();
    let table = StackTable::new(&stream, &dims, &stacks).unwrap();
    let plane = table.nf_max() as usize * nmode * npix;
    let staged: Vec<DeviceBuffer<C32>> = stacks
        .iter()
        .map(|r| {
            let mut v =
                psi[r.start as usize * nmode * npix..r.end as usize * nmode * npix].to_vec();
            v.resize(plane, C32::default());
            DeviceBuffer::from_host(&stream, &v).unwrap()
        })
        .collect();
    let dev_counts = DeviceBuffer::from_host(&stream, counts.as_slice()).unwrap();
    let mut scratch = LlkScratch::new(&stream, &dims, table.nf_max()).unwrap();
    let mut partials =
        DeviceBuffer::<LlkPartial>::zeroed(&stream, table.nf_max() as usize).unwrap();
    let mut ring = LlkRing::new(&kernels, &stream, &dims, 2).unwrap();

    let ns = [10, nframe as u32];
    let mut counted = 0;
    for (r, &n) in ns.iter().enumerate() {
        let landed: Vec<usize> = (0..stacks.len()).filter(|&s| stacks[s].start < n).collect();
        let end = stacks[*landed.last().unwrap()].end;
        ring.device().count(&ex, counted..end, &dev_counts).unwrap();
        counted = end;
        let mut rec = ring.device().record();
        for &s in &landed {
            let stack = table.at(s);
            kernels
                .llk(
                    &ex,
                    stack,
                    &dev_counts,
                    &staged[s],
                    &mut scratch,
                    &mut partials,
                )
                .unwrap();
            rec.push(&ex, &dims, stack, &partials).unwrap();
        }
        let ended = rec.end(&ex).unwrap();
        ring.enqueued(&ex, &ended, r, n).unwrap();
    }

    let obs = Observed::from_counts(&counts);
    let got = ring.flush(&ex).unwrap();
    assert_eq!(got.len(), ns.len());
    for (r, t) in got {
        let n = ns[r] as usize;
        let cpu = llk(&obs, 0..n, nmode, &psi64[..n * nmode * npix]);
        let nb_obs = counts.as_slice()[..n * npix]
            .iter()
            .filter(|&&c| c != MASKED)
            .count();
        assert_eq!((t.nframe as usize, t.nb_obs), (n, nb_obs), "record {r}");
        let s = t.sums;
        for (what, g, c) in [
            ("poisson", s.poisson, cpu.poisson),
            ("gaussian", s.gaussian, cpu.gaussian),
            ("euclidean", s.euclidean, cpu.euclidian),
            ("icalc", s.icalc, cpu.nb_photons_calc),
        ] {
            let rel = (g - c).abs() / c.abs();
            println!("record {r} (n {n}) {what}: rel {rel:.2e}");
            assert!(rel < 1e-5, "record {r} (n {n}) {what}: {g} vs {c}");
        }
    }
}

/// Frame `f`'s flux in [`featured_start`]: 0.6 to 1.4 times the scenario's, unordered.
fn flux(f: usize) -> f64 {
    1.0 + 0.4 * (2.3 * f as f64).sin()
}

/// Detector pixel `(y, x)`'s background in [`featured_start`]: 1 to 5 counts, smooth.
fn truth_background(y: usize, x: usize) -> f64 {
    3.0 + 2.0 * (0.21 * y as f64).sin() * (0.13 * x as f64 + 0.5).cos()
}

/// Whether pixel `(y, x)` is masked in every frame of [`featured_start`]'s scan.
fn featured_mask(y: usize, x: usize) -> bool {
    y == 3 || x == 20
}

/// [`start`]'s scan on the fused side masked by [`featured_mask`], with frame `f`'s counts
/// times [`flux`]`(f)` where `with_flux`, plus [`truth_background`] where `with_background`,
/// rounded; and the truth object and probe it was drawn from scaled to that scan, so that the
/// flux and the background are what the model misses.
fn featured_start(nprobe: usize, with_flux: bool, with_background: bool) -> (Scan, Object, Probe) {
    let frame = FUSED.frame;
    let (sim, object, probe) =
        common::scenario::truth(frame, NPOS, nprobe, 2e5, frame as f64 / 8.0);
    let mut scan = mask(sim.scan, featured_mask);
    let dims = scan.counts.dims();
    let [_, ny, nx] = dims;
    let data = scan
        .counts
        .as_slice()
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            if c == MASKED {
                return c;
            }
            let (f, j) = (i / (ny * nx), i % (ny * nx));
            let mut c = f64::from(c);
            if with_flux {
                c *= flux(f);
            }
            if with_background {
                c += truth_background(j / nx, j % nx);
            }
            let c = c.round();
            assert!(c < f64::from(MASKED), "count {c} overflows");
            c as u16
        })
        .collect();
    scan.counts = Counts::from_raw(dims, data).unwrap();
    let mut p = Ptycho::new(&scan, object, probe).unwrap();
    p.scale_obj_probe();
    let (object, probe) = p.into_modes();
    (scan, object, probe)
}

/// AP, DM or ML cycles on either backend.
#[derive(Clone, Copy, Debug)]
enum Cycles {
    Ap(ApParams),
    Dm(DmParams),
    Ml(MlParams),
}

impl Cycles {
    fn cpu(self, p: &mut Ptycho, ncycle: usize) {
        match self {
            Cycles::Ap(a) => p.ap(&a, ncycle),
            Cycles::Dm(d) => p.dm(&d, ncycle),
            Cycles::Ml(m) => p.ml(&m, ncycle),
        }
    }

    fn gpu(self, g: &mut GpuPtycho, ncycle: usize) {
        match self {
            Cycles::Ap(a) => g.ap(&a, ncycle),
            Cycles::Dm(d) => g.dm(&d, ncycle),
            Cycles::Ml(m) => g.ml(&m, ncycle),
        }
        .unwrap()
    }
}

/// AP, DM and ML updating the object and the probe, each with an LLK every `calc_llk` cycles.
fn feature_cycles(calc_llk: usize) -> [Cycles; 3] {
    [
        Cycles::Ap(ApParams {
            obj_inertia: Some(0.01),
            probe_inertia: Some(0.001),
            calc_llk,
            zero_phase_ramp: true,
        }),
        Cycles::Dm(DmParams {
            calc_llk,
            ..DmParams::default()
        }),
        Cycles::Ml(MlParams {
            update_object: true,
            update_probe: true,
            calc_llk,
            zero_phase_ramp: true,
        }),
    ]
}

/// `ncycle` of `cycles` from `start` modelling `features` on the GPU and on the CPU, and
/// without them on the CPU. Checks the GPU's LLK records, object, probe and features within
/// 1e-4 of the CPU's `f64`, and returns the CPU runs with and without the features.
fn features_match_cpu(
    cycles: Cycles,
    (scan, object, probe): &(Scan, Object, Probe),
    features: &Features,
    ncycle: usize,
) -> (Ptycho, Ptycho) {
    let new = || Ptycho::new(scan, object.clone(), probe.clone()).unwrap();
    let (mut cpu, mut plain) = (new(), new());
    cpu.set_features(features.clone()).unwrap();
    let mut gpu =
        GpuPtycho::new(&ctx(), scan, object.clone(), probe.clone(), FUSED.options).unwrap();
    gpu.set_features(features).unwrap();
    cycles.cpu(&mut cpu, ncycle);
    cycles.cpu(&mut plain, ncycle);
    cycles.gpu(&mut gpu, ncycle);
    assert_eq!(gpu.cycle(), cpu.cycle());
    let errs = [
        llk_rel(gpu.history(), cpu.history()),
        rel_l2(&gpu.object().unwrap(), cpu.object()),
        rel_l2(&gpu.probe().unwrap(), cpu.probe()),
    ];
    let (g, c) = (gpu.features().unwrap(), cpu.features());
    let worst = |g: &Option<Vec<f64>>, c: &Option<Vec<f64>>| {
        assert_eq!(g.is_some(), c.is_some());
        (g.iter().flatten().zip(c.iter().flatten()))
            .map(|(a, b)| ((a - b) / b).abs())
            .fold(0.0, f64::max)
    };
    let scale = worst(&g.scale, &c.scale);
    let background = worst(&g.background, &c.background);
    println!(
        "{cycles:?}: llk {:.2e} object {:.2e} probe {:.2e} scale {scale:.2e} \
         background {background:.2e}",
        errs[0], errs[1], errs[2],
    );
    check(errs, [1e-4, 1e-4, 1e-4]);
    assert!(scale < 1e-4, "scale: {scale:e}");
    assert!(background < 1e-4, "background: {background:e}");
    (cpu, plain)
}

/// The last LLK record's Poisson LLK.
fn last_llk(p: &Ptycho) -> f64 {
    p.history().last().unwrap().llk.poisson
}

/// A per-frame intensity scale refined from 1 by 20 AP, DM and ML cycles on a scan whose
/// frames' flux varies, as [`features_match_cpu`] checks it. On the CPU the scale follows the
/// frames' flux relative to its mean and brings the last LLK below that of a run without it.
#[test]
fn scale_matches_cpu() {
    const NCYCLE: usize = 20;
    let start = featured_start(2, true, false);
    let ones = Features {
        scale: Some(vec![1.0; NPOS]),
        ..Features::default()
    };
    let mean = (0..NPOS).map(flux).sum::<f64>() / NPOS as f64;
    // Measured 2.2e-2, 9.3e-2 and 2.7e-2 from 0.67 at the start.
    for (cycles, flux_tol) in feature_cycles(1).into_iter().zip([0.05, 0.15, 0.05]) {
        let (cpu, plain) = features_match_cpu(cycles, &start, &ones, NCYCLE);
        let off_flux = (cpu.features().scale.as_ref().unwrap().iter().enumerate())
            .map(|(f, s)| (s * mean / flux(f) - 1.0).abs())
            .fold(0.0, f64::max);
        println!(
            "scale {cycles:?}: off flux {off_flux:.2e}; last llk {:.4e}, {:.4e} without",
            last_llk(&cpu),
            last_llk(&plain),
        );
        assert!(off_flux < flux_tol, "{cycles:?}: off flux {off_flux:e}");
        assert!(
            last_llk(&cpu) < last_llk(&plain),
            "{cycles:?}: the scale left the LLK"
        );
    }
}

/// A flat background of 1 count on `scan`'s detector.
fn flat_background(scan: &Scan) -> Vec<f64> {
    let [_, ny, nx] = scan.counts.dims();
    vec![1.0; ny * nx]
}

/// The RMS of `b`'s error relative to [`truth_background`] over the pixels [`featured_mask`]
/// leaves, `[ny, nx]` in detector order; asserts the masked pixels kept the flat start.
fn off_truth_background(b: &[f64], [ny, nx]: [usize; 2]) -> f64 {
    let (mut e, mut t) = (0.0, 0.0);
    for (j, &b) in b.iter().enumerate().take(ny * nx) {
        let (y, x) = (j / nx, j % nx);
        if featured_mask(y, x) {
            assert_eq!(b, 1.0, "masked pixel ({y}, {x}) stepped");
            continue;
        }
        let w = truth_background(y, x);
        e += (b - w).powi(2);
        t += w * w;
    }
    (e / t).sqrt()
}

/// A detector background refined from a flat 1 count by 20 AP, DM and ML cycles on a scan with
/// one, as [`features_match_cpu`] checks it. On the CPU the background nears the truth's on the
/// unmasked pixels, the masked ones keep the start, and the last LLK falls below that of a run
/// without it.
#[test]
fn background_matches_cpu() {
    const NCYCLE: usize = 20;
    let start = featured_start(2, false, true);
    let [_, ny, nx] = start.0.counts.dims();
    let flat = Features {
        background: Some(flat_background(&start.0)),
        ..Features::default()
    };
    let at_start = off_truth_background(flat.background.as_ref().unwrap(), [ny, nx]);
    // Measured 2.8e-1, 3.0e-1 and 2.7e-1 from 7.0e-1 at the start.
    for (cycles, truth_tol) in feature_cycles(1).into_iter().zip([0.4, 0.4, 0.4]) {
        let (cpu, plain) = features_match_cpu(cycles, &start, &flat, NCYCLE);
        let off = off_truth_background(cpu.features().background.as_ref().unwrap(), [ny, nx]);
        println!(
            "background {cycles:?}: off truth {off:.2e} from {at_start:.2e}; \
             last llk {:.4e}, {:.4e} without",
            last_llk(&cpu),
            last_llk(&plain),
        );
        assert!(off < truth_tol, "{cycles:?}: off truth {off:e}");
        assert!(
            last_llk(&cpu) < last_llk(&plain),
            "{cycles:?}: the background left the LLK"
        );
    }
}

/// Both features at once on a scan with both effects, as [`features_match_cpu`] checks it:
/// every feature's step takes its sums before any steps, on both backends.
#[test]
fn scale_and_background_match_cpu() {
    const NCYCLE: usize = 20;
    let start = featured_start(2, true, true);
    let both = Features {
        scale: Some(vec![1.0; NPOS]),
        background: Some(flat_background(&start.0)),
    };
    for cycles in feature_cycles(1) {
        let (cpu, plain) = features_match_cpu(cycles, &start, &both, NCYCLE);
        println!(
            "both {cycles:?}: last llk {:.4e}, {:.4e} without",
            last_llk(&cpu),
            last_llk(&plain),
        );
        assert!(
            last_llk(&cpu) < last_llk(&plain),
            "{cycles:?}: the features left the LLK"
        );
    }
}

/// Features need the fused passes, one finite, positive scale per frame and one finite,
/// non-negative background value per detector pixel; rPIE and BH model none; a refused set
/// leaves the features modelled.
#[test]
fn features_are_refused_where_they_cannot_run() {
    let (scan, object, probe) = start(M1.frame, 1, false);
    let mut m1 = GpuPtycho::new(&ctx(), &scan, object, probe, M1.options).unwrap();
    for features in [
        Features {
            scale: Some(vec![1.0; NPOS]),
            ..Features::default()
        },
        Features {
            background: Some(flat_background(&scan)),
            ..Features::default()
        },
    ] {
        let err = m1.set_features(&features).unwrap_err().to_string();
        assert!(err.contains("fused: true"), "{err}");
    }
    m1.set_features(&Features::default()).unwrap();

    let (scan, object, probe) = featured_start(1, true, true);
    let mut cpu = Ptycho::new(&scan, object.clone(), probe.clone()).unwrap();
    let mut gpu = GpuPtycho::new(&ctx(), &scan, object, probe, FUSED.options).unwrap();
    let flat = flat_background(&scan);
    let half = Features {
        scale: Some(vec![0.5; NPOS]),
        background: Some(flat.iter().map(|b| 2.0 * b).collect()),
    };
    cpu.set_features(half.clone()).unwrap();
    gpu.set_features(&half).unwrap();
    let one_bad = |mut v: Vec<f64>, bad: f64| {
        let n = v.len();
        v[n / 2] = bad;
        v
    };
    let bad_scales = [
        vec![1.0; NPOS - 1],
        one_bad(vec![1.0; NPOS], 0.0),
        one_bad(vec![1.0; NPOS], -1.0),
        one_bad(vec![1.0; NPOS], f64::NAN),
        one_bad(vec![1.0; NPOS], f64::INFINITY),
    ];
    let bad_backgrounds = [
        flat[1..].to_vec(),
        one_bad(flat.clone(), -1.0),
        one_bad(flat.clone(), f64::NAN),
        one_bad(flat.clone(), f64::INFINITY),
    ];
    let bad = (bad_scales.into_iter()).map(|s| Features {
        scale: Some(s),
        background: Some(flat.clone()),
    });
    let bad = bad.chain(bad_backgrounds.into_iter().map(|b| Features {
        scale: Some(vec![1.0; NPOS]),
        background: Some(b),
    }));
    for bad in bad {
        assert!(cpu.set_features(bad.clone()).is_err(), "{bad:?}");
        assert!(gpu.set_features(&bad).is_err(), "{bad:?}");
    }
    assert_eq!(cpu.features(), &half);
    assert_eq!(gpu.features().unwrap(), half);
    assert!(cpu.rpie(&ptychi::RPIE, 1).is_err());
    assert!(gpu.rpie(&ptychi::RPIE, 1).is_err());
    let bh = BhParams {
        batch_size: NPOS,
        probe_rho: 0.01,
        seed: 7,
    };
    assert!(gpu.bh(&bh, 1, 1).is_err());
    assert_eq!(gpu.cycle(), 0);
}

/// AP, DM and ML cycles modelling a scale and a background replay from graphs the eager bits:
/// the object, the probe, both features and every LLK record, across features written in place
/// between calls and the public LLK query. Dropping the features before any cycle gives a fresh
/// start's bits.
#[test]
fn feature_graphs_replay_bitwise_like_eager() {
    let (scan, object, probe) = featured_start(2, true, true);
    let flat = flat_background(&scan);
    let bits = |s: &Stack<C64>| -> Vec<[u64; 2]> {
        s.as_slice()
            .iter()
            .map(|v| [v.re.to_bits(), v.im.to_bits()])
            .collect()
    };
    let llk_bits =
        |l: Llk| [l.poisson, l.gaussian, l.euclidian, l.nb_photons_calc].map(f64::to_bits);
    let state = |g: &mut GpuPtycho| {
        let mut llk: Vec<_> = g
            .history()
            .iter()
            .map(|r| (Some((r.cycle, r.algorithm)), llk_bits(r.llk)))
            .collect();
        llk.push((None, llk_bits(g.llk().unwrap())));
        let features = g.features().unwrap();
        let feature_bits =
            |v: Option<Vec<f64>>| v.map(|v| v.into_iter().map(f64::to_bits).collect::<Vec<_>>());
        (
            bits(&g.object().unwrap()),
            bits(&g.probe().unwrap()),
            [
                feature_bits(features.scale),
                feature_bits(features.background),
            ],
            llk,
        )
    };
    let new = |graph: bool| {
        let options = GpuOptions {
            graph,
            ..FUSED.options
        };
        GpuPtycho::new(&ctx(), &scan, object.clone(), probe.clone(), options).unwrap()
    };
    let start = Features {
        scale: Some(vec![1.0; NPOS]),
        background: Some(flat.clone()),
    };
    let run = |graph: bool| {
        let mut g = new(graph);
        g.set_features(&start).unwrap();
        let [ap, dm, ml] = feature_cycles(3);
        ap.gpu(&mut g, 4);
        let query = g.llk().unwrap();
        g.set_features(&Features {
            scale: Some((0..NPOS).map(flux).collect()),
            background: Some(flat.iter().map(|b| 0.5 * b).collect()),
        })
        .unwrap();
        dm.gpu(&mut g, 4);
        ml.gpu(&mut g, 4);
        ap.gpu(&mut g, 4);
        let mut s = state(&mut g);
        s.3.push((None, llk_bits(query)));
        s
    };
    let (eager, graph) = (run(false), run(true));
    assert_eq!(eager.3.len(), 2 + 2 + 2 + 2 + 2);
    assert!(eager.3 == graph.3, "LLK records differ");
    assert!(eager.0 == graph.0, "object differs");
    assert!(eager.1 == graph.1, "probe differs");
    assert!(eager.2 == graph.2, "features differ");

    let [ap, ..] = feature_cycles(3);
    let mut fresh = new(true);
    ap.gpu(&mut fresh, 4);
    let mut dropped = new(true);
    dropped.set_features(&start).unwrap();
    dropped.set_features(&Features::default()).unwrap();
    ap.gpu(&mut dropped, 4);
    let (fresh, dropped) = (state(&mut fresh), state(&mut dropped));
    assert!(dropped.2 == [None, None]);
    assert!(
        fresh == dropped,
        "dropping the features differs from a fresh start"
    );
}
