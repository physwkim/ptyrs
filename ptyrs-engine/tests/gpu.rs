//! The GPU backend against the CPU oracle: per-record LLK and the final object and probe of AP,
//! DM, rPIE and ML updating the object, the probe or both, from the same scaled start; AP and
//! DM by M1's launches and by the fused passes; and the ML gradients.

use std::sync::Arc;

use cuda_core::CudaContext;
use ptyrs_core::{C64, Counts, MASKED, Object, Probe, Scan, Stack};
use ptyrs_engine::{
    ApParams, Backend, DmParams, GpuOptions, GpuPtycho, Llk, LlkRecord, MlParams, Params, Ptycho,
    RpieParams, Step,
};

mod common;

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

/// The fused passes on 128-pixel frames, the smallest side they run.
const FUSED: Setup = Setup {
    frame: 128,
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
        let dims = scan.counts.dims();
        let data = scan
            .counts
            .as_slice()
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let (y, x) = ((i / frame) % frame, i % frame);
                if y == 3 || x == 20 { MASKED } else { c }
            })
            .collect();
        scan.counts = Counts::from_raw(dims, data).unwrap();
    }
    let mut p = Ptycho::new(&scan, object, probe).unwrap();
    p.scale_obj_probe();
    let (object, probe) = p.into_modes();
    (scan, object, probe)
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
            ..common::RPIE
        };
        check(
            compare(M1, 2, true, Params::Rpie(params), 10),
            [1e-4, 1e-4, 1e-4],
        );
    }
}

/// ML's gradients of the same estimate, in `f32` against `f64`, on M1's and the fused passes'
/// setups (ML itself always runs M1's launches).
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

/// ML's direction (β from the last gradient across both arrays) and Poisson quadratic step
/// against the CPU, for each pair of updated arrays.
#[test]
fn ml_matches_cpu() {
    for (o, p) in UPDATES {
        let params = MlParams {
            update_object: o.is_some(),
            update_probe: p.is_some(),
            calc_llk: 1,
            zero_phase_ramp: true,
        };
        let (scan, object, probe) = start(M1.frame, 2, true);
        let mut cpu = Ptycho::new(&scan, object.clone(), probe.clone()).unwrap();
        let mut gpu = GpuPtycho::new(&ctx(), &scan, object, probe, M1.options).unwrap();
        cpu.ml(&params, 10);
        gpu.ml(&params, 10).unwrap();
        assert_eq!(gpu.cycle(), cpu.cycle());
        let errs = [
            llk_rel(gpu.history(), cpu.history()),
            rel_l2(&gpu.object().unwrap(), cpu.object()),
            rel_l2(&gpu.probe().unwrap(), cpu.probe()),
        ];
        eprintln!(
            "ml {params:?}: llk {:.2e} object {:.2e} probe {:.2e}",
            errs[0], errs[1], errs[2]
        );
        check(errs, [1e-4, 1e-4, 1e-4]);
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
        Params::Rpie(common::RPIE),
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
        Params::Rpie(common::RPIE),
        Params::Rpie(RpieParams {
            batch_size: 3,
            ..common::RPIE
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

#[test]
fn gpu_rejects_what_it_cannot_run() {
    let (scan, object, probe) = start(M1.frame, 1, false);
    let two = Object::new(Stack::filled(
        [2, object.dims()[1], object.dims()[2]],
        C64::new(1.0, 0.0),
    ));
    let mut g = GpuPtycho::new(&ctx(), &scan, two, probe.clone(), M1.options).unwrap();
    assert!(g.rpie(&common::RPIE, 1).is_err());
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
    // the fused passes have no 32-point FFT
    assert!(GpuPtycho::new(&ctx(), &scan, object, probe, FUSED.options).is_err());
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
            ..common::RPIE
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
