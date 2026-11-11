//! The GPU backend against the CPU oracle: per-record LLK and the final object and probe of AP,
//! DM and rPIE updating the object, the probe or both, from the same scaled start.

use std::sync::Arc;

use cuda_core::CudaContext;
use ptyrs_core::{C64, Counts, MASKED, Object, Probe, Scan, Stack};
use ptyrs_engine::{ApParams, DmParams, GpuOptions, GpuPtycho, LlkRecord, Ptycho, RpieParams};

mod common;

/// 30 positions: stacks of 7 and batches of 4 both leave a shorter last one.
const NPOS: usize = 30;
const STACK: GpuOptions = GpuOptions { stack_size: 7 };

fn ctx() -> Arc<CudaContext> {
    CudaContext::new(0).expect("CUDA device 0")
}

/// The parity scenario at 32 pixels, scaled on the CPU; `masked` masks row 3 and column 20
/// of every frame.
fn start(nprobe: usize, masked: bool) -> (Scan, Object, Probe) {
    let (mut scan, object, probe) = common::dm_scenario(32, NPOS, nprobe, 2e5);
    if masked {
        let dims = scan.counts.dims();
        let data = scan
            .counts
            .as_slice()
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let (y, x) = ((i / 32) % 32, i % 32);
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

#[derive(Clone, Copy, Debug)]
enum Run {
    Ap(ApParams),
    Dm(DmParams),
    Rpie(RpieParams),
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

/// Runs `run` for `ncycle` on both backends and returns the worst LLK and the object and probe
/// relative errors.
fn compare(nprobe: usize, masked: bool, run: Run, ncycle: usize) -> [f64; 3] {
    let (scan, object, probe) = start(nprobe, masked);
    let mut cpu = Ptycho::new(&scan, object.clone(), probe.clone()).unwrap();
    let mut gpu = GpuPtycho::new(&ctx(), &scan, object, probe, STACK).unwrap();
    match run {
        Run::Ap(p) => {
            cpu.ap(&p, ncycle);
            gpu.ap(&p, ncycle).unwrap();
        }
        Run::Dm(p) => {
            cpu.dm(&p, ncycle);
            gpu.dm(&p, ncycle).unwrap();
        }
        Run::Rpie(p) => {
            cpu.rpie(&p, ncycle).unwrap();
            gpu.rpie(&p, ncycle).unwrap();
        }
    }
    assert_eq!(gpu.cycle(), cpu.cycle());
    let errs = [
        llk_rel(gpu.history(), cpu.history()),
        rel_l2(&gpu.object().unwrap(), cpu.object()),
        rel_l2(&gpu.probe().unwrap(), cpu.probe()),
    ];
    eprintln!(
        "{run:?}: llk {:.2e} object {:.2e} probe {:.2e}",
        errs[0], errs[1], errs[2]
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
    for (o, p) in UPDATES {
        let params = ApParams {
            obj_inertia: o,
            probe_inertia: p,
            calc_llk: 1,
            zero_phase_ramp: true,
        };
        check(compare(2, true, Run::Ap(params), 10), [1e-4, 1e-4, 1e-4]);
    }
}

#[test]
fn dm_matches_cpu() {
    for (o, p) in UPDATES {
        let params = DmParams {
            obj_inertia: o,
            probe_inertia: p,
            calc_llk: 1,
            ..DmParams::default()
        };
        check(compare(2, true, Run::Dm(params), 10), [1e-4, 1e-4, 1e-4]);
    }
    // two object-then-probe rounds per cycle, one mode, no mask
    let params = DmParams {
        loop_obj_probe: 2,
        calc_llk: 3,
        ..DmParams::default()
    };
    check(compare(1, false, Run::Dm(params), 7), [1e-4, 1e-4, 1e-4]);
}

#[test]
fn rpie_matches_cpu() {
    for (o, p) in UPDATES {
        let params = RpieParams {
            obj_step: o.map(|_| 1.0),
            probe_step: p.map(|_| 1.0),
            ..common::RPIE
        };
        check(compare(2, true, Run::Rpie(params), 10), [1e-4, 1e-4, 1e-4]);
    }
}

#[test]
fn gpu_rejects_what_it_cannot_run() {
    let (scan, object, probe) = start(1, false);
    let two = Object::new(Stack::filled(
        [2, object.dims()[1], object.dims()[2]],
        C64::new(1.0, 0.0),
    ));
    let mut g = GpuPtycho::new(&ctx(), &scan, two, probe.clone(), STACK).unwrap();
    assert!(g.rpie(&common::RPIE, 1).is_err());
    let zero = GpuOptions { stack_size: 0 };
    assert!(GpuPtycho::new(&ctx(), &scan, object, probe, zero).is_err());
}
