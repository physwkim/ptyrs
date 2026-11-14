//! The GPU passes that run a propagator's plane stage besides LSQML's and BH's (rPIE's, and
//! AP's, DM's and RAAR's), on the cases of `common/near.rs`.

use ptyrs::base::{C64, Stack};
use ptyrs::engine::{ApParams, DmParams, Features, GpuPtycho, Ptycho, RaarParams, RpieParams};

#[path = "common/near.rs"]
mod near;
use near::{NPOS, cases, drawn_offsets};

#[path = "common/worst.rs"]
mod worst;
use worst::worst_of;

fn rel_l2(a: &Stack<C64>, b: &Stack<C64>) -> f64 {
    let (mut e, mut t) = (0.0, 0.0);
    for (x, y) in a.as_slice().iter().zip(b.as_slice()) {
        e += (x - y).norm_sqr();
        t += y.norm_sqr();
    }
    (e / t).sqrt()
}

const RPIE: RpieParams = RpieParams {
    obj_step: Some(1.0),
    probe_step: Some(1.0),
    obj_alpha: 0.5,
    probe_alpha: 0.5,
    batch_size: 4,
    calc_llk: 1,
};

/// rPIE under each propagator, with fixed positions and refining drawn offsets: every epoch
/// lowers the LLK the device records, the last is the CPU's of the arrays it ends on (fixed
/// positions), and the graphs replay the eager run bit for bit.
#[test]
fn rpie_descends_and_replays_under_every_propagator() {
    const NEPOCH: usize = 6;
    let moved = Features {
        position: Some(drawn_offsets()),
        ..Features::default()
    };
    for c in cases() {
        for features in [None, Some(&moved)] {
            let what = format!("{} positions {}", c.name, features.is_some());
            let run = |graph| {
                let mut gpu = c.gpu(graph);
                if let Some(f) = features {
                    gpu.set_features(f).unwrap();
                }
                let start = gpu.llk().unwrap().poisson;
                gpu.rpie(&RPIE, NEPOCH).unwrap();
                let trace: Vec<f64> = gpu.history().iter().map(|r| r.llk.poisson).collect();
                let arrays = (gpu.object().unwrap(), gpu.probe().unwrap());
                (start, trace, arrays, gpu.features().unwrap().position)
            };
            let (start, trace, arrays, offsets) = run(false);
            eprintln!("{what}: start {start:.4e}, epochs {trace:.4?}");
            assert_eq!(trace.len(), NEPOCH, "{what}");
            let mut last = start;
            for (e, &v) in trace.iter().enumerate() {
                assert!(v < last, "{what} epoch {e}: {v:e} after {last:e}");
                last = v;
            }
            if features.is_none() {
                let want = c.llk(&arrays.0, &arrays.1);
                let err = (last - want).abs() / want;
                eprintln!("{what}: CPU LLK of the result {want:.6e}, {err:.2e}");
                assert!(err < 1e-5, "{what}: recorded {last:e}, CPU {want:e}");
            }
            let graphs = run(true);
            assert_eq!(graphs.1, trace, "{what}: LLK from graphs");
            assert_eq!(graphs.2.0.as_slice(), arrays.0.as_slice(), "{what}: object");
            assert_eq!(graphs.2.1.as_slice(), arrays.1.as_slice(), "{what}: probe");
            assert_eq!(graphs.3, offsets, "{what}: offsets");
        }
    }
}

/// rPIE under each propagator against the CPU's, with fixed positions and refining drawn
/// offsets, eagerly and from graphs: each epoch's recorded LLK, and the object, the probe
/// and the offsets (over the largest) after the last.
#[test]
fn rpie_matches_cpu_under_every_propagator() {
    const NEPOCH: usize = 6;
    // Measured worst: LLK 3.7e-6, object 3.4e-7, probe 4.6e-6, offsets 3.1e-7.
    const TOL: [f64; 4] = [2e-5, 2e-6, 2.5e-5, 2e-6];
    let moved = Features {
        position: Some(drawn_offsets()),
        ..Features::default()
    };
    for c in cases() {
        for features in [None, Some(&moved)] {
            let (object, probe) = (c.object.clone(), c.probe.clone());
            let mut cpu = Ptycho::with_optics(&c.scan, c.optics(), object, probe).unwrap();
            if let Some(f) = features {
                cpu.set_features(f.clone()).unwrap();
            }
            cpu.rpie(&RPIE, NEPOCH).unwrap();
            let want: Vec<f64> = cpu.history().iter().map(|r| r.llk.poisson).collect();
            assert_eq!(want.len(), NEPOCH);
            for graph in [false, true] {
                let what = format!("{} positions {} graph {graph}", c.name, features.is_some());
                let mut gpu = c.gpu(graph);
                if let Some(f) = features {
                    gpu.set_features(f).unwrap();
                }
                gpu.rpie(&RPIE, NEPOCH).unwrap();
                let got: Vec<f64> = gpu.history().iter().map(|r| r.llk.poisson).collect();
                assert_eq!(got.len(), NEPOCH, "{what}");
                let offset = match &cpu.features().position {
                    Some(want) => {
                        let got = gpu.features().unwrap().position.unwrap();
                        let pairs = || got.iter().flatten().zip(want.iter().flatten());
                        worst_of(pairs().map(|(g, w)| (g - w).abs()))
                            / worst_of(pairs().map(|(_, w)| w.abs()))
                    }
                    None => 0.0,
                };
                let errs = [
                    worst_of(got.iter().zip(&want).map(|(g, w)| (g - w).abs() / w)),
                    rel_l2(&gpu.object().unwrap(), cpu.object()),
                    rel_l2(&gpu.probe().unwrap(), cpu.probe()),
                    offset,
                ];
                eprintln!(
                    "{what}: llk {:.2e} object {:.2e} probe {:.2e} offset {:.2e}, llk {:.4e} -> {:.4e}",
                    errs[0],
                    errs[1],
                    errs[2],
                    errs[3],
                    want[0],
                    want[NEPOCH - 1]
                );
                for ((name, e), tol) in ["llk", "object", "probe", "offset"]
                    .iter()
                    .zip(errs)
                    .zip(TOL)
                {
                    assert!(e < tol, "{what} {name}: {e:e}");
                }
            }
        }
    }
}

/// AP, DM or RAAR on the CPU and on the GPU.
type Projections = (
    &'static str,
    fn(&mut Ptycho, usize),
    fn(&mut GpuPtycho, usize),
);

const AP: ApParams = ApParams {
    obj_inertia: Some(0.01),
    probe_inertia: Some(0.001),
    calc_llk: 1,
    zero_phase_ramp: true,
};

const DM: DmParams = DmParams {
    obj_inertia: Some(0.01),
    probe_inertia: Some(0.001),
    alpha: 0.02,
    loop_obj_probe: 1,
    calc_llk: 1,
    zero_phase_ramp: true,
};

const RAAR: RaarParams = RaarParams {
    obj_inertia: Some(0.01),
    probe_inertia: Some(0.001),
    beta: 0.9,
    loop_obj_probe: 1,
    calc_llk: 1,
    zero_phase_ramp: true,
};

const PROJECTIONS: [Projections; 3] = [
    ("AP", |p, n| p.ap(&AP, n), |g, n| g.ap(&AP, n).unwrap()),
    ("DM", |p, n| p.dm(&DM, n), |g, n| g.dm(&DM, n).unwrap()),
    (
        "RAAR",
        |p, n| p.raar(&RAAR, n),
        |g, n| g.raar(&RAAR, n).unwrap(),
    ),
];

/// AP, DM and RAAR under each propagator against the CPU's at fixed positions, plain and
/// stepping a scale and a background, eagerly and from graphs: each recorded LLK, and the
/// object, the probe and the features after the last cycle.
#[test]
fn projections_match_cpu_under_every_propagator() {
    const NCYCLE: usize = 8;
    // Measured worst: LLK 6.7e-6, object 1.6e-6, probe 1.5e-6, features 1.7e-5.
    const TOL: [f64; 4] = [4e-5, 1e-5, 1e-5, 1e-4];
    let stepped = |frame: usize| Features {
        scale: Some(vec![1.0; NPOS]),
        background: Some(vec![1.0; frame * frame]),
        position: None,
    };
    for c in cases() {
        let [_, ny, _] = c.scan.counts.dims();
        let stepped = stepped(ny);
        for (name, on_cpu, on_gpu) in PROJECTIONS {
            for features in [None, Some(&stepped)] {
                let (object, probe) = (c.object.clone(), c.probe.clone());
                let mut cpu = Ptycho::with_optics(&c.scan, c.optics(), object, probe).unwrap();
                if let Some(f) = features {
                    cpu.set_features(f.clone()).unwrap();
                }
                on_cpu(&mut cpu, NCYCLE);
                let want: Vec<f64> = cpu.history().iter().map(|r| r.llk.poisson).collect();
                assert_eq!(want.len(), NCYCLE);
                for graph in [false, true] {
                    let what = format!(
                        "{name} {} features {} graph {graph}",
                        c.name,
                        features.is_some()
                    );
                    let mut gpu = c.gpu(graph);
                    if let Some(f) = features {
                        gpu.set_features(f).unwrap();
                    }
                    on_gpu(&mut gpu, NCYCLE);
                    let got: Vec<f64> = gpu.history().iter().map(|r| r.llk.poisson).collect();
                    assert_eq!(got.len(), NCYCLE, "{what}");
                    let (gf, cf) = (gpu.features().unwrap(), cpu.features());
                    let flat = |f: &Features| -> Vec<f64> {
                        let (s, b) = (f.scale.iter().flatten(), f.background.iter().flatten());
                        s.chain(b).copied().collect()
                    };
                    let (gf, cf) = (flat(&gf), flat(cf));
                    assert_eq!(gf.len(), cf.len(), "{what}");
                    let errs = [
                        worst_of(got.iter().zip(&want).map(|(g, w)| (g - w).abs() / w)),
                        rel_l2(&gpu.object().unwrap(), cpu.object()),
                        rel_l2(&gpu.probe().unwrap(), cpu.probe()),
                        match cf.is_empty() {
                            true => 0.0,
                            false => {
                                worst_of(gf.iter().zip(&cf).map(|(g, w)| (g - w).abs()))
                                    / worst_of(cf.iter().map(|w| w.abs()))
                            }
                        },
                    ];
                    eprintln!(
                        "{what}: llk {:.2e} object {:.2e} probe {:.2e} features {:.2e}, llk {:.4e} -> {:.4e}",
                        errs[0],
                        errs[1],
                        errs[2],
                        errs[3],
                        want[0],
                        want[NCYCLE - 1]
                    );
                    for ((name, e), tol) in ["llk", "object", "probe", "features"]
                        .iter()
                        .zip(errs)
                        .zip(TOL)
                    {
                        assert!(e < tol, "{what} {name}: {e:e}");
                    }
                }
            }
        }
    }
}

/// AP, DM and RAAR with a position are refused under a plane stage, before they touch
/// anything, and run under the Fresnel transform, which has none.
#[test]
fn projections_with_a_position_are_refused_under_a_plane_stage() {
    let moved = Features {
        position: Some(drawn_offsets()),
        ..Features::default()
    };
    for c in cases() {
        let staged = c.name != "Fresnel";
        let mut gpu = c.gpu(true);
        gpu.set_features(&moved).unwrap();
        let before = gpu.object().unwrap();
        let runs = [
            gpu.ap(&AP, 1).map_err(|e| e.to_string()),
            gpu.dm(&DM, 1).map_err(|e| e.to_string()),
            gpu.raar(&RAAR, 1).map_err(|e| e.to_string()),
        ];
        for (r, name) in runs.into_iter().zip(["Ap", "Dm", "Raar"]) {
            match r {
                Ok(()) => assert!(!staged, "{} {name} ran", c.name),
                Err(e) => {
                    let want = format!("{name} with a position runs no propagator");
                    assert!(staged && e.contains(&want), "{} {name}: {e}", c.name);
                }
            }
        }
        if staged {
            assert_eq!(gpu.object().unwrap().as_slice(), before.as_slice());
            assert!(gpu.history().is_empty(), "{}", c.name);
        }
    }
}
