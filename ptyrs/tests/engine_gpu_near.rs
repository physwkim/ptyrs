//! The GPU passes that run a propagator's plane stage besides LSQML's and BH's (rPIE's, and
//! AP's, DM's and RAAR's), on the cases of `common/near.rs`, and the sampling a position's
//! offsets enter by.

use cuda_core::CudaContext;
use ptyrs::base::{C64, Counts, Geometry, Scan, Stack, required_obj_shape, sim};
use ptyrs::engine::{
    ApParams, BhParams, Cost, DmParams, EngineError, Features, GpuPtycho, LsqmlParams, MlParams,
    Optics, Ptycho, RaarParams, RpieParams, Sampling,
};
use ptyrs::gpu::FUSED_SIDES;

#[path = "common/near.rs"]
mod near;
use near::{NPOS, OPTIONS, PIXEL, WAVELENGTH, cases, drawn_offsets};

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

/// AP, DM and RAAR under each propagator against the CPU's, plain, stepping a scale and a
/// background, and refining drawn offsets, eagerly and from graphs: each recorded LLK, and
/// the object, the probe, the scale and background, and the offsets (over the largest) after
/// the last cycle.
#[test]
fn projections_match_cpu_under_every_propagator() {
    const NCYCLE: usize = 8;
    // Measured worst: LLK 6.7e-6, object 1.6e-6, probe 2.3e-6, features 1.7e-5, offsets
    // 7.3e-7.
    const TOL: [f64; 5] = [4e-5, 1e-5, 1.5e-5, 1e-4, 4e-6];
    let stepped = |frame: usize| Features {
        scale: Some(vec![1.0; NPOS]),
        background: Some(vec![1.0; frame * frame]),
        position: None,
    };
    let moved = Features {
        position: Some(drawn_offsets()),
        ..Features::default()
    };
    for c in cases() {
        let [_, ny, _] = c.scan.counts.dims();
        let stepped = stepped(ny);
        for (name, on_cpu, on_gpu) in PROJECTIONS {
            for (label, features) in [
                ("none", None),
                ("stepped", Some(&stepped)),
                ("moved", Some(&moved)),
            ] {
                let (object, probe) = (c.object.clone(), c.probe.clone());
                let mut cpu = Ptycho::with_optics(&c.scan, c.optics(), object, probe).unwrap();
                if let Some(f) = features {
                    cpu.set_features(f.clone()).unwrap();
                }
                on_cpu(&mut cpu, NCYCLE);
                let want: Vec<f64> = cpu.history().iter().map(|r| r.llk.poisson).collect();
                assert_eq!(want.len(), NCYCLE);
                for graph in [false, true] {
                    let what = format!("{name} {} features {label} graph {graph}", c.name);
                    let mut gpu = c.gpu(graph);
                    if let Some(f) = features {
                        gpu.set_features(f).unwrap();
                    }
                    on_gpu(&mut gpu, NCYCLE);
                    let got: Vec<f64> = gpu.history().iter().map(|r| r.llk.poisson).collect();
                    assert_eq!(got.len(), NCYCLE, "{what}");
                    let (gf, cf) = (gpu.features().unwrap(), cpu.features());
                    let scaled = |f: &Features| -> Vec<f64> {
                        let (s, b) = (f.scale.iter().flatten(), f.background.iter().flatten());
                        s.chain(b).copied().collect()
                    };
                    let moved = |f: &Features| -> Vec<f64> {
                        f.position.iter().flatten().flatten().copied().collect()
                    };
                    let worst = |g: Vec<f64>, w: Vec<f64>| match w.is_empty() {
                        true => 0.0,
                        false => {
                            assert_eq!(g.len(), w.len(), "{what}");
                            worst_of(g.iter().zip(&w).map(|(g, w)| (g - w).abs()))
                                / worst_of(w.iter().map(|w| w.abs()))
                        }
                    };
                    let errs = [
                        worst_of(got.iter().zip(&want).map(|(g, w)| (g - w).abs() / w)),
                        rel_l2(&gpu.object().unwrap(), cpu.object()),
                        rel_l2(&gpu.probe().unwrap(), cpu.probe()),
                        worst(scaled(&gf), scaled(cf)),
                        worst(moved(&gf), moved(cf)),
                    ];
                    eprintln!(
                        "{what}: llk {:.2e} object {:.2e} probe {:.2e} features {:.2e} offsets {:.2e}, llk {:.4e} -> {:.4e}",
                        errs[0],
                        errs[1],
                        errs[2],
                        errs[3],
                        errs[4],
                        want[0],
                        want[NCYCLE - 1]
                    );
                    for ((name, e), tol) in ["llk", "object", "probe", "features", "offsets"]
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

/// The sampling is the CPU's [`Sampling::auto`] at build: `FullObject` of a near-field scan
/// whose object is under 5/4 of the frame, `Patch` of the same scan in the far field and of
/// a case's wider scan; `set_sampling` overrides it either way and back.
#[test]
fn sampling_is_the_cpus_auto_until_set() {
    let frame = FUSED_SIDES[0] as usize;
    let pixel = [PIXEL, PIXEL];
    let raw = sim::fermat(NPOS, PIXEL / 2.0);
    let shape = required_obj_shape(&raw.pynx_preprocess(pixel), pixel, [frame, frame], 4);
    let counts = vec![0.0; NPOS * frame * frame];
    let counts = Counts::from_f64([NPOS, frame, frame], &counts, None).unwrap();
    let geometry = Geometry::new(WAVELENGTH, 1.0, PIXEL, [frame, frame]).unwrap();
    let tight = Scan::new(geometry, raw, counts).unwrap();
    let wide = &cases()[1];
    let far = || Optics::far_field(&tight.geometry);
    let near = || wide.optics();
    let ctx = CudaContext::new(0).unwrap();
    let runs: [(_, _, _, &dyn Fn() -> Optics, _); 3] = [
        (
            "tight near",
            &tight,
            sim::object(shape),
            &near,
            Sampling::FullObject,
        ),
        (
            "tight far",
            &tight,
            sim::object(shape),
            &far,
            Sampling::Patch,
        ),
        (
            "wide near",
            &wide.scan,
            wide.object.clone(),
            &near,
            Sampling::Patch,
        ),
    ];
    for (what, scan, object, optics, auto) in runs {
        let probe = wide.probe.clone();
        let cpu = Ptycho::with_optics(scan, optics(), object.clone(), probe.clone()).unwrap();
        assert_eq!(cpu.sampling(), auto, "{what}: CPU");
        let mut gpu =
            GpuPtycho::with_optics(&ctx, scan, &optics(), object, probe, OPTIONS).unwrap();
        assert_eq!(gpu.sampling(), auto, "{what}");
        let other = match auto {
            Sampling::Patch => Sampling::FullObject,
            Sampling::FullObject => Sampling::Patch,
        };
        for s in [other, auto] {
            gpu.set_sampling(s).unwrap();
            assert_eq!(gpu.sampling(), s, "{what}");
        }
    }
}

/// `set_sampling` there and back between the calls of a run that models positions drops the
/// runs, graphs and states built under it; their rebuilds give the records, arrays and
/// offsets of the run left alone bit for bit.
#[test]
fn sampling_round_trip_rebuilds_the_same_run() {
    let c = &cases()[0];
    let moved = Features {
        position: Some(drawn_offsets()),
        ..Features::default()
    };
    let bh = BhParams {
        batch_size: 8,
        probe_rho: 0.01,
        position_rho: 0.1,
        seed: 7,
    };
    let lsqml = LsqmlParams {
        batch_size: 8,
        cost: Cost::Poisson,
        seed: 7,
    };
    let run = |toggle: bool| {
        let mut gpu = c.gpu(true);
        gpu.set_features(&moved).unwrap();
        let mut records = Vec::new();
        for _ in 0..2 {
            gpu.dm(&DmParams::default(), 2).unwrap();
            gpu.ml(&MlParams::default(), 2).unwrap();
            gpu.bh(&bh, 1, 1).unwrap();
            gpu.lsqml(&lsqml, 1, 1).unwrap();
            if toggle {
                gpu.set_sampling(Sampling::FullObject).unwrap();
                gpu.set_sampling(Sampling::Patch).unwrap();
            }
            records.push(gpu.llk().unwrap().poisson);
        }
        records.extend(gpu.history().iter().map(|r| r.llk.poisson));
        let arrays = (gpu.object().unwrap(), gpu.probe().unwrap());
        (records, arrays, gpu.features().unwrap().position)
    };
    let (kept, toggled) = (run(false), run(true));
    assert_eq!(toggled.0, kept.0, "records");
    assert_eq!(toggled.1.0.as_slice(), kept.1.0.as_slice(), "object");
    assert_eq!(toggled.1.1.as_slice(), kept.1.1.as_slice(), "probe");
    assert_eq!(toggled.2, kept.2, "offsets");
}

/// While a position samples the object, which no GPU pass models yet, every algorithm, the
/// LLK and ML's gradients fail and change nothing, after a call of each, which built its run:
/// switched to `Sampling::FullObject` after calls under `Patch` with the position, or the
/// position set after calls under `FullObject` without one. Back under `Patch` the same calls
/// give the records, arrays, gradients and offsets of a run never switched, bit for bit.
#[test]
fn full_object_position_fails_every_pass_until_patch() {
    let c = &cases()[0];
    let moved = Features {
        position: Some(drawn_offsets()),
        ..Features::default()
    };
    let bh = BhParams {
        batch_size: 8,
        probe_rho: 0.01,
        position_rho: 0.1,
        seed: 7,
    };
    let lsqml = LsqmlParams {
        batch_size: 8,
        cost: Cost::Poisson,
        seed: 7,
    };
    type Call<'a> = &'a dyn Fn(&mut GpuPtycho) -> Result<Vec<f64>, EngineError>;
    let calls: [(&str, Call); 9] = [
        ("AP", &|g| g.ap(&AP, 1).map(|()| Vec::new())),
        ("DM", &|g| g.dm(&DM, 1).map(|()| Vec::new())),
        ("RAAR", &|g| g.raar(&RAAR, 1).map(|()| Vec::new())),
        ("rPIE", &|g| g.rpie(&RPIE, 1).map(|()| Vec::new())),
        ("ML", &|g| {
            g.ml(&MlParams::default(), 1).map(|()| Vec::new())
        }),
        ("BH", &|g| g.bh(&bh, 1, 1).map(|()| Vec::new())),
        ("LSQML", &|g| g.lsqml(&lsqml, 1, 1).map(|()| Vec::new())),
        ("LLK", &|g| g.llk().map(|l| vec![l.poisson])),
        ("ML gradients", &|g| {
            let grads = g.ml_grads()?;
            let parts = grads.iter().flat_map(|s| s.as_slice().iter());
            Ok(parts.flat_map(|z| [z.re, z.im]).collect())
        }),
    ];
    let run = |full_object_first: bool, switched: bool| {
        let mut gpu = c.gpu(true);
        let sampling = match switched {
            true => Sampling::FullObject,
            false => Sampling::Patch,
        };
        match full_object_first {
            true => gpu.set_sampling(sampling).unwrap(),
            false => gpu.set_features(&moved).unwrap(),
        }
        for (what, call) in calls {
            call(&mut gpu).unwrap_or_else(|e| panic!("{what} before: {e}"));
        }
        match full_object_first {
            true => gpu.set_features(&moved).unwrap(),
            false => gpu.set_sampling(sampling).unwrap(),
        }
        if switched {
            let start = (gpu.object().unwrap(), gpu.probe().unwrap());
            let records = gpu.history().len();
            for (what, call) in calls {
                let e = call(&mut gpu).unwrap_err().to_string();
                assert!(e.contains("Sampling::FullObject"), "{what}: {e}");
            }
            assert_eq!(gpu.history().len(), records, "records");
            assert_eq!(gpu.object().unwrap().as_slice(), start.0.as_slice());
            assert_eq!(gpu.probe().unwrap().as_slice(), start.1.as_slice());
            gpu.set_sampling(Sampling::Patch).unwrap();
        }
        let values: Vec<Vec<f64>> = calls
            .iter()
            .map(|(_, call)| call(&mut gpu).unwrap())
            .collect();
        let records: Vec<f64> = gpu.history().iter().map(|r| r.llk.poisson).collect();
        let arrays = (gpu.object().unwrap(), gpu.probe().unwrap());
        (values, records, arrays, gpu.features().unwrap().position)
    };
    for full_object_first in [false, true] {
        let first = ["Patch", "FullObject"][usize::from(full_object_first)];
        let (kept, switched) = (run(full_object_first, false), run(full_object_first, true));
        for ((what, _), (s, k)) in calls.iter().zip(switched.0.iter().zip(&kept.0)) {
            assert_eq!(s, k, "{first} first: {what}");
        }
        assert_eq!(switched.1, kept.1, "{first} first: records");
        let (s, k) = (&switched.2, &kept.2);
        assert_eq!(s.0.as_slice(), k.0.as_slice(), "{first} first: object");
        assert_eq!(s.1.as_slice(), k.1.as_slice(), "{first} first: probe");
        assert_eq!(switched.3, kept.3, "{first} first: offsets");
    }
}
