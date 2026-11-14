//! The GPU passes that run a propagator's plane stage besides LSQML's and BH's, on the cases
//! of `common/near.rs`.

use ptyrs::base::{C64, Stack};
use ptyrs::engine::{Features, Ptycho, RpieParams};

#[path = "common/near.rs"]
mod near;
use near::{cases, drawn_offsets};

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
