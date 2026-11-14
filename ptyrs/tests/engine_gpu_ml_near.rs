//! GPU ML under a propagator's plane stage against the CPU's, on the cases of
//! `common/near.rs`.

use ptyrs::base::{C64, Stack};
use ptyrs::engine::{Features, MlParams, Ptycho};

#[path = "common/near.rs"]
mod near;
use near::{Case, NPOS, cases, drawn_offsets};

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

const ML: MlParams = MlParams {
    update_object: true,
    update_probe: true,
    calc_llk: 1,
    zero_phase_ramp: false,
};

/// No feature, a stepped scale and background, and refined drawn offsets.
fn models(frame: usize) -> [(&'static str, Features); 3] {
    let stepped = Features {
        scale: Some(vec![1.0; NPOS]),
        background: Some(vec![1.0; frame * frame]),
        position: None,
    };
    let moved = Features {
        position: Some(drawn_offsets()),
        ..Features::default()
    };
    [
        ("plain", Features::default()),
        ("stepped", stepped),
        ("moved", moved),
    ]
}

fn cpu(c: &Case, features: &Features) -> Ptycho {
    let (object, probe) = (c.object.clone(), c.probe.clone());
    let mut p = Ptycho::with_optics(&c.scan, c.optics(), object, probe).unwrap();
    p.set_features(features.clone()).unwrap();
    p
}

/// The opening scale (a call of no cycle), then the LLK and the gradients of the scaled
/// estimate, under each propagator and model.
#[test]
fn ml_scale_and_grads_match_cpu_under_every_propagator() {
    // Measured worst: LLK 4.8e-8, object 1.6e-7, probe 2.0e-7, object gradient 2.1e-6, probe
    // gradient 5.0e-6.
    const TOL: [f64; 5] = [5e-7, 1e-6, 1e-6, 1.5e-5, 2.5e-5];
    for c in cases() {
        let [_, ny, _] = c.scan.counts.dims();
        for (model, features) in models(ny) {
            let what = format!("{} {model}", c.name);
            let mut cpu = cpu(&c, &features);
            cpu.ml(&ML, 0);
            let mut gpu = c.gpu(true);
            gpu.set_features(&features).unwrap();
            gpu.ml(&ML, 0).unwrap();
            let [go, gp] = gpu.ml_grads().unwrap();
            // The f64 LLK models no feature.
            let llk = match model {
                "plain" => {
                    let want = c.llk(cpu.object(), cpu.probe());
                    (gpu.llk().unwrap().poisson - want).abs() / want
                }
                _ => 0.0,
            };
            let errs = [
                llk,
                rel_l2(&gpu.object().unwrap(), cpu.object()),
                rel_l2(&gpu.probe().unwrap(), cpu.probe()),
                rel_l2(&go, &cpu.obj_grad()),
                rel_l2(&gp, &cpu.probe_grad()),
            ];
            eprintln!(
                "{what}: llk {:.2e} object {:.2e} probe {:.2e} object grad {:.2e} probe grad {:.2e}",
                errs[0], errs[1], errs[2], errs[3], errs[4]
            );
            for ((name, e), tol) in ["llk", "object", "probe", "object grad", "probe grad"]
                .iter()
                .zip(errs)
                .zip(TOL)
            {
                assert!(e < tol, "{what} {name}: {e:e}");
            }
        }
    }
}

/// ML cycles under each propagator against the CPU's, plain, stepping a scale and a
/// background, and refining drawn offsets, eagerly and from graphs: each recorded LLK, and
/// the object, the probe and the features after the last cycle.
#[test]
fn ml_matches_cpu_under_every_propagator() {
    const NCYCLE: usize = 8;
    // Measured worst: LLK 4.0e-5, object 1.1e-5, probe 6.2e-7, features 1.3e-4, the first
    // and the last under the Fresnel transform back stepping a scale and a background.
    const TOL: [f64; 4] = [2e-4, 6e-5, 5e-6, 1e-3];
    for c in cases() {
        let [_, ny, _] = c.scan.counts.dims();
        for (model, features) in models(ny) {
            let mut cpu = cpu(&c, &features);
            cpu.ml(&ML, NCYCLE);
            let want: Vec<f64> = cpu.history().iter().map(|r| r.llk.poisson).collect();
            assert_eq!(want.len(), NCYCLE);
            for graph in [false, true] {
                let what = format!("{} {model} graph {graph}", c.name);
                let mut gpu = c.gpu(graph);
                gpu.set_features(&features).unwrap();
                gpu.ml(&ML, NCYCLE).unwrap();
                let got: Vec<f64> = gpu.history().iter().map(|r| r.llk.poisson).collect();
                assert_eq!(got.len(), NCYCLE, "{what}");
                let flat = |f: &Features| -> Vec<f64> {
                    let (s, b) = (f.scale.iter().flatten(), f.background.iter().flatten());
                    let p = f.position.iter().flatten().flatten();
                    s.chain(b).chain(p).copied().collect()
                };
                let (gf, cf) = (flat(&gpu.features().unwrap()), flat(cpu.features()));
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
