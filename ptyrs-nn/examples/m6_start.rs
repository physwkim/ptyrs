//! The M6 start measure (DESIGN, M6) on the held-out scans of `train_ptychonn`. The target is
//! the Poisson LLK the flat start reaches after N epochs of an algorithm; the measure is the
//! epochs the predicted start needs to reach it with the same algorithm and data, on the
//! engine's CPU path. The first table holds the known probe fixed in DM and single-frame rPIE;
//! the second starts DM from it and updates it at DM's default probe inertia.
//!
//! `cargo run --release -p ptyrs-nn --example m6_start -- <predictor_dir>`

mod common;

use std::path::PathBuf;
use std::time::Instant;

use burn::prelude::*;
use ptyrs_core::{Object, Probe, Scan, sim};
use ptyrs_engine::{DmParams, Ptycho, RpieParams};
use ptyrs_nn::metric::iterations_to_llk;
use ptyrs_nn::predictor::{Predictor, flat_object, mean_frame_total};

/// Flat-start epochs whose LLK becomes the target.
const NS: [usize; 2] = [20, 100];
/// Epochs run from the predicted start.
const MAX: usize = 200;

#[derive(Clone, Copy, Debug)]
enum Alg {
    Dm,
    Rpie,
    /// DM updating the probe.
    DmProbe,
}
/// In discriminant order.
const ALGS: [Alg; 3] = [Alg::Dm, Alg::Rpie, Alg::DmProbe];
const TABLES: [(&str, &[Alg]); 2] = [
    ("probe fixed", &[Alg::Dm, Alg::Rpie]),
    ("probe updated", &[Alg::DmProbe]),
];

/// The LLK of `start` and after each of `epochs` epochs of `alg`.
fn run(alg: Alg, scan: &Scan, start: Object, probe: &Probe, epochs: usize) -> (f64, Vec<f64>) {
    let mut p = Ptycho::new(scan, start, probe.clone()).unwrap();
    let start = p.llk().poisson;
    match alg {
        Alg::Dm => p.dm(
            &DmParams {
                probe_inertia: None,
                calc_llk: 1,
                ..DmParams::default()
            },
            epochs,
        ),
        Alg::DmProbe => p.dm(
            &DmParams {
                calc_llk: 1,
                ..DmParams::default()
            },
            epochs,
        ),
        Alg::Rpie => p
            .rpie(
                &RpieParams {
                    probe_step: None,
                    batch_size: 1,
                    calc_llk: 1,
                    ..RpieParams::default()
                },
                epochs,
            )
            .unwrap(),
    }
    let history: Vec<f64> = p.history().iter().map(|r| r.llk.poisson).collect();
    assert_eq!(history.len(), epochs);
    (start, history)
}

/// The probe that reproduces `scan`'s counts from `truth`: the simulator scales the intensity
/// to its photon count.
fn known_probe(scan: &Scan, truth: Object, probe: &Probe) -> Probe {
    let p = Ptycho::new(scan, truth, probe.clone()).unwrap();
    let s = (p.observed().total() / p.llk().nb_photons_calc).sqrt();
    let mut probe = probe.clone();
    probe.as_mut_slice().iter_mut().for_each(|c| *c *= s);
    probe
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("usage: m6_start <predictor_dir>")?,
    );
    let predictor = Predictor::load(&dir, &Device::cuda(0))?;
    let frame = predictor.config().frame[0];
    let probe = sim::probe([frame, frame], 1)?;
    let mut cases = Vec::new();
    for (name, object, seed) in common::held_out(frame, &probe) {
        let (scan, truth, addr) = common::scan(frame, &probe, seed, object);
        let predicted =
            predictor.initial_object(&scan.counts, mean_frame_total(&scan.counts), &addr)?;
        let known = known_probe(&scan, truth, &probe);
        cases.push((name, scan, flat_object(&addr), predicted, known));
    }

    // One thread per reconstruction: the engine's CPU path is single-threaded.
    let t = Instant::now();
    // runs[case][alg as usize] = (flat run, predicted run).
    let runs: Vec<Vec<_>> = std::thread::scope(|s| {
        let handles: Vec<Vec<_>> = cases
            .iter()
            .map(|(_, scan, flat, predicted, probe)| {
                ALGS.map(|alg| {
                    (
                        s.spawn(move || run(alg, scan, flat.clone(), probe, NS[1])),
                        s.spawn(move || run(alg, scan, predicted.clone(), probe, MAX)),
                    )
                })
                .into()
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.into_iter()
                    .map(|(f, p)| (f.join().unwrap(), p.join().unwrap()))
                    .collect()
            })
            .collect()
    });
    eprintln!(
        "{} reconstructions in {:.0} s",
        2 * cases.len() * ALGS.len(),
        t.elapsed().as_secs_f64()
    );

    for (title, algs) in TABLES {
        println!("{title}");
        table(&cases, &runs, algs);
    }
    Ok(())
}

type Run = (f64, Vec<f64>);

/// One row per case, algorithm of `algs` and N.
fn table(cases: &[(&str, Scan, Object, Object, Probe)], runs: &[Vec<(Run, Run)>], algs: &[Alg]) {
    println!(
        "{:12} {:7} {:>3}  {:>8}  {:>8} {:>8}  {:>11} {:>11}",
        "object", "alg", "N", "target", "LLK flat", "LLK pred", "flat epochs", "pred epochs"
    );
    for ((name, ..), runs) in cases.iter().zip(runs) {
        for &alg in algs {
            let ((flat0, flat), (pred0, pred)) = &runs[alg as usize];
            for n in NS {
                let target = flat[n - 1];
                // The flat run's own first epoch at its epoch-N LLK; N when the LLK only falls.
                let flat_n = iterations_to_llk(*flat0, target, flat.iter().copied()).unwrap();
                let pred_n = iterations_to_llk(*pred0, target, pred.iter().copied())
                    .map_or(format!(">{MAX}"), |k| k.to_string());
                println!(
                    "{name:12} {:7} {n:>3}  {target:>8.4}  {flat0:>8.4} {pred0:>8.4}  {flat_n:>11} {pred_n:>11}",
                    format!("{alg:?}")
                );
            }
        }
    }
}
