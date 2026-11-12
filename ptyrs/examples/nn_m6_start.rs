//! The M6 start measure (DESIGN, M6) on the held-out scans of `nn_train_ptychonn`. The target is
//! the Poisson LLK the flat start reaches after N epochs of an algorithm; the measure is the
//! epochs the predicted start needs to reach it with the same algorithm and data, on the
//! engine's CPU path. The first table holds the known probe fixed in DM and single-frame rPIE;
//! the second starts DM from it and updates it at DM's default probe inertia. rPIE runs at
//! object alpha 0.5 and 1.0: at its default 0.1 it climbs back from its minimum on these scans
//! and oscillates, in pty-chi as in the engine. A start reaches the target once its LLK is
//! within [`LLK_MARGIN`] above it: the LLK of rPIE and of DM passes a minimum and drifts, and a
//! strict count would measure the flat run's own dip. Every LLK curve goes to `<curves.tsv>`,
//! one row per epoch, epoch 0 being the start.
//!
//! `cargo run --release -p ptyrs --features nn --example nn_m6_start -- <predictor_dir> <curves.tsv>`

#[path = "common/nn/mod.rs"]
mod common;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::prelude::*;
use ptyrs::base::{Object, Probe, Scan, sim};
use ptyrs::engine::{DmParams, Ptycho, RpieParams};
use ptyrs::nn::metric::{LLK_MARGIN, iterations_to_llk};
use ptyrs::nn::predictor::{Predictor, flat_object, mean_frame_total};

/// Flat-start epochs whose LLK becomes the target.
const NS: [usize; 2] = [20, 100];
/// Epochs run from the predicted start.
const MAX: usize = 200;

#[derive(Clone, Copy, Debug)]
enum Alg {
    Dm,
    /// rPIE at this object alpha.
    Rpie(f64),
    /// DM updating the probe.
    DmProbe,
}
const ALGS: [Alg; 4] = [Alg::Dm, Alg::Rpie(0.5), Alg::Rpie(1.0), Alg::DmProbe];
/// Each table's rows, as a range of `ALGS`.
const TABLES: [(&str, Range<usize>); 2] = [("probe fixed", 0..3), ("probe updated", 3..4)];

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
        Alg::Rpie(obj_alpha) => p
            .rpie(
                &RpieParams {
                    obj_alpha,
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

/// A held-out scan, its two starts and the known probe.
struct Case {
    name: &'static str,
    seed: u64,
    scan: Scan,
    flat: Object,
    predicted: Object,
    probe: Probe,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: m6_start <predictor_dir> <curves.tsv>";
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().ok_or(usage)?);
    let curves = PathBuf::from(args.next().ok_or(usage)?);
    let predictor = Predictor::load(&dir, &Device::cuda(0))?;
    let frame = predictor.config().frame[0];
    let probe = sim::probe([frame, frame], 1)?;
    let mut cases = Vec::new();
    for (name, seed, scan, _, addr, probe) in common::held_out(frame, &probe) {
        let predicted =
            predictor.initial_object(&scan.counts, mean_frame_total(&scan.counts), &addr)?;
        let flat = flat_object(&addr);
        cases.push(Case {
            name,
            seed,
            scan,
            flat,
            predicted,
            probe,
        });
    }

    // One thread per reconstruction: the engine's CPU path is single-threaded.
    let t = Instant::now();
    // runs[case][k] = (flat run, predicted run) of ALGS[k].
    let runs: Vec<Vec<_>> = std::thread::scope(|s| {
        let handles: Vec<Vec<_>> = cases
            .iter()
            .map(|c| {
                ALGS.map(|alg| {
                    (
                        s.spawn(move || run(alg, &c.scan, c.flat.clone(), &c.probe, NS[1])),
                        s.spawn(move || run(alg, &c.scan, c.predicted.clone(), &c.probe, MAX)),
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

    write_curves(&curves, &cases, &runs)?;
    for (title, algs) in TABLES {
        println!("{title}");
        table(&cases, &runs, algs);
    }
    Ok(())
}

type Run = (f64, Vec<f64>);

/// `object seed alg start epoch llk` rows, the LLK in Rust's round-trip notation.
fn write_curves(path: &Path, cases: &[Case], runs: &[Vec<(Run, Run)>]) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "object\tseed\talg\tstart\tepoch\tllk")?;
    for (c, runs) in cases.iter().zip(runs) {
        for (alg, (flat, pred)) in ALGS.iter().zip(runs) {
            for (start, (llk0, llk)) in [("flat", flat), ("pred", pred)] {
                for (epoch, l) in std::iter::once(llk0).chain(llk).enumerate() {
                    writeln!(w, "{}\t{}\t{alg:?}\t{start}\t{epoch}\t{l}", c.name, c.seed)?;
                }
            }
        }
    }
    w.flush()
}

/// One row per case, algorithm of `ALGS[algs]` and N: the epochs to the target within
/// [`LLK_MARGIN`].
fn table(cases: &[Case], runs: &[Vec<(Run, Run)>], algs: Range<usize>) {
    println!(
        "{:12} {:9} {:>3}  {:>8}  {:>8} {:>8}  {:>7} {:>7}",
        "object", "alg", "N", "target", "LLK flat", "LLK pred", "flat 1%", "pred 1%"
    );
    for (c, runs) in cases.iter().zip(runs) {
        for (alg, ((flat0, flat), (pred0, pred))) in
            ALGS[algs.clone()].iter().zip(&runs[algs.clone()])
        {
            for n in NS {
                let target = flat[n - 1];
                // The flat run reaches its own epoch-N LLK by epoch N; earlier when it rises after.
                let flat_m =
                    iterations_to_llk(*flat0, target, LLK_MARGIN, flat.iter().copied()).unwrap();
                let pred_m = iterations_to_llk(*pred0, target, LLK_MARGIN, pred.iter().copied())
                    .map_or(format!(">{MAX}"), |k| k.to_string());
                println!(
                    "{:12} {:9} {n:>3}  {target:>8.4}  {flat0:>8.4} {pred0:>8.4}  {flat_m:>7} {pred_m:>7}",
                    c.name,
                    format!("{alg:?}")
                );
            }
        }
    }
}
