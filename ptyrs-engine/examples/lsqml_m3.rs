//! LSQML on a measured scan (the M3 case): the per-epoch LLK curve of GPU LSQML on CUDA device
//! 0 or of the CPU's `f64` [`Lsqml`] on the same batches, the epochs to 1.01×, 1.02× and 1.05×
//! the true LLK, and the GPU's time per epoch.
//!
//! `cargo run --release -p ptyrs-engine --example lsqml_m3 -- <cxi> <init.h5> <truth_poisson> <truth_gaussian> <truth_euclidian> <gaussian|poisson> <batch> [gpu|cpu] [nepoch] [stack_size]`
//! (defaults `gpu`, 300, 16) starts from `init.h5` as it is, as pty-chi's LSQML does, seed 7.
//! `gpu` times `nepoch` epochs with no LLK after an untimed call that captures the epoch
//! graphs, then runs `nepoch` epochs from the start with an LLK after every epoch; its seconds
//! to a target are epochs × the timed ms/epoch, which leaves out the LLK passes, as pty-chi's
//! `time_to_target` leaves out its evaluation. `cpu` prints each epoch's LLK as it lands.

use std::time::Instant;

use cuda_core::CudaContext;
use ptyrs_engine::{
    Cost, EngineError, GpuOptions, GpuPtycho, Llk, Lsqml, LsqmlParams, ProbeVariation, Ptycho,
};

/// Epochs in the untimed call: the preconditioner's build, update and keep graphs.
const WARMUP: usize = 12;
const TARGETS: [f64; 3] = [1.01, 1.02, 1.05];

fn kinds(l: &Llk) -> [f64; 3] {
    [l.poisson, l.gaussian, l.euclidian]
}

fn print_epoch(e: usize, l: &[f64; 3]) {
    println!("curve epoch {e} {:.10e} {:.10e} {:.10e}", l[0], l[1], l[2]);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: lsqml_m3 <cxi> <init.h5> <truth_poisson> <truth_gaussian> <truth_euclidian> <gaussian|poisson> <batch> [gpu|cpu] [nepoch] [stack_size]";
    let mut args = std::env::args().skip(1);
    let (cxi, init) = (args.next().ok_or(usage)?, args.next().ok_or(usage)?);
    let mut truth = [0.0; 3];
    for t in &mut truth {
        *t = args.next().ok_or(usage)?.parse()?;
    }
    let cost = match args.next().as_deref() {
        Some("gaussian") => Cost::Amplitude,
        Some("poisson") => Cost::Poisson,
        _ => return Err(usage.into()),
    };
    let batch_size = args.next().ok_or(usage)?.parse()?;
    let gpu = match args.next().as_deref() {
        None | Some("gpu") => true,
        Some("cpu") => false,
        Some(_) => return Err(usage.into()),
    };
    let mut next = |default: usize| args.next().map_or(Ok(default), |s| s.parse());
    let (nepoch, stack_size) = (next(300)?, next(16)?);
    let params = LsqmlParams {
        batch_size,
        cost,
        seed: 7,
    };

    let scan = ptyrs_io::read_cxi(&cxi)?;
    let (object, probe) = ptyrs_io::read_modes(&init)?;
    eprintln!(
        "{} frames {:?}, {} probe modes, object {:?}, {cost:?} batch {batch_size}, {}",
        scan.counts.nframe(),
        scan.geometry.frame,
        probe.nmode(),
        object.plane_shape(),
        if gpu { "gpu" } else { "cpu" }
    );
    let at = Ptycho::new(&scan, object.clone(), probe.clone())?;
    let mut curve = vec![kinds(&at.llk())];
    print_epoch(0, &curve[0]);
    let mut ms = None;
    if gpu {
        let ctx = CudaContext::new(0).map_err(EngineError::from)?;
        let options = GpuOptions {
            stack_size,
            ..GpuOptions::default()
        };
        let mut g = GpuPtycho::new(&ctx, &scan, object.clone(), probe.clone(), options)?;
        let t = Instant::now();
        g.lsqml(&params, WARMUP, 0)?;
        g.synchronize()?;
        let warm = t.elapsed().as_secs_f64();
        let mut g = GpuPtycho::new(&ctx, &scan, object.clone(), probe.clone(), options)?;
        let t = Instant::now();
        g.lsqml(&params, nepoch, 0)?;
        g.synchronize()?;
        let per = 1e3 * t.elapsed().as_secs_f64() / nepoch as f64;
        println!(
            "time: {per:.3} ms/epoch over {nepoch} epochs, stack {stack_size} \
             ({WARMUP} warm-up epochs {warm:.3} s)"
        );
        ms = Some(per);

        let mut g = GpuPtycho::new(&ctx, &scan, object, probe, options)?;
        g.lsqml(&params, nepoch, 1)?;
        for (e, r) in g.history().iter().enumerate() {
            curve.push(kinds(&r.llk));
            print_epoch(e + 1, &curve[e + 1]);
        }
    } else {
        let (mut o, mut p) = (object, probe);
        let mut var = ProbeVariation::none(p.plane_shape());
        let mut cpu = Lsqml::new(params);
        let t = Instant::now();
        for e in 1..=nepoch {
            cpu.epoch(
                at.addr(),
                at.observed(),
                at.quad(),
                &mut o,
                &mut p,
                &mut var,
            );
            curve.push(kinds(&Ptycho::new(&scan, o.clone(), p.clone())?.llk()));
            print_epoch(e, &curve[e]);
        }
        println!(
            "time: {:.3} s/epoch with the LLK, {nepoch} epochs",
            t.elapsed().as_secs_f64() / nepoch as f64
        );
    }

    for (k, name) in ["poisson", "gaussian", "euclidian"].iter().enumerate() {
        let last = curve[nepoch][k] / truth[k];
        let hits = TARGETS.map(
            |f| match (curve.iter().position(|l| l[k] <= f * truth[k]), ms) {
                (Some(e), Some(ms)) => format!("{f}x epoch {e} {:.2} s", e as f64 * ms * 1e-3),
                (Some(e), None) => format!("{f}x epoch {e}"),
                (None, _) => format!("{f}x not reached"),
            },
        );
        println!(
            "target {name}: {}; {last:.4}x truth at epoch {nepoch}",
            hits.join(", ")
        );
    }
    Ok(())
}
