//! GPU BH on a measured scan (the M3 case), on CUDA device 0: time per epoch and epochs to
//! 1.01× the true LLK.
//!
//! `cargo run --release -p ptyrs-engine --example bh_m3 -- <cxi> <init.h5> <truth_poisson> <truth_gaussian> <truth_euclidian> [scaled|init] [batches] [ntime] [nepoch] [stack_size] [fused|m1]`
//! (defaults `scaled`, `96,1024`, 20, 300, 16, `fused`) starts from `init.h5` after
//! [`Ptycho::scale_obj_probe`] as solve's CPU M3 curves do, or with `init` from `init.h5` as
//! it is, as the pty-chi M3 runs do; probe rho 0.01 and seed 7. For each comma-separated
//! batch size it times `ntime` epochs after 5 untimed ones with no LLK, then runs `nepoch`
//! epochs from the start with an LLK after every epoch and prints the curve and, per LLK
//! kind, the first epoch at or below 1.01× the truth. Its seconds are epochs × the timed
//! ms/epoch, which leaves out the LLK passes, as pty-chi's `time_to_target` leaves out its
//! evaluation.

use std::time::Instant;

use cuda_core::CudaContext;
use ptyrs_engine::{BhParams, EngineError, GpuOptions, GpuPtycho, Llk, Ptycho};

const WARMUP: usize = 5;
const TARGET: f64 = 1.01;

fn kinds(l: &Llk) -> [f64; 3] {
    [l.poisson, l.gaussian, l.euclidian]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: bh_m3 <cxi> <init.h5> <truth_poisson> <truth_gaussian> <truth_euclidian> [scaled|init] [batches] [ntime] [nepoch] [stack_size] [fused|m1]";
    let mut args = std::env::args().skip(1);
    let (cxi, init) = (args.next().ok_or(usage)?, args.next().ok_or(usage)?);
    let mut truth = [0.0; 3];
    for t in &mut truth {
        *t = args.next().ok_or(usage)?.parse()?;
    }
    let scaled = match args.next().as_deref() {
        None | Some("scaled") => true,
        Some("init") => false,
        Some(_) => return Err(usage.into()),
    };
    let batches = args.next().unwrap_or_else(|| "96,1024".into());
    let batches = batches
        .split(',')
        .map(str::parse)
        .collect::<Result<Vec<usize>, _>>()?;
    let mut next = |default: usize| args.next().map_or(Ok(default), |s| s.parse());
    let (ntime, nepoch, stack_size) = (next(20)?, next(300)?, next(16)?);
    let fused = match args.next().as_deref() {
        None | Some("fused") => true,
        Some("m1") => false,
        Some(_) => return Err(usage.into()),
    };

    let scan = ptyrs_io::read_cxi(&cxi)?;
    let (object, probe) = ptyrs_io::read_modes(&init)?;
    let mut p = Ptycho::new(&scan, object, probe)?;
    if scaled {
        p.scale_obj_probe();
    }
    let (object, probe) = p.into_modes();
    let ctx = CudaContext::new(0).map_err(EngineError::from)?;
    let options = GpuOptions {
        stack_size,
        fused,
        ..GpuOptions::default()
    };
    let passes = if fused { "fused" } else { "m1" };
    eprintln!(
        "{} frames {:?}, {} probe modes, object {:?}, stack {stack_size}, {passes}, scaled start {scaled}",
        scan.counts.nframe(),
        scan.geometry.frame,
        probe.nmode(),
        object.plane_shape()
    );
    for batch_size in batches {
        let params = BhParams {
            batch_size,
            probe_rho: 0.01,
            seed: 7,
        };
        let mut g = GpuPtycho::new(&ctx, &scan, object.clone(), probe.clone(), options)?;
        let t = Instant::now();
        g.bh(&params, WARMUP, 0)?;
        g.synchronize()?;
        let warm = t.elapsed().as_secs_f64();
        let t = Instant::now();
        g.bh(&params, ntime, 0)?;
        g.synchronize()?;
        let ms = 1e3 * t.elapsed().as_secs_f64() / ntime as f64;
        println!(
            "bh batch {batch_size}: {ms:.3} ms/epoch over {ntime} epochs ({WARMUP} warm-up epochs {warm:.3} s)"
        );

        let mut g = GpuPtycho::new(&ctx, &scan, object.clone(), probe.clone(), options)?;
        let start = g.llk()?;
        g.bh(&params, nepoch, 1)?;
        let curve: Vec<[f64; 3]> = std::iter::once(kinds(&start))
            .chain(g.history().iter().map(|r| kinds(&r.llk)))
            .collect();
        for (e, l) in curve.iter().enumerate() {
            println!(
                "curve batch {batch_size} epoch {e} {:.6e} {:.6e} {:.6e}",
                l[0], l[1], l[2]
            );
        }
        for (k, name) in ["poisson", "gaussian", "euclidian"].iter().enumerate() {
            let hit = curve.iter().position(|l| l[k] <= TARGET * truth[k]);
            let last = curve[curve.len() - 1][k] / truth[k];
            match hit {
                Some(e) => println!(
                    "target batch {batch_size} {name}: epoch {e}, {:.2} s ({last:.4}x truth at epoch {nepoch})",
                    e as f64 * ms * 1e-3
                ),
                None => println!(
                    "target batch {batch_size} {name}: not reached in {nepoch} epochs ({last:.4}x truth at the end)"
                ),
            }
        }
    }
    Ok(())
}
