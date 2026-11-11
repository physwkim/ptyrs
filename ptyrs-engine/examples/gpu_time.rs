//! Per-cycle time of [`GpuPtycho`] on CUDA device 0, the M1 baseline for M2.
//!
//! `cargo run --release -p ptyrs-engine --example gpu_time -- <dm|ap|rpie> [ncycle] [nprobe] [frame] [npos] [step] [sizes]`
//! (defaults 50, 2, 256, 400, 16 object pixels, `16`) builds the `llk_trace` scan, and for
//! each comma-separated size (frames per stack; rPIE frames per batch) runs 5 untimed cycles,
//! then times `ncycle` cycles without LLK or phase-ramp removal, as `scripts/compare_pynx.py
//! --time` times PyNX.

use std::time::Instant;

use cuda_core::CudaContext;
use ptyrs_engine::{ApParams, DmParams, EngineError, GpuOptions, GpuPtycho, Ptycho, RpieParams};

#[path = "../tests/common/mod.rs"]
mod common;

const WARMUP: usize = 5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: gpu_time <dm|ap|rpie> [ncycle] [nprobe] [frame] [npos] [step] [sizes]";
    let mut args = std::env::args().skip(1);
    let algo = args.next().ok_or(usage)?;
    if !["dm", "ap", "rpie"].contains(&algo.as_str()) {
        return Err(usage.into());
    }
    let mut next = |default: usize| args.next().map_or(Ok(default), |s| s.parse());
    let (ncycle, nprobe, frame, npos) = (next(50)?, next(2)?, next(256)?, next(400)?);
    let step = args.next().map_or(Ok(16.0), |s| s.parse::<f64>())?;
    let sizes = args.next().unwrap_or_else(|| "16".into());
    let sizes = sizes
        .split(',')
        .map(str::parse)
        .collect::<Result<Vec<usize>, _>>()?;

    let (scan, object, probe) = common::dm_scenario(frame, npos, nprobe, 2e5, step);
    let mut p = Ptycho::new(&scan, object, probe)?;
    p.scale_obj_probe();
    let (object, probe) = p.into_modes();
    let ctx = CudaContext::new(0).map_err(EngineError::from)?;
    eprintln!(
        "{algo}: {npos} frames {frame}x{frame}, {nprobe} probe modes, object {:?}",
        object.plane_shape()
    );

    let ap = ApParams {
        obj_inertia: Some(0.01),
        probe_inertia: Some(0.001),
        calc_llk: 0,
        zero_phase_ramp: false,
    };
    let dm = DmParams {
        calc_llk: 0,
        zero_phase_ramp: false,
        ..DmParams::default()
    };
    for size in sizes {
        let stack_size = if algo == "rpie" { 16 } else { size };
        let options = GpuOptions { stack_size };
        let mut g = GpuPtycho::new(&ctx, &scan, object.clone(), probe.clone(), options)?;
        let rpie = RpieParams {
            batch_size: size,
            calc_llk: 0,
            ..common::RPIE
        };
        let run = |g: &mut GpuPtycho, n: usize| -> Result<(), EngineError> {
            match algo.as_str() {
                "dm" => g.dm(&dm, n),
                "ap" => g.ap(&ap, n),
                _ => g.rpie(&rpie, n),
            }
        };
        run(&mut g, WARMUP)?;
        g.synchronize()?;
        let t = Instant::now();
        run(&mut g, ncycle)?;
        g.synchronize()?;
        let ms = 1e3 * t.elapsed().as_secs_f64() / ncycle as f64;
        println!("{algo} size {size}: {ms:.3} ms/cycle over {ncycle} cycles");
    }
    Ok(())
}
