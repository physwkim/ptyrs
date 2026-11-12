//! Per-cycle time of [`GpuPtycho`] on CUDA device 0, the M1 baseline for M2.
//!
//! `cargo run --release -p ptyrs --example engine_gpu_time -- <dm|ap|ml|rpie> [ncycle] [nprobe] [frame] [npos] [step] [sizes] [graph|eager] [calc_llk] [fused|m1] [features]`
//! (defaults 50, 2, 256, 400, 16 object pixels, `16`, `graph`, 0, `fused`, `none`) builds the
//! `engine_llk_trace` scan, and for each comma-separated size (frames per stack; rPIE frames per
//! batch) runs 5 untimed cycles, then times `ncycle` cycles without phase-ramp removal, with
//! an LLK record every `calc_llk` cycles (0: none, as `scripts/compare_pynx.py --time` times
//! PyNX, and `scripts/compare_pynx_ml.py --time` PyNX ML). `graph` replays steady-state AP,
//! DM and ML cycles, rPIE epochs and LLK passes from the graphs the untimed cycles captured
//! ([`GpuOptions::graph`]); `fused` runs AP and DM stacks, ML's gradients and the LLK pass by
//! the fused passes, `m1` by M1's launches ([`GpuOptions::fused`]). ML updates object and
//! probe. `features`, comma-separated, models [`Features`] from the start: `scale` (1 per
//! frame), `background` (1 count per pixel), `position` (offsets of 0.1 pixels alternating in
//! sign, AP only) or `none`.

use std::time::Instant;

use cuda_core::CudaContext;
use ptyrs::engine::{
    ApParams, DmParams, EngineError, Features, GpuOptions, GpuPtycho, MlParams, Ptycho, RpieParams,
};

#[path = "../tests/common/engine/mod.rs"]
mod common;
#[path = "../tests/common/ptychi/mod.rs"]
mod ptychi;

const WARMUP: usize = 5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: gpu_time <dm|ap|ml|rpie> [ncycle] [nprobe] [frame] [npos] [step] [sizes] [graph|eager] [calc_llk] [fused|m1] [features]";
    let mut args = std::env::args().skip(1);
    let algo = args.next().ok_or(usage)?;
    if !["dm", "ap", "ml", "rpie"].contains(&algo.as_str()) {
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
    let graph = match args.next().as_deref() {
        None | Some("graph") => true,
        Some("eager") => false,
        Some(_) => return Err(usage.into()),
    };
    let calc_llk = args.next().map_or(Ok(0), |s| s.parse::<usize>())?;
    let fused = match args.next().as_deref() {
        None | Some("fused") => true,
        Some("m1") => false,
        Some(_) => return Err(usage.into()),
    };
    let features = args.next().unwrap_or_else(|| "none".into());
    let mut want = Features::default();
    for f in features.split(',') {
        match f {
            "none" => {}
            "scale" => want.scale = Some(vec![1.0; npos]),
            "background" => want.background = Some(vec![1.0; frame * frame]),
            "position" => {
                let d = |f: usize| if f.is_multiple_of(2) { 0.1 } else { -0.1 };
                want.position = Some((0..npos).map(|f| [d(f), -d(f)]).collect());
            }
            _ => return Err(usage.into()),
        }
    }

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
        calc_llk,
        zero_phase_ramp: false,
    };
    let dm = DmParams {
        calc_llk,
        zero_phase_ramp: false,
        ..DmParams::default()
    };
    let ml = MlParams {
        update_object: true,
        update_probe: true,
        calc_llk,
        zero_phase_ramp: false,
    };
    for size in sizes {
        let stack_size = if algo == "rpie" { 16 } else { size };
        let options = GpuOptions {
            stack_size,
            graph,
            fused,
            ..GpuOptions::default()
        };
        let mut g = GpuPtycho::new(&ctx, &scan, object.clone(), probe.clone(), options)?;
        g.set_features(&want)?;
        let rpie = RpieParams {
            batch_size: size,
            calc_llk,
            ..ptychi::RPIE
        };
        let run = |g: &mut GpuPtycho, n: usize| -> Result<(), EngineError> {
            match algo.as_str() {
                "dm" => g.dm(&dm, n),
                "ap" => g.ap(&ap, n),
                "ml" => g.ml(&ml, n),
                _ => g.rpie(&rpie, n),
            }
        };
        run(&mut g, WARMUP)?;
        g.synchronize()?;
        let t = Instant::now();
        run(&mut g, ncycle)?;
        g.synchronize()?;
        let ms = 1e3 * t.elapsed().as_secs_f64() / ncycle as f64;
        let mode = if graph { "graph" } else { "eager" };
        let passes = if fused { "fused" } else { "m1" };
        println!(
            "{algo} size {size} {mode} {passes} calc_llk {calc_llk} features {features}: {ms:.3} ms/cycle over {ncycle} cycles, {} LLK records",
            g.history().len()
        );
    }
    Ok(())
}
