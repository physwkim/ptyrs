//! DM, AP, ML or rPIE on a simulated scan, the Rust side of `scripts/compare_pynx.py` (DM,
//! AP), `scripts/compare_pynx_ml.py` (ML) and `scripts/compare_ptychi_rpie.py` (rPIE).
//!
//! `cargo run --release -p ptyrs --example engine_llk_trace -- <out_dir> <dm|ap|ml|rpie> [ncycle] [nprobe] [frame] [npos] [step] [--gpu]`
//! (defaults 50, 1, 64, 100, 4 object pixels) writes `scan.cxi`, `init.h5` (the scaled
//! starting object and probe), `llk_rust.txt` (cycle and the normalised Poisson, Gaussian,
//! Euclidian LLK and photon count per cycle), `llk_truth.txt` (the same four for the truth the
//! counts were drawn from, [`ptyrs::engine::sim::Simulation::truth_llk`]) and `final_rust.h5`,
//! from the CPU oracle or, with `--gpu`, from [`GpuPtycho`] on CUDA device 0 with the default
//! stacks, by the fused passes on [`ptyrs::gpu::FUSED_SIDES`] frames, M1's launches on others. ML
//! updates object and probe from the end of `ML_WARMUP` AP cycles, which `init.h5` holds.
//! rPIE also writes
//! `views.txt` (object window corner `oy ox` per frame), `quad.txt` (`re im` of the quadratic
//! phase, natural order), which pty-chi has no notion of, and `rpie.txt` (`ptychi::RPIE`,
//! step 0 for a fixed array).

use std::fmt::Write as _;
use std::path::PathBuf;

use cuda_core::CudaContext;
use ptyrs::engine::{
    ApParams, DmParams, EngineError, GpuOptions, GpuPtycho, Llk, MlParams, Ptycho, RpieParams,
    quad_phase,
};

#[path = "../tests/common/ptychi/mod.rs"]
mod ptychi;
#[path = "../tests/common/engine/scenario.rs"]
mod scenario;

/// AP updates the probe too, so both halves of `psi_to_obj_probe` are compared.
const AP: ApParams = ApParams {
    obj_inertia: Some(0.01),
    probe_inertia: Some(0.001),
    calc_llk: 1,
    zero_phase_ramp: true,
};

const ML: MlParams = MlParams {
    update_object: true,
    update_probe: true,
    calc_llk: 1,
    zero_phase_ramp: true,
};

/// AP cycles before ML, whose end is ML's `init.h5`.
const ML_WARMUP: usize = 20;

const RPIE: RpieParams = ptychi::RPIE;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: llk_trace <out_dir> <dm|ap|ml|rpie> [ncycle] [nprobe] [frame] [npos] [step] [--gpu]";
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let gpu = args
        .iter()
        .position(|a| a == "--gpu")
        .map(|i| args.remove(i))
        .is_some();
    let mut args = args.into_iter();
    let out = PathBuf::from(args.next().ok_or(usage)?);
    let algo = args.next().ok_or(usage)?;
    if !["dm", "ap", "ml", "rpie"].contains(&algo.as_str()) {
        return Err(usage.into());
    }
    let mut next = |default: usize| args.next().map_or(Ok(default), |s| s.parse());
    let (ncycle, nprobe, frame, npos) = (next(50)?, next(1)?, next(64)?, next(100)?);
    let step = args.next().map_or(Ok(4.0), |s| s.parse::<f64>())?;
    std::fs::create_dir_all(&out)?;

    let (scan, object, probe, truth) = scenario::dm_scenario_truth(frame, npos, nprobe, 2e5, step);
    let cxi = out.join("scan.cxi");
    ptyrs::io::write_cxi(&cxi, &scan)?;
    let scan = ptyrs::io::read_cxi(&cxi)?;
    std::fs::write(out.join("llk_truth.txt"), llk_line(&truth))?;
    let mut p = Ptycho::new(&scan, object, probe)?;
    p.scale_obj_probe();
    if algo == "ml" {
        // From the flat object ML's trajectory is chaotic (PyNX against itself with the
        // object nudged by 1e-7 differs by 1e-1 within 10 cycles), so it starts from AP's.
        p.ap(&AP, ML_WARMUP);
        let (object, probe) = p.into_modes();
        p = Ptycho::new(&scan, object, probe)?;
    }
    ptyrs::io::write_modes(out.join("init.h5"), p.object(), p.probe())?;
    if algo == "rpie" {
        let mut s = String::new();
        for v in p.addr().views().iter().step_by(p.addr().nmode()) {
            writeln!(s, "{} {}", v.oy, v.ox)?;
        }
        std::fs::write(out.join("views.txt"), s)?;
        let mut s = String::new();
        for q in quad_phase(&scan.geometry) {
            writeln!(s, "{:.17e} {:.17e}", q.re, q.im)?;
        }
        std::fs::write(out.join("quad.txt"), s)?;
        let (os, ps) = (RPIE.obj_step.unwrap_or(0.0), RPIE.probe_step.unwrap_or(0.0));
        std::fs::write(
            out.join("rpie.txt"),
            format!(
                "obj_step {os}\nprobe_step {ps}\nobj_alpha {}\nprobe_alpha {}\nbatch_size {}\n",
                RPIE.obj_alpha, RPIE.probe_alpha, RPIE.batch_size
            ),
        )?;
    }

    let dm = DmParams {
        calc_llk: 1,
        ..DmParams::default()
    };
    let backend = if gpu { "gpu" } else { "cpu" };
    let (history, object, probe, secs) = if gpu {
        let ctx = CudaContext::new(0).map_err(EngineError::from)?;
        let (object, probe) = p.into_modes();
        let options = GpuOptions {
            fused: ptyrs::gpu::FUSED_SIDES.contains(&(frame as u32)),
            ..GpuOptions::default()
        };
        eprintln!("GPU passes: {}", if options.fused { "fused" } else { "m1" });
        let mut g = GpuPtycho::new(&ctx, &scan, object, probe, options)?;
        let t = std::time::Instant::now();
        match algo.as_str() {
            "dm" => g.dm(&dm, ncycle)?,
            "ap" => g.ap(&AP, ncycle)?,
            "ml" => g.ml(&ML, ncycle)?,
            _ => g.rpie(&RPIE, ncycle)?,
        }
        g.synchronize()?;
        let secs = t.elapsed().as_secs_f64();
        (g.history().to_vec(), g.object()?, g.probe()?, secs)
    } else {
        let t = std::time::Instant::now();
        match algo.as_str() {
            "dm" => p.dm(&dm, ncycle),
            "ap" => p.ap(&AP, ncycle),
            "ml" => p.ml(&ML, ncycle)?,
            _ => p.rpie(&RPIE, ncycle)?,
        }
        let secs = t.elapsed().as_secs_f64();
        let history = p.history().to_vec();
        let (object, probe) = p.into_modes();
        (history, object, probe, secs)
    };
    eprintln!("{algo} x{ncycle} on {backend}: {secs:.2} s");

    let mut s = String::new();
    for r in &history {
        write!(s, "{} {}", r.cycle, llk_line(&r.llk))?;
    }
    std::fs::write(out.join("llk_rust.txt"), s)?;
    ptyrs::io::write_modes(out.join("final_rust.h5"), &object, &probe)?;
    Ok(())
}

/// The normalised Poisson, Gaussian, Euclidian LLK and photon count, one line.
fn llk_line(l: &Llk) -> String {
    format!(
        "{:.17e} {:.17e} {:.17e} {:.17e}\n",
        l.poisson, l.gaussian, l.euclidian, l.nb_photons_calc
    )
}
