//! [`GpuPtycho`] per-cycle time by stack size on CUDA device 0, from `scan.cxi` and `init.h5`
//! (`dist_m2_time`'s or `engine_llk_trace`'s), with a hash of the result bits: the timing
//! behind DESIGN's stack-rule and RAAR numbers.
//!
//! `cargo run --release -p ptyrs --example engine_carried_time -- <dir> <algos> <ncycle> <sizes>
//! <repeats> [fused|f16|m1] [pos]`: for each comma-separated algorithm (`ap`, `dm`, `raar`, `ml`,
//! `bh`, `lsqml`; BH and Poisson LSQML in batches of 96 frames) and size (`default` for
//! `GpuOptions::stack_size: None`), 5 untimed cycles (epochs), then `repeats` timed calls of
//! `ncycle`; prints the median ms per cycle and a hash of the final object, probe, offsets and
//! one LLK pass's record. `f16` is `fused` with [`GpuOptions::psi_f16`]; `pos` models
//! positions from zero offsets (fused passes only).

use std::error::Error;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::Path;
use std::time::Instant;

use cuda_core::CudaContext;
use ptyrs::engine::{
    ApParams, BhParams, Cost, DmParams, EngineError, Features, GpuOptions, GpuPtycho, LsqmlParams,
    MlParams, RaarParams,
};
use ptyrs::io::{read_cxi, read_modes};

const WARMUP: usize = 5;
const BATCH: usize = 96;

fn cycles(g: &mut GpuPtycho, algo: &str, n: usize) -> Result<(), EngineError> {
    match algo {
        "ap" => g.ap(
            &ApParams {
                obj_inertia: Some(0.01),
                probe_inertia: Some(0.001),
                calc_llk: 0,
                zero_phase_ramp: false,
            },
            n,
        ),
        "dm" => g.dm(
            &DmParams {
                calc_llk: 0,
                zero_phase_ramp: false,
                ..DmParams::default()
            },
            n,
        ),
        "raar" => g.raar(
            &RaarParams {
                calc_llk: 0,
                zero_phase_ramp: false,
                ..RaarParams::default()
            },
            n,
        ),
        "ml" => g.ml(
            &MlParams {
                update_object: true,
                update_probe: true,
                calc_llk: 0,
                zero_phase_ramp: false,
            },
            n,
        ),
        "bh" => g.bh(
            &BhParams {
                batch_size: BATCH,
                probe_rho: 0.01,
                position_rho: 0.1,
                seed: 7,
            },
            n,
            0,
        ),
        "lsqml" => g.lsqml(
            &LsqmlParams {
                batch_size: BATCH,
                cost: Cost::Poisson,
                seed: 7,
            },
            n,
            0,
        ),
        _ => Err(EngineError::Invalid(format!("unknown algorithm {algo}"))),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let usage =
        "usage: engine_carried_time <dir> <algos> <ncycle> <sizes> <repeats> [fused|f16|m1] [pos]";
    if a.len() < 5 {
        return Err(usage.into());
    }
    let dir = Path::new(&a[0]);
    let (ncycle, repeats): (usize, usize) = (a[2].parse()?, a[4].parse()?);
    let (mode, fused, psi_f16) = match a.get(5).map(String::as_str) {
        None | Some("fused") => ("fused", true, false),
        Some("f16") => ("f16", true, true),
        Some("m1") => ("m1", false, false),
        _ => return Err(usage.into()),
    };
    let pos = match a.get(6).map(String::as_str) {
        None => false,
        Some("pos") => true,
        _ => return Err(usage.into()),
    };
    let scan = read_cxi(dir.join("scan.cxi"))?;
    let (object, probe) = read_modes(dir.join("init.h5"))?;
    let nframe = scan.counts.nframe();
    let ctx = CudaContext::new(0).map_err(EngineError::from)?;
    for algo in a[1].split(',') {
        for size in a[3].split(',') {
            let stack_size = (size != "default").then(|| size.parse()).transpose()?;
            let options = GpuOptions {
                stack_size,
                graph: true,
                fused,
                psi_f16,
                ..GpuOptions::default()
            };
            let mut g = GpuPtycho::new(&ctx, &scan, object.clone(), probe.clone(), options)?;
            if pos {
                g.set_features(&Features {
                    position: Some(vec![[0.0; 2]; nframe]),
                    ..Features::default()
                })?;
            }
            cycles(&mut g, algo, WARMUP)?;
            g.synchronize()?;
            let mut ms = Vec::new();
            for _ in 0..repeats {
                let t = Instant::now();
                cycles(&mut g, algo, ncycle)?;
                g.synchronize()?;
                ms.push(1e3 * t.elapsed().as_secs_f64() / ncycle as f64);
            }
            let llk = g.llk()?;
            let mut h = DefaultHasher::new();
            for s in [g.object()?.as_slice(), g.probe()?.as_slice()] {
                for v in s {
                    (v.re.to_bits(), v.im.to_bits()).hash(&mut h);
                }
            }
            for d in g.features()?.position.iter().flatten().flatten() {
                d.to_bits().hash(&mut h);
            }
            for v in [
                llk.poisson,
                llk.gaussian,
                llk.euclidian,
                llk.nb_photons_calc,
            ] {
                v.to_bits().hash(&mut h);
            }
            let runs: Vec<String> = ms.iter().map(|m| format!("{m:.3}")).collect();
            ms.sort_by(f64::total_cmp);
            println!(
                "{algo} size {size} {mode}{}: {nframe} frames, median {:.3} ms/cycle (runs {}) over {ncycle} cycles, hash {:016x}, llk {:.6e}",
                if pos { " pos" } else { "" },
                ms[ms.len() / 2],
                runs.join(" "),
                h.finish(),
                llk.poisson
            );
        }
    }
    Ok(())
}
