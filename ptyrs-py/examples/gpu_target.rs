//! The GPU engine's BH or rPIE on a measured scan from its init as it is, on CUDA device 0, for
//! `refs/py/time_to_target.py`.
//!
//! `gpu_target <cxi> <init.h5> <nepoch> <target|-> bh <batch_size> <probe_rho> <seed>`
//! `gpu_target <cxi> <init.h5> <nepoch> <target|-> rpie <obj_step> <probe_step> <obj_alpha> <probe_alpha> <batch_size>`
//!
//! Prints `setup_s` (reading the files, the CUDA context and [`GpuPtycho::new`], synchronised),
//! with `read_s` and `context_s` the time to the end of the first two, and `llk E G`, the
//! Gaussian LLK after epoch `E` of `nepoch`, an LLK pass after every epoch.
//! With a target, `target E` for the first epoch at or below it (or `target none`), then a new
//! [`GpuPtycho`] from the same init runs those `E` epochs with no LLK pass: `wall_s` is their
//! synchronised wall time, graph captures included, and `end_llk` the LLK they end at.

use std::iter::once;
use std::time::Instant;

use cuda_core::CudaContext;
use ptyrs::engine::{BhParams, EngineError, GpuOptions, GpuPtycho, RpieParams};

const USAGE: &str = "usage: gpu_target <cxi> <init.h5> <nepoch> <target|-> \
    (bh <batch_size> <probe_rho> <seed> | \
    rpie <obj_step> <probe_step> <obj_alpha> <probe_alpha> <batch_size>)";

enum Method {
    Bh(BhParams),
    Rpie(RpieParams),
}

impl Method {
    fn parse(name: &str, p: &[String]) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(match (name, p) {
            ("bh", [batch, rho, seed]) => Self::Bh(BhParams {
                batch_size: batch.parse()?,
                probe_rho: rho.parse()?,
                seed: seed.parse()?,
            }),
            ("rpie", [obj_step, probe_step, obj_alpha, probe_alpha, batch]) => {
                Self::Rpie(RpieParams {
                    obj_step: Some(obj_step.parse()?),
                    probe_step: Some(probe_step.parse()?),
                    obj_alpha: obj_alpha.parse()?,
                    probe_alpha: probe_alpha.parse()?,
                    batch_size: batch.parse()?,
                    calc_llk: 0,
                })
            }
            _ => return Err(USAGE.into()),
        })
    }

    /// `nepoch` epochs in one call: BH restarts its CG direction at every call.
    fn run(&self, g: &mut GpuPtycho, nepoch: usize, calc_llk: usize) -> Result<(), EngineError> {
        match self {
            Self::Bh(p) => g.bh(p, nepoch, calc_llk),
            Self::Rpie(p) => g.rpie(&RpieParams { calc_llk, ..*p }, nepoch),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [cxi, init, nepoch, target, name, params @ ..] = args.as_slice() else {
        return Err(USAGE.into());
    };
    let nepoch: usize = nepoch.parse()?;
    let target: Option<f64> = match target.as_str() {
        "-" => None,
        t => Some(t.parse()?),
    };
    let method = Method::parse(name, params)?;
    let options = GpuOptions::default();

    let t = Instant::now();
    let scan = ptyrs::io::read_cxi(cxi)?;
    let (object, probe) = ptyrs::io::read_modes(init)?;
    println!("read_s {}", t.elapsed().as_secs_f64());
    let ctx = CudaContext::new(0).map_err(EngineError::from)?;
    println!("context_s {}", t.elapsed().as_secs_f64());
    let mut g = GpuPtycho::new(&ctx, &scan, object.clone(), probe.clone(), options)?;
    g.synchronize()?;
    println!("setup_s {}", t.elapsed().as_secs_f64());

    let start = g.llk()?.gaussian;
    method.run(&mut g, nepoch, 1)?;
    let curve: Vec<f64> = once(start)
        .chain(g.history().iter().map(|r| r.llk.gaussian))
        .collect();
    for (e, v) in curve.iter().enumerate() {
        println!("llk {e} {v}");
    }
    let Some(target) = target else {
        return Ok(());
    };
    let Some(epochs) = curve.iter().position(|&v| v <= target) else {
        println!("target none");
        return Ok(());
    };
    println!("target {epochs}");

    drop(g);
    let mut g = GpuPtycho::new(&ctx, &scan, object, probe, options)?;
    g.synchronize()?;
    let t = Instant::now();
    method.run(&mut g, epochs, 0)?;
    g.synchronize()?;
    println!("wall_s {}", t.elapsed().as_secs_f64());
    println!("end_llk {}", g.llk()?.gaussian);
    Ok(())
}
