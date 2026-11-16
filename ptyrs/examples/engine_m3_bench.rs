//! Setup and steady epoch time of GPU LSQML or BH on an M3 scan, one engine per process on CUDA
//! device 0, for the comparison with pty-chi (`refs/ptychi/run_m3.py bench`): the settings of
//! `engine_lsqml_m3` and `engine_bh_m3`, from `init.h5` as it is.
//!
//! `cargo run --release -p ptyrs --example engine_m3_bench -- <cxi> <init.h5> <lsqml-gaussian|lsqml-poisson|bh> <batch> [calls] [nprof] [stack_size]`
//! (defaults `20,60`, 0, 16) prints `setup_s`, from the start of `main` to the start of epoch
//! 1: reading the files, the CUDA context and [`GpuPtycho::new`], synchronised. After an untimed
//! call that captures the epoch graphs it times one call per comma-separated epoch count, none
//! if `calls` is empty; every call starts the algorithm afresh, so a steady epoch is the
//! difference of two calls over the difference of their epochs. With `nprof` > 0 a last call of
//! `nprof` epochs runs between `cuProfilerStart` and `cuProfilerStop`, for
//! `ncu --profile-from-start off`.

use std::time::Instant;

use cuda_core::CudaContext;
use ptyrs::engine::{
    BhParams, Cost, EngineError, GpuOptions, GpuPtycho, LsqmlParams, ProbeVariation,
};

#[link(name = "cuda")]
unsafe extern "C" {
    fn cuProfilerStart() -> i32;
    fn cuProfilerStop() -> i32;
}

enum Algo {
    Lsqml(LsqmlParams),
    Bh(BhParams),
}

impl Algo {
    /// The untimed epochs of `engine_lsqml_m3` and `engine_bh_m3`.
    fn warmup(&self) -> usize {
        match self {
            Self::Lsqml(_) => 12,
            Self::Bh(_) => 5,
        }
    }

    /// `plane` is the probe's.
    fn run(&self, g: &mut GpuPtycho, plane: [usize; 2], nepoch: usize) -> Result<(), EngineError> {
        match self {
            Self::Lsqml(p) => g.lsqml(p, &mut ProbeVariation::none(plane), nepoch, 0)?,
            Self::Bh(p) => g.bh(p, nepoch, 0)?,
        }
        g.synchronize()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let usage = "usage: m3_bench <cxi> <init.h5> <lsqml-gaussian|lsqml-poisson|bh> <batch> [calls] [nprof] [stack_size]";
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [cxi, init, algo, batch, rest @ ..] = args.as_slice() else {
        return Err(usage.into());
    };
    if rest.len() > 3 {
        return Err(usage.into());
    }
    let batch_size = batch.parse()?;
    let lsqml = |cost| {
        Algo::Lsqml(LsqmlParams {
            batch_size,
            cost,
            seed: 7,
        })
    };
    let algo = match algo.as_str() {
        "lsqml-gaussian" => lsqml(Cost::Amplitude),
        "lsqml-poisson" => lsqml(Cost::Poisson),
        "bh" => Algo::Bh(BhParams {
            batch_size,
            probe_rho: 0.01,
            position_rho: 0.1,
            seed: 7,
        }),
        _ => return Err(usage.into()),
    };
    let calls = rest.first().map_or("20,60", String::as_str);
    let calls = calls
        .split(',')
        .filter(|n| !n.is_empty())
        .map(str::parse)
        .collect::<Result<Vec<usize>, _>>()?;
    let nprof: usize = rest.get(1).map_or(Ok(0), |s| s.parse())?;
    let stack_size = rest.get(2).map_or(Ok(16), |s| s.parse())?;

    let scan = ptyrs::io::read_cxi(cxi)?;
    let (object, probe) = ptyrs::io::read_modes(init)?;
    let plane = probe.plane_shape();
    let ctx = CudaContext::new(0).map_err(EngineError::from)?;
    let options = GpuOptions {
        stack_size: Some(stack_size),
        ..GpuOptions::default()
    };
    let mut g = GpuPtycho::new(&ctx, &scan, object, probe, options)?;
    g.synchronize()?;
    println!("setup_s {}", start.elapsed().as_secs_f64());

    let t = Instant::now();
    algo.run(&mut g, plane, algo.warmup())?;
    println!(
        "warmup_s {} epochs {}",
        t.elapsed().as_secs_f64(),
        algo.warmup()
    );
    for n in calls {
        let t = Instant::now();
        algo.run(&mut g, plane, n)?;
        println!("call_s {} epochs {n}", t.elapsed().as_secs_f64());
    }
    if nprof > 0 {
        // SAFETY: plain driver calls on the current context; an error only means no profiler.
        unsafe { cuProfilerStart() };
        algo.run(&mut g, plane, nprof)?;
        // SAFETY: as above.
        unsafe { cuProfilerStop() };
        println!("profiled epochs {nprof}");
    }
    Ok(())
}
