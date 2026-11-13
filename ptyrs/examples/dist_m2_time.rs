//! M2 (DESIGN §0 fold condition 2): [`GpuPtycho`] AP and DM per-cycle time on CUDA device 0,
//! graph replays and fused passes, from the `scan.cxi` and `init.h5` that
//! `scripts/engine/pynx_m2.py` times PyNX CUDA on.
//!
//! `cargo run --release -p ptyrs --example dist_m2_time -- export <ap_scaling_dir> <out_dir>`
//! writes `dist_ap_scaling`'s fermat 4096 counts (256x256, 2 probe modes, 80 px steps) as
//! `scan.cxi`, and its starting point (flat object, truth probe scaled to the counts) as
//! `init.h5`. M1's are `engine_llk_trace <dir> dm <ncycle> 2 256 400 16`'s.
//!
//! `... -- time <dir> <ap|dm> <ncycle> <sizes> <repeats>` runs, per comma-separated stack
//! size, 5 untimed cycles, then `repeats` timed calls of `ncycle` cycles, and prints their
//! median.
//!
//! `... -- prof <dir> <ap|dm> <size> <k>` runs 5 untimed cycles, then one call of `k` cycles
//! between `cuProfilerStart` and `cuProfilerStop`, for `ncu --profile-from-start off`.

use std::error::Error;
use std::path::Path;
use std::time::Instant;

use cuda_core::CudaContext;
use ptyrs::base::{
    C64, Counts, Geometry, Object, Positions, Probe, Scan, Stack, required_obj_shape, sim,
};
use ptyrs::engine::{ApParams, DmParams, EngineError, GpuOptions, GpuPtycho};
use ptyrs::io::{read_cxi, read_modes, write_cxi, write_modes};

#[link(name = "cuda")]
unsafe extern "C" {
    fn cuProfilerStart() -> i32;
    fn cuProfilerStop() -> i32;
}

const WARMUP: usize = 5;
const FRAME: usize = 256;
const NMODE: usize = 2;
const NFERMAT: usize = 4096;
const AP: ApParams = ApParams {
    obj_inertia: Some(0.01),
    probe_inertia: Some(0.001),
    calc_llk: 0,
    zero_phase_ramp: false,
};

fn export(src: &Path, out: &Path) -> Result<(), Box<dyn Error>> {
    let g = Geometry::new(1.5e-10, 1.0, 75e-6, [FRAME, FRAME])?;
    let pixel = g.pixel_size_object();
    let f = sim::fermat(NFERMAT, 80.0 * pixel[1]);
    // CXI stores translations as f32
    let round = |v: Vec<f64>| v.into_iter().map(|x| x as f32 as f64).collect();
    let pos = Positions::new(round(f.y), round(f.x))?;
    let shape = required_obj_shape(&pos.pynx_preprocess(pixel), pixel, g.frame, 16);
    let raw: Vec<u16> = std::fs::read(src.join(format!("fermat_{NFERMAT}_{FRAME}_{NMODE}.u16")))?
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect();
    let counts = Counts::from_raw([NFERMAT, FRAME, FRAME], raw)?;
    let total: f64 = counts.as_slice().iter().map(|&c| f64::from(c)).sum();
    let probe = sim::probe(g.frame, NMODE)?;
    let p2: f64 = probe.as_slice().iter().map(C64::norm_sqr).sum();
    let s = (total / (NFERMAT as f64 * p2)).sqrt();
    let probe = Probe::new(Stack::from_vec(
        [NMODE, FRAME, FRAME],
        probe.as_slice().iter().map(|c| c * s).collect(),
    )?);
    let object = Object::new(Stack::from_vec(
        [1, shape[0], shape[1]],
        vec![C64::new(1.0, 0.0); shape[0] * shape[1]],
    )?);
    std::fs::create_dir_all(out)?;
    write_cxi(out.join("scan.cxi"), &Scan::new(g, pos, counts)?)?;
    write_modes(out.join("init.h5"), &object, &probe)?;
    println!("fermat {NFERMAT}: object {shape:?}, probe scale {s:.6e}");
    Ok(())
}

/// The engine on `dir`'s scan and starting point, after the untimed cycles, and its frame
/// count.
fn warm(dir: &Path, algo: &str, size: usize) -> Result<(GpuPtycho, usize), Box<dyn Error>> {
    let scan = read_cxi(dir.join("scan.cxi"))?;
    let (object, probe) = read_modes(dir.join("init.h5"))?;
    let ctx = CudaContext::new(0).map_err(EngineError::from)?;
    let options = GpuOptions {
        stack_size: size,
        graph: true,
        fused: true,
        ..GpuOptions::default()
    };
    let mut g = GpuPtycho::new(&ctx, &scan, object, probe, options)?;
    cycles(&mut g, algo, WARMUP)?;
    g.synchronize()?;
    Ok((g, scan.counts.nframe()))
}

fn cycles(g: &mut GpuPtycho, algo: &str, n: usize) -> Result<(), EngineError> {
    let dm = DmParams {
        calc_llk: 0,
        zero_phase_ramp: false,
        ..DmParams::default()
    };
    match algo {
        "dm" => g.dm(&dm, n),
        _ => g.ap(&AP, n),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["export", src, out] => export(Path::new(src), Path::new(out)),
        ["time", dir, algo @ ("ap" | "dm"), ncycle, sizes, repeats] => {
            let (ncycle, repeats): (usize, usize) = (ncycle.parse()?, repeats.parse()?);
            for size in sizes.split(',') {
                let size: usize = size.parse()?;
                let (mut g, nframe) = warm(Path::new(dir), algo, size)?;
                let mut ms = Vec::new();
                for _ in 0..repeats {
                    let t = Instant::now();
                    cycles(&mut g, algo, ncycle)?;
                    g.synchronize()?;
                    ms.push(1e3 * t.elapsed().as_secs_f64() / ncycle as f64);
                }
                let runs: Vec<String> = ms.iter().map(|m| format!("{m:.3}")).collect();
                ms.sort_by(f64::total_cmp);
                println!(
                    "{algo} size {size} graph fused: {nframe} frames, median {:.3} ms/cycle (runs {}) over {ncycle} cycles",
                    ms[ms.len() / 2],
                    runs.join(" ")
                );
            }
            Ok(())
        }
        ["prof", dir, algo @ ("ap" | "dm"), size, k] => {
            let (size, k): (usize, usize) = (size.parse()?, k.parse()?);
            let (mut g, _) = warm(Path::new(dir), algo, size)?;
            // SAFETY: argument-free driver calls; a failure only means no profiler is attached
            unsafe { cuProfilerStart() };
            cycles(&mut g, algo, k)?;
            g.synchronize()?;
            // SAFETY: as above
            unsafe { cuProfilerStop() };
            println!("{algo} size {size}: profiled {k} cycles");
            Ok(())
        }
        _ => Err("usage: dist_m2_time export <src> <out> | time <dir> <ap|dm> <ncycle> <sizes> <repeats> | prof <dir> <ap|dm> <size> <k>".into()),
    }
}
