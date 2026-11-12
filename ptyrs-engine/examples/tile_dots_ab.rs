//! M6: time per cycle of GPU ML and per epoch of GPU BH with each Tile dot product move
//! ([`GpuOptions::tile_dots`]) against SIMT, on CUDA device 0, in one process.
//!
//! `cargo run --release -p ptyrs-engine --example tile_dots_ab -- ml [ncycle] [rounds]`
//! (defaults 50, 21) runs ML updating object and probe on `gpu_time`'s M1 case (400 frames
//! of 256x256 on a Fermat spiral of step 16 pixels, 2 probe modes, stacks of 16, fused,
//! graphs); its moves are `beta` (`cg_beta_parts`, every cycle) and `norm`
//! (`cg_norm_parts`, once per call in its start).
//!
//! `... tile_dots_ab -- bh <cxi> <init.h5> <batch_size> [nepoch] [rounds]` (defaults 20, 21)
//! runs BH from `init.h5` as it is, as `bh_m3` does (probe rho 0.01, seed 7, stacks of 16,
//! fused, graphs); its move is `bh` (`bh_dots`, once per batch).
//!
//! Every variant, and a second all-SIMT one (`simt2`) for the noise between two devices of
//! the same options, is built first and warmed up by 5 cycles. Each round then times one
//! call of `ncycle` cycles per variant, in an order rotated per round: the wall time of the
//! call and a sync, as `gpu_time` times it, without LLK. All variants run the same calls, so
//! their arrays must end bit-identical, which is checked last. Printed per variant: the
//! median time per cycle and its range, and the median over rounds of its difference from
//! `simt` in the same round.

use std::time::Instant;

use cuda_core::CudaContext;
use ptyrs_engine::{BhParams, EngineError, GpuOptions, GpuPtycho, MlParams, Ptycho, TileDotsOn};

#[path = "../tests/common/mod.rs"]
mod common;

const WARMUP: usize = 5;

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: tile_dots_ab ml [ncycle] [rounds] | tile_dots_ab bh <cxi> <init.h5> <batch_size> [nepoch] [rounds]";
    let mut args = std::env::args().skip(1);
    let algo = args.next().ok_or(usage)?;
    let ctx = CudaContext::new(0).map_err(EngineError::from)?;
    let simt = TileDotsOn::SIMT;
    let (scan, object, probe, moves, bh) = match algo.as_str() {
        "ml" => {
            let (scan, object, probe) = common::dm_scenario(256, 400, 2, 2e5, 16.0);
            let mut p = Ptycho::new(&scan, object, probe)?;
            p.scale_obj_probe();
            let (object, probe) = p.into_modes();
            let moves = vec![
                ("beta", TileDotsOn { beta: true, ..simt }),
                ("norm", TileDotsOn { norm: true, ..simt }),
            ];
            (scan, object, probe, moves, None)
        }
        "bh" => {
            let (cxi, init) = (args.next().ok_or(usage)?, args.next().ok_or(usage)?);
            let batch_size = args.next().ok_or(usage)?.parse()?;
            let scan = ptyrs_io::read_cxi(&cxi)?;
            let (object, probe) = ptyrs_io::read_modes(&init)?;
            let params = BhParams {
                batch_size,
                probe_rho: 0.01,
                seed: 7,
            };
            let moves = vec![("bh", TileDotsOn { bh: true, ..simt })];
            (scan, object, probe, moves, Some(params))
        }
        _ => return Err(usage.into()),
    };
    let mut next = |default: usize| args.next().map_or(Ok(default), |s| s.parse());
    let (ncycle, rounds) = (next(if bh.is_some() { 20 } else { 50 })?, next(21)?);
    let ml = MlParams {
        update_object: true,
        update_probe: true,
        calc_llk: 0,
        zero_phase_ramp: false,
    };
    let run = |g: &mut GpuPtycho, n: usize| -> Result<(), EngineError> {
        match &bh {
            Some(p) => g.bh(p, n, 0),
            None => g.ml(&ml, n),
        }
    };

    let mut variants = vec![("simt", simt), ("simt2", simt)];
    variants.extend(moves);
    let mut gs = Vec::new();
    for &(name, tile_dots) in &variants {
        let options = GpuOptions {
            tile_dots,
            ..GpuOptions::default()
        };
        let mut g = GpuPtycho::new(&ctx, &scan, object.clone(), probe.clone(), options)?;
        run(&mut g, WARMUP)?;
        g.synchronize()?;
        eprintln!("{name}: built and warmed up");
        gs.push(g);
    }
    eprintln!(
        "{algo}{}: {} frames {:?}, {} probe modes, object {:?}; {rounds} rounds of {ncycle} cycles",
        bh.map_or(String::new(), |p| format!(" batch {}", p.batch_size)),
        scan.counts.nframe(),
        scan.geometry.frame,
        probe.nmode(),
        object.plane_shape()
    );

    let nv = gs.len();
    let mut ms = vec![Vec::new(); nv];
    for r in 0..rounds {
        for k in 0..nv {
            let v = (r + k) % nv;
            let t = Instant::now();
            run(&mut gs[v], ncycle)?;
            gs[v].synchronize()?;
            ms[v].push(1e3 * t.elapsed().as_secs_f64() / ncycle as f64);
        }
    }

    let bits = |g: &GpuPtycho| -> Result<Vec<u64>, EngineError> {
        let (o, p) = (g.object()?, g.probe()?);
        Ok(o.as_slice()
            .iter()
            .chain(p.as_slice())
            .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
            .collect())
    };
    let want = bits(&gs[0])?;
    for (g, (name, _)) in gs.iter().zip(&variants).skip(1) {
        if bits(g)? != want {
            return Err(format!("{name}: object or probe differs from simt").into());
        }
    }
    println!("{algo}: object and probe bit-identical across variants");
    for (v, (name, _)) in variants.iter().enumerate() {
        let d: Vec<f64> = ms[v].iter().zip(&ms[0]).map(|(a, b)| a - b).collect();
        let lo = ms[v].iter().copied().fold(f64::INFINITY, f64::min);
        let hi = ms[v].iter().copied().fold(0.0, f64::max);
        println!(
            "{algo} {name:6}: {:.4} ms/cycle ({lo:.4}..{hi:.4}), {:+.4} ms/cycle vs simt (round median)",
            median(&ms[v]),
            median(&d)
        );
    }
    Ok(())
}
