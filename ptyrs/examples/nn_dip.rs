//! DIP (DESIGN §4) against the engine's ML on two held-out scans of `nn_train_ptychonn`, a random
//! object and `sim::object`, both on the GPU with the known probe fixed. DIP trains its network
//! by Adam on the object gradient of `GpuPtycho::ml_grads`. ML starts from the flat object after
//! `DM_CYCLES` DM cycles, as PyNX's recipes run DM or AP first: at the flat start the narrow
//! probe's far field underflows f32 at pixels with counts, so the GPU Poisson LLK is infinite
//! and ML's line search does not move. The DM cycles count as iterations of ML's curve. Every
//! LLK curve goes to `<curves.tsv>`, one row per iteration, iteration 0 being the start; the
//! table gives the LLK at a few iterations and the iterations DIP takes to come within
//! [`LLK_MARGIN`] of ML's LLK after N.
//!
//! `cargo run --release -p ptyrs --features nn --example nn_dip -- <curves.tsv> [iterations=1000] [lr=0.01]`

#[path = "common/nn/mod.rs"]
mod common;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::time::Instant;

use burn::prelude::*;
use cuda_core::CudaContext;
use ptyrs::base::{C64, Object, Probe, Scan, Stack, sim};
use ptyrs::engine::{DmParams, GpuOptions, GpuPtycho, MlParams, Ptycho};
use ptyrs::nn::dip::{Dip, DipConfig, LossError, ObjectLoss};
use ptyrs::nn::metric::{LLK_MARGIN, iterations_to_llk};
use ptyrs::nn::predictor::flat_object;

const FRAME: usize = 256;
/// DM cycles before ML: one makes the LLK finite, and of 1 to 50 it leaves ML lowest after 20,
/// 50 and 100 iterations on both scans.
const DM_CYCLES: usize = 1;
/// Iterations at which the table gives the LLK.
const AT: [usize; 7] = [0, 10, 20, 50, 100, 200, 500];
/// ML iterations whose LLK becomes DIP's target.
const NS: [usize; 2] = [20, 100];

/// The engine's normalised Poisson LLK and its object gradient: `ml_grads` gives that of the
/// LLK sum.
struct GpuLoss {
    ptycho: GpuPtycho,
    nb_obs: f64,
}

impl ObjectLoss for GpuLoss {
    fn gradient(&mut self, object: &Tensor<3>) -> Result<(f64, Tensor<3>), LossError> {
        let [ny, nx, _] = object.dims();
        let host = object
            .to_data()
            .convert::<f64>()
            .try_into_vec::<f64>()
            .map_err(|e| format!("{e:?}"))?;
        let data = host.chunks(2).map(|z| C64::new(z[0], z[1])).collect();
        self.ptycho
            .set_object(&Object::new(Stack::from_vec([1, ny, nx], data)?))?;
        let llk = self.ptycho.llk()?.poisson;
        let [g, _] = self.ptycho.ml_grads()?;
        let n = self.nb_obs;
        let g: Vec<f64> = g
            .as_slice()
            .iter()
            .flat_map(|c| [c.re / n, c.im / n])
            .collect();
        let g = TensorData::new(g, object.shape());
        Ok((
            llk,
            Tensor::from_data(g, (&object.device(), object.dtype())),
        ))
    }
}

/// The LLK of the flat start and after each of `n` iterations, [`DM_CYCLES`] of DM then ML.
fn ml(
    ctx: &Arc<CudaContext>,
    scan: &Scan,
    flat: Object,
    probe: &Probe,
    n: usize,
) -> Result<Vec<f64>, LossError> {
    let mut g = GpuPtycho::new(ctx, scan, flat, probe.clone(), GpuOptions::default())?;
    let mut llk = vec![g.llk()?.poisson];
    let dm = DmParams {
        probe_inertia: None,
        calc_llk: 1,
        zero_phase_ramp: false,
        ..DmParams::default()
    };
    g.dm(&dm, DM_CYCLES)?;
    let ml = MlParams {
        calc_llk: 1,
        zero_phase_ramp: false,
        ..MlParams::default()
    };
    g.ml(&ml, n - DM_CYCLES)?;
    // DM records the estimate its cycle ends with, ML the one its cycle starts from: ML's first
    // record is DM's last estimate, rescaled.
    let history = g.history();
    for r in history[..DM_CYCLES].iter().chain(&history[DM_CYCLES + 1..]) {
        llk.push(r.llk.poisson);
    }
    llk.push(g.llk()?.poisson);
    Ok(llk)
}

/// The LLK after each of `0..=n` DIP steps of size `lr`.
fn dip(
    ctx: &Arc<CudaContext>,
    scan: &Scan,
    start: Object,
    probe: &Probe,
    n: usize,
    lr: f64,
) -> Result<Vec<f64>, LossError> {
    let [_, ny, nx] = start.dims();
    let ptycho = GpuPtycho::new(ctx, scan, start, probe.clone(), GpuOptions::default())?;
    let mut loss = GpuLoss {
        ptycho,
        nb_obs: scan.counts.nb_obs() as f64,
    };
    let mut net = Dip::new(&DipConfig::new([ny, nx]), &Device::cuda(0))?;
    let mut llk = Vec::with_capacity(n + 1);
    for _ in 0..n {
        llk.push(net.step(&mut loss, lr)?);
    }
    llk.push(loss.gradient(&net.object())?.0);
    Ok(llk)
}

fn main() -> Result<(), LossError> {
    let usage = "usage: dip <curves.tsv> [iterations] [lr]";
    let mut args = std::env::args().skip(1);
    let curves = args.next().ok_or(usage)?;
    let n: usize = args.next().map_or(Ok(1000), |a| a.parse())?;
    let lr: f64 = args.next().map_or(Ok(0.01), |a| a.parse())?;
    if n <= DM_CYCLES {
        return Err(format!("iterations must exceed the {DM_CYCLES} DM cycles").into());
    }
    let ctx = CudaContext::new(0)?;
    let probe = sim::probe([FRAME, FRAME], 1)?;

    let mut w = BufWriter::new(File::create(&curves)?);
    writeln!(w, "object\tseed\talg\titer\tllk")?;
    // The table's iterations, the last being n.
    let at: Vec<usize> = AT.into_iter().filter(|&i| i < n).chain([n]).collect();
    let header: String = at.iter().map(|i| format!("{i:>8}")).collect();
    println!("dip at lr {lr}; ml after {DM_CYCLES} dm cycles from flat; probe known and fixed");
    for (name, seed, scan, truth, addr, probe) in common::held_out(FRAME, &probe)
        .into_iter()
        .filter(|&(_, seed, ..)| seed != 2_000_001)
    {
        let at_truth = Ptycho::new(&scan, truth, probe.clone())?.llk().poisson;
        let flat = flat_object(&addr);
        let t = Instant::now();
        let ml = ml(&ctx, &scan, flat.clone(), &probe, n)?;
        let t_ml = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let dip = dip(&ctx, &scan, flat, &probe, n, lr)?;
        let t_dip = t.elapsed().as_secs_f64();
        for (alg, llk) in [("ml", &ml), ("dip", &dip)] {
            for (i, l) in llk.iter().enumerate() {
                writeln!(w, "{name}\t{seed}\t{alg}\t{i}\t{l}")?;
            }
        }

        println!(
            "\n{name} (seed {seed}), object {:?}, LLK of the truth {at_truth:.4}",
            addr.obj_shape()
        );
        println!("{:4} {header}  {:>7}", "", "s/iter");
        for (alg, llk, t) in [("ml", &ml, t_ml), ("dip", &dip, t_dip)] {
            let row: String = at.iter().map(|&i| format!("{:>8.4}", llk[i])).collect();
            println!("{alg:4} {row}  {:>7.4}", t / n as f64);
        }
        for m in NS.into_iter().filter(|&m| m <= n) {
            let k = iterations_to_llk(dip[0], ml[m], LLK_MARGIN, dip[1..].iter().copied())
                .map_or(format!(">{n}"), |k| k.to_string());
            println!("dip to within 1% of ml's LLK at {m:>3} ({:.4}): {k}", ml[m]);
        }
    }
    w.flush()?;
    Ok(())
}
