//! Position refinement on a scan with known position errors: the RMS distance of the modelled
//! positions (view corner plus [`Features::position`]) from the truth, after removing the mean
//! offset, as `refs/ptychi/run_m3.py`'s `position_error` measures pty-chi's.
//!
//! `cargo run --release -p ptyrs --example engine_pos_fermat -- <cxi> <init.h5> <json> <plan>
//! [positions.txt]`, the first three an M3 dataset (`refs/ptychi/m3ref.py`), `fermat_poserr`
//! for its position errors, where `<plan>` is a comma list of `dm:N`, `ml:N`, `bh:N`,
//! `dmoff:N`, `mloff:N`, `bhoff:N` (`off`: positions kept; `bh`: batches of 96, as
//! `engine_bh_m3`), each one call of N cycles on CUDA device 0; `positions.txt` gets the last
//! modelled positions and the truth. The offsets start at the nominal positions' sub-pixel
//! part, so the first modelled positions are the nominal ones.

use std::time::Instant;

use cuda_core::CudaContext;
use ptyrs::base::view_corner;
use ptyrs::engine::{BhParams, DmParams, EngineError, Features, GpuOptions, GpuPtycho, MlParams};

fn pairs(v: &serde_json::Value) -> Vec<[f64; 2]> {
    let rows = v.as_array().expect("[n, 2] array");
    rows.iter()
        .map(|r| [r[0].as_f64().unwrap(), r[1].as_f64().unwrap()])
        .collect()
}

/// RMS of `|a − b − mean(a − b)|`.
fn rms(a: &[[f64; 2]], b: &[[f64; 2]]) -> f64 {
    let n = a.len() as f64;
    let d: Vec<[f64; 2]> = a
        .iter()
        .zip(b)
        .map(|(a, b)| [a[0] - b[0], a[1] - b[1]])
        .collect();
    let m = [0, 1].map(|i| d.iter().map(|d| d[i]).sum::<f64>() / n);
    let s: f64 = d
        .iter()
        .map(|d| (d[0] - m[0]).powi(2) + (d[1] - m[1]).powi(2))
        .sum();
    (s / n).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: engine_pos_fermat <cxi> <init.h5> <json> <plan> [positions.txt]";
    let mut args = std::env::args().skip(1);
    let mut next = || args.next().ok_or(usage);
    let (cxi, init, json, plan) = (next()?, next()?, next()?, next()?);
    let out = args.next();
    let scan = ptyrs::io::read_cxi(&cxi)?;
    let (object, probe) = ptyrs::io::read_modes(&init)?;
    let meta: serde_json::Value = serde_json::from_reader(std::fs::File::open(&json)?)?;
    let truth = pairs(&meta["positions"]["true_in_nominal_frame_px"]);
    let nominal = pairs(&meta["positions"]["nominal_pp_px"]);

    let pixel = scan.geometry.pixel_size_object();
    let p = scan.positions.pynx_preprocess(pixel).in_pixels(pixel);
    let off = p
        .iter()
        .zip(&nominal)
        .flat_map(|(a, b)| [(a[0] - b[0]).abs(), (a[1] - b[1]).abs()])
        .fold(0.0, f64::max);
    let (obj, frame) = (object.plane_shape(), scan.geometry.frame);
    let base: Vec<[f64; 2]> = p
        .iter()
        .map(|p| {
            [0, 1].map(|a| {
                let c = view_corner(obj[a], frame[a], p[a]).expect("window inside the object");
                c as f64 - ((obj[a] - frame[a]) / 2) as f64
            })
        })
        .collect();
    let d0: Vec<[f64; 2]> = p
        .iter()
        .zip(&base)
        .map(|(p, b)| [p[0] - b[0], p[1] - b[1]])
        .collect();
    let modelled = |d: &[[f64; 2]]| -> Vec<[f64; 2]> {
        base.iter()
            .zip(d)
            .map(|(b, d)| [b[0] + d[0], b[1] + d[1]])
            .collect()
    };
    println!(
        "{} frames, object {obj:?}, {} probe modes; |ptyrs - json nominal| max {off:.2e} px",
        p.len(),
        probe.nmode()
    );
    println!(
        "start: nominal rms {:.4} px, corners only {:.4} px",
        rms(&modelled(&d0), &truth),
        rms(&base, &truth)
    );

    let ctx = CudaContext::new(0).map_err(EngineError::from)?;
    let mut g = GpuPtycho::new(&ctx, &scan, object, probe, GpuOptions::default())?;
    let features = Features {
        position: Some(d0),
        ..Features::default()
    };
    g.set_features(&features)?;
    println!("start llk {:.6e}", g.llk()?.poisson);
    let dm = DmParams {
        zero_phase_ramp: false,
        ..DmParams::default()
    };
    let ml = MlParams {
        update_object: true,
        update_probe: true,
        calc_llk: 0,
        zero_phase_ramp: false,
    };
    let bh = BhParams {
        batch_size: 96,
        probe_rho: 0.01,
        position_rho: 0.1,
        seed: 7,
    };
    for step in plan.split(',') {
        let (algo, n) = step.split_once(':').ok_or(usage)?;
        let n: usize = n.parse()?;
        let kept = g.features()?;
        if algo.ends_with("off") {
            g.set_features(&Features::default())?;
        }
        let t = Instant::now();
        match algo {
            "dm" | "dmoff" => g.dm(&dm, n)?,
            "ml" | "mloff" => g.ml(&ml, n)?,
            "bh" | "bhoff" => g.bh(&bh, n, 0)?,
            _ => return Err(usage.into()),
        }
        g.synchronize()?;
        let s = t.elapsed().as_secs_f64();
        if algo.ends_with("off") {
            g.set_features(&kept)?;
        }
        let d = g.features()?.position.expect("positions");
        let llk = g.llk()?;
        println!(
            "{algo}:{n} -> cycle {} rms {:.4} px, llk {:.6e}, {s:.2} s",
            g.cycle(),
            rms(&modelled(&d), &truth),
            llk.poisson
        );
    }
    if let Some(out) = out {
        let d = g.features()?.position.expect("positions");
        let rows: Vec<String> = (modelled(&d).iter().zip(&truth))
            .map(|(m, t)| format!("{:.9} {:.9} {:.9} {:.9}", m[0], m[1], t[0], t[1]))
            .collect();
        std::fs::write(out, rows.join("\n") + "\n")?;
    }
    Ok(())
}
