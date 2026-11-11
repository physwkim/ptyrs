//! DM or AP on a simulated scan, the Rust side of `scripts/compare_pynx.py`.
//!
//! `cargo run --release -p ptyrs-engine --example llk_trace -- <out_dir> <dm|ap> [ncycle] [nprobe] [frame] [npos]`
//! (defaults 50, 1, 64, 100) writes `scan.cxi`, `init.h5` (the scaled starting object and
//! probe), `llk_rust.txt` (cycle and the normalised Poisson, Gaussian, Euclidian LLK and photon
//! count per cycle) and `final_rust.h5`.

use std::fmt::Write as _;
use std::path::PathBuf;

use ptyrs_engine::{ApParams, DmParams, Ptycho};

#[path = "../tests/common/mod.rs"]
mod common;

/// AP updates the probe too, so both halves of `psi_to_obj_probe` are compared.
const AP: ApParams = ApParams {
    obj_inertia: Some(0.01),
    probe_inertia: Some(0.001),
    calc_llk: 1,
    zero_phase_ramp: true,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: llk_trace <out_dir> <dm|ap> [ncycle] [nprobe] [frame] [npos]";
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(args.next().ok_or(usage)?);
    let algo = args.next().ok_or(usage)?;
    let mut next = |default: usize| args.next().map_or(Ok(default), |s| s.parse());
    let (ncycle, nprobe, frame, npos) = (next(50)?, next(1)?, next(64)?, next(100)?);
    std::fs::create_dir_all(&out)?;

    let (scan, object, probe) = common::dm_scenario(frame, npos, nprobe, 2e5);
    let cxi = out.join("scan.cxi");
    ptyrs_io::write_cxi(&cxi, &scan)?;
    let scan = ptyrs_io::read_cxi(&cxi)?;
    let mut p = Ptycho::new(&scan, object, probe)?;
    p.scale_obj_probe();
    ptyrs_io::write_modes(out.join("init.h5"), p.object(), p.probe())?;

    let t = std::time::Instant::now();
    match algo.as_str() {
        "dm" => p.dm(
            &DmParams {
                calc_llk: 1,
                ..DmParams::default()
            },
            ncycle,
        ),
        "ap" => p.ap(&AP, ncycle),
        _ => return Err(usage.into()),
    }
    eprintln!("{algo} x{ncycle}: {:.2} s", t.elapsed().as_secs_f64());

    let mut s = String::new();
    for r in p.history() {
        let l = r.llk;
        writeln!(
            s,
            "{} {:.17e} {:.17e} {:.17e} {:.17e}",
            r.cycle, l.poisson, l.gaussian, l.euclidian, l.nb_photons_calc
        )?;
    }
    std::fs::write(out.join("llk_rust.txt"), s)?;
    ptyrs_io::write_modes(out.join("final_rust.h5"), p.object(), p.probe())?;
    Ok(())
}
