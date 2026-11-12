//! The truth LLK of an `llk_trace` run, the target of `refs/ptychi/trace_m3.py`.
//!
//! `cargo run --release -p ptyrs-engine --example scenario_truth -- <llk_trace out_dir>`
//! rebuilds `common::dm_scenario`'s truth object and probe from the shapes in `scan.cxi` and
//! `init.h5`, checks that `sim::simulate` of them reproduces `scan.cxi`'s counts exactly, and
//! writes `llk_truth.txt`: the normalised Poisson, Gaussian, Euclidian LLK and photon count of
//! the truth in `llk_rust.txt`'s columns. The probe is scaled by `simulate`'s flux factor, so
//! the truth's `icalc` is the mean the counts were drawn from.

use std::path::PathBuf;

use ptyrs_engine::Ptycho;

/// `llk_trace`'s photons per frame and `dm_scenario`'s noise seed.
const PHOTONS: f64 = 2e5;
const SEED: u64 = 1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("usage: scenario_truth <llk_trace out_dir>")?,
    );
    let scan = ptyrs_io::read_cxi(dir.join("scan.cxi"))?;
    let (init_obj, init_probe) = ptyrs_io::read_modes(dir.join("init.h5"))?;
    let object = ptyrs_core::sim::object(init_obj.plane_shape());
    let mut probe = ptyrs_core::sim::probe(scan.geometry.frame, init_probe.nmode())?;
    let again = ptyrs_engine::sim::simulate(
        scan.geometry,
        scan.positions.clone(),
        &object,
        &probe,
        PHOTONS,
        None,
        SEED,
    )?;
    if again.counts != scan.counts {
        return Err("the rebuilt truth does not reproduce scan.cxi".into());
    }

    let nframe = scan.counts.nframe();
    let p = Ptycho::new(&scan, object.clone(), probe.clone())?;
    // simulate's flux factor is over every pixel, the LLK's photon sum over unmasked ones
    if p.observed().nb_obs() != scan.counts.as_slice().len() {
        return Err("scan.cxi has masked pixels".into());
    }
    let r = (PHOTONS * nframe as f64 / p.llk().nb_photons_calc).sqrt();
    probe.as_mut_slice().iter_mut().for_each(|v| *v *= r);
    let l = Ptycho::new(&scan, object, probe)?.llk();
    let line = format!(
        "{:.17e} {:.17e} {:.17e} {:.17e}\n",
        l.poisson, l.gaussian, l.euclidian, l.nb_photons_calc
    );
    std::fs::write(dir.join("llk_truth.txt"), &line)?;
    eprint!("truth LLK (poisson gaussian euclidian photons): {line}");
    Ok(())
}
