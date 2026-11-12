//! The ptyrs CPU f64 side of `refs/ptychi/run_near.py`: BH and Gaussian LSQML on
//! `refs/f64/lsqml_near.py`'s angular-spectrum case (its case 0, saved in `tests/data`), the
//! normalised LLK after every epoch.
//!
//! `cargo run --release -p ptyrs --example near_speed -- <out_dir> <epochs> <run>...`
//! with each run `lsqml:<batch>` or `bh:<batch>:<probe rho>` writes
//! `<out_dir>/ptyrs_<run, ':' as '_'>.txt`, `epoch poisson gaussian euclidian` per line from
//! epoch 0, the start. Batches are `random_batches(16, batch, SEED, epoch)`, as the pty-chi runs
//! of `run_near.py` take them; BH runs CG when one batch holds every frame, else steepest descent.

use std::error::Error;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::{fs, slice};

use ptyrs::base::{AddrTable, C64, Counts, Object, Positions, Probe, Stack};
use ptyrs::engine::{
    Bh, BhParams, Cost, Llk, Lsqml, LsqmlParams, Observed, Optics, ProbeVariation, Wavefield, llk,
    obj_probe_to_psi,
};

const SHAPE: [usize; 2] = [24, 24];
const OBJ: [usize; 2] = [40, 40];
const NPROBE: usize = 2;
/// lsqml_near.py's angular-spectrum case.
const CASE: f64 = 0.0;
const SEED: u64 = 7;

fn table(text: &str) -> Vec<Vec<f64>> {
    text.lines()
        .map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect())
        .collect()
}

fn stack(dims: [usize; 3], rows: &[Vec<f64>], lead: usize) -> Stack<C64> {
    let data = rows
        .iter()
        .map(|r| C64::new(r[lead], r[lead + 1]))
        .collect();
    Stack::from_vec(dims, data).unwrap()
}

struct State {
    addr: AddrTable,
    object: Object,
    probe: Probe,
    optics: Optics,
    obs: Observed,
}

fn state() -> State {
    // corner = (OBJ − SHAPE)/2 + position at a pixel of 1
    let corners = table(include_str!(
        "../../tests/engine/data/lsqml_near_corners_f64.txt"
    ));
    let at = |axis: usize| -> Vec<f64> {
        let centre = ((OBJ[axis] - SHAPE[axis]) / 2) as f64;
        corners.iter().map(|r| r[1 + axis] - centre).collect()
    };
    let positions = Positions::new(at(0), at(1)).unwrap();
    let addr = AddrTable::new(&positions, [1.0, 1.0], SHAPE, OBJ, NPROBE, 1).unwrap();
    let object = Object::new(stack(
        [1, OBJ[0], OBJ[1]],
        &table(include_str!(
            "../../tests/engine/data/lsqml_near_obj_f64.txt"
        )),
        2,
    ));
    let probe = Probe::new(stack(
        [NPROBE, SHAPE[0], SHAPE[1]],
        &table(include_str!(
            "../../tests/engine/data/lsqml_near_probe_f64.txt"
        )),
        3,
    ));
    let cases = table(include_str!(
        "../../tests/engine/data/lsqml_near_cases_f64.txt"
    ));
    let c = cases.iter().find(|c| c[0] == CASE).unwrap();
    let optics = Optics::free_space(&Wavefield::new(SHAPE, c[1], [c[2], c[2]], c[3]).unwrap());
    let values: Vec<f64> = table(include_str!(
        "../../tests/engine/data/lsqml_near_counts_f64.txt"
    ))
    .iter()
    .filter(|r| r[0] == CASE)
    .flat_map(|r| r[2..].iter().copied())
    .collect();
    let dims = [corners.len(), SHAPE[0], SHAPE[1]];
    let obs = Observed::from_counts(&Counts::from_f64(dims, &values, None).unwrap());
    State {
        addr,
        object,
        probe,
        optics,
        obs,
    }
}

/// The normalised LLK of every frame.
fn llk_of(s: &State, object: &Object, probe: &Probe) -> Llk {
    let views = s.addr.views();
    let mut psi = vec![C64::default(); views.len() * probe.plane_len()];
    obj_probe_to_psi(views, object, probe, s.optics.quad(), &mut psi);
    s.optics.propagator().forward(&mut psi);
    llk(&s.obs, 0..s.addr.nframe(), s.addr.nmode(), &psi).normalised(s.obs.nb_obs())
}

enum Method {
    Lsqml(Lsqml, ProbeVariation),
    Bh(Bh),
}

fn main() -> Result<(), Box<dyn Error>> {
    let usage = "usage: near_speed <out_dir> <epochs> <lsqml:BATCH | bh:BATCH:RHO>...";
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(args.next().ok_or(usage)?);
    let epochs: usize = args.next().ok_or(usage)?.parse()?;
    let s = state();
    for run in args {
        let mut method = match run.split(':').collect::<Vec<_>>()[..] {
            ["lsqml", batch] => Method::Lsqml(
                Lsqml::new(LsqmlParams {
                    batch_size: batch.parse()?,
                    cost: Cost::Amplitude,
                    seed: SEED,
                }),
                ProbeVariation::none(s.probe.plane_shape()),
            ),
            ["bh", batch, rho] => Method::Bh(Bh::new(BhParams {
                batch_size: batch.parse()?,
                probe_rho: rho.parse()?,
                seed: SEED,
            })),
            _ => return Err(usage.into()),
        };
        let (mut object, mut probe) = (s.object.clone(), s.probe.clone());
        let mut text = String::new();
        for epoch in 0..=epochs {
            if epoch > 0 {
                let objects = slice::from_mut(&mut object);
                match &mut method {
                    Method::Lsqml(l, var) => {
                        l.epoch(&s.addr, &s.obs, &s.optics, objects, &mut probe, var);
                    }
                    Method::Bh(b) => {
                        b.epoch(&s.addr, &s.obs, &s.optics, objects, &mut probe);
                    }
                }
            }
            let l = llk_of(&s, &object, &probe);
            writeln!(
                text,
                "{epoch} {:.17e} {:.17e} {:.17e}",
                l.poisson, l.gaussian, l.euclidian
            )?;
        }
        let path = out.join(format!("ptyrs_{}.txt", run.replace(':', "_")));
        fs::write(&path, text)?;
        println!("wrote {}", path.display());
    }
    Ok(())
}
