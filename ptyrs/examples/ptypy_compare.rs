//! Plain DM or RAAR against PtyPy's, the Rust side of `scripts/engine/compare_ptypy.py`, on
//! M1's scan (256² frames, 400 positions 4 object pixels apart, 2 probe modes, 2e5 photons per
//! frame).
//!
//! `cargo run --release -p ptyrs --example ptypy_compare -- <dir> prepare <dm|raar> [ncycle]`
//! (default 100) writes `scan.cxi`, `init.h5` (the scaled start), `views.txt` (object window
//! corner `oy ox` per frame), `truth.h5` and `algo.txt` (the algorithm and its parameter as
//! PtyPy takes it: `dm 1` for DM with alpha 0, `raar 0.9` for RAAR with PyNX's default beta),
//! and runs it without phase-ramp removal on the CPU in f64, writing `llk_rust.txt` (cycle and
//! the normalised Poisson, Gaussian, Euclidian LLK and photon count of the estimate each cycle
//! starts from) and `final_rust.h5`.
//!
//! `... -- <dir> compare` reads PtyPy's state after each iteration `i`,
//! `ptypy/ptypy_<i>.ptyr`, maps its object onto ptyrs's pixels by the storage origin, and
//! prints its LLK (taken by ptyrs) against ptyrs's after `i` cycles (row `i` of
//! `llk_rust.txt`, or `final_rust.h5` for the last); then the FRC between the final objects of
//! ptyrs, PtyPy and the truth over the bounding box of the window centres (the region the M5
//! clock compares).

use std::fmt::Write as _;
use std::ops::Range;
use std::path::{Path, PathBuf};

use ptyrs::base::frc::Frc;
use ptyrs::base::{C64, Object, Probe, Scan};
use ptyrs::engine::{DmParams, Llk, Ptycho, RaarParams};

#[path = "../tests/common/engine/scenario.rs"]
mod scenario;

const FRAME: usize = 256;
const NPOS: usize = 400;
const NPROBE: usize = 2;
const PHOTONS: f64 = 2e5;
const STEP: f64 = 4.0;

type Error = Box<dyn std::error::Error>;

enum Algo {
    Dm,
    Raar,
}

fn main() -> Result<(), Error> {
    let usage =
        "usage: ptypy_compare <dir> prepare <dm|raar> [ncycle] | ptypy_compare <dir> compare";
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().ok_or(usage)?);
    match args.next().as_deref() {
        Some("prepare") => {
            let algo = match args.next().as_deref() {
                Some("dm") => Algo::Dm,
                Some("raar") => Algo::Raar,
                _ => return Err(usage.into()),
            };
            prepare(&dir, algo, args.next().map_or(Ok(100), |s| s.parse())?)
        }
        Some("compare") => compare(&dir),
        _ => Err(usage.into()),
    }
}

fn prepare(dir: &Path, algo: Algo, ncycle: usize) -> Result<(), Error> {
    std::fs::create_dir_all(dir)?;
    let (scan, object, probe, _) = scenario::dm_scenario_truth(FRAME, NPOS, NPROBE, PHOTONS, STEP);
    let cxi = dir.join("scan.cxi");
    ptyrs::io::write_cxi(&cxi, &scan)?;
    let scan = ptyrs::io::read_cxi(&cxi)?;
    let (_, truth_obj, truth_probe) = scenario::truth(FRAME, NPOS, NPROBE, PHOTONS, STEP);
    ptyrs::io::write_modes(dir.join("truth.h5"), &truth_obj, &truth_probe)?;
    let mut p = Ptycho::new(&scan, object, probe)?;
    p.scale_obj_probe();
    ptyrs::io::write_modes(dir.join("init.h5"), p.object(), p.probe())?;
    let mut s = String::new();
    for v in p.addr().views().iter().step_by(p.addr().nmode()) {
        writeln!(s, "{} {}", v.oy, v.ox)?;
    }
    std::fs::write(dir.join("views.txt"), s)?;

    // PtyPy's DM with alpha 1 is PyNX's (ptyrs's) with alpha 0.
    let dm = DmParams {
        alpha: 0.0,
        calc_llk: 1,
        zero_phase_ramp: false,
        ..DmParams::default()
    };
    let raar = RaarParams {
        calc_llk: 1,
        zero_phase_ramp: false,
        ..RaarParams::default()
    };
    let line = match algo {
        Algo::Dm => "dm 1".to_string(),
        Algo::Raar => format!("raar {}", raar.beta),
    };
    std::fs::write(dir.join("algo.txt"), format!("{line}\n"))?;
    let t = std::time::Instant::now();
    match algo {
        Algo::Dm => p.dm(&dm, ncycle),
        Algo::Raar => p.raar(&raar, ncycle),
    }
    eprintln!("{line} x{ncycle}: {:.1} s", t.elapsed().as_secs_f64());
    let mut s = String::new();
    for r in p.history() {
        writeln!(s, "{} {}", r.cycle, llk_line(&r.llk))?;
    }
    std::fs::write(dir.join("llk_rust.txt"), s)?;
    let (object, probe) = p.into_modes();
    ptyrs::io::write_modes(dir.join("final_rust.h5"), &object, &probe).map_err(Into::into)
}

fn compare(dir: &Path) -> Result<(), Error> {
    let scan = ptyrs::io::read_cxi(dir.join("scan.cxi"))?;
    let (start, _) = ptyrs::io::read_modes(dir.join("init.h5"))?;
    let rust = read_rows::<4>(&dir.join("llk_rust.txt"), 1)?;
    let (rust_final, rust_probe) = ptyrs::io::read_modes(dir.join("final_rust.h5"))?;
    let n = rust.len();
    let rust_final_llk = Ptycho::new(&scan, rust_final.clone(), rust_probe)?.llk();

    println!(
        "iteration, rel. diff of the LLK of PtyPy's state from ptyrs's: Poisson Gaussian Euclidian photons"
    );
    let mut worst = [0.0f64; 4];
    let mut ptypy_final = None;
    for i in 0..=n {
        let (object, probe) = ptypy_state(dir, i, &scan, &start)?;
        let l = Ptycho::new(&scan, object.clone(), probe)?.llk();
        let r = rust.get(i).copied().unwrap_or(llk_array(&rust_final_llk));
        let rel: Vec<f64> = llk_array(&l)
            .iter()
            .zip(&r)
            .map(|(a, b)| (a / b - 1.0).abs())
            .collect();
        for (w, d) in worst.iter_mut().zip(&rel) {
            *w = w.max(*d);
        }
        println!(
            "{i:4}  Poisson {:.6e} vs {:.6e}  rel. diff {:.2e} {:.2e} {:.2e} {:.2e}",
            l.poisson, r[0], rel[0], rel[1], rel[2], rel[3]
        );
        ptypy_final = Some(object);
    }
    println!(
        "max rel. diff over iterations 0..={n}: {:.2e} {:.2e} {:.2e} {:.2e}",
        worst[0], worst[1], worst[2], worst[3]
    );

    let ptypy_final = ptypy_final.ok_or("no PtyPy state")?;
    let (truth, _) = ptyrs::io::read_modes(dir.join("truth.h5"))?;
    let corners = read_rows::<2>(&dir.join("views.txt"), 0)?;
    let region = [0, 1].map(|d| centres(corners.iter().map(|c| c[d] as usize)));
    let shape = [region[0].len(), region[1].len()];
    println!(
        "FRC after {n} cycles over rows {:?}, columns {:?}:",
        region[0], region[1]
    );
    for (name, a, b) in [
        ("ptyrs vs PtyPy", &rust_final, &ptypy_final),
        ("truth vs ptyrs", &truth, &rust_final),
        ("truth vs PtyPy", &truth, &ptypy_final),
    ] {
        let f = Frc::compute(&crop(a, &region), &crop(b, &region), shape)?;
        let rings = &f.rings[1..];
        let min = rings.iter().map(|r| r.frc).fold(f64::INFINITY, f64::min);
        let mean = rings.iter().map(|r| r.frc).sum::<f64>() / rings.len() as f64;
        println!(
            "  {name}: min ring {min:.6}, mean {mean:.6}, first ring under half-bit {:?} of {}",
            f.crossing(),
            rings.len()
        );
    }
    Ok(())
}

/// The whitespace-separated numbers of each line of `path`, after `skip` leading ones.
fn read_rows<const N: usize>(path: &Path, skip: usize) -> Result<Vec<[f64; N]>, Error> {
    std::fs::read_to_string(path)?
        .lines()
        .map(|l| {
            let v: Vec<f64> = l
                .split_whitespace()
                .skip(skip)
                .map(str::parse)
                .collect::<Result<_, _>>()?;
            v.try_into().map_err(|_| {
                format!("{}: line {l:?} does not hold {N} values", path.display()).into()
            })
        })
        .collect()
}

/// PtyPy's object and probe in a `.ptyr`, the object moved onto ptyrs's pixels by its storage
/// origin (ptyrs pixel `[0, 0]` at 0, as `compare_ptypy.py` places the views); ptyrs pixels
/// outside PtyPy's storage, which no view reads, keep `start`'s values.
fn ptypy_state(
    dir: &Path,
    i: usize,
    scan: &Scan,
    start: &Object,
) -> Result<(Object, Probe), Error> {
    let path = dir.join(format!("ptypy/ptypy_{i:04}.ptyr"));
    let r = ptyrs::io::read_ptyr(&path)?;
    let pixel = scan.geometry.pixel_size_object();
    let mut off = [0isize; 2];
    for d in 0..2 {
        let shift = -r.origin[d] / r.psize[d];
        if (r.psize[d] / pixel[d] - 1.0).abs() > 1e-12 || (shift - shift.round()).abs() > 1e-6 {
            return Err(format!(
                "{}: psize {:?} origin {:?} is not on ptyrs's {pixel:?} grid",
                path.display(),
                r.psize,
                r.origin
            )
            .into());
        }
        off[d] = shift.round() as isize;
    }
    let [_, ny, nx] = start.0.dims();
    let [_, py, px] = r.object.0.dims();
    let mut object = start.0.clone();
    let (o, p) = (object.as_mut_slice(), r.object.0.as_slice());
    for y in 0..ny {
        for x in 0..nx {
            let (yy, xx) = (y as isize + off[0], x as isize + off[1]);
            if (0..py as isize).contains(&yy) && (0..px as isize).contains(&xx) {
                o[y * nx + x] = p[yy as usize * px + xx as usize];
            }
        }
    }
    Ok((Object::new(object), r.probe))
}

/// Rows (or columns) spanned by the window centres of the views at `corners`.
fn centres(corners: impl Iterator<Item = usize> + Clone) -> Range<usize> {
    let lo = corners.clone().min().unwrap_or(0) + FRAME / 2;
    lo..corners.max().unwrap_or(0) + FRAME / 2 + 1
}

fn crop(o: &Object, [rows, cols]: &[Range<usize>; 2]) -> Vec<C64> {
    let nx = o.0.dims()[2];
    rows.clone()
        .flat_map(|y| {
            o.0.as_slice()[y * nx + cols.start..y * nx + cols.end]
                .iter()
                .copied()
        })
        .collect()
}

/// The normalised Poisson, Gaussian, Euclidian LLK and photon count.
fn llk_array(l: &Llk) -> [f64; 4] {
    [l.poisson, l.gaussian, l.euclidian, l.nb_photons_calc]
}

fn llk_line(l: &Llk) -> String {
    llk_array(l).map(|v| format!("{v:.17e}")).join(" ")
}
