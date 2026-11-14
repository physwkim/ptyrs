//! The ptyrs GPU side of `refs/ptychi/run_optics_target.py`: Poisson LSQML's epochs and time
//! to 1.01× the true LLK through a multislice object to the far field, and in the near field
//! by angular spectrum.
//!
//! `cargo run --release -p ptyrs --example engine_optics_target -- gen <dir>` simulates both
//! cases with the CPU `f64` forward model ([`SliceWaves`] then [`Optics::exit`]) and writes,
//! per case `ms` and `near`, `<case>.cxi` (Poisson counts, natural layout), `<case>_truth.h5`
//! and `<case>_start.h5` (the slices as the object's modes) and `<case>.txt` (optics, `f64`
//! LLKs, view corners), which pty-chi reads too. `count <dir> <case> <nepoch> <batch>...` runs
//! each batch for `nepoch` epochs with the device's LLK after every one and prints the first
//! epoch at the target. `time <dir> <case> <batch>:<epochs>...` times each run from the start
//! with no LLK pass, after an untimed run that loads the modules and plans, then checks the
//! end state's LLK. Batches are `random_batches(nframe, batch, SEED, epoch)`.

use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use cuda_core::CudaContext;
use ptyrs::base::{
    AddrTable, C64, Geometry, Object, Positions, Probe, Scan, Stack, required_obj_shape,
};
use ptyrs::engine::{
    Cost, EngineError, GpuOptions, GpuPtycho, Llk, LsqmlParams, Observed, Optics, SliceWaves,
    Wavefield, llk, shift,
};

const FRAME: usize = 256;
const NPROBE: usize = 2;
/// A jittered GRID × GRID raster of STEP pixels.
const GRID: usize = 15;
const STEP: f64 = 40.0;
const WAVELENGTH: f64 = 1e-10;
const PIXEL: f64 = 50e-9;
/// Photons the probe sends through a frame.
const PHOTONS: f64 = 1e6;
/// lsqml_multislice.py's gaps; slice `s` is the simulated object, reflected by `s`, to the
/// power `WEIGHTS[s]`.
const SPACINGS: [f64; 2] = [4e-5, 6e-5];
const WEIGHTS: [f64; 3] = [0.5, 0.3, 0.2];
/// lsqml_near.py's angular-spectrum distance.
const NEAR: f64 = 5e-4;
const SEED: u64 = 7;
const TARGET: f64 = 1.01;
/// Epochs of the untimed run.
const WARMUP: usize = 12;
/// Frames per CPU forward chunk.
const CHUNK: usize = 15;

/// `case`'s exit distance (infinite: the plain Fourier far field) and slice gaps.
fn spec(case: &str) -> Result<(f64, &'static [f64]), Box<dyn Error>> {
    match case {
        "ms" => Ok((f64::INFINITY, &SPACINGS)),
        "near" => Ok((NEAR, &[])),
        _ => Err(format!("no case {case}: ms or near").into()),
    }
}

fn optics(case: &str) -> Result<Optics, Box<dyn Error>> {
    let (distance, spacings) = spec(case)?;
    let plane = |d: f64| Wavefield::new([FRAME; 2], WAVELENGTH, [PIXEL; 2], d);
    let gaps = spacings
        .iter()
        .map(|&d| plane(d))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Optics::free_space(&plane(distance)?).with_slices(&gaps)?)
}

/// The integer-pixel corners of `scan`'s views, frame by frame.
fn corners(scan: &Scan, optics: &Optics, obj: [usize; 2]) -> Result<Vec<[u32; 2]>, EngineError> {
    let positions = scan.positions.pynx_preprocess(optics.pixel());
    let addr = AddrTable::new(&positions, optics.pixel(), [FRAME; 2], obj, NPROBE, 1)?;
    Ok(addr
        .views()
        .iter()
        .step_by(NPROBE)
        .map(|v| [v.oy, v.ox])
        .collect())
}

/// The normalised `f64` LLK of `objects` and `probe` against `obs`, CHUNK frames at a time.
fn llk_of(
    optics: &Optics,
    addr: &AddrTable,
    obs: &Observed,
    objects: &[Object],
    probe: &Probe,
) -> Llk {
    let mut sum = Llk::default();
    let per = CHUNK * addr.nmode();
    for (c, views) in addr.views().chunks(per).enumerate() {
        let mut psi = SliceWaves::new(optics, views, objects, probe, None)
            .psi()
            .to_vec();
        optics.exit().forward(&mut psi);
        let first = c * CHUNK;
        let l = llk(
            obs,
            first..first + views.len() / addr.nmode(),
            addr.nmode(),
            &psi,
        );
        sum.poisson += l.poisson;
        sum.gaussian += l.gaussian;
        sum.euclidian += l.euclidian;
        sum.nb_photons_calc += l.nb_photons_calc;
    }
    sum.normalised(obs.nb_obs())
}

fn reflect(o: &Object, rows: bool, cols: bool) -> Vec<C64> {
    let [ny, nx] = o.plane_shape();
    let a = o.as_slice();
    (0..ny * nx)
        .map(|i| {
            let (y, x) = (i / nx, i % nx);
            a[(if rows { ny - 1 - y } else { y }) * nx + if cols { nx - 1 - x } else { x }]
        })
        .collect()
}

/// Packs slices into one object of as many modes, as `write_modes` stores them.
fn pack(slices: &[Object]) -> Result<Object, Box<dyn Error>> {
    let [ny, nx] = slices[0].plane_shape();
    let data = slices
        .iter()
        .flat_map(|o| o.as_slice().iter().copied())
        .collect();
    Ok(Object::new(Stack::from_vec([slices.len(), ny, nx], data)?))
}

fn unpack(o: &Object) -> Result<Vec<Object>, Box<dyn Error>> {
    let [n, ny, nx] = o.dims();
    let plane = ny * nx;
    (0..n)
        .map(|s| {
            Ok(Object::new(Stack::from_vec(
                [1, ny, nx],
                o.as_slice()[s * plane..][..plane].to_vec(),
            )?))
        })
        .collect()
}

fn generate(dir: &Path) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(dir)?;
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut jitter = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 61) as f64 % 5.0 - 2.0
    };
    let (mut y, mut x) = (Vec::new(), Vec::new());
    for r in 0..GRID {
        for c in 0..GRID {
            y.push((r as f64 * STEP + jitter()) * PIXEL);
            x.push((c as f64 * STEP + jitter()) * PIXEL);
        }
    }
    let raw = Positions::new(y, x)?;
    let pre = raw.pynx_preprocess([PIXEL; 2]);
    let obj = required_obj_shape(&pre, [PIXEL; 2], [FRAME; 2], 16);
    let addr = AddrTable::new(&pre, [PIXEL; 2], [FRAME; 2], obj, NPROBE, 1)?;
    let whole = ptyrs::base::sim::object(obj);
    let mut truth_probe = ptyrs::base::sim::probe([FRAME; 2], NPROBE)?;
    truth_probe
        .as_mut_slice()
        .iter_mut()
        .for_each(|v| *v *= PHOTONS.sqrt());
    // The start probe: the truth's under a quadratic phase, as engine_gpu_lsqml_multislice.
    let mut start_probe = truth_probe.clone();
    let sigma = FRAME as f64 / 12.0;
    for m in 0..NPROBE {
        for (i, v) in start_probe.plane_mut(m).iter_mut().enumerate() {
            let u = ((i % FRAME) as f64 - FRAME as f64 / 2.0) / sigma;
            let w = ((i / FRAME) as f64 - FRAME as f64 / 2.0) / sigma;
            *v *= C64::from_polar(1.0, -0.5 * (u * u + w * w));
        }
    }
    let geometry = Geometry::new(
        WAVELENGTH,
        1.0,
        WAVELENGTH / (FRAME as f64 * PIXEL),
        [FRAME; 2],
    )?;
    let (nframe, n) = (addr.nframe(), FRAME * FRAME);
    for (case, seed) in [("ms", 1u64), ("near", 2)] {
        let optics = optics(case)?;
        let truth: Vec<Object> = match optics.nslice() {
            1 => vec![whole.clone()],
            _ => (WEIGHTS.iter().enumerate())
                .map(|(s, &w)| {
                    let data = reflect(&whole, s > 0, s > 1)
                        .into_iter()
                        .map(|c| c.powf(w))
                        .collect();
                    Ok(Object::new(Stack::from_vec([1, obj[0], obj[1]], data)?))
                })
                .collect::<Result<_, Box<dyn Error>>>()?,
        };
        let ones = vec![C64::new(1.0, 0.0); obj[0] * obj[1]];
        let start = vec![Object::new(Stack::from_vec([1, obj[0], obj[1]], ones)?); truth.len()];
        let mut icalc = vec![0.0; nframe * n];
        let per = CHUNK * addr.nmode();
        for (c, views) in addr.views().chunks(per).enumerate() {
            let mut psi = SliceWaves::new(&optics, views, &truth, &truth_probe, None)
                .psi()
                .to_vec();
            optics.exit().forward(&mut psi);
            for (k, plane) in psi.chunks_exact(n).enumerate() {
                let f = c * CHUNK + k / addr.nmode();
                for (j, v) in plane.iter().enumerate() {
                    icalc[f * n + shift(j, FRAME, FRAME)] += v.norm_sqr();
                }
            }
        }
        let counts = ptyrs::base::sim::poisson([nframe, FRAME, FRAME], &icalc, None, seed)?;
        let scan = Scan::new(geometry, raw.clone(), counts)?;
        let obs = Observed::from_counts(&scan.counts);
        let t = llk_of(&optics, &addr, &obs, &truth, &truth_probe);
        let s = llk_of(&optics, &addr, &obs, &start, &start_probe);
        let mut meta = String::new();
        writeln!(meta, "wavelength {WAVELENGTH:e}")?;
        writeln!(meta, "pixel {PIXEL:e}")?;
        let (distance, spacings) = spec(case)?;
        writeln!(meta, "distance {distance:e}")?;
        let spacings: Vec<String> = spacings.iter().map(|d| format!("{d:e}")).collect();
        writeln!(meta, "spacings {}", spacings.join(" "))?;
        writeln!(meta, "scale {:.17e}", optics.exit().intensity_scale())?;
        writeln!(
            meta,
            "truth_llk {:.17e} {:.17e} {:.17e}",
            t.poisson, t.gaussian, t.euclidian
        )?;
        writeln!(
            meta,
            "start_llk {:.17e} {:.17e} {:.17e}",
            s.poisson, s.gaussian, s.euclidian
        )?;
        for (f, [oy, ox]) in corners(&scan, &optics, obj)?.iter().enumerate() {
            writeln!(meta, "corner {f} {oy} {ox}")?;
        }
        ptyrs::io::write_cxi(dir.join(format!("{case}.cxi")), &scan)?;
        ptyrs::io::write_modes(
            dir.join(format!("{case}_truth.h5")),
            &pack(&truth)?,
            &truth_probe,
        )?;
        ptyrs::io::write_modes(
            dir.join(format!("{case}_start.h5")),
            &pack(&start)?,
            &start_probe,
        )?;
        fs::write(dir.join(format!("{case}.txt")), meta)?;
        let max = scan.counts.as_slice().iter().max().copied().unwrap_or(0);
        println!(
            "{case}: {nframe} frames {FRAME}x{FRAME}, object {obj:?} x {} slices, counts max {max}, \
             Poisson LLK truth {:.6} start {:.6}",
            truth.len(),
            t.poisson,
            s.poisson
        );
    }
    Ok(())
}

/// A case read back: the scan, its optics, the start and the `f64` LLKs of `<case>.txt`.
struct Case {
    scan: Scan,
    optics: Optics,
    objects: Vec<Object>,
    probe: Probe,
    truth: f64,
    start: f64,
}

fn load(dir: &Path, case: &str) -> Result<Case, Box<dyn Error>> {
    let scan = ptyrs::io::read_cxi(dir.join(format!("{case}.cxi")))?;
    let (packed, probe) = ptyrs::io::read_modes(dir.join(format!("{case}_start.h5")))?;
    let objects = unpack(&packed)?;
    let optics = optics(case)?;
    let meta = fs::read_to_string(dir.join(format!("{case}.txt")))?;
    let field = |key: &str| -> Result<Vec<f64>, Box<dyn Error>> {
        let line = meta
            .lines()
            .find(|l| l.split_whitespace().next() == Some(key));
        let line = line.ok_or(format!("{case}.txt has no {key}"))?;
        Ok(line
            .split_whitespace()
            .skip(1)
            .map(str::parse)
            .collect::<Result<_, _>>()?)
    };
    let want = meta
        .lines()
        .filter_map(|l| l.strip_prefix("corner "))
        .map(|l| {
            let v: Vec<u32> = l
                .split_whitespace()
                .skip(1)
                .map(str::parse)
                .collect::<Result<_, _>>()?;
            Ok([v[0], v[1]])
        })
        .collect::<Result<Vec<[u32; 2]>, Box<dyn Error>>>()?;
    if corners(&scan, &optics, packed.plane_shape())? != want {
        return Err(format!("{case}.cxi's positions give other corners than {case}.txt").into());
    }
    Ok(Case {
        scan,
        optics,
        objects,
        probe,
        truth: field("truth_llk")?[0],
        start: field("start_llk")?[0],
    })
}

impl Case {
    fn gpu(&self, ctx: &std::sync::Arc<CudaContext>) -> Result<GpuPtycho, EngineError> {
        let options = GpuOptions::default();
        GpuPtycho::with_slices(
            ctx,
            &self.scan,
            &self.optics,
            &self.objects,
            self.probe.clone(),
            options,
        )
    }
}

fn params(batch_size: usize) -> LsqmlParams {
    LsqmlParams {
        batch_size,
        cost: Cost::Poisson,
        seed: SEED,
    }
}

fn load_average() -> String {
    fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next().map(str::to_owned))
        .unwrap_or_default()
}

fn count(dir: &Path, case: &str, nepoch: usize, batches: &[usize]) -> Result<(), Box<dyn Error>> {
    let c = load(dir, case)?;
    let ctx = CudaContext::new(0).map_err(EngineError::from)?;
    for &batch in batches {
        let mut g = c.gpu(&ctx)?;
        let first = g.llk()?.poisson;
        g.lsqml(&params(batch), nepoch, 1)?;
        let curve: Vec<f64> = g.history().iter().map(|r| r.llk.poisson).collect();
        let hit = curve
            .iter()
            .position(|&l| l <= TARGET * c.truth)
            .map(|e| e + 1);
        let mut text = format!("0 {first:.9e}\n");
        for (e, l) in curve.iter().enumerate() {
            writeln!(text, "{} {l:.9e}", e + 1)?;
        }
        fs::write(dir.join(format!("ptyrs_{case}_b{batch}.txt")), text)?;
        println!(
            "count {case} batch {batch}: target epoch {}, start {:.6} (f64 {:.6}), end {:.4}x truth",
            hit.map_or("none".into(), |e| e.to_string()),
            first,
            c.start,
            curve.last().copied().unwrap_or(f64::NAN) / c.truth
        );
    }
    Ok(())
}

fn time(dir: &Path, case: &str, runs: &[(usize, usize)]) -> Result<(), Box<dyn Error>> {
    let c = load(dir, case)?;
    let ctx = CudaContext::new(0).map_err(EngineError::from)?;
    for &(batch, nepoch) in runs {
        let mut warm = c.gpu(&ctx)?;
        warm.lsqml(&params(batch), WARMUP, 0)?;
        warm.synchronize()?;
        drop(warm);
        let load = load_average();
        let mut g = c.gpu(&ctx)?;
        let t = Instant::now();
        g.lsqml(&params(batch), nepoch, 0)?;
        g.synchronize()?;
        let s = t.elapsed().as_secs_f64();
        let end = g.llk()?.poisson / c.truth;
        println!(
            "time {case} batch {batch}: {nepoch} epochs {s:.3} s, {:.3} ms/epoch, end {end:.4}x \
             truth{}, load {load}",
            1e3 * s / nepoch as f64,
            if end <= TARGET {
                ""
            } else {
                " (above the target)"
            }
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let usage = "usage: optics_target gen <dir> | count <dir> <ms|near> <nepoch> <batch>... | \
                 time <dir> <ms|near> <batch>:<epochs>...";
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = PathBuf::from(args.get(1).ok_or(usage)?);
    match (args[0].as_str(), &args[2..]) {
        ("gen", []) => generate(&dir),
        ("count", [case, nepoch, batches @ ..]) if !batches.is_empty() => {
            let batches = batches
                .iter()
                .map(|b| b.parse())
                .collect::<Result<Vec<usize>, _>>()?;
            count(&dir, case, nepoch.parse()?, &batches)
        }
        ("time", [case, runs @ ..]) if !runs.is_empty() => {
            let runs = runs
                .iter()
                .map(|r| {
                    let (b, e) = r.split_once(':').ok_or(usage)?;
                    Ok((b.parse()?, e.parse()?))
                })
                .collect::<Result<Vec<(usize, usize)>, Box<dyn Error>>>()?;
            time(&dir, case, &runs)
        }
        _ => Err(usage.into()),
    }
}
