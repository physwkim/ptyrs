//! Strong scaling of [`DistAp`] (DESIGN §7 M4): AP time per cycle, object and probe updated,
//! on every visible GPU in NUMA order. 4096 frames of 256x256, 2 probe modes, 80 px steps
//! (as `dist_live_feed`), stacks of 32 frames, fused passes.
//!
//! `dist_ap_scaling prepare <dir> <fermat|raster>` simulates the counts once on the CPU into
//! `<dir>`, so timed runs hold the GPUs only for the reconstruction.
//! `dist_ap_scaling run <dir> <fermat|raster> <sectors|strips|grid> <ncycle> <repeats>` uploads,
//! warms up 2 cycles, then times `repeats` runs of `ncycle` cycles; it ends with a hash of
//! the object and probe bits, equal on 1, 2 and 4 GPUs.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ptyrs::base::{AddrTable, C64, Geometry, Positions, required_obj_shape, sim};
use ptyrs::dist::{ApConfig, ApProblem, DistAp, Scheme, topo};
use ptyrs::gpu::C32;

const FRAME: usize = 256;
const NMODE: usize = 2;
const STEP_PX: f64 = 80.0;
const PHOTONS: f64 = 1e5;
const STACK: usize = 32;
const WARMUP: usize = 2;

fn geometry() -> Geometry {
    Geometry::new(1.5e-10, 1.0, 75e-6, [FRAME, FRAME]).unwrap()
}

fn positions(pattern: &str) -> Result<Positions, Box<dyn Error>> {
    let step = STEP_PX * geometry().pixel_size_object()[1];
    Ok(match pattern {
        "fermat" => sim::fermat(4096, step),
        "raster" => sim::raster(64, 64, step),
        _ => return Err(format!("unknown pattern {pattern}").into()),
    })
}

fn obj_shape(pos: &Positions) -> [usize; 2] {
    let g = geometry();
    let pixel = g.pixel_size_object();
    required_obj_shape(&pos.pynx_preprocess(pixel), pixel, g.frame, 4)
}

fn counts_path(dir: &Path, pattern: &str) -> PathBuf {
    dir.join(format!("{pattern}_4096_{FRAME}_{NMODE}.u16"))
}

fn prepare(dir: &Path, pattern: &str) -> Result<(), Box<dyn Error>> {
    let t = Instant::now();
    let pos = positions(pattern)?;
    let object = sim::object(obj_shape(&pos));
    let probe = sim::probe([FRAME, FRAME], NMODE)?;
    let scan = ptyrs::engine::sim::simulate(geometry(), pos, &object, &probe, PHOTONS, None, 7)?;
    let bytes: Vec<u8> = scan
        .counts
        .as_slice()
        .iter()
        .flat_map(|c| c.to_le_bytes())
        .collect();
    std::fs::create_dir_all(dir)?;
    std::fs::write(counts_path(dir, pattern), bytes)?;
    println!(
        "{pattern}: object {:?}, simulated in {:.1} s",
        object.plane_shape(),
        t.elapsed().as_secs_f64()
    );
    Ok(())
}

/// FNV-1a over the bits.
fn hash(v: &[C32]) -> u64 {
    v.iter()
        .flat_map(|c| [c.re.to_bits(), c.im.to_bits()])
        .flat_map(u32::to_le_bytes)
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        })
}

fn run(dir: &Path, pattern: &str, args: &[String]) -> Result<(), Box<dyn Error>> {
    let [scheme, ncycle, repeats] = args else {
        return Err("run needs <scheme> <ncycle> <repeats>".into());
    };
    let scheme = match scheme.as_str() {
        "sectors" => Scheme::Sectors { centre: None },
        "strips" => Scheme::Strips,
        "grid" => Scheme::Grid,
        s => return Err(format!("unknown scheme {s}").into()),
    };
    let (ncycle, repeats): (usize, usize) = (ncycle.parse()?, repeats.parse()?);
    let g = geometry();
    let pixel = g.pixel_size_object();
    let pos = positions(pattern)?;
    let shape = obj_shape(&pos);
    let addr = AddrTable::new(&pos.pynx_preprocess(pixel), pixel, g.frame, shape, NMODE, 1)?;
    let counts: Vec<u16> = std::fs::read(counts_path(dir, pattern))?
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect();
    // Flat object; truth probe scaled so the calculated photons match the observed.
    let probe = sim::probe(g.frame, NMODE)?;
    let p2: f64 = probe.as_slice().iter().map(C64::norm_sqr).sum();
    let total: f64 = counts.iter().map(|&c| f64::from(c)).sum();
    let s = (total / (addr.nframe() as f64 * p2)).sqrt();
    let c32 = |c: &C64| C32 {
        re: (c.re * s) as f32,
        im: (c.im * s) as f32,
    };
    let problem = ApProblem {
        object: vec![C32 { re: 1.0, im: 0.0 }; shape[0] * shape[1]],
        probe: probe.as_slice().iter().map(c32).collect(),
        counts,
        quad_phase: (std::f64::consts::PI / (g.wavelength * g.distance) * pixel[1] * pixel[1])
            as f32,
        addr,
    };
    let devs = topo::devices()?;
    let ordinals = topo::numa_grouped(&devs);
    let cfg = ApConfig {
        scheme,
        stack: STACK,
        obj_inertia: Some(0.01),
        probe_inertia: Some(0.001),
        calc_llk: 0,
    };
    let t = Instant::now();
    let mut ap = DistAp::new(&ordinals, problem, cfg)?;
    let setup = t.elapsed().as_secs_f64();
    ap.run(WARMUP)?;
    let mut ms: Vec<f64> = (0..repeats)
        .map(|_| {
            let t = Instant::now();
            ap.run(ncycle)?;
            Ok(t.elapsed().as_secs_f64() * 1e3 / ncycle as f64)
        })
        .collect::<Result<_, Box<dyn Error>>>()?;
    let runs = ms.iter().map(|m| format!("{m:.2}")).collect::<Vec<_>>();
    ms.sort_by(f64::total_cmp);
    println!(
        "{pattern} {scheme:?} gpus {} object {shape:?}: setup {setup:.2} s, ms/cycle min {:.2} median {:.2} (runs {})",
        ap.ngpu(),
        ms[0],
        ms[ms.len() / 2],
        runs.join(" "),
    );
    for (r, (l, o)) in ap.loads().iter().zip(&ordinals).enumerate() {
        println!(
            "  region {r} on {} ({}): computes {} frames ({} interior), recomputes {}, pulls {} halo tiles",
            devs[*o].pci_bus_id, devs[*o].name, l.computed, l.interior, l.recomputed, l.halo_tiles
        );
    }
    println!(
        "  after {} cycles: object hash {:016x}, probe hash {:016x}",
        WARMUP + repeats * ncycle,
        hash(&ap.object()?),
        hash(&ap.probe()?)
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [mode, dir, pattern, rest @ ..] = args.as_slice() else {
        return Err("usage: ap_scaling prepare|run <dir> <fermat|raster> [...]".into());
    };
    let dir = Path::new(dir);
    match mode.as_str() {
        "prepare" => prepare(dir, pattern),
        "run" => run(dir, pattern, rest),
        m => Err(format!("unknown mode {m}").into()),
    }
}
