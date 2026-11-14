//! A live AP scan on every visible GPU in NUMA order (DESIGN §7 M4): [`DistLiveAp`] over the
//! M5 simulator's frames, 4096 frames of 256x256 at 32 px steps, two probe modes, the whole
//! trajectory planned before the first frame. Frames go `batch` at a time through the ring
//! and the uploader into the stores of the GPUs that iterate them, each batch followed by `epochs` epochs over every frame landed so far; prints per batch the frames
//! new to each GPU, the halo rows each lists, the ms to upload and land the batch and the ms
//! per epoch,
//! and per run the graph recaptures after setup, the frames lost and a hash of the object
//! bits, equal on 1, 2 and 4 GPUs.
//!
//! `dist_live_ap prepare <dir> <raster|fermat>` simulates the counts once on the CPU into
//! `<dir>`, so timed runs hold the GPUs only for the reconstruction.
//! `dist_live_ap run <dir> <raster|fermat> <batch> <epochs>` lands a raster in strips, a
//! Fermat spiral in sectors.

#![warn(clippy::disallowed_methods)]

use std::error::Error;
use std::num::{NonZeroU32, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use cuda_core::Context;

use ptyrs::dist::{DistLiveAp, Scheme, topo};
use ptyrs::gpu::{C32, ViewAddr};
use ptyrs::live::farfield::{Ptycho, PtychoSpec, ScanPath};
use ptyrs::live::graph::WarmUp;
use ptyrs::live::pos::{Pos, PosTable};
use ptyrs::live::recon::{ApConfig, ViewPlacement};
use ptyrs::live::ring::{Ring, UploaderConfig, run_region_uploader};
use ptyrs::live::scan::ScanShape;

const FRAME: usize = 256;
const POINTS: usize = 4096;
const STEP_PX: f64 = 32.0;

fn sim(pattern: &str) -> Result<(Ptycho, Scheme), Box<dyn Error>> {
    let (path, scheme) = match pattern {
        "raster" => (ScanPath::Raster { cols: 64 }, Scheme::Strips),
        "fermat" => (ScanPath::Fermat, Scheme::Sectors { centre: None }),
        _ => return Err(format!("unknown pattern {pattern}").into()),
    };
    let p = Ptycho::new(PtychoSpec {
        n: FRAME,
        path,
        points: POINTS,
        step: STEP_PX,
        photons: 1e6,
        seed: 5,
        wavelength: 1.5498e-10,
        distance: 5.0,
        det_pixel: 75e-6,
    });
    Ok((p, scheme))
}

fn counts_path(dir: &Path, pattern: &str) -> PathBuf {
    dir.join(format!("live_{pattern}_{POINTS}_{FRAME}.u16"))
}

fn prepare(dir: &Path, pattern: &str) -> Result<(), Box<dyn Error>> {
    let t = Instant::now();
    let (p, _) = sim(pattern)?;
    let threads = std::thread::available_parallelism()?.get();
    let per = POINTS.div_ceil(threads);
    let parts: Vec<Vec<u8>> = std::thread::scope(|s| {
        let p = &p;
        let jobs: Vec<_> = (0..POINTS)
            .step_by(per)
            .map(|at| {
                s.spawn(move || {
                    (at..(at + per).min(POINTS))
                        .flat_map(|k| p.frame(k))
                        .flat_map(u16::to_le_bytes)
                        .collect()
                })
            })
            .collect();
        jobs.into_iter().map(|j| j.join().unwrap()).collect()
    });
    std::fs::create_dir_all(dir)?;
    std::fs::write(counts_path(dir, pattern), parts.concat())?;
    println!(
        "{pattern}: {}, simulated in {:.1} s",
        p.describe(),
        t.elapsed().as_secs_f64()
    );
    Ok(())
}

/// FNV-1a over the bits.
fn hash(v: &[[f32; 2]]) -> u64 {
    v.iter()
        .flat_map(|c| c.map(f32::to_bits))
        .flat_map(u32::to_le_bytes)
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        })
}

fn run(dir: &Path, pattern: &str, args: &[String]) -> Result<(), Box<dyn Error>> {
    let [batch, epochs] = args else {
        return Err("run needs <batch> <epochs>".into());
    };
    let (batch, epochs): (usize, usize) = (batch.parse()?, epochs.parse()?);
    let (p, scheme) = sim(pattern)?;
    let pos: Vec<(f64, f64)> = (0..POINTS).map(|k| p.position(k)).collect();
    let dx = p.spec().dx();
    let (place, reach) = ViewPlacement::plan([dx, dx], &pos);
    let mut views: Vec<ViewAddr> = Vec::new();
    for &q in &pos {
        place.views(q, 2, 1, &mut views);
    }
    let shape = ScanShape {
        nframe: POINTS as u32,
        nmode: 2,
        nprobe: 2,
        nobj: 1,
        frame: [FRAME as u32; 2],
        reach,
        quad_phase: 0.0,
    };
    let cfg = ApConfig {
        shape,
        nf_max: NonZeroU32::new(32).unwrap(),
        cap: shape.tile_cap(&views),
        fill: [1.0, 0.0],
        chunk: 2 << 20,
        inertia: 0.01,
        stacks: NonZeroU32::new(16).unwrap(),
    };
    let counts: Vec<u16> = std::fs::read(counts_path(dir, pattern))?
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect();
    let probe: Vec<C32> = p
        .probe()
        .iter()
        .map(|c| C32 { re: c.re, im: c.im })
        .collect();
    let devs = topo::devices()?;
    let ordinals = topo::numa_grouped(&devs);
    let t = Instant::now();
    let mut ap = DistLiveAp::new(&ordinals, &cfg, &views, scheme, &probe, false)?;
    let setup = t.elapsed().as_secs_f64();
    let warm = WarmUp::end();
    let ngpu = ap.ngpu();
    let obj = shape.dims().obj;
    println!(
        "{pattern} {scheme:?} gpus {ngpu} object {obj:?}: {POINTS} frames of {FRAME}^2, \
         {STEP_PX} px steps, batches of {batch}, {epochs} epochs each; setup {setup:.2} s"
    );
    for (r, o) in ordinals.iter().enumerate() {
        println!(
            "  region {r} on {} ({})",
            devs[*o].pci_bus_id, devs[*o].name
        );
    }
    println!(
        "{:>5} {:>6} {:>23} {:>31} {:>8} {:>9}",
        "batch",
        "landed",
        "new frames per GPU",
        "halo rows unpacked per GPU",
        "land ms",
        "epoch ms"
    );
    let n = FRAME * FRAME;
    let ctx = Context::new(ordinals[0])?;
    let ring = Arc::new(Ring::new(&ctx, 2 * batch.max(64), n)?);
    let table = Arc::new(PosTable::new(1 << 16));
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, landed) = crossbeam_channel::unbounded();
    let uploader = {
        let (ring, table, stop) = (ring.clone(), table.clone(), stop.clone());
        let (stores, route) = (ap.stores(), ap.route(place));
        let up = UploaderConfig {
            batch: NonZeroUsize::new(64).unwrap(),
            flush_ns: 100_000,
            idle: Duration::from_micros(20),
            events: 4,
            pos_wait_ns: 1_000_000_000,
        };
        std::thread::spawn(move || {
            run_region_uploader(ring, stores, route, table, up, stop, Some(tx))
        })
    };
    let (mut at, mut epoch_ms) = (0, Vec::new());
    for k in 0.. {
        if at == POINTS {
            break;
        }
        let end = (at + batch).min(POINTS);
        let before = ap.landed().to_vec();
        let t = Instant::now();
        for k in at..end {
            let (x, y) = pos[k];
            table.insert(k as i32, Pos::new(x, y, 0));
            let mut s = ring.reserve();
            s.frame().copy_from_slice(&counts[k * n..(k + 1) * n]);
            s.commit(k as i32, 0, 0);
        }
        let mut up = at;
        while up < end {
            up += landed.recv()?.n as usize;
        }
        ap.push(&views[2 * at..2 * end])?;
        ap.synchronize()?;
        let land = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        for _ in 0..epochs {
            ap.epoch()?;
        }
        ap.synchronize()?;
        let ms = t.elapsed().as_secs_f64() * 1e3 / epochs as f64;
        epoch_ms.push(ms);
        let new: Vec<String> = ap
            .landed()
            .iter()
            .zip(&before)
            .map(|(l, b)| (l - b).to_string())
            .collect();
        let rows: Vec<String> = ap.halo_rows().iter().map(|r| r[1].to_string()).collect();
        println!(
            "{k:>5} {end:>6} {:>23} {:>31} {land:>8.2} {ms:>9.2}",
            new.join("/"),
            rows.join("/"),
        );
        at = end;
    }
    stop.store(true, Ordering::Release);
    let up = uploader.join().expect("the uploader")?;
    let stored: Vec<String> = up.regions.iter().map(|r| r.frames.to_string()).collect();
    let landed: Vec<String> = ap.landed().iter().map(u32::to_string).collect();
    println!(
        "  frames stored per GPU {}, uploader dropped {}",
        stored.join("/"),
        up.pos_missing
    );
    println!(
        "  frames per GPU {}; recaptures {}, lost {}, closed {:?}; last epoch {:.2} ms; \
         after {} epochs: object hash {:016x}",
        landed.join("/"),
        warm.recaptures(),
        ap.lost(),
        ap.closed(),
        epoch_ms.last().copied().unwrap_or(f64::NAN),
        ap.epochs(),
        hash(&ap.object()?),
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [mode, dir, pattern, rest @ ..] = args.as_slice() else {
        return Err("usage: dist_live_ap prepare|run <dir> <raster|fermat> [...]".into());
    };
    let dir = Path::new(dir);
    match mode.as_str() {
        "prepare" => prepare(dir, pattern),
        "run" => run(dir, pattern, rest),
        m => Err(format!("unknown mode {m}").into()),
    }
}
