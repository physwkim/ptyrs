//! New frames per GPU during a live scan (DESIGN §7 M4): Strips over a raster and Sectors over
//! a Fermat spiral, 2 and 4 regions, replayed in arrival order. Prints the spread (most minus
//! fewest new frames over the regions) per window of N consecutive arrivals and writes the
//! spread over time to `<out>/<scan>_<regions>.csv`. CPU only.
//!
//! `cargo run --release -p ptyrs-dist --example live_feed <out>`

use std::fmt::Write as _;
use std::path::PathBuf;

use ptyrs_core::{AddrTable, required_obj_shape, sim};
use ptyrs_dist::{DEFAULT_TILE, Partition, Scheme, feed};

const PIXEL: [f64; 2] = [1e-8, 1e-8];
const FRAME: usize = 256;
/// 80 px between neighbouring positions: about 69 % linear overlap of 256-px frames.
const STEP: f64 = 8e-7;
const RATE_HZ: f64 = 3000.0;
const RASTER: [usize; 2] = [100, 100];
const WINDOWS: [usize; 5] = [16, 64, 100, 256, 1024];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("usage: live_feed <out dir>")?,
    );
    std::fs::create_dir_all(&out)?;
    let scans = [
        (
            "raster",
            sim::raster(RASTER[0], RASTER[1], STEP),
            Scheme::Strips,
        ),
        (
            "fermat",
            sim::fermat(RASTER[0] * RASTER[1], STEP),
            Scheme::Sectors { centre: None },
        ),
    ];
    println!(
        "Frames arrive in index order at {RATE_HZ} Hz; {FRAME}^2 frames, step {} px, tile {DEFAULT_TILE} px; \
         raster {}x{} (row = {} arrivals), Fermat {} points.",
        STEP / PIXEL[1],
        RASTER[0],
        RASTER[1],
        RASTER[1],
        RASTER[0] * RASTER[1],
    );
    println!(
        "spread = most - fewest new frames over the regions in one window; every window start."
    );
    println!(
        "{:<7} {:>4} {:>5} {:>7} {:>6} {:>8} {:>5} {:>5} {:>8}",
        "scan", "gpus", "N", "ms", "share", "mean", "p95", "max", "idle %"
    );
    for (name, pos, scheme) in scans {
        let pos = pos.pynx_preprocess(PIXEL);
        let obj = required_obj_shape(&pos, PIXEL, [FRAME, FRAME], 0);
        let addr = AddrTable::new(&pos, PIXEL, [FRAME, FRAME], obj, 1, 1)?;
        for nregion in [2, 4] {
            let part = Partition::from_addr(&addr, DEFAULT_TILE, scheme, nregion)?;
            let series: Vec<Vec<usize>> = WINDOWS
                .iter()
                .map(|&len| {
                    let w = feed::windows(&part, len);
                    let spreads: Vec<usize> = w.iter().map(|c| feed::spread(c)).collect();
                    let idle = w.iter().filter(|c| c.contains(&0)).count();
                    let ordered = {
                        let mut s = spreads.clone();
                        s.sort_unstable();
                        s
                    };
                    println!(
                        "{:<7} {:>4} {:>5} {:>7.1} {:>6.1} {:>8.2} {:>5} {:>5} {:>8.1}",
                        name,
                        nregion,
                        len,
                        len as f64 / RATE_HZ * 1e3,
                        len as f64 / nregion as f64,
                        spreads.iter().sum::<usize>() as f64 / spreads.len() as f64,
                        ordered[(ordered.len() - 1) * 95 / 100],
                        ordered[ordered.len() - 1],
                        idle as f64 * 100.0 / w.len() as f64,
                    );
                    spreads
                })
                .collect();
            let mut csv = String::from("start");
            for len in WINDOWS {
                write!(csv, ",spread_{len}")?;
            }
            csv.push('\n');
            for start in 0..part.nframe() {
                write!(csv, "{start}")?;
                for s in &series {
                    match s.get(start) {
                        Some(v) => write!(csv, ",{v}")?,
                        None => csv.push(','),
                    }
                }
                csv.push('\n');
            }
            std::fs::write(out.join(format!("{name}_{nregion}.csv")), csv)?;
        }
    }
    Ok(())
}
