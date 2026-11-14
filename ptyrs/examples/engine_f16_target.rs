//! Cycles to a target LLK for DM or RAAR with f32 and f16 carried waves
//! ([`GpuOptions::psi_f16`]), from `scan.cxi` and `init.h5` (`engine_carried_time`'s): the
//! convergence behind DESIGN §8's f16 time to target. ptyrs is bit-deterministic, so the
//! spread comes from perturbed starts, as in the PyNX f16 study.
//!
//! `cargo run --release -p ptyrs --example engine_f16_target -- run <dir> <dm|raar> <ncycle>
//! <first>..<end> <out_dir>` runs `ncycle` cycles in one call on CUDA device 0 from each start
//! of the range: 0 is `init.h5`'s, s > 0 has every object value x made x (1 + KICK (u + i v)),
//! u and v uniform in [-1, 1) from seed s (the first exit waves perturbed once as f16 rounds
//! them). Each start runs with f32 and with f16 carried waves, and each run's Poisson LLK per
//! cycle goes to `<out_dir>/<algo>_<f32|f16>_<start>.txt`.
//!
//! `... -- target <out_dir> <dm|raar>` reads them back. The target is 1.01 times the lower of
//! start 0's two mean LLKs over the last 50 cycles; a run's cycles to target is its first
//! crossing, linear between cycles. Prints each arm's median and range of the cycles and of the
//! last-50 mean LLKs, and for each the per-start f16 - f32 differences with their two-sided
//! exact Wilcoxon signed-rank p (both arms run every start) and the unpaired two-sided exact
//! Mann-Whitney p (the PyNX study's test). DM's LLK is lowest early and then rises, so its runs
//! cross in the first descent and only the last-50 means tell its arms apart.

use std::error::Error;
use std::fs;
use std::path::Path;

use cuda_core::CudaContext;
use ptyrs::base::{C64, Object, Stack};
use ptyrs::engine::{DmParams, EngineError, GpuOptions, GpuPtycho, RaarParams};
use ptyrs::io::{read_cxi, read_modes};

/// The relative RMS error of one f16 rounding of a complex value, 2.06e-4, is √(2/3) of this.
const KICK: f64 = 2.52e-4;
const TAIL: usize = 50;
const MARGIN: f64 = 1.01;
const ARMS: [(&str, bool); 2] = [("f32", false), ("f16", true)];

fn kicked(object: &Object, seed: u64) -> Object {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        2.0 * ((state >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    };
    let data = object
        .as_slice()
        .iter()
        .map(|&x| x * (1.0 + KICK * C64::new(uniform(), uniform())))
        .collect();
    Object::new(Stack::from_vec(object.dims(), data).expect("the object's own dims"))
}

fn run(a: &[String]) -> Result<(), Box<dyn Error>> {
    let usage = "run <dir> <dm|raar> <ncycle> <first>..<end> <out_dir>";
    let [dir, algo, ncycle, starts, out] = a else {
        return Err(usage.into());
    };
    let ncycle: usize = ncycle.parse()?;
    let (first, end) = starts.split_once("..").ok_or(usage)?;
    let (first, end): (u64, u64) = (first.parse()?, end.parse()?);
    let dir = Path::new(dir);
    let scan = read_cxi(dir.join("scan.cxi"))?;
    let (object, probe) = read_modes(dir.join("init.h5"))?;
    let ctx = CudaContext::new(0).map_err(EngineError::from)?;
    fs::create_dir_all(out)?;
    for start in first..end {
        let o = match start {
            0 => object.clone(),
            s => kicked(&object, s),
        };
        for (arm, psi_f16) in ARMS {
            let options = GpuOptions {
                psi_f16,
                ..GpuOptions::default()
            };
            let mut g = GpuPtycho::new(&ctx, &scan, o.clone(), probe.clone(), options)?;
            match algo.as_str() {
                "dm" => g.dm(
                    &DmParams {
                        calc_llk: 1,
                        ..DmParams::default()
                    },
                    ncycle,
                ),
                "raar" => g.raar(
                    &RaarParams {
                        calc_llk: 1,
                        ..RaarParams::default()
                    },
                    ncycle,
                ),
                _ => return Err(format!("unknown algorithm {algo}").into()),
            }?;
            let llk: Vec<String> = g
                .history()
                .iter()
                .map(|r| format!("{:e}", r.llk.poisson))
                .collect();
            fs::write(
                Path::new(out).join(format!("{algo}_{arm}_{start}.txt")),
                llk.join("\n"),
            )?;
            println!(
                "{algo} {arm} start {start}: final LLK {}",
                llk[llk.len() - 1]
            );
        }
    }
    Ok(())
}

/// The first cycle (1-based, linear between cycles) at which `llk` reaches `target`.
fn crossing(llk: &[f64], target: f64) -> Option<f64> {
    let j = llk.iter().position(|&v| v <= target)?;
    Some(match j {
        0 => 1.0,
        j => j as f64 + (llk[j - 1] - target) / (llk[j - 1] - llk[j]),
    })
}

/// The two-sided exact Mann-Whitney p of `b` against `a`, and the number of tied pairs (the
/// exact distribution assumes none).
fn mann_whitney(a: &[f64], b: &[f64]) -> (f64, usize) {
    let (m, n) = (a.len(), b.len());
    let mut u = 0.0;
    let mut ties = 0;
    for x in a {
        for y in b {
            if x > y {
                u += 1.0;
            } else if x == y {
                u += 0.5;
                ties += 1;
            }
        }
    }
    // f[i][j][k]: orderings of i values of a and j of b with k pairs where a's is the larger
    let mut f = vec![vec![vec![1.0]; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            f[i][j] = (0..=i * j)
                .map(|k| {
                    // the largest value is a's (it beats all j of b) or b's
                    let top_a = k.checked_sub(j).and_then(|k| f[i - 1][j].get(k));
                    top_a.unwrap_or(&0.0) + f[i][j - 1].get(k).unwrap_or(&0.0)
                })
                .collect();
        }
    }
    let all: f64 = f[m][n].iter().sum();
    let (le, ge): (f64, f64) = f[m][n]
        .iter()
        .enumerate()
        .fold((0.0, 0.0), |(le, ge), (k, c)| {
            let k = k as f64;
            (
                le + if k <= u { *c } else { 0.0 },
                ge + if k >= u { *c } else { 0.0 },
            )
        });
    ((2.0 * le.min(ge) / all).min(1.0), ties)
}

fn target(a: &[String]) -> Result<(), Box<dyn Error>> {
    let [out, algo] = a else {
        return Err("target <out_dir> <dm|raar>".into());
    };
    let read = |arm: &str, start: usize| -> Result<Option<Vec<f64>>, Box<dyn Error>> {
        let path = Path::new(out).join(format!("{algo}_{arm}_{start}.txt"));
        if !path.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(path)?;
        Ok(Some(
            text.lines().map(str::parse).collect::<Result<_, _>>()?,
        ))
    };
    let mut runs: [Vec<Vec<f64>>; 2] = Default::default();
    for (i, (arm, _)) in ARMS.iter().enumerate() {
        while let Some(llk) = read(arm, runs[i].len())? {
            runs[i].push(llk);
        }
    }
    let tail = |llk: &[f64]| llk[llk.len() - TAIL..].iter().sum::<f64>() / TAIL as f64;
    let target = MARGIN * tail(&runs[0][0]).min(tail(&runs[1][0]));
    println!(
        "{algo}: target {target:.6e} = {MARGIN} x the lower start-0 mean of the last {TAIL} \
         of {} cycles",
        runs[0][0].len()
    );
    let mut cycles: [Vec<f64>; 2] = Default::default();
    let mut tails: [Vec<f64>; 2] = Default::default();
    for (i, (arm, _)) in ARMS.iter().enumerate() {
        for (start, llk) in runs[i].iter().enumerate() {
            let c = crossing(llk, target);
            println!(
                "  {arm} start {start}: cycles {}, last-{TAIL} mean {:.6e}",
                c.map_or("never".into(), |c| format!("{c:.4}")),
                tail(llk)
            );
            cycles[i].push(c.unwrap_or(f64::INFINITY));
            tails[i].push(tail(llk));
        }
        let ([median, lo, hi], [t_median, t_lo, t_hi]) = (spread(&cycles[i]), spread(&tails[i]));
        println!(
            "{arm}: {} runs, cycles to target median {median:.2}, range {lo:.2}..{hi:.2}; \
             last-{TAIL} mean median {t_median:.6e}, range {t_lo:.6e}..{t_hi:.6e}",
            cycles[i].len()
        );
    }
    for (what, v) in [("cycles", &cycles), ("last-50 means", &tails)] {
        let d: Vec<f64> = v[1].iter().zip(&v[0]).map(|(b, a)| b - a).collect();
        let [median, lo, hi] = spread(&d);
        let (p, ties) = wilcoxon(&d);
        let (u_p, u_ties) = mann_whitney(&v[0], &v[1]);
        println!(
            "f16 - f32 {what} per start: median {median:.4e}, range {lo:.4e}..{hi:.4e}; paired \
             exact Wilcoxon p {p:.3e} ({ties} tied |d|), unpaired exact Mann-Whitney p \
             {u_p:.3e} ({u_ties} tied pairs)"
        );
    }
    Ok(())
}

/// The two-sided exact Wilcoxon signed-rank p of the paired differences `d`, zeros dropped, and
/// the number of tied |d| (the exact distribution assumes none).
fn wilcoxon(d: &[f64]) -> (f64, usize) {
    let mut d: Vec<f64> = d.iter().copied().filter(|&x| x != 0.0).collect();
    d.sort_by(|x, y| x.abs().total_cmp(&y.abs()));
    let n = d.len();
    let ties = d.windows(2).filter(|w| w[0].abs() == w[1].abs()).count();
    let w: usize = (1..=n).filter(|&r| d[r - 1] > 0.0).sum();
    // f[k]: subsets of the ranks 1..=n summing to k
    let top = n * (n + 1) / 2;
    let mut f = vec![0.0; top + 1];
    f[0] = 1.0;
    for r in 1..=n {
        for k in (r..=top).rev() {
            f[k] += f[k - r];
        }
    }
    let all = 2f64.powi(n as i32);
    let (le, ge): (f64, f64) = (f[..=w].iter().sum(), f[w..].iter().sum());
    ((2.0 * le.min(ge) / all).min(1.0), ties)
}

/// The median, least and greatest of `v`.
fn spread(v: &[f64]) -> [f64; 3] {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    let median = match n % 2 {
        1 => s[n / 2],
        _ => (s[n / 2 - 1] + s[n / 2]) / 2.0,
    };
    [median, s[0], s[n - 1]]
}

fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a.first().map(String::as_str) {
        Some("run") => run(&a[1..]),
        Some("target") => target(&a[1..]),
        _ => Err("usage: engine_f16_target run|target ...".into()),
    }
}
