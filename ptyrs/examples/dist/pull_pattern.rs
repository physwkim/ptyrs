//! Concurrent peer traffic in the AP gather's patterns: every flow moves `MiB` from `src` to
//! `dst` as 1 MiB runs, all flows at once. Methods: SM pull (`copy_runs` on `dst`, the gather
//! path), copy-engine push (`cuMemcpyPeerAsync` on `src`'s stream), copy-engine pull (on
//! `dst`'s stream). Reports the wall time of the slowest flow as GB/s per flow.
//!
//! `pull_pattern [MiB] [reps]`, with all four GPUs visible in ordinal order.

use std::sync::Arc;
use std::time::Instant;

use cuda_core::{Context, DeviceBuffer, IntoResult, Stream, sys};
use ptyrs::dist::exchange::{Run, Runs};
use ptyrs::dist::{Exchange, topo};

const RUN_WORDS: usize = (1 << 20) / 8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).map(|a| a.parse::<usize>());
    let mib = args.next().transpose()?.unwrap_or(128);
    let reps = args.next().transpose()?.unwrap_or(5);
    let ctxs: Vec<Arc<Context>> = (0..4).map(Context::new).collect::<Result<_, _>>()?;
    topo::enable_all_pairs(&ctxs)?;
    let streams: Vec<Arc<Stream>> = ctxs
        .iter()
        .map(|c| c.new_stream())
        .collect::<Result<_, _>>()?;
    let ex: Vec<Exchange> = ctxs.iter().map(Exchange::load).collect::<Result<_, _>>()?;
    let words = mib * RUN_WORDS;
    let src: Vec<DeviceBuffer<u64>> = streams
        .iter()
        .map(|s| DeviceBuffer::zeroed(s, words))
        .collect::<Result<_, _>>()?;
    // One slot per source on every destination.
    let dst: Vec<DeviceBuffer<u64>> = streams
        .iter()
        .map(|s| DeviceBuffer::zeroed(s, 4 * words))
        .collect::<Result<_, _>>()?;
    let all: Vec<(usize, usize)> = (0..4)
        .flat_map(|d| (0..4).filter(move |&s| s != d).map(move |s| (d, s)))
        .collect();
    let patterns: Vec<(&str, Vec<(usize, usize)>)> = vec![
        ("0<2", vec![(0, 2)]),
        ("2<3", vec![(2, 3)]),
        ("0<1", vec![(0, 1)]),
        ("0<2 2<0", vec![(0, 2), (2, 0)]),
        ("2<3 3<2", vec![(2, 3), (3, 2)]),
        ("same socket", vec![(0, 1), (1, 0), (2, 3), (3, 2)]),
        ("cross", vec![(0, 2), (2, 0), (1, 3), (3, 1)]),
        (
            "ring 0-1-2-3-0",
            vec![
                (0, 1),
                (0, 3),
                (1, 0),
                (1, 2),
                (2, 1),
                (2, 3),
                (3, 2),
                (3, 0),
            ],
        ),
        ("all to all", all),
    ];
    let wslot = |d: usize, s: usize| dst[d].cu_deviceptr() + (s * words * 8) as u64;
    println!(
        "{mib} MiB per flow, best of {reps}; GB/s per flow (slowest flow); measured on shared GPUs"
    );
    println!(
        "{:>16} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "pattern", "flows", "SM pull", "CE push", "CE pull", "SM push", "SM pull/1"
    );
    for (name, flows) in &patterns {
        let runs: Vec<Runs> = (0..4)
            .map(|d| {
                let r: Vec<Run> = flows
                    .iter()
                    .filter(|f| f.0 == d)
                    .flat_map(|&(d, s)| (0..mib).map(move |i| (d, s, i)))
                    .map(|(d, s, i)| Run {
                        dst: wslot(d, s) + (i * RUN_WORDS * 8) as u64,
                        src: src[s].cu_deviceptr() + (i * RUN_WORDS * 8) as u64,
                        len: RUN_WORDS as u64,
                    })
                    .collect();
                Runs::new(&streams[d], &r)
            })
            .collect::<Result<_, _>>()?;
        let push: Vec<Runs> = (0..4)
            .map(|s| {
                let r: Vec<Run> = flows
                    .iter()
                    .filter(|f| f.1 == s)
                    .flat_map(|&(d, s)| (0..mib).map(move |i| (d, s, i)))
                    .map(|(d, s, i)| Run {
                        dst: wslot(d, s) + (i * RUN_WORDS * 8) as u64,
                        src: src[s].cu_deviceptr() + (i * RUN_WORDS * 8) as u64,
                        len: RUN_WORDS as u64,
                    })
                    .collect();
                Runs::new(&streams[s], &r)
            })
            .collect::<Result<_, _>>()?;
        // Per destination, one table per source, launched one after another.
        let seq: Vec<Vec<Runs>> = (0..4)
            .map(|d| {
                flows
                    .iter()
                    .filter(|f| f.0 == d)
                    .map(|&(d, s)| {
                        let r: Vec<Run> = (0..mib)
                            .map(|i| Run {
                                dst: wslot(d, s) + (i * RUN_WORDS * 8) as u64,
                                src: src[s].cu_deviceptr() + (i * RUN_WORDS * 8) as u64,
                                len: RUN_WORDS as u64,
                            })
                            .collect();
                        Runs::new(&streams[d], &r)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, _>>()?;
        let mut best = [f64::INFINITY; 5];
        for _ in 0..reps {
            for (m, b) in best.iter_mut().enumerate() {
                for s in &streams {
                    s.synchronize()?;
                }
                let t = Instant::now();
                match m {
                    0 => {
                        for d in 0..4 {
                            // SAFETY: every run addresses 1 MiB inside a live buffer of its
                            // device; no other work touches them.
                            unsafe { ex[d].copy_runs(&streams[d], &runs[d])? };
                        }
                    }
                    4 => {
                        for d in 0..4 {
                            for r in &seq[d] {
                                // SAFETY: as for method 0.
                                unsafe { ex[d].copy_runs(&streams[d], r)? };
                            }
                        }
                    }
                    3 => {
                        for s in 0..4 {
                            // SAFETY: as above, with the destinations on peers.
                            unsafe { ex[s].copy_runs(&streams[s], &push[s])? };
                        }
                    }
                    _ => {
                        for &(d, s) in flows {
                            let on = if m == 1 { &streams[s] } else { &streams[d] };
                            // SAFETY: both ranges lie inside live buffers of their contexts.
                            unsafe {
                                sys::cuMemcpyPeerAsync(
                                    wslot(d, s),
                                    ctxs[d].cu_ctx(),
                                    src[s].cu_deviceptr(),
                                    ctxs[s].cu_ctx(),
                                    words * 8,
                                    on.cu_stream(),
                                )
                                .result()?
                            };
                        }
                    }
                }
                for s in &streams {
                    s.synchronize()?;
                }
                *b = b.min(t.elapsed().as_secs_f64());
            }
        }
        let gbps = |t: f64| (words * 8) as f64 / t / 1e9;
        println!(
            "{name:>16} {:>6} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            flows.len(),
            gbps(best[0]),
            gbps(best[1]),
            gbps(best[2]),
            gbps(best[3]),
            gbps(best[4])
        );
    }
    Ok(())
}
