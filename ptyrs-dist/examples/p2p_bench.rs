//! Peer copy bandwidth and latency for every ordered GPU pair.
//!
//! `cargo run --release -p ptyrs-dist --example p2p_bench [MiB] [reps]`

use cuda_core::Context;
use ptyrs_dist::p2p::{self, Outcome};
use ptyrs_dist::topo;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).map(|a| a.parse::<u32>());
    let mib = args.next().transpose()?.unwrap_or(64);
    let reps = args.next().transpose()?.unwrap_or(20);
    let devs = topo::devices()?;
    let ctxs = devs
        .iter()
        .map(|d| Context::new(d.ordinal))
        .collect::<Result<Vec<_>, _>>()?;
    let rows = p2p::bench(&ctxs, (mib as usize) << 20, reps)?;
    println!("P2P, {reps} x {mib} MiB per bulk timing; measured on shared GPUs");
    for d in &devs {
        println!(
            "  GPU{} {} pci {} numa {:?}",
            d.ordinal, d.name, d.pci_bus_id, d.numa
        );
    }
    println!(
        "{:>7} {:>5} {:>10} {:>10} {:>10} {:>10}  data",
        "src>dst", "path", "copy GB/s", "copy us", "pull GB/s", "load us"
    );
    for r in &rows {
        let link = if devs[r.src].numa.is_some() && devs[r.src].numa == devs[r.dst].numa {
            "numa"
        } else {
            "sys"
        };
        let (m, data) = match r.outcome {
            Outcome::ProbeFailed => {
                println!(
                    "{:>3}>{:<3} {:>5} {:>43}  PROBE FAILED, not timed",
                    r.src, r.dst, link, ""
                );
                continue;
            }
            Outcome::Corrupt(m) => (m, "CORRUPT"),
            Outcome::Intact(m) => (m, "ok"),
        };
        println!(
            "{:>3}>{:<3} {:>5} {:>10.2} {:>10.2} {:>10.2} {:>10.2}  {}",
            r.src,
            r.dst,
            link,
            m.copy_gbps,
            m.copy_latency_us,
            m.pull_gbps,
            m.load_latency_us,
            data
        );
    }
    Ok(())
}
