//! Writes the M5 scan simulator's ground truth (`Ptycho::dump`) into
//! `--truth DIR`, from the same scan arguments pty-sim takes with `--model
//! ptycho` (which it also takes), so a dump matches the stream frame for frame.

#![warn(clippy::disallowed_methods)]

use std::path::Path;
use std::time::Instant;

use ptyrs::live::args::cli;
use ptyrs::live::model::PtychoCfg;

fn main() {
    let (dir, cfg) = cli("live_ptycho_truth", |a| {
        a.choice("model", &[("ptycho", ())])?;
        Ok((a.str("truth", "ptycho-truth"), PtychoCfg::from_args(a)?))
    });
    let t = Instant::now();
    let p = cfg.build();
    let built = t.elapsed();
    p.dump(Path::new(&dir)).expect("dump");
    println!(
        "ptycho_truth {} dir={dir} build_ms={} dump_ms={}",
        p.describe(),
        built.as_millis(),
        (t.elapsed() - built).as_millis()
    );
}
