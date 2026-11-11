//! Writes the M5 scan simulator's ground truth (`Ptycho::dump`) into
//! `--truth DIR`, from the same scan arguments pty-sim takes with `--model
//! ptycho`, so a dump matches the stream frame for frame.

use std::path::Path;
use std::time::Instant;

use ptyrs_live::args::Args;
use ptyrs_live::model::ptycho_from_args;

fn main() {
    let a = Args::parse();
    let dir = a.str("truth", "ptycho-truth");
    let t = Instant::now();
    let p = ptycho_from_args(&a);
    let built = t.elapsed();
    p.dump(Path::new(&dir)).expect("dump");
    println!(
        "ptycho_truth {} dir={dir} build_ms={} dump_ms={}",
        p.describe(),
        built.as_millis(),
        (t.elapsed() - built).as_millis()
    );
}
