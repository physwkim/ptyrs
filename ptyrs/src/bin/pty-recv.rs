//! Receiver: [`ptyrs::live::recv::run`] with the `--key value` flags of
//! [`RecvCfg::from_args`]. Exits with [`RecvReport::exit_code`], or 1 if a
//! step failed.
//!
//! [`RecvReport::exit_code`]: ptyrs::live::recv::RecvReport::exit_code

#![warn(clippy::disallowed_methods)]

use std::process::ExitCode;

use ptyrs::live::args::Args;
use ptyrs::live::recv::{RecvCfg, run};

fn main() -> ExitCode {
    match run(&RecvCfg::from_args(&Args::parse())) {
        Ok(r) => {
            print!("{}", r.lines);
            ExitCode::from(r.exit_code())
        }
        Err(e) => {
            eprintln!("pty-recv: {e}");
            ExitCode::FAILURE
        }
    }
}
