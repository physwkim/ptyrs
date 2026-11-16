//! Receiver: [`ptyrs::live::recv::run`] with the `--key value` flags of
//! [`RecvCfg::from_args`]. Exits with [`RecvReport::exit_code`], 1 if a step
//! failed, or [`EX_USAGE`] on a refused flag; `--help` lists the flags.
//!
//! [`RecvReport::exit_code`]: ptyrs::live::recv::RecvReport::exit_code
//! [`EX_USAGE`]: ptyrs::live::args::EX_USAGE

#![warn(clippy::disallowed_methods)]

use std::process::ExitCode;

use ptyrs::live::args::cli;
use ptyrs::live::recv::{RecvCfg, run};

fn main() -> ExitCode {
    match run(&cli("pty-recv", RecvCfg::from_args)) {
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
