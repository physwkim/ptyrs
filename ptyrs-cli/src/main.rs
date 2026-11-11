//! `ptyrs live CONFIG.toml` receives one scan through
//! [`ptyrs_live::recv::run`] as the file's `[recv]` table says, prints the
//! `recv ...` report and exits like pty-recv: 2 if a trigger never reached
//! the GPU, 3 if one reached it wrong, 1 on a bad file or a failed step.
//! `examples/live.toml` lists every key at its default.

use std::path::Path;
use std::process::ExitCode;

use ptyrs_live::recv::{RecvCfg, run};
use serde::Deserialize;

/// The stages that follow the receive path get tables beside `[recv]`.
#[derive(Debug, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Config {
    recv: RecvCfg,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [cmd, path] = args.as_slice() else {
        eprintln!("usage: ptyrs live CONFIG.toml");
        return ExitCode::FAILURE;
    };
    if cmd != "live" {
        eprintln!("ptyrs: unknown command {cmd}; usage: ptyrs live CONFIG.toml");
        return ExitCode::FAILURE;
    }
    match load(Path::new(path)).and_then(|c| run(&c.recv).map_err(|e| e.to_string())) {
        Ok(r) => {
            print!("{}", r.lines);
            ExitCode::from(r.exit_code())
        }
        Err(e) => {
            eprintln!("ptyrs: {e}");
            ExitCode::FAILURE
        }
    }
}

fn load(path: &Path) -> Result<Config, String> {
    let at = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
    let text = std::fs::read_to_string(path).map_err(|e| at(&e))?;
    toml::from_str(&text).map_err(|e| at(&e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_example_file_is_the_defaults() {
        let c: Config = toml::from_str(include_str!("../examples/live.toml")).unwrap();
        assert_eq!(c, Config::default());
    }
}
