#![warn(clippy::disallowed_methods)]

use std::process::Command;

use ptyrs::live::args::EX_USAGE;

/// `bin`'s exit code, stdout and stderr with `args`.
fn run(bin: &str, args: &[&str]) -> (Option<i32>, String, String) {
    let out = Command::new(bin).args(args).output().unwrap();
    let text = |b: &[u8]| String::from_utf8_lossy(b).into_owned();
    (out.status.code(), text(&out.stdout), text(&out.stderr))
}

/// The message `bin` refuses `args` with, after checking it exits [`EX_USAGE`] unpanicked.
fn refused(bin: &str, args: &[&str]) -> String {
    let (code, _, err) = run(bin, args);
    assert_eq!(code, Some(EX_USAGE), "{args:?}: {err}");
    assert!(!err.contains("panicked"), "{args:?}: {err}");
    err
}

const RECV: &str = env!("CARGO_BIN_EXE_pty-recv");
const SIM: &str = env!("CARGO_BIN_EXE_pty-sim");

#[test]
fn zero_workers_is_refused_before_any_frame() {
    let err = refused(
        RECV,
        &[
            "--workers",
            "0",
            "--connect-timeout-s",
            "1",
            "--util-min",
            "0",
        ],
    );
    assert!(err.starts_with("pty-recv: --workers 0: "), "{err}");
}

#[test]
fn help_prints_the_usage_and_exits_0() {
    for (bin, name, choice) in [
        (RECV, "pty-recv", "--source pva|zmq|h5"),
        (SIM, "pty-sim", "--out pva|zmq|h5"),
    ] {
        let (code, out, err) = run(bin, &["--help"]);
        assert_eq!(code, Some(0), "{name}: {err}");
        assert!(out.starts_with(&format!("usage: {name} ")), "{out}");
        assert!(out.contains(&format!("\n  {choice}\n")), "{out}");
    }
}

#[test]
fn an_unknown_flag_a_missing_value_and_a_bad_value_exit_with_ex_usage() {
    let e = refused(RECV, &["--frame", "5"]);
    assert!(
        e.starts_with("pty-recv: not read with the flags given: --frame "),
        "{e}"
    );
    assert_eq!(
        refused(RECV, &["--frames"]),
        "pty-recv: --frames needs a value\n"
    );
    assert_eq!(
        refused(SIM, &["--codec", "x"]),
        "pty-sim: --codec x: expected one of bslz4, none, lz4\n"
    );
    assert_eq!(
        refused(SIM, &["--out", "h5"]),
        "pty-sim: --out h5 needs --h5 PATH\n"
    );
    let e = refused(SIM, &["--h5-flush", "4"]);
    assert!(
        e.starts_with("pty-sim: not read with the flags given: --h5-flush "),
        "{e}"
    );
}
