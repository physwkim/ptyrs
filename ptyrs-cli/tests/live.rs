use std::path::PathBuf;
use std::process::{Command, Output};

fn ptyrs(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ptyrs"))
        .args(args)
        .output()
        .unwrap()
}

/// `text` as a config file under the target dir, named per test.
fn config(name: &str, text: &str) -> PathBuf {
    let p = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}.toml"));
    std::fs::write(&p, text).unwrap();
    p
}

fn refused(out: &Output) -> String {
    assert!(!out.status.success());
    assert!(out.stdout.is_empty());
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn a_misspelt_key_is_refused_with_its_line() {
    let p = config("misspelt", "[recv]\nframes = 10\nworkerz = 4\n");
    let err = refused(&ptyrs(&["live", p.to_str().unwrap()]));
    assert!(err.contains("workerz") && err.contains("line 3"), "{err}");
}

#[test]
fn zero_workers_is_refused_before_any_frame() {
    let p = config(
        "zero_workers",
        "[recv]\nworkers = 0\nconnect-timeout-s = 1\n",
    );
    let err = refused(&ptyrs(&["live", p.to_str().unwrap()]));
    assert!(err.contains("workers"), "{err}");
}

#[test]
fn a_missing_file_or_command_is_refused() {
    let err = refused(&ptyrs(&["live", "/nonexistent/live.toml"]));
    assert!(err.contains("/nonexistent/live.toml"), "{err}");
    let err = refused(&ptyrs(&["recv", "live.toml"]));
    assert!(err.contains("unknown command recv"), "{err}");
    let err = refused(&ptyrs(&[]));
    assert!(err.contains("usage"), "{err}");
}
