use std::process::Command;

#[test]
fn zero_workers_is_refused_before_any_frame() {
    let out = Command::new(env!("CARGO_BIN_EXE_pty-recv"))
        .args([
            "--workers",
            "0",
            "--connect-timeout-s",
            "1",
            "--util-min",
            "0",
        ])
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(err.contains("--workers 0"), "{err}");
}
