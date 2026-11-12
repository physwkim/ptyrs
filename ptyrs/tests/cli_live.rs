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

/// Needs a GPU. Nothing serves `addr`, so no frame arrives: every rate
/// over the receive span is reported as none, not as a number.
#[test]
fn a_run_without_frames_reports_no_rates() {
    let p = config(
        "no_frames",
        "[recv]\nframes = 10\naddr = \"127.0.0.1:15999\"\nconnect-timeout-s = 1\nring-slots = 16\n",
    );
    let out = ptyrs(&["live", p.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(2), "every trigger is lost");
    let text = String::from_utf8_lossy(&out.stdout);
    let rates = [
        "rx_rate_hz",
        "cpu_cores_steady",
        "cpu_cores_total",
        "worker_cpu_cores",
        "h2d_cpu_cores",
        "mean_batch",
        "h2d_MBps",
    ];
    for k in rates {
        assert!(text.contains(&format!(" {k}=none")), "{k}: {text}");
    }
}

/// Needs a GPU. The PULL socket fails at connect: the run stops with that
/// error at once instead of waiting out the 30 s connect timeout.
#[test]
fn a_failing_zmq_source_stops_the_run_at_once() {
    let p = config(
        "zmq_bad_endpoint",
        "[recv]\nframes = 10\naddr = \"127.0.0.1:15999\"\nconnect-timeout-s = 30\n\
         ring-slots = 16\n[recv.source.zmq]\nendpoint = \"tcp://127.0.0.1:notaport\"\n",
    );
    let t0 = std::time::Instant::now();
    let out = ptyrs(&["live", p.to_str().unwrap()]);
    let took = t0.elapsed();
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(err.contains("ZMQ PULL"), "{err}");
    assert!(took.as_secs() < 10, "stopped after {took:?}");
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
