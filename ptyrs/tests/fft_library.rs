//! `ptyrs::fft` runs on the libcufft build.rs read, whatever LD_LIBRARY_PATH names.

use std::ffi::{CStr, c_int, c_void};
use std::path::PathBuf;
use std::process::Command;

use ptyrs::fft::sys::LIBRARY;

const TEST: &str = "libcufft_is_the_build_file_whatever_ld_library_path_names";
const CHILD: &str = "PTYRS_FFT_LIBRARY_CHILD";

/// The name the loader opened each loaded object by, symlinks unresolved.
fn loaded_objects() -> Vec<String> {
    unsafe extern "C" fn push(info: *mut libc::dl_phdr_info, _: usize, out: *mut c_void) -> c_int {
        // SAFETY: the loader passes a valid entry, and `out` is the `Vec` below.
        unsafe {
            let name = CStr::from_ptr((*info).dlpi_name).to_string_lossy();
            (*out.cast::<Vec<String>>()).push(name.into_owned());
        }
        0
    }
    let mut names = Vec::new();
    // SAFETY: `push` only appends to `names`, which outlives the call.
    unsafe { libc::dl_iterate_phdr(Some(push), (&raw mut names).cast()) };
    names
}

#[test]
fn libcufft_is_the_build_file_whatever_ld_library_path_names() {
    if std::env::var_os(CHILD).is_some() {
        ptyrs::fft::version().unwrap();
        let cufft: Vec<String> = loaded_objects()
            .into_iter()
            .filter(|name| name.contains("libcufft"))
            .collect();
        assert_eq!(cufft, [LIBRARY]);
        println!("{CHILD}: {LIBRARY}");
        return;
    }
    // A working libcufft under another path, first on the loader's search: glibc reads
    // LD_LIBRARY_PATH only at startup, so the check runs in a child.
    let decoy = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("fft-library-decoy");
    std::fs::create_dir_all(&decoy).unwrap();
    let soname = decoy.join("libcufft.so.12");
    let _ = std::fs::remove_file(&soname);
    std::os::unix::fs::symlink(LIBRARY, &soname).unwrap();
    let out = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", TEST, "--nocapture"])
        .env(CHILD, "1")
        .env("LD_LIBRARY_PATH", &decoy)
        .output()
        .unwrap();
    let (stdout, stderr) = (
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(
        out.status.success() && stdout.contains(&format!("{CHILD}: {LIBRARY}")),
        "child {}:\n{stdout}{stderr}",
        out.status
    );
}
