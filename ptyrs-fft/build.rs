//! Links libcufft from the toolkit named by CUDA_TOOLKIT_PATH (or CUDA_HOME).
use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    let root = env::var("CUDA_TOOLKIT_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .expect("set CUDA_TOOLKIT_PATH or CUDA_HOME to the CUDA toolkit root");
    let lib = PathBuf::from(root).join("lib64");
    assert!(
        lib.join("libcufft.so").exists(),
        "libcufft.so not in {}",
        lib.display()
    );
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=cufft");
    // Tests/examples run without LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
}
