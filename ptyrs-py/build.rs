fn main() {
    // The module exports only `PyInit_ptyrs`. Upstream `#[no_mangle]` kernel host stubs would
    // otherwise be exported and pull in device-only cuFFT symbols the loader cannot resolve.
    println!("cargo:rustc-link-arg-cdylib=-Wl,--exclude-libs,ALL");
}
