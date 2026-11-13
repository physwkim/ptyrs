//! Hand-written subset of cufft.h (CUDA 13.x, libcufft.so.12), resolved from [`LIBRARY`].
#![allow(non_camel_case_types)]

use std::os::raw::{c_int, c_void};
use std::sync::OnceLock;

use libloading::Library;

/// The libcufft file build.rs read the cubin's FFTs from, symlinks resolved. It is opened by
/// this path, never searched for by SONAME, so neither LD_LIBRARY_PATH nor a toolkit symlink
/// moved since the build can put another libcufft under the binding.
pub const LIBRARY: &str = env!("PTYRS_LIBCUFFT");

pub type cufftHandle = c_int;
pub type cufftResult = c_int;
pub type cufftType = c_int;

pub const CUFFT_SUCCESS: cufftResult = 0;
pub const CUFFT_C2C: cufftType = 0x29;
pub const CUFFT_FORWARD: c_int = -1;
pub const CUFFT_INVERSE: c_int = 1;

/// The cufft.h functions the binding calls, from [`LIBRARY`].
pub struct Cufft {
    pub create: unsafe extern "C" fn(plan: *mut cufftHandle) -> cufftResult,
    pub set_auto_allocation:
        unsafe extern "C" fn(plan: cufftHandle, auto_allocate: c_int) -> cufftResult,
    pub make_plan_many: unsafe extern "C" fn(
        plan: cufftHandle,
        rank: c_int,
        n: *mut c_int,
        inembed: *mut c_int,
        istride: c_int,
        idist: c_int,
        onembed: *mut c_int,
        ostride: c_int,
        odist: c_int,
        ty: cufftType,
        batch: c_int,
        work_size: *mut usize,
    ) -> cufftResult,
    pub set_work_area:
        unsafe extern "C" fn(plan: cufftHandle, work_area: *mut c_void) -> cufftResult,
    /// `stream` is a `cudaStream_t`, which is the same handle as a driver `CUstream`.
    pub set_stream: unsafe extern "C" fn(plan: cufftHandle, stream: *mut c_void) -> cufftResult,
    pub exec_c2c: unsafe extern "C" fn(
        plan: cufftHandle,
        idata: *mut c_void,
        odata: *mut c_void,
        direction: c_int,
    ) -> cufftResult,
    pub destroy: unsafe extern "C" fn(plan: cufftHandle) -> cufftResult,
    pub get_version: unsafe extern "C" fn(version: *mut c_int) -> cufftResult,
    // Keeps the function pointers above valid.
    _lib: Library,
}

/// [`LIBRARY`]'s functions, loaded on first use, or why they could not be.
pub fn cufft() -> Result<&'static Cufft, &'static str> {
    static CUFFT: OnceLock<Result<Cufft, String>> = OnceLock::new();
    CUFFT
        .get_or_init(|| load().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(String::as_str)
}

fn load() -> Result<Cufft, libloading::Error> {
    // SAFETY: loading libcufft runs no initialisers with preconditions, and every symbol type
    // below matches its declaration in cufft.h.
    unsafe {
        let lib = Library::new(LIBRARY)?;
        Ok(Cufft {
            create: *lib.get(b"cufftCreate\0")?,
            set_auto_allocation: *lib.get(b"cufftSetAutoAllocation\0")?,
            make_plan_many: *lib.get(b"cufftMakePlanMany\0")?,
            set_work_area: *lib.get(b"cufftSetWorkArea\0")?,
            set_stream: *lib.get(b"cufftSetStream\0")?,
            exec_c2c: *lib.get(b"cufftExecC2C\0")?,
            destroy: *lib.get(b"cufftDestroy\0")?,
            get_version: *lib.get(b"cufftGetVersion\0")?,
            _lib: lib,
        })
    }
}
