//! Hand-written subset of cufft.h (CUDA 13.x, libcufft.so.12).
#![allow(non_camel_case_types)]

use std::os::raw::{c_int, c_void};

pub type cufftHandle = c_int;
pub type cufftResult = c_int;
pub type cufftType = c_int;

pub const CUFFT_SUCCESS: cufftResult = 0;
pub const CUFFT_C2C: cufftType = 0x29;
pub const CUFFT_FORWARD: c_int = -1;
pub const CUFFT_INVERSE: c_int = 1;

unsafe extern "C" {
    pub fn cufftCreate(plan: *mut cufftHandle) -> cufftResult;
    pub fn cufftSetAutoAllocation(plan: cufftHandle, auto_allocate: c_int) -> cufftResult;
    pub fn cufftMakePlanMany(
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
    ) -> cufftResult;
    pub fn cufftSetWorkArea(plan: cufftHandle, work_area: *mut c_void) -> cufftResult;
    /// `stream` is a `cudaStream_t`, which is the same handle as a driver `CUstream`.
    pub fn cufftSetStream(plan: cufftHandle, stream: *mut c_void) -> cufftResult;
    pub fn cufftExecC2C(
        plan: cufftHandle,
        idata: *mut c_void,
        odata: *mut c_void,
        direction: c_int,
    ) -> cufftResult;
    pub fn cufftDestroy(plan: cufftHandle) -> cufftResult;
    pub fn cufftGetVersion(version: *mut c_int) -> cufftResult;
}
