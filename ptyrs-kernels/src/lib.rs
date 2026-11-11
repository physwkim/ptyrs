//! ptyrs-kernels: cuda-oxide SIMT device code.
//!
//! This crate is compiled twice. Plain `rustc` builds it as a host library whose
//! `#[cuda_module]` expansion provides the typed launchers ([`kernels::LoadedModule`]); the
//! cuda-oxide backend builds it again as device code (NVVM IR). Host crates do not drive
//! either step by hand: `ptyrs-gpu`'s build script runs the device build, links the result
//! with the cuFFT device-API code and embeds the cubin, and `ptyrs-gpu` wraps every kernel
//! in a checked call.
//!
//! Buffers are raw device pointers. A complex value is two interleaved `f32` (re, im), the
//! layout of `ptyrs_fft::C32`.

use cuda_device::{device, kernel, shared::DynamicSharedArray, thread};
use cuda_host::cuda_module;

/// Complex elements per thread of every in-kernel FFT (cuFFT device API, BY_COMPONENT).
pub const FFT_EPT: u32 = 16;

// In-kernel FFT entry points. Each name spells size, elements per thread and direction;
// ptyrs-gpu's build script defines it as a call to the libcufft function with exactly
// those traits for the target architecture. Arguments: registers (`2 * FFT_EPT` floats,
// re/im interleaved; thread `t` holds elements `t + i * N / FFT_EPT`), dynamic shared
// memory, FFT index within the block, thread index within the FFT, FFTs per block. Every
// thread of the block must call it: it synchronizes the block.
#[device]
unsafe extern "C" {
    fn ptyrs_cufft_c2c_f32_n128_e16_fwd(r: *mut f32, s: *mut u8, b: u32, f: u32, bpb: u32);
    fn ptyrs_cufft_c2c_f32_n128_e16_inv(r: *mut f32, s: *mut u8, b: u32, f: u32, bpb: u32);
    fn ptyrs_cufft_c2c_f32_n256_e16_fwd(r: *mut f32, s: *mut u8, b: u32, f: u32, bpb: u32);
    fn ptyrs_cufft_c2c_f32_n256_e16_inv(r: *mut f32, s: *mut u8, b: u32, f: u32, bpb: u32);
}

/// Line handled by this thread's FFT: blockDim = (threads per FFT, FFTs per block).
#[device]
fn line_index() -> u32 {
    thread::blockIdx_x() * thread::blockDim_y() + thread::threadIdx_y()
}

/// Loads this thread's registers of line `line_index()`: complex element `j` of line `l` is
/// at `l * line_stride + j * elem_stride` (complex units). Lines past `nline` load zeros.
///
/// # Safety
///
/// `input` must hold every element of that layout for `nline` lines.
#[device]
unsafe fn load_line(
    input: *const f32,
    n: u32,
    nline: u32,
    line_stride: u32,
    elem_stride: u32,
) -> [f32; 2 * FFT_EPT as usize] {
    let mut r = [0.0f32; 2 * FFT_EPT as usize];
    let l = line_index();
    if l < nline {
        for i in 0..FFT_EPT {
            let j = thread::threadIdx_x() + i * (n / FFT_EPT);
            let at = 2 * (l as usize * line_stride as usize + j as usize * elem_stride as usize);
            let k = 2 * i as usize;
            unsafe {
                r[k] = *input.add(at);
                r[k + 1] = *input.add(at + 1);
            }
        }
    }
    r
}

/// Stores this thread's registers with the layout of [`load_line`].
///
/// # Safety
///
/// `output` must hold every element of that layout for `nline` lines.
#[device]
unsafe fn store_line(
    output: *mut f32,
    r: &[f32; 2 * FFT_EPT as usize],
    n: u32,
    nline: u32,
    line_stride: u32,
    elem_stride: u32,
) {
    let l = line_index();
    if l < nline {
        for i in 0..FFT_EPT {
            let j = thread::threadIdx_x() + i * (n / FFT_EPT);
            let at = 2 * (l as usize * line_stride as usize + j as usize * elem_stride as usize);
            let k = 2 * i as usize;
            unsafe {
                *output.add(at) = r[k];
                *output.add(at + 1) = r[k + 1];
            }
        }
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `x[i] *= s` for `n` complex values.
    ///
    /// # Safety
    ///
    /// `x` must hold `n` complex values.
    #[kernel]
    pub unsafe fn scale_c32(x: *mut f32, n: u32, s: f32) {
        let i = thread::index_1d().get();
        if i < n as usize {
            unsafe {
                *x.add(2 * i) *= s;
                *x.add(2 * i + 1) *= s;
            }
        }
    }

    // Unnormalized C2C FFT of `nline` strided lines (see `load_line`); forward uses
    // exp(-2 pi i jk / N). `input` may equal `output`: a thread reads all its elements
    // before the FFT's block barrier and writes after it, and lines must not overlap.
    // Launch: blockDim = (N / FFT_EPT, bpb), gridDim.x = ceil(nline / bpb), dynamic shared
    // memory = bpb * smem_per_fft bytes (ptyrs-gpu's BLOCK_FFTS).

    /// 128-point forward line FFT.
    ///
    /// # Safety
    ///
    /// The buffers must hold `nline` lines of the stated layout; launch as above.
    #[kernel]
    pub unsafe fn fft128_lines_fwd(
        input: *const f32,
        output: *mut f32,
        nline: u32,
        line_stride: u32,
        elem_stride: u32,
    ) {
        unsafe {
            let mut r = load_line(input, 128, nline, line_stride, elem_stride);
            ptyrs_cufft_c2c_f32_n128_e16_fwd(
                r.as_mut_ptr(),
                DynamicSharedArray::<u8, 16>::get(),
                thread::threadIdx_y(),
                thread::threadIdx_x(),
                thread::blockDim_y(),
            );
            store_line(output, &r, 128, nline, line_stride, elem_stride);
        }
    }

    /// 128-point inverse line FFT.
    ///
    /// # Safety
    ///
    /// As [`fft128_lines_fwd`].
    #[kernel]
    pub unsafe fn fft128_lines_inv(
        input: *const f32,
        output: *mut f32,
        nline: u32,
        line_stride: u32,
        elem_stride: u32,
    ) {
        unsafe {
            let mut r = load_line(input, 128, nline, line_stride, elem_stride);
            ptyrs_cufft_c2c_f32_n128_e16_inv(
                r.as_mut_ptr(),
                DynamicSharedArray::<u8, 16>::get(),
                thread::threadIdx_y(),
                thread::threadIdx_x(),
                thread::blockDim_y(),
            );
            store_line(output, &r, 128, nline, line_stride, elem_stride);
        }
    }

    /// 256-point forward line FFT.
    ///
    /// # Safety
    ///
    /// As [`fft128_lines_fwd`].
    #[kernel]
    pub unsafe fn fft256_lines_fwd(
        input: *const f32,
        output: *mut f32,
        nline: u32,
        line_stride: u32,
        elem_stride: u32,
    ) {
        unsafe {
            let mut r = load_line(input, 256, nline, line_stride, elem_stride);
            ptyrs_cufft_c2c_f32_n256_e16_fwd(
                r.as_mut_ptr(),
                DynamicSharedArray::<u8, 16>::get(),
                thread::threadIdx_y(),
                thread::threadIdx_x(),
                thread::blockDim_y(),
            );
            store_line(output, &r, 256, nline, line_stride, elem_stride);
        }
    }

    /// 256-point inverse line FFT.
    ///
    /// # Safety
    ///
    /// As [`fft128_lines_fwd`].
    #[kernel]
    pub unsafe fn fft256_lines_inv(
        input: *const f32,
        output: *mut f32,
        nline: u32,
        line_stride: u32,
        elem_stride: u32,
    ) {
        unsafe {
            let mut r = load_line(input, 256, nline, line_stride, elem_stride);
            ptyrs_cufft_c2c_f32_n256_e16_inv(
                r.as_mut_ptr(),
                DynamicSharedArray::<u8, 16>::get(),
                thread::threadIdx_y(),
                thread::threadIdx_x(),
                thread::blockDim_y(),
            );
            store_line(output, &r, 256, nline, line_stride, elem_stride);
        }
    }
}
