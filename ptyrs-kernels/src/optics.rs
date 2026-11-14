//! Propagator stages (ptyrs::engine `propagate`): what stands between a stack's exit waves
//! and the fused far-field transform when the propagator is not the Fourier transform.
//!
//! A unitary propagator is `U = D F V`: `F` the transform of the fused passes, `D` a
//! unit-modulus detector factor no intensity sees, and `V` acting on the exit-wave plane.
//! The far field and the Fresnel transform have `V = 1` (their plane factor is the passes'
//! `quad` phase), a Fresnel transform back in distance the index reversal, and angular
//! spectrum `V = T F⁻¹` with `T` its transfer function, even in both frequencies. The passes
//! hand over `F_x w` and take `F_x` of the correction, so a stage maps row transforms to row
//! transforms; the in-object propagation of a multislice gap is the same line pass three
//! times over plain planes.
//!
//! A stack's planes are its `nf * nmode` views; planes past them are not touched.

use cuda_device::{device, shared::DynamicSharedArray, thread};
use cuda_host::cuda_module;

use crate::fused::block_fft;
use crate::pixel::{StackCtl, StackGeom, cmul, cmul_conj, load_c, stack_frames, store_c};
use crate::{FFT_EPT_MAX, fft_ept};

/// The lines are columns, of `ny` points; without it rows, of `nx`.
pub const STAGE_COLS: u32 = 1;
/// A forward FFT of each line before the multiplication.
pub const STAGE_PRE_FWD: u32 = 2;
/// An inverse FFT of each line before the multiplication; never with `STAGE_PRE_FWD`.
pub const STAGE_PRE_INV: u32 = 4;
/// Each value is multiplied by `scale mul`, the plane's value at its pixel.
pub const STAGE_MUL: u32 = 8;
/// With `STAGE_MUL`: by `scale conj(mul)`.
pub const STAGE_CONJ: u32 = 16;
/// A forward FFT of each line after the multiplication.
pub const STAGE_POST_FWD: u32 = 32;
/// An inverse FFT of each line after the multiplication; never with `STAGE_POST_FWD`.
pub const STAGE_POST_INV: u32 = 64;

/// One line pass in place over the stack's planes of `buf`, `[nf_max * nmode, ny, nx]`: per
/// line of `N` points the `flags`' FFT (unnormalized, forward `exp(-2 pi i jk / N)`), the
/// multiplication by the `[ny, nx]` plane `mul`, and their second FFT. Block `(N /
/// fft_ept(N), bpb)`, grid x `ceil(nf_max * nmode * lines / bpb)` with `lines` the lines of
/// a plane, dynamic shared memory `bpb` FFTs.
///
/// # Safety
///
/// `N` is the line length `flags` name in `g`; `buf` holds its shape for `g`, `mul` one
/// plane under `STAGE_MUL`; `ctl` one entry; launch as above.
#[device]
pub unsafe fn stage<const N: u32>(
    buf: *mut f32,
    mul: *const f32,
    scale: f32,
    ctl: *const StackCtl,
    g: StackGeom,
    flags: u32,
) {
    let (_, nf) = unsafe { stack_frames(ctl, g) };
    let cols = flags & STAGE_COLS != 0;
    let per = if cols { g.nx } else { g.ny } as u64;
    let nline = nf as u64 * g.nmode as u64 * per;
    let bpb = thread::blockDim_y();
    let first = thread::blockIdx_x() as u64 * bpb as u64;
    // The whole block leaves together, before its first barrier.
    if first >= nline {
        return;
    }
    let l = first + thread::threadIdx_y() as u64;
    let live = l < nline;
    let n = g.ny as usize * g.nx as usize;
    let (plane, i) = ((l / per) as usize * n, (l % per) as usize);
    let (base, stride) = if cols {
        (i, g.nx as usize)
    } else {
        (i * g.nx as usize, 1)
    };
    let ept = const { fft_ept(N) };
    let tpf = N / ept;
    let (b, f) = (thread::threadIdx_y(), thread::threadIdx_x());
    let at = |e: u32| base + (f + e * tpf) as usize * stride;
    let mut r = [0.0f32; 2 * FFT_EPT_MAX as usize];
    unsafe {
        let smem = DynamicSharedArray::<u8, 16>::get();
        if live {
            for e in 0..ept {
                let k = 2 * e as usize;
                (r[k], r[k + 1]) = load_c(buf, plane + at(e));
            }
        }
        if flags & STAGE_PRE_FWD != 0 {
            block_fft::<N, false>(&mut r, smem, b, f, bpb);
        } else if flags & STAGE_PRE_INV != 0 {
            block_fft::<N, true>(&mut r, smem, b, f, bpb);
        }
        if live && flags & STAGE_MUL != 0 {
            for e in 0..ept {
                let k = 2 * e as usize;
                let m = load_c(mul, at(e));
                let m = (scale * m.0, scale * m.1);
                (r[k], r[k + 1]) = if flags & STAGE_CONJ != 0 {
                    cmul_conj(m, (r[k], r[k + 1]))
                } else {
                    cmul(m, (r[k], r[k + 1]))
                };
            }
        }
        if flags & STAGE_POST_FWD != 0 {
            block_fft::<N, false>(&mut r, smem, b, f, bpb);
        } else if flags & STAGE_POST_INV != 0 {
            block_fft::<N, true>(&mut r, smem, b, f, bpb);
        }
        if live {
            for e in 0..ept {
                let k = 2 * e as usize;
                store_c(buf, plane + at(e), (r[k], r[k + 1]));
            }
        }
    }
}

#[cuda_module]
pub mod optics_kernels {
    use super::stage;
    use crate::pixel::*;
    use crate::*;

    /// [`super::stage`] for 64-point lines.
    ///
    /// # Safety
    ///
    /// As [`super::stage`].
    #[kernel]
    pub unsafe fn optics_stage_64(
        buf: *mut f32,
        mul: *const f32,
        scale: f32,
        ctl: *const StackCtl,
        g: StackGeom,
        flags: u32,
    ) {
        unsafe { stage::<64>(buf, mul, scale, ctl, g, flags) }
    }

    /// [`super::stage`] for 128-point lines.
    ///
    /// # Safety
    ///
    /// As [`super::stage`].
    #[kernel]
    pub unsafe fn optics_stage_128(
        buf: *mut f32,
        mul: *const f32,
        scale: f32,
        ctl: *const StackCtl,
        g: StackGeom,
        flags: u32,
    ) {
        unsafe { stage::<128>(buf, mul, scale, ctl, g, flags) }
    }

    /// [`super::stage`] for 256-point lines.
    ///
    /// # Safety
    ///
    /// As [`super::stage`].
    #[kernel]
    pub unsafe fn optics_stage_256(
        buf: *mut f32,
        mul: *const f32,
        scale: f32,
        ctl: *const StackCtl,
        g: StackGeom,
        flags: u32,
    ) {
        unsafe { stage::<256>(buf, mul, scale, ctl, g, flags) }
    }

    /// [`super::stage`] for 512-point lines.
    ///
    /// # Safety
    ///
    /// As [`super::stage`].
    #[kernel]
    pub unsafe fn optics_stage_512(
        buf: *mut f32,
        mul: *const f32,
        scale: f32,
        ctl: *const StackCtl,
        g: StackGeom,
        flags: u32,
    ) {
        unsafe { stage::<512>(buf, mul, scale, ctl, g, flags) }
    }

    /// The index reversal of each of the stack's planes of `buf`, `[nf_max * nmode, ny, nx]`,
    /// in place: pixel `(v, u)` trades places with `((ny - v) % ny, (nx - u) % nx)`. One
    /// thread per value of `buf`.
    ///
    /// # Safety
    ///
    /// `buf` holds its shape for `g`; `ctl` one entry.
    #[kernel]
    pub unsafe fn optics_reverse(buf: *mut f32, ctl: *const StackCtl, g: StackGeom) {
        let (_, nf) = unsafe { stack_frames(ctl, g) };
        let n = g.ny as usize * g.nx as usize;
        let i = global_index();
        if i >= nf as usize * g.nmode as usize * n {
            return;
        }
        let (plane, j) = (i / n * n, i % n);
        let (v, u) = (j as u32 / g.nx, j as u32 % g.nx);
        let k = (((g.ny - v) % g.ny) * g.nx + (g.nx - u) % g.nx) as usize;
        // The lower index of a pair moves both.
        if j < k {
            unsafe {
                let (a, b) = (load_c(buf, plane + j), load_c(buf, plane + k));
                store_c(buf, plane + j, b);
                store_c(buf, plane + k, a);
            }
        }
    }
}
