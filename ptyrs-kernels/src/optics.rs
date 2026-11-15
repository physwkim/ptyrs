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
//! Angular spectrum with an evanescent band, where `T` is 0, has `Vᴴ V` the projection off
//! the band. A line pass under [`STAGE_SAVE`] keeps the band's values on their way into the
//! forward multiplication in a [`Band`]'s carry, and one under [`STAGE_RESTORE`] writes
//! them, scaled, over the adjoint multiplication's in the same bins.
//!
//! A multislice object is a product per slice with that gap between: [`optics_slice_mul`]
//! is a slice's product and its adjoint, [`optics_slice_phase`] the exit plane's phase,
//! which the fused row pass applies only to a single slice, and [`optics_slice_pair`] the
//! object adjoint's integrand.
//!
//! A stack's planes are its `nf * nmode` views; planes past them are not touched.
//!
//! [`optics_slice_mul`]: optics_kernels::LoadedModule::optics_slice_mul
//! [`optics_slice_phase`]: optics_kernels::LoadedModule::optics_slice_phase
//! [`optics_slice_pair`]: optics_kernels::LoadedModule::optics_slice_pair

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
/// Ahead of the multiplication, each value on the [`Band`] into its slot of the carry.
pub const STAGE_SAVE: u32 = 128;
/// After the multiplication, each value on the [`Band`] replaced by its slot of the carry
/// times `band.scale`.
pub const STAGE_RESTORE: u32 = 256;

/// [`Band::slots`] of a pixel off the band.
pub const OFF_BAND: u32 = u32::MAX;

/// The pixels of a `[ny, nx]` plane a line pass carries, and the carry: pixel `j`'s slot is
/// `slots[j]`, [`OFF_BAND`] for none, and slot `s` of the stack's plane `p` is `carry[p *
/// nband + s]`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Band {
    pub slots: *const u32,
    pub carry: *mut f32,
    pub nband: u32,
    /// [`STAGE_RESTORE`]'s factor.
    pub scale: f32,
}

/// [`optics_slice_mul`], [`optics_slice_phase`]: by the conjugate of the object window or
/// of the phase.
///
/// [`optics_slice_mul`]: optics_kernels::LoadedModule::optics_slice_mul
/// [`optics_slice_phase`]: optics_kernels::LoadedModule::optics_slice_phase
pub const SLICE_CONJ: u32 = 1;
/// [`optics_kernels::LoadedModule::optics_slice_mul`]: `src` is the probe, `[nprobe, ny,
/// nx]` in the natural layout, each view reading its mode; without it the stack's waves.
pub const SLICE_PROBE: u32 = 2;
/// [`optics_kernels::LoadedModule::optics_slice_phase`]: each frame's phase is displaced
/// by its offset.
pub const SLICE_OFFSET: u32 = 2;

/// A stack's waves `w`, `[nf_max * nw, ny, nx]` at the exit wave's stored pixels: `nw`
/// planes per frame, `nmode` for a wave per view and `nprobe` for one per probe mode. View
/// `m` of a frame reads [`wave_plane`], its own or its probe mode's (`m % nprobe`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct WavePlanes {
    pub w: *const f32,
    pub nw: u32,
}

/// The plane of `w` view `m` of frame `fl` reads: `fl * nw + m % nw`.
#[device]
pub fn wave_plane(w: WavePlanes, fl: usize, m: usize) -> usize {
    let nw = w.nw as usize;
    fl * nw + m % nw
}

/// One line pass in place over the stack's planes of `buf`, `[nf_max * nmode, ny, nx]`: per
/// line of `N` points the `flags`' FFT (unnormalized, forward `exp(-2 pi i jk / N)`), the
/// multiplication by the `[ny, nx]` plane `mul` with `band`'s save before it and restore
/// after, and their second FFT. Block `(N / fft_ept(N), bpb)`, grid x `ceil(nf_max * nmode
/// * lines / bpb)` with `lines` the lines of a plane, dynamic shared memory `bpb` FFTs.
///
/// # Safety
///
/// `N` is the line length `flags` name in `g`; `buf` holds its shape for `g`, `mul` one
/// plane under `STAGE_MUL`; under `STAGE_SAVE` or `STAGE_RESTORE` `band.slots` one plane of
/// slots, each [`OFF_BAND`] or below `band.nband`, and `band.carry` `nband` values per
/// plane of `buf`; `ctl` one entry; launch as above.
#[device]
pub unsafe fn stage<const N: u32>(
    buf: *mut f32,
    mul: *const f32,
    scale: f32,
    ctl: *const StackCtl,
    g: StackGeom,
    flags: u32,
    band: Band,
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
    let (p, i) = ((l / per) as usize, (l % per) as usize);
    let plane = p * n;
    let (base, stride) = if cols {
        (i, g.nx as usize)
    } else {
        (i * g.nx as usize, 1)
    };
    let ept = const { fft_ept(N) };
    let tpf = N / ept;
    let (b, f) = (thread::threadIdx_y(), thread::threadIdx_x());
    let at = |e: u32| base + (f + e * tpf) as usize * stride;
    // Value `e`'s index in `band.carry`, on the band.
    let carried = |e: u32| {
        let s = unsafe { *band.slots.add(at(e)) };
        (s != OFF_BAND).then(|| p * band.nband as usize + s as usize)
    };
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
        if live && flags & STAGE_SAVE != 0 {
            for e in 0..ept {
                if let Some(c) = carried(e) {
                    let k = 2 * e as usize;
                    store_c(band.carry, c, (r[k], r[k + 1]));
                }
            }
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
        if live && flags & STAGE_RESTORE != 0 {
            for e in 0..ept {
                if let Some(c) = carried(e) {
                    let k = 2 * e as usize;
                    let v = load_c(band.carry, c);
                    (r[k], r[k + 1]) = (band.scale * v.0, band.scale * v.1);
                }
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
    use super::{Band, SLICE_CONJ, SLICE_OFFSET, SLICE_PROBE, WavePlanes, stage, wave_plane};
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
        band: Band,
    ) {
        unsafe { stage::<64>(buf, mul, scale, ctl, g, flags, band) }
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
        band: Band,
    ) {
        unsafe { stage::<128>(buf, mul, scale, ctl, g, flags, band) }
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
        band: Band,
    ) {
        unsafe { stage::<256>(buf, mul, scale, ctl, g, flags, band) }
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
        band: Band,
    ) {
        unsafe { stage::<512>(buf, mul, scale, ctl, g, flags, band) }
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

    /// A slice's product over the stack's views: `dst`'s plane of view `k` is `obj`'s
    /// bilinear window of that view times `src` (under `SLICE_CONJ` the window's conjugate
    /// times it), `src.w` the probe under `SLICE_PROBE` (`src.nw` unread), else the stack's
    /// waves. `dst` is `[nf_max * nmode, ny, nx]` at the exit wave's stored pixels, `views`
    /// in epoch order. One thread per value of `dst`.
    ///
    /// # Safety
    ///
    /// `dst` and `src` hold their shapes for `g`, `src.nw` `nmode` or `nprobe` unless under
    /// `SLICE_PROBE`, `obj` `[nobj, nyo, nxo]`, `views` the scan's, each window inside the
    /// object; `ctl` one entry.
    #[kernel]
    pub unsafe fn optics_slice_mul(
        dst: *mut f32,
        src: WavePlanes,
        obj: *const f32,
        views: *const ViewAddr,
        ctl: *const StackCtl,
        g: StackGeom,
        flags: u32,
    ) {
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let n = g.ny as usize * g.nx as usize;
        let i = global_index();
        if i >= nf as usize * g.nmode as usize * n {
            return;
        }
        let (kl, j) = (i / n, i % n);
        let (y, x) = (j as u32 / g.nx, j as u32 % g.nx);
        let s = psi_index(y, x, g.ny, g.nx);
        unsafe {
            let v = *views.add(f0 as usize * g.nmode as usize + kl);
            let o = sample_obj(obj, v, y, x, g.nyo, g.nxo);
            let w = if flags & SLICE_PROBE != 0 {
                load_c(src.w, v.probe as usize * n + j)
            } else {
                let nmode = g.nmode as usize;
                load_c(src.w, wave_plane(src, kl / nmode, kl % nmode) * n + s)
            };
            let w = if flags & SLICE_CONJ != 0 {
                cmul_conj(o, w)
            } else {
                cmul(o, w)
            };
            store_c(dst, kl * n + s, w);
        }
    }

    /// The exit plane's phase on the stack's planes of `buf`, `[nf_max * nmode, ny, nx]` at
    /// the exit wave's stored pixels, in place: each value times `exp(i quad |r|^2)`, `r`
    /// the pixel from the frame's centre, and under `SLICE_OFFSET` times `exp(-2i quad r.d)`,
    /// `d` the frame's `(dy, dx)` of `offsets`, `[nframe, 2]` by epoch position: the phase
    /// displaced by `d` but for the frame's constant `exp(i quad |d|^2)`, which no intensity
    /// sees and the conjugate takes back. Under `SLICE_CONJ` by the conjugate of all that.
    /// One thread per value of `buf`.
    ///
    /// # Safety
    ///
    /// `buf` holds its shape for `g`, `offsets` its own under `SLICE_OFFSET`; `ctl` one entry.
    #[kernel]
    pub unsafe fn optics_slice_phase(
        buf: *mut f32,
        offsets: *const f32,
        ctl: *const StackCtl,
        g: StackGeom,
        quad: f32,
        flags: u32,
    ) {
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let n = g.ny as usize * g.nx as usize;
        let i = global_index();
        if i >= nf as usize * g.nmode as usize * n {
            return;
        }
        let (kl, j) = (i / n, i % n);
        let (y, x) = (j as u32 / g.nx, j as u32 % g.nx);
        let mut q = quad_phase(quad, y, x, g.ny, g.nx);
        unsafe {
            if flags & SLICE_OFFSET != 0 {
                let f = f0 as usize + kl / g.nmode as usize;
                let (dy, dx) = (*offsets.add(2 * f), *offsets.add(2 * f + 1));
                // `freq` of the stored index is the centred pixel over the side.
                let c = quad * core::f32::consts::FRAC_1_PI;
                let ky = (y + g.ny / 2) % g.ny;
                let kx = (x + g.nx / 2) % g.nx;
                q = cmul(q, ramp(c * dy * g.ny as f32, ky, g.ny));
                q = cmul(q, ramp(c * dx * g.nx as f32, kx, g.nx));
            }
            let at = kl * n + psi_index(y, x, g.ny, g.nx);
            let w = load_c(buf, at);
            let w = if flags & SLICE_CONJ != 0 {
                cmul_conj(q, w)
            } else {
                cmul(q, w)
            };
            store_c(buf, at, w);
        }
    }

    /// `dst = conj(a) b` over the stack's planes, all `[nf_max * nmode, ny, nx]`. One thread
    /// per value of `dst`.
    ///
    /// # Safety
    ///
    /// The three hold their shape for `g`; `ctl` one entry.
    #[kernel]
    pub unsafe fn optics_slice_pair(
        dst: *mut f32,
        a: *const f32,
        b: *const f32,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        let (_, nf) = unsafe { stack_frames(ctl, g) };
        let i = global_index();
        if i >= nf as usize * g.nmode as usize * g.ny as usize * g.nx as usize {
            return;
        }
        unsafe { store_c(dst, i, cmul_conj(load_c(a, i), load_c(b, i))) };
    }
}
