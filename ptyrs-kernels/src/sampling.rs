//! Full-object sampling (pty-chi `extract_patches_full_object` /
//! `place_patches_full_object`), on the layouts of [`crate::pixel`]: a view's window is its
//! object plane Fourier-interpolated at the view's corner plus its frame's offset, a sample
//! past the plane's edge wrapping.
//!
//! A pass reads `modes` object modes per frame, `nobj` or the first only: its stack's point
//! `q = o * nf_max + fl`, object-mode major, is object mode `o` at frame `f0 + fl`'s position,
//! which every view of the frame shares (`Views` checks it); the views `(f0 + fl) * nmode + o *
//! nprobe + p` read its window, plane `q` of a window buffer. A launch covers the pass `p0 ..
//! p0 + npass` of the points, each in its `[nyo, nxo]` plane of a scratch buffer; the points
//! of frames from `nf` on are skipped.

use cuda_device::{device, thread};
use cuda_host::cuda_module;

use crate::pixel::{StackGeom, ViewAddr, cmul, freq};

/// A pass over the stack's points `p0 .. p0 + npass` of `modes` object modes per frame, read
/// with the derivative of order `ord_y`, `ord_x` over the position.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SamplePass {
    pub p0: u32,
    pub npass: u32,
    pub modes: u32,
    pub ord_y: u32,
    pub ord_x: u32,
}

// SAFETY: repr(C) of five u32, no padding; every bit pattern is a valid value.
unsafe impl cuda_core::DeviceCopy for SamplePass {}

/// The object solve's buffers, at addresses fixed for the solve: `n` complex values each of
/// the object `x`, the residual `res`, the direction `dir` and `ap`, a pass of the normal
/// operator times `1 / scale`; `n` floats of `AᴴA`'s diagonal `den`; a block sum per block
/// `part`; and the scalars `s` at [`PCG_REG`], [`PCG_RZ`] and [`PCG_PAP`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Pcg {
    pub x: *mut f32,
    pub res: *mut f32,
    pub dir: *mut f32,
    pub ap: *mut f32,
    pub den: *const f32,
    pub part: *mut f32,
    pub s: *mut f32,
    pub scale: f32,
    pub n: u32,
}

/// [`Pcg`]'s scalar slots: the regularization, the residual's preconditioned power before
/// alternate steps, and the curvature along the direction; [`PCG_SCALARS`] of them.
pub const PCG_REG: u32 = 0;
pub const PCG_RZ: [u32; 2] = [1, 2];
pub const PCG_PAP: u32 = 3;
pub const PCG_SCALARS: usize = 4;

/// Where a point samples its plane on one axis: whole pixels `c` plus `d` in `[0, 1)`.
#[derive(Clone, Copy)]
struct Axis {
    c: i64,
    d: f32,
}

/// The axis at corner `c` plus `d`, `d`'s whole pixels moved into `c`.
#[device]
fn axis(c: u32, d: f32) -> Axis {
    let i = d as i32;
    let i = i - (i as f32 > d) as i32;
    Axis {
        c: c as i64 + i as i64,
        d: d - i as f32,
    }
}

/// Object mode and position of point `q` of the stack at frame `f0`: view
/// `(f0 + q % nf_max) * nmode + (q / nf_max) * nprobe`'s corner and fraction plus its frame's
/// offset.
///
/// # Safety
///
/// That view and `offsets[2 * frame + 1]` must exist.
#[device]
unsafe fn point(
    views: *const ViewAddr,
    offsets: *const f32,
    f0: u32,
    q: u32,
    g: StackGeom,
) -> (u32, Axis, Axis) {
    let (f, o) = ((f0 + q % g.nf_max) as usize, q / g.nf_max);
    unsafe {
        let v = *views.add(f * g.nmode as usize + (o * g.nprobe) as usize);
        let (dy, dx) = (*offsets.add(2 * f), *offsets.add(2 * f + 1));
        (v.obj, axis(v.oy, v.fy + dy), axis(v.ox, v.fx + dx))
    }
}

/// Cycles of `exp(+2 pi i k t / n)` for DFT index `j` of an axis of `n` (signed bin `k`, the
/// split of [`freq`]) at `t = a.c + a.d`, the whole pixels reduced mod `n` exactly.
#[device]
fn cycles(j: u32, n: u32, a: Axis) -> f32 {
    turns(
        j as i64 - if j < n.div_ceil(2) { 0 } else { n as i64 },
        n,
        a,
    )
}

/// Cycles of `exp(+2 pi i k t / n)` for bin `k` at `t = a.c + a.d`, the whole pixels reduced
/// mod `n` exactly.
#[device]
fn turns(k: i64, n: u32, a: Axis) -> f32 {
    let whole = (k * a.c).rem_euclid(n as i64) as f32;
    (whole + k as f32 * a.d) / n as f32
}

/// The spectrum of `|h_t[-v]|^2` at index `m` of an axis of `n`, `h_t` the one-axis kernel
/// that samples at `t`: `(1 - m/n) exp(-2 pi i t m/n) + (m/n) exp(-2 pi i t (m - n)/n)`, the
/// pairs of bins `k - k'` that are `m` and `m - n`.
#[device]
fn square(m: u32, n: u32, a: Axis) -> (f32, f32) {
    let w = m as f32 / n as f32;
    let (mut s0, mut c0, mut s1, mut c1) = (0.0, 0.0, 0.0, 0.0);
    // SAFETY: libdevice function of one f32, writing only the two locals.
    unsafe {
        __nv_sincospif(2.0 * turns(m as i64, n, a), &mut s0, &mut c0);
        __nv_sincospif(2.0 * turns(m as i64 - n as i64, n, a), &mut s1, &mut c1);
    }
    ((1.0 - w) * c0 + w * c1, -((1.0 - w) * s0 + w * s1))
}

/// The preconditioner of the object solve at a pixel of `AᴴA`'s diagonal `d`: `max(d + reg,
/// 1e-12)`.
#[device]
fn pre(d: f32, reg: f32) -> f32 {
    let v = d + reg;
    if v > 1e-12 { v } else { 1e-12 }
}

/// The block's sum of each thread's `v`, which thread 0 writes to `*out`. Every thread of the
/// block must call it.
///
/// # Safety
///
/// `s` holds `PIXEL_THREADS` shared floats, `out` one float; block `(PIXEL_THREADS, 1, 1)`.
#[device]
unsafe fn store_sum(s: *mut f32, v: f32, out: *mut f32) {
    let t = thread::threadIdx_x() as usize;
    unsafe {
        *s.add(t) = v;
        crate::block_sum::<1>(s);
        if t == 0 {
            *out = *s;
        }
    }
}

/// The sampling phase of spectrum index `j` of an `[nyo, nxo]` plane at `(y, x)`, times
/// `scale` and differentiated `ord_y`, `ord_x` times over the position: `scale exp(+2 pi i
/// f.t) (2 pi i f_y)^ord_y (2 pi i f_x)^ord_x`.
#[device]
fn phase(j: u32, nyo: u32, nxo: u32, y: Axis, x: Axis, s: SamplePass, scale: f32) -> (f32, f32) {
    let (jy, jx) = (j / nxo, j % nxo);
    let (mut sin, mut cos) = (0.0, 0.0);
    // SAFETY: libdevice function of one f32, writing only the two locals.
    unsafe {
        __nv_sincospif(
            2.0 * (cycles(jy, nyo, y) + cycles(jx, nxo, x)),
            &mut sin,
            &mut cos,
        )
    };
    let mut e = (scale * cos, scale * sin);
    let tau = 2.0 * core::f32::consts::PI;
    for _ in 0..s.ord_y {
        e = cmul(e, (0.0, tau * freq(jy, nyo)));
    }
    for _ in 0..s.ord_x {
        e = cmul(e, (0.0, tau * freq(jx, nxo)));
    }
    e
}

// libdevice's `sincospif` reduces its argument exactly. `pixel` declares `cospif` and
// `sinpif`; the device build rejects a second declaration of a symbol.
#[device]
unsafe extern "C" {
    fn __nv_sincospif(x: f32, sin: *mut f32, cos: *mut f32);
}

#[cuda_module]
pub mod sampling_kernels {
    use super::{PCG_PAP, PCG_REG, Pcg, SamplePass, phase, point, pre, square, store_sum};
    use crate::pixel::*;
    use crate::*;
    use cuda_device::{SharedArray, thread};

    /// `scratch[i] = spectra[o] * phase(q) / (nyo * nxo)` for every pass point `q = p0 + i`
    /// of a frame below `nf`, `o` its object mode, `phase` its sampling phase of the pass's
    /// derivative order: an unnormalized inverse FFT of `scratch` leaves each point's plane
    /// interpolated at its position. One thread per element of `[npass, nyo, nxo]`.
    ///
    /// # Safety
    ///
    /// `spectra` `[nobj, nyo, nxo]`, `views` `nframe * nmode` records with `obj < nobj`,
    /// `offsets` `[nframe, 2]`, `scratch` `[npass, nyo, nxo]`, `ctl` one entry.
    #[kernel]
    pub unsafe fn sample_shift(
        spectra: *const f32,
        views: *const ViewAddr,
        offsets: *const f32,
        scratch: *mut f32,
        ctl: *const StackCtl,
        g: StackGeom,
        s: SamplePass,
    ) {
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let n = g.nyo as usize * g.nxo as usize;
        let t = global_index();
        if t >= s.npass as usize * n {
            return;
        }
        let (i, j) = (t / n, t % n);
        let q = s.p0 + i as u32;
        if q % g.nf_max >= nf {
            return;
        }
        unsafe {
            let (o, y, x) = point(views, offsets, f0, q, g);
            let e = phase(j as u32, g.nyo, g.nxo, y, x, s, 1.0 / n as f32);
            store_c(scratch, t, cmul(load_c(spectra, o as usize * n + j), e));
        }
    }

    /// The pass's windows out of the inverse-FFT'd `scratch`: `win[q, y, x] = scratch[i, y,
    /// x]` for pass point `q = p0 + i` of a frame below `nf`, `(y, x)` in `[ny, nx]`. One
    /// thread per element of `[npass, ny, nx]`.
    ///
    /// # Safety
    ///
    /// `scratch` `[npass, nyo, nxo]`, `win` `[modes * nf_max, ny, nx]`, `ctl` one entry.
    #[kernel]
    pub unsafe fn sample_crop(
        scratch: *const f32,
        win: *mut f32,
        ctl: *const StackCtl,
        g: StackGeom,
        s: SamplePass,
    ) {
        let (_, nf) = unsafe { stack_frames(ctl, g) };
        let nw = g.ny as usize * g.nx as usize;
        let t = global_index();
        if t >= s.npass as usize * nw {
            return;
        }
        let (i, w) = (t / nw, t % nw);
        let q = s.p0 + i as u32;
        if q % g.nf_max >= nf {
            return;
        }
        let (y, x) = (w / g.nx as usize, w % g.nx as usize);
        let at = i * g.nyo as usize * g.nxo as usize + y * g.nxo as usize + x;
        unsafe { store_c(win, q as usize * nw + w, load_c(scratch, at)) };
    }

    /// Each pass point's patch embedded at its plane's origin: `scratch[i, y, x] = patches[q,
    /// y, x]` over `(y, x)` in `[ny, nx]`, else 0, for pass point `q = p0 + i`, `modes` all
    /// `nobj`; 0 for a point of a frame from `nf` on. One thread per element of `[npass, nyo,
    /// nxo]`.
    ///
    /// # Safety
    ///
    /// `patches` `[nobj * nf_max, ny, nx]`, `scratch` `[npass, nyo, nxo]`, `ctl` one entry.
    #[kernel]
    pub unsafe fn place_embed(
        patches: *const f32,
        scratch: *mut f32,
        ctl: *const StackCtl,
        g: StackGeom,
        s: SamplePass,
    ) {
        let (_, nf) = unsafe { stack_frames(ctl, g) };
        let n = g.nyo as usize * g.nxo as usize;
        let t = global_index();
        if t >= s.npass as usize * n {
            return;
        }
        let (i, j) = (t / n, t % n);
        let q = s.p0 + i as u32;
        let (y, x) = ((j / g.nxo as usize) as u32, (j % g.nxo as usize) as u32);
        let mut v = (0.0, 0.0);
        if q % g.nf_max < nf && y < g.ny && x < g.nx {
            let nw = g.ny as usize * g.nx as usize;
            v = unsafe { load_c(patches, q as usize * nw + (y * g.nx + x) as usize) };
        }
        unsafe { store_c(scratch, t, v) };
    }

    /// `acc[o] += conj(phase(q)) * scratch[i]` over the pass points `q = p0 + i` of object
    /// mode `o` and a frame below `nf`, `modes` all `nobj`, in ascending `q`: the forward-FFT'd
    /// embedded patches moved back by each point's position. One thread per element of
    /// `[nobj, nyo, nxo]`.
    ///
    /// # Safety
    ///
    /// As [`sample_shift`], `scratch` the forward-FFT'd [`place_embed`] and `acc` `[nobj,
    /// nyo, nxo]`.
    ///
    /// [`sample_shift`]: LoadedModule::sample_shift
    /// [`place_embed`]: LoadedModule::place_embed
    #[kernel]
    pub unsafe fn place_accumulate(
        scratch: *const f32,
        views: *const ViewAddr,
        offsets: *const f32,
        acc: *mut f32,
        ctl: *const StackCtl,
        g: StackGeom,
        s: SamplePass,
    ) {
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let n = g.nyo as usize * g.nxo as usize;
        let t = global_index();
        if t >= s.modes as usize * n {
            return;
        }
        let (o, j) = ((t / n) as u32, t % n);
        let first = o * g.nf_max;
        let end = (s.p0 + s.npass).min(first + nf);
        let mut a = unsafe { load_c(acc, t) };
        for q in s.p0.max(first)..end {
            unsafe {
                let (_, y, x) = point(views, offsets, f0, q, g);
                let e = phase(j as u32, g.nyo, g.nxo, y, x, s, 1.0);
                let v = cmul_conj(e, load_c(scratch, (q - s.p0) as usize * n + j));
                a = (a.0 + v.0, a.1 + v.1);
            }
        }
        unsafe { store_c(acc, t, a) };
    }

    /// `obj[t] += acc[t] * scale` for `t < len`.
    ///
    /// # Safety
    ///
    /// `obj` and `acc` hold `len` complex values.
    #[kernel]
    pub unsafe fn place_add(obj: *mut f32, acc: *const f32, len: u32, scale: f32) {
        let t = global_index();
        if t >= len as usize {
            return;
        }
        unsafe {
            let (o, a) = (load_c(obj, t), load_c(acc, t));
            store_c(obj, t, (o.0 + a.0 * scale, o.1 + a.1 * scale));
        }
    }

    /// `acc[o, y, x] = sum_p |probe[p, y, x]|^2` over `(y, x)` in `[ny, nx]`, else 0, for
    /// every object mode `o`. One thread per element of `[nobj, nyo, nxo]`.
    ///
    /// # Safety
    ///
    /// `probe` `[nprobe, ny, nx]`, `acc` `[nobj, nyo, nxo]`.
    #[kernel]
    pub unsafe fn illum_embed(probe: *const f32, acc: *mut f32, g: StackGeom, nobj: u32) {
        let n = g.nyo as usize * g.nxo as usize;
        let t = global_index();
        if t >= nobj as usize * n {
            return;
        }
        let j = t % n;
        let (y, x) = ((j / g.nxo as usize) as u32, (j % g.nxo as usize) as u32);
        let mut s = 0.0;
        if y < g.ny && x < g.nx {
            let nw = g.ny as usize * g.nx as usize;
            for p in 0..g.nprobe as usize {
                let v = unsafe { load_c(probe, p * nw + (y * g.nx + x) as usize) };
                s += v.0 * v.0 + v.1 * v.1;
            }
        }
        unsafe { store_c(acc, t, (s, 0.0)) };
    }

    /// `acc[o, j] *= sum_f conj(phase_f(j))` for every object mode `o`, `phase_f` the
    /// sampling phase of frame `f`'s position over the whole scan, in ascending `f`. One
    /// thread per element of `[nyo, nxo]`.
    ///
    /// # Safety
    ///
    /// `g` the scan's geometry, `nf_max == nframe`; `views` `nframe * nmode` records, `offsets`
    /// `[nframe, 2]`, `acc` `[nobj, nyo, nxo]`.
    #[kernel]
    pub unsafe fn illum_phase(
        views: *const ViewAddr,
        offsets: *const f32,
        acc: *mut f32,
        g: StackGeom,
        nobj: u32,
    ) {
        let n = g.nyo as usize * g.nxo as usize;
        let t = global_index();
        if t >= n {
            return;
        }
        let s = SamplePass {
            p0: 0,
            npass: 0,
            modes: nobj,
            ord_y: 0,
            ord_x: 0,
        };
        let mut sum = (0.0, 0.0);
        for f in 0..g.nframe {
            unsafe {
                let (_, y, x) = point(views, offsets, 0, f, g);
                let e = phase(t as u32, g.nyo, g.nxo, y, x, s, 1.0);
                sum = (sum.0 + e.0, sum.1 - e.1);
            }
        }
        for o in 0..nobj as usize {
            unsafe {
                let v = load_c(acc, o * n + t);
                store_c(acc, o * n + t, cmul(v, sum));
            }
        }
    }

    /// `out[t] = max(re(acc[t]) * scale, 0)` for `t < len`.
    ///
    /// # Safety
    ///
    /// `acc` holds `len` complex values, `out` `len` floats.
    #[kernel]
    pub unsafe fn illum_clamp(acc: *const f32, out: *mut f32, len: u32, scale: f32) {
        let t = global_index();
        if t >= len as usize {
            return;
        }
        let v = unsafe { load_c(acc, t).0 } * scale;
        unsafe { *out.add(t) = if v > 0.0 { v } else { 0.0 } };
    }

    /// `acc[o, j] *= sum_f M_f(j)` for every object mode `o`, `M_f` the spectrum of
    /// `|h_t[-v]|^2` of frame `f`'s position `t` over the whole scan, the product of its axes'
    /// [`square`], in ascending `f`: with `acc` the transformed [`illum_embed`], an inverse
    /// transform leaves the diagonal of `AᴴA`. One thread per element of `[nyo, nxo]`.
    ///
    /// # Safety
    ///
    /// As [`illum_phase`].
    ///
    /// [`illum_embed`]: LoadedModule::illum_embed
    /// [`illum_phase`]: LoadedModule::illum_phase
    #[kernel]
    pub unsafe fn diag_phase(
        views: *const ViewAddr,
        offsets: *const f32,
        acc: *mut f32,
        g: StackGeom,
        nobj: u32,
    ) {
        let n = g.nyo as usize * g.nxo as usize;
        let t = global_index();
        if t >= n {
            return;
        }
        let (jy, jx) = ((t / g.nxo as usize) as u32, (t % g.nxo as usize) as u32);
        let mut sum = (0.0, 0.0);
        for f in 0..g.nframe {
            unsafe {
                let (_, y, x) = point(views, offsets, 0, f, g);
                let e = cmul(square(jy, g.nyo, y), square(jx, g.nxo, x));
                sum = (sum.0 + e.0, sum.1 + e.1);
            }
        }
        for o in 0..nobj as usize {
            unsafe {
                let v = load_c(acc, o * n + t);
                store_c(acc, o * n + t, cmul(v, sum));
            }
        }
    }

    /// The pass's planes interpolated in `scratch` weighted on their windows by the probe's
    /// power and zeroed past them: `scratch[i, y, x] *= sum_p |probe[p, y, x]|^2` over `(y, x)`
    /// in `[ny, nx]`, else 0, `A`'s windows then `Aᴴ`'s patches embedded at the origin. One
    /// thread per element of `[npass, nyo, nxo]`.
    ///
    /// # Safety
    ///
    /// `probe` `[nprobe, ny, nx]`, `scratch` `[npass, nyo, nxo]`.
    #[kernel]
    pub unsafe fn normal_mask(probe: *const f32, scratch: *mut f32, g: StackGeom, s: SamplePass) {
        let n = g.nyo as usize * g.nxo as usize;
        let t = global_index();
        if t >= s.npass as usize * n {
            return;
        }
        let j = t % n;
        let (y, x) = ((j / g.nxo as usize) as u32, (j % g.nxo as usize) as u32);
        let mut v = (0.0, 0.0);
        if y < g.ny && x < g.nx {
            let nw = g.ny as usize * g.nx as usize;
            let mut w = 0.0;
            for p in 0..g.nprobe as usize {
                let q = unsafe { load_c(probe, p * nw + (y * g.nx + x) as usize) };
                w += q.0 * q.0 + q.1 * q.1;
            }
            let a = unsafe { load_c(scratch, t) };
            v = (a.0 * w, a.1 * w);
        }
        unsafe { store_c(scratch, t, v) };
    }

    /// `s[PCG_REG] = inertia * f32::from_bits(*max_bits)`, the object solve's regularization
    /// from the bits of `max(D)`. One thread.
    ///
    /// # Safety
    ///
    /// `max_bits` one word, `pcg` as [`Pcg`] says.
    #[kernel]
    pub unsafe fn pcg_reg(max_bits: *const u32, inertia: f32, pcg: Pcg) {
        if global_index() == 0 {
            unsafe { *pcg.s.add(PCG_REG as usize) = inertia * f32::from_bits(*max_bits) };
        }
    }

    /// The object solve's start from `ap = AᴴA x / scale`: `res = res - scale ap`, `dir = res
    /// / pre`, `pre` [`pre`] of `den` and the regularization, and block `b`'s `sum |res|^2 /
    /// pre` over the elements `i ≡ b * PIXEL_THREADS + t (mod gridDim * PIXEL_THREADS)` into
    /// `part[b]`.
    ///
    /// # Safety
    ///
    /// `pcg` as [`Pcg`] says, `part` `gridDim.x` floats; block `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn pcg_start(pcg: Pcg) {
        static mut RED: SharedArray<f32, 256> = SharedArray::UNINIT;
        let stride = thread::gridDim_x() as usize * PIXEL_THREADS as usize;
        let reg = unsafe { *pcg.s.add(PCG_REG as usize) };
        let mut sum = 0.0;
        let mut i = global_index();
        while i < pcg.n as usize {
            unsafe {
                let (b, a) = (load_c(pcg.res, i), load_c(pcg.ap, i));
                let r = (b.0 - a.0 * pcg.scale, b.1 - a.1 * pcg.scale);
                let m = pre(*pcg.den.add(i), reg);
                store_c(pcg.res, i, r);
                store_c(pcg.dir, i, (r.0 / m, r.1 / m));
                sum += (r.0 * r.0 + r.1 * r.1) / m;
            }
            i += stride;
        }
        let out = unsafe { pcg.part.add(thread::blockIdx_x() as usize) };
        unsafe { store_sum((&raw mut RED).cast::<f32>(), sum, out) }
    }

    /// `*out = sum part[0 .. npart]`, in a fixed order.
    ///
    /// # Safety
    ///
    /// `part` holds `npart` floats, `out` one; grid `(1, 1, 1)`, block `(PIXEL_THREADS, 1,
    /// 1)`.
    #[kernel]
    pub unsafe fn pcg_sum(part: *const f32, npart: u32, out: *mut f32) {
        static mut RED: SharedArray<f32, 256> = SharedArray::UNINIT;
        let mut sum = 0.0;
        let mut b = thread::threadIdx_x() as usize;
        while b < npart as usize {
            sum += unsafe { *part.add(b) };
            b += PIXEL_THREADS as usize;
        }
        unsafe { store_sum((&raw mut RED).cast::<f32>(), sum, out) }
    }

    /// The curvature along the object solve's direction from `ap = AᴴA dir / scale`: `ap =
    /// scale ap + reg dir`, and block `b`'s `sum Re(conj(dir) ap)` into `part[b]`, over the
    /// elements of [`pcg_start`].
    ///
    /// # Safety
    ///
    /// As [`pcg_start`].
    ///
    /// [`pcg_start`]: LoadedModule::pcg_start
    #[kernel]
    pub unsafe fn pcg_curv(pcg: Pcg) {
        static mut RED: SharedArray<f32, 256> = SharedArray::UNINIT;
        let stride = thread::gridDim_x() as usize * PIXEL_THREADS as usize;
        let reg = unsafe { *pcg.s.add(PCG_REG as usize) };
        let mut sum = 0.0;
        let mut i = global_index();
        while i < pcg.n as usize {
            unsafe {
                let (a, p) = (load_c(pcg.ap, i), load_c(pcg.dir, i));
                let v = (a.0 * pcg.scale + reg * p.0, a.1 * pcg.scale + reg * p.1);
                store_c(pcg.ap, i, v);
                sum += p.0 * v.0 + p.1 * v.1;
            }
            i += stride;
        }
        let out = unsafe { pcg.part.add(thread::blockIdx_x() as usize) };
        unsafe { store_sum((&raw mut RED).cast::<f32>(), sum, out) }
    }

    /// The object solve's step along `dir`: `alpha = s[rz] / s[PCG_PAP]` if the curvature is
    /// positive, else 0; `x += alpha dir`, `res -= alpha ap`, and block `b`'s `sum |res|^2 /
    /// pre` into `part[b]`, over the elements of [`pcg_start`].
    ///
    /// # Safety
    ///
    /// As [`pcg_start`], `rz` a slot of [`PCG_RZ`](super::PCG_RZ).
    ///
    /// [`pcg_start`]: LoadedModule::pcg_start
    #[kernel]
    pub unsafe fn pcg_step(pcg: Pcg, rz: u32) {
        static mut RED: SharedArray<f32, 256> = SharedArray::UNINIT;
        let stride = thread::gridDim_x() as usize * PIXEL_THREADS as usize;
        let at = |slot: u32| pcg.s.wrapping_add(slot as usize);
        let (reg, rz, pap) = unsafe { (*at(PCG_REG), *at(rz), *at(PCG_PAP)) };
        let alpha = if pap > 0.0 { rz / pap } else { 0.0 };
        let mut sum = 0.0;
        let mut i = global_index();
        while i < pcg.n as usize {
            unsafe {
                let (v, p) = (load_c(pcg.x, i), load_c(pcg.dir, i));
                let (a, r) = (load_c(pcg.ap, i), load_c(pcg.res, i));
                store_c(pcg.x, i, (v.0 + alpha * p.0, v.1 + alpha * p.1));
                let r = (r.0 - alpha * a.0, r.1 - alpha * a.1);
                store_c(pcg.res, i, r);
                sum += (r.0 * r.0 + r.1 * r.1) / pre(*pcg.den.add(i), reg);
            }
            i += stride;
        }
        let out = unsafe { pcg.part.add(thread::blockIdx_x() as usize) };
        unsafe { store_sum((&raw mut RED).cast::<f32>(), sum, out) }
    }

    /// The object solve's next direction: `beta = s[next] / s[rz]` if `s[rz] > 0`, else 0;
    /// `dir = res / pre + beta dir`, over the elements of [`pcg_start`].
    ///
    /// # Safety
    ///
    /// As [`pcg_start`], `rz` and `next` slots of [`PCG_RZ`](super::PCG_RZ).
    ///
    /// [`pcg_start`]: LoadedModule::pcg_start
    #[kernel]
    pub unsafe fn pcg_dir(pcg: Pcg, rz: u32, next: u32) {
        let stride = thread::gridDim_x() as usize * PIXEL_THREADS as usize;
        let at = |slot: u32| pcg.s.wrapping_add(slot as usize);
        let (reg, rz, next) = unsafe { (*at(PCG_REG), *at(rz), *at(next)) };
        let beta = if rz > 0.0 { next / rz } else { 0.0 };
        let mut i = global_index();
        while i < pcg.n as usize {
            unsafe {
                let (r, p) = (load_c(pcg.res, i), load_c(pcg.dir, i));
                let m = pre(*pcg.den.add(i), reg);
                store_c(pcg.dir, i, (r.0 / m + beta * p.0, r.1 / m + beta * p.1));
            }
            i += stride;
        }
    }
}
