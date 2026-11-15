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

use cuda_device::device;
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
    let k = j as i64 - if j < n.div_ceil(2) { 0 } else { n as i64 };
    let whole = (k * a.c).rem_euclid(n as i64) as f32;
    (whole + k as f32 * a.d) / n as f32
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
    use super::{SamplePass, phase, point};
    use crate::pixel::*;
    use crate::*;

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
}
