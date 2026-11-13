//! pty-chi's gradient position correction on the far-field Fourier path, shared by the
//! methods that model [`crate::fused`]'s moved probe: the moved probe `W` at the exit wave's
//! stored pixels, each frame's object slopes at its first view, the per-frame position sums
//! of an exit-wave update `chi` against them, the clipped step of a batch's offsets, and the
//! offsets gathered by epoch position. Frames: as in [`crate::bh`], a stack's frame `fl` is
//! epoch position `f0 + fl`; its views are at that position of `views`.

use cuda_device::{device, shared::DynamicSharedArray, thread};
use cuda_host::cuda_module;

use crate::fused::{FUSED_COLS, col_fft};
use crate::pixel::{StackCtl, StackGeom, ViewAddr, freq, load_c, stack_frames, store_c};
use crate::{FFT_EPT_MAX, fft_ept};

/// Floats [`position_kernels::LoadedModule::position_sums`] writes per frame and chunk:
/// `Σ Re(conj(u_y) chi)`, `Σ Re(conj(u_x) chi)`, `Σ |u_y|²`, `Σ |u_x|²`.
pub const POSITION_SUMS: u32 = 4;

/// pty-chi's floor on the position direction's denominator, per pixel.
const EPS_POSITION: f32 = 1e-6;

/// A batch: epoch positions `p0 .. p0 + nb`, each with `nchunk` reduction chunks.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BatchSpan {
    pub p0: u32,
    pub nb: u32,
    pub nchunk: u32,
}

/// pty-chi's position step: `dx = step Δ`, clamped to `±min(10 mad, limit)`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PositionRule {
    pub step: f32,
    pub limit: f32,
}

/// What [`position_kernels::LoadedModule::position_sums`] reads and writes: `chi` the
/// stack's exit-wave update `[nf_max * nmode, ny, nx]` and `w` its moved probe `[nf_max *
/// nprobe, ny, nx]`, both at the exit wave's stored pixels, `slopes` its frames' object
/// slopes `[nf_max, 2, ny, nx]` ([`slopes`]), the epoch's `views`, and `sums` the
/// [`POSITION_SUMS`] of every frame's chunks by epoch position.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PositionSums {
    pub chi: *const f32,
    pub w: *const f32,
    pub slopes: *const f32,
    pub views: *const ViewAddr,
    pub sums: *mut f32,
}

/// The stack's shifted probe from `T` of `Kernels::shifted_rows_fwd`, in place: the
/// unnormalized row IDFT of rows `ys = blockIdx.x * FUSED_COLS + b` of plane `blockIdx.y` (`fl
/// * nprobe + p`, for `fl < nf`), `W_p` at its stored pixels. Thread `b * tpf + f` (`tpf =
/// NX / fft_ept(NX)`); grid `(ny / FUSED_COLS, nf_max * nprobe)`, block `FUSED_COLS * tpf`,
/// dynamic shared memory `FUSED_COLS` `NX`-point FFTs ([`crate::fused::FFT_SKEW`]).
///
/// # Safety
///
/// `NX == g.nx`; `t` holds `[nf_max * nprobe, ny, nx]`; `ctl` one entry; launch as above.
#[device]
pub unsafe fn shift_w<const NX: u32>(t: *mut f32, ctl: *const StackCtl, g: StackGeom) {
    let (_, nf) = unsafe { stack_frames(ctl, g) };
    let plane = thread::blockIdx_y();
    if plane >= nf * g.nprobe {
        return;
    }
    let ept = const { fft_ept(NX) };
    let tpf = NX / ept;
    let tid = thread::threadIdx_x();
    let (b, f) = (tid / tpf, tid % tpf);
    let ys = thread::blockIdx_x() * FUSED_COLS + b;
    let row = (plane as usize * g.ny as usize + ys as usize) * NX as usize;
    let mut r = [0.0f32; 2 * FFT_EPT_MAX as usize];
    unsafe {
        let smem = DynamicSharedArray::<u8, 16>::get();
        #[unroll]
        for i in 0..ept {
            let v = load_c(t, row + (f + i * tpf) as usize);
            r[2 * i as usize] = v.0;
            r[2 * i as usize + 1] = v.1;
        }
        col_fft::<NX, true>(&mut r, smem, b, f, FUSED_COLS);
        #[unroll]
        for i in 0..ept {
            let k = 2 * i as usize;
            store_c(t, row + (f + i * tpf) as usize, (r[k], r[k + 1]));
        }
    }
}

/// pty-chi's object slope along `axis` (0 `y`, 1 `x`) of each frame `fl = blockIdx.y` of the
/// stack: with `v` the frame's first view, the patch `O[v.obj, v.oy + y, v.ox + x]` (`fy` and
/// `fx` not read), Fourier-differentiated as periodic, `IDFT(DFT(patch) i 2π freq(k, N)) /
/// N` along the axis, into `out[fl, axis]` at natural `(y, x)`. Line `l = blockIdx.x *
/// FUSED_COLS + b` across the axis; thread `f * FUSED_COLS + b` along `y`, `b * tpf + f`
/// along `x` (`tpf = N / fft_ept(N)`), so either reads its rows in order. Grid `(m /
/// FUSED_COLS, nf_max)`, `m` the side across the axis, block `FUSED_COLS * tpf`, dynamic shared
/// memory `FUSED_COLS` `N`-point FFTs ([`crate::fused::FFT_SKEW`]).
///
/// # Safety
///
/// `N` is `g.ny` along `y`, `g.nx` along `x`; `obj` holds `[nobj, nyo, nxo]` with every
/// frame's first view's window inside, `views` the epoch's views, `out` `[nf_max, 2, ny,
/// nx]`; `ctl` one entry; launch as above.
#[device]
pub unsafe fn slopes<const N: u32>(
    obj: *const f32,
    views: *const ViewAddr,
    out: *mut f32,
    axis: u32,
    ctl: *const StackCtl,
    g: StackGeom,
) {
    let (f0, nf) = unsafe { stack_frames(ctl, g) };
    let fl = thread::blockIdx_y();
    if fl >= nf {
        return;
    }
    let ept = const { fft_ept(N) };
    let tpf = N / ept;
    let tid = thread::threadIdx_x();
    let (b, f) = if axis == 0 {
        (tid % FUSED_COLS, tid / FUSED_COLS)
    } else {
        (tid / tpf, tid % tpf)
    };
    let l = thread::blockIdx_x() * FUSED_COLS + b;
    let at = |e: u32| if axis == 0 { (e, l) } else { (l, e) };
    let n = g.ny as usize * g.nx as usize;
    let dst = (2 * fl + axis) as usize * n;
    let tau = 2.0 * core::f32::consts::PI / N as f32;
    let mut r = [0.0f32; 2 * FFT_EPT_MAX as usize];
    unsafe {
        let v = *views.add((f0 + fl) as usize * g.nmode as usize);
        let src =
            (v.obj as usize * g.nyo as usize + v.oy as usize) * g.nxo as usize + v.ox as usize;
        #[unroll]
        for i in 0..ept {
            let (y, x) = at(f + i * tpf);
            let o = load_c(obj, src + y as usize * g.nxo as usize + x as usize);
            r[2 * i as usize] = o.0;
            r[2 * i as usize + 1] = o.1;
        }
        let smem = DynamicSharedArray::<u8, 16>::get();
        col_fft::<N, false>(&mut r, smem, b, f, FUSED_COLS);
        #[unroll]
        for i in 0..ept {
            let w = tau * freq(f + i * tpf, N);
            let (re, im) = (r[2 * i as usize], r[2 * i as usize + 1]);
            r[2 * i as usize] = -im * w;
            r[2 * i as usize + 1] = re * w;
        }
        thread::sync_threads();
        col_fft::<N, true>(&mut r, smem, b, f, FUSED_COLS);
        #[unroll]
        for i in 0..ept {
            let (y, x) = at(f + i * tpf);
            let k = 2 * i as usize;
            store_c(out, dst + (y * g.nx + x) as usize, (r[k], r[k + 1]));
        }
    }
}

#[cuda_module]
pub mod position_kernels {
    use super::{
        BatchSpan, EPS_POSITION, POSITION_SUMS, PositionRule, PositionSums, shift_w, slopes,
    };
    use crate::bh::store_sums;
    use crate::grad::{GAMMA_CHUNK, block_max};
    use crate::pixel::*;
    use crate::*;
    use cuda_device::SharedArray;

    /// [`super::shift_w`] for 64-point rows.
    ///
    /// # Safety
    ///
    /// As [`super::shift_w`].
    #[kernel]
    pub unsafe fn position_shift_w_64(t: *mut f32, ctl: *const StackCtl, g: StackGeom) {
        unsafe { shift_w::<64>(t, ctl, g) }
    }

    /// [`super::shift_w`] for 128-point rows.
    ///
    /// # Safety
    ///
    /// As [`super::shift_w`].
    #[kernel]
    pub unsafe fn position_shift_w_128(t: *mut f32, ctl: *const StackCtl, g: StackGeom) {
        unsafe { shift_w::<128>(t, ctl, g) }
    }

    /// [`super::shift_w`] for 256-point rows.
    ///
    /// # Safety
    ///
    /// As [`super::shift_w`].
    #[kernel]
    pub unsafe fn position_shift_w_256(t: *mut f32, ctl: *const StackCtl, g: StackGeom) {
        unsafe { shift_w::<256>(t, ctl, g) }
    }

    /// [`super::shift_w`] for 512-point rows.
    ///
    /// # Safety
    ///
    /// As [`super::shift_w`].
    #[kernel]
    pub unsafe fn position_shift_w_512(t: *mut f32, ctl: *const StackCtl, g: StackGeom) {
        unsafe { shift_w::<512>(t, ctl, g) }
    }

    /// [`super::slopes`] for 64-point lines.
    ///
    /// # Safety
    ///
    /// As [`super::slopes`].
    #[kernel]
    pub unsafe fn position_slopes_64(
        obj: *const f32,
        views: *const ViewAddr,
        out: *mut f32,
        axis: u32,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        unsafe { slopes::<64>(obj, views, out, axis, ctl, g) }
    }

    /// [`super::slopes`] for 128-point lines.
    ///
    /// # Safety
    ///
    /// As [`super::slopes`].
    #[kernel]
    pub unsafe fn position_slopes_128(
        obj: *const f32,
        views: *const ViewAddr,
        out: *mut f32,
        axis: u32,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        unsafe { slopes::<128>(obj, views, out, axis, ctl, g) }
    }

    /// [`super::slopes`] for 256-point lines.
    ///
    /// # Safety
    ///
    /// As [`super::slopes`].
    #[kernel]
    pub unsafe fn position_slopes_256(
        obj: *const f32,
        views: *const ViewAddr,
        out: *mut f32,
        axis: u32,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        unsafe { slopes::<256>(obj, views, out, axis, ctl, g) }
    }

    /// [`super::slopes`] for 512-point lines.
    ///
    /// # Safety
    ///
    /// As [`super::slopes`].
    #[kernel]
    pub unsafe fn position_slopes_512(
        obj: *const f32,
        views: *const ViewAddr,
        out: *mut f32,
        axis: u32,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        unsafe { slopes::<512>(obj, views, out, axis, ctl, g) }
    }

    /// The position sums of chunk `c = blockIdx.x` of frame `f0 + fl`, `fl = blockIdx.y`,
    /// over the pixels `i` in `c * GAMMA_CHUNK .. min((c + 1) * GAMMA_CHUNK, n)` of its first
    /// view `v` (mode 0): with `u_ax = slope_ax W_v` (the slope at `i`, `W` and `chi` at the
    /// exit-wave index of `i`), `sums[4p .. 4p + 4]` gets the [`POSITION_SUMS`], `p = (f0 +
    /// fl) * nchunk + c`.
    ///
    /// # Safety
    ///
    /// `p` as [`PositionSums`] says for this stack, `views` `nframe * nmode` records, `sums`
    /// `4 * nframe * nchunk` floats, `ctl` one entry; grid `(nchunk, nf_max, 1)`, block
    /// `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn position_sums(p: PositionSums, ctl: *const StackCtl, g: StackGeom) {
        static mut RED: SharedArray<f32, 1024> = SharedArray::UNINIT;
        let PositionSums {
            chi,
            w,
            slopes,
            views,
            sums,
        } = p;
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let (ny, nx, nmode) = (g.ny, g.nx, g.nmode as usize);
        let n = ny as usize * nx as usize;
        let fl = thread::blockIdx_y() as usize;
        // The whole block leaves together, before its first barrier.
        if fl >= nf as usize {
            return;
        }
        let c = thread::blockIdx_x() as usize;
        let chunk = GAMMA_CHUNK as usize;
        let end = n.min((c + 1) * chunk);
        let pos = f0 as usize + fl;
        let v = unsafe { *views.add(pos * nmode) };
        let mut acc = [0.0f32; POSITION_SUMS as usize];
        let mut i = c * chunk + thread::threadIdx_x() as usize;
        while i < end {
            let (py, px) = ((i / nx as usize) as u32, (i % nx as usize) as u32);
            let at = psi_index(py, px, ny, nx);
            unsafe {
                let ws = load_c(w, (fl * g.nprobe as usize + v.probe as usize) * n + at);
                let ch = load_c(chi, fl * nmode * n + at);
                // Unrolled so `acc` stays in registers.
                #[unroll]
                for ax in 0..2 {
                    let s = load_c(slopes, (2 * fl + ax) * n + i);
                    let gu = cmul(s, ws);
                    acc[ax] += cmul_conj(gu, ch).0;
                    acc[2 + ax] += gu.0 * gu.0 + gu.1 * gu.1;
                }
            }
            i += PIXEL_THREADS as usize;
        }
        let at = pos * n.div_ceil(chunk) + c;
        unsafe {
            store_sums::<4>(
                (&raw mut RED).cast::<f32>(),
                acc,
                sums.add(POSITION_SUMS as usize * at),
            );
        }
    }

    /// pty-chi's position step on the batch's positions `p` from the [`position_sums`],
    /// each frame's chunks summed in order (and left in its chunk 0): per axis `Δ = num /
    /// (den + max(1e-6 ny nx, max_batch den))` and `dx = rule.step Δ`, 0 at the scan's other
    /// frames, then `offsets[order[p]] += clamp(dx, ±min(10 mean |dx − mean dx|,
    /// rule.limit))`, both means over the scan's `nframe` frames.
    ///
    /// # Safety
    ///
    /// `sums` holds `4 * span.nchunk` floats per position, `order` the epoch's `nframe`
    /// frames, `offsets` `[nframe, 2]` by frame; grid `(1, 1, 1)`, block `(PIXEL_THREADS, 1,
    /// 1)`.
    ///
    /// [`position_sums`]: LoadedModule::position_sums
    #[kernel]
    pub unsafe fn position_step(
        sums: *mut f32,
        span: BatchSpan,
        order: *const u32,
        offsets: *mut f32,
        nframe: u32,
        rule: PositionRule,
        g: StackGeom,
    ) {
        static mut RED: SharedArray<f32, 1024> = SharedArray::UNINIT;
        let PositionRule { step, limit } = rule;
        let t = thread::threadIdx_x() as usize;
        let seg = PIXEL_THREADS as usize;
        let k = POSITION_SUMS as usize;
        let (p0, len, nchunk) = (span.p0 as usize, span.nb as usize, span.nchunk as usize);
        let s = (&raw mut RED).cast::<f32>();
        let eps = EPS_POSITION * g.ny as f32 * g.nx as f32;
        let mut max = [eps; 2];
        let mut i = t;
        while i < len {
            let at = unsafe { sums.add(k * (p0 + i) * nchunk) };
            let mut tot = [0.0f32; POSITION_SUMS as usize];
            for c in 0..nchunk {
                // Unrolled, as every loop over `tot` and `max`, so they stay in registers.
                #[unroll]
                for j in 0..POSITION_SUMS {
                    let j = j as usize;
                    tot[j] += unsafe { *at.add(k * c + j) };
                }
            }
            #[unroll]
            for j in 0..POSITION_SUMS {
                let j = j as usize;
                unsafe { *at.add(j) = tot[j] };
            }
            #[unroll]
            for ax in 0..2 {
                if tot[2 + ax] > max[ax] {
                    max[ax] = tot[2 + ax];
                }
            }
            i += seg;
        }
        let max = unsafe {
            *s.add(t) = max[0];
            *s.add(seg + t) = max[1];
            block_max::<2>(s);
            [*s, *s.add(seg)]
        };
        let dx = |p: usize| {
            let at = unsafe { sums.add(k * p * nchunk) };
            let d = |ax: usize, m: f32| unsafe { step * *at.add(ax) / (*at.add(2 + ax) + m) };
            [d(0, max[0]), d(1, max[1])]
        };
        let mut acc = [0.0f32; 2];
        let mut i = t;
        while i < len {
            let d = dx(p0 + i);
            acc = [acc[0] + d[0], acc[1] + d[1]];
            i += seg;
        }
        // The sums take the next two segments: the maxima just read may still be unread.
        let mean = unsafe {
            let s = s.add(2 * seg);
            *s.add(t) = acc[0];
            *s.add(seg + t) = acc[1];
            crate::block_sum::<2>(s);
            [*s / nframe as f32, *s.add(seg) / nframe as f32]
        };
        let mut acc = [0.0f32; 2];
        let mut i = t;
        while i < len {
            let d = dx(p0 + i);
            acc = [
                acc[0] + (d[0] - mean[0]).abs(),
                acc[1] + (d[1] - mean[1]).abs(),
            ];
            i += seg;
        }
        // Every thread read the maxima before the sums above let any thread past them.
        let rest = (nframe - span.nb) as f32;
        let lim = unsafe {
            *s.add(t) = acc[0];
            *s.add(seg + t) = acc[1];
            crate::block_sum::<2>(s);
            let mad = |ax: usize, m: f32| (*s.add(ax * seg) + rest * m.abs()) / nframe as f32;
            [
                (10.0 * mad(0, mean[0])).min(limit),
                (10.0 * mad(1, mean[1])).min(limit),
            ]
        };
        let mut i = t;
        while i < len {
            let d = dx(p0 + i);
            unsafe {
                let o = offsets.add(2 * *order.add(p0 + i) as usize);
                *o += d[0].max(-lim[0]).min(lim[0]);
                *o.add(1) += d[1].max(-lim[1]).min(lim[1]);
            }
            i += seg;
        }
    }

    /// `out[2p .. 2p + 2] = offsets[2 order[p] .. 2 order[p] + 2]` for the epoch's `nframe`
    /// positions `p`: the offsets by epoch position, as the shift passes read them.
    ///
    /// # Safety
    ///
    /// `offsets` and `out` hold `2 * nframe` floats, `order` `nframe` frames.
    #[kernel]
    pub unsafe fn position_gather(
        offsets: *const f32,
        order: *const u32,
        out: *mut f32,
        nframe: u32,
    ) {
        let i = global_index();
        if i < 2 * nframe as usize {
            unsafe { *out.add(i) = *offsets.add(2 * *order.add(i / 2) as usize + i % 2) };
        }
    }
}
