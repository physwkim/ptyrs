//! pty-chi LSQML kernels (ptyrs-engine `lsqml`) on the far-field Fourier path: the column
//! pass that turns a stack's row-FFT'd exit waves into the exit-wave update `chi` between
//! the fused row passes, the Poisson `a_far` solver, the object preconditioner, the per-frame
//! step-size sums and 2×2 solve, and the update. A step's scalars stay on the device;
//! partial sums are reduced in a fixed order.
//!
//! With `Psi = DFT(psi)` unnormalized, the orthonormal far field's intensity is `icalc =
//! Σ_m |Psi_m|² / n` and `chi = IDFT(−a_far c Psi) / n`, both unnormalized, `c` the cost's
//! residual. Frames: as in [`crate::bh`], a stack's frame `fl` is epoch position `f0 + fl`;
//! its views are at that position of `views`, its counts at frame `order[f0 + fl]` of
//! `counts`.

use cuda_device::{async_copy, device, shared::DynamicSharedArray, thread};
use cuda_host::cuda_module;

use crate::FFT_EPT;
use crate::bh::BhObs;
use crate::fused::{FUSED_COLS, block_fft};
use crate::pixel::{MASKED, PIXEL_THREADS, StackCtl, StackGeom, load_c, stack_frames, store_c};

/// [`lsqml_kernels::LoadedModule::lsqml_cols_64`] writes `icalc`.
pub const LSQML_ICALC: u32 = 1;
/// The column pass scales every mode by `−a_far c / n`, inverse-FFTs and stores it; without
/// it `work` is only read.
pub const LSQML_CHI: u32 = 2;
/// `c` is the Poisson residual `1 − obs / (icalc + 1e-6)`; without it the Gaussian `2 (1 −
/// √obs / (√icalc + 1e-9))`.
pub const LSQML_POISSON: u32 = 4;

/// Floats a [`lsqml_kernels::LoadedModule::lsqml_dots`] block writes: `Σ |u|²`, `Σ |v|²`,
/// `Re` and `Im` of `Σ u conj(v)`, `Σ Re(conj(u) chi)`, `Σ Re(conj(v) chi)`.
pub const LSQML_DOTS: u32 = 6;
/// Floats a [`lsqml_kernels::LoadedModule::lsqml_xi`] block writes: the `a_far` numerator and
/// denominator.
pub const LSQML_XI: u32 = 2;
/// Floats of one frame's step record: `a_far`, `a_obj`, `a_probe`.
pub const LSQML_REC: u32 = 3;

/// pty-chi's offsets: on `√icalc` in the Gaussian residual, on `icalc` in the Poisson
/// residual and in the Poisson `xi`.
const EPS_AMPLITUDE: f32 = 1e-9;
const EPS_POISSON: f32 = 1e-6;
const EPS_XI: f32 = 1e-5;
/// pty-chi's `λ0` per pixel and mode in the step-size system.
const LAMBDA0: f32 = 1.2e-7;

/// The buffers of the column pass: `work` the stack's `[nf_max * nmode, ny, nx]` transforms,
/// `icalc` `[nf_max, ny, nx]` in detector order, `obs` the scan's counts in epoch order and
/// `a_far` by epoch position.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LsqmlCols {
    pub work: *mut f32,
    pub icalc: *mut f32,
    pub obs: BhObs,
    pub a_far: *const f32,
}

/// A batch: epoch positions `p0 .. p0 + nb`, each with `nchunk` reduction chunks.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LsqmlSpan {
    pub p0: u32,
    pub nb: u32,
    pub nchunk: u32,
}

/// The Poisson `a_far` state: `order` the epoch's frames by position, `a_scan` `a_far` by
/// frame (from each frame's last step), `a_pos` by epoch position, `prev` one float, the
/// running `a′`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LsqmlFar {
    pub order: *const u32,
    pub a_scan: *mut f32,
    pub a_pos: *mut f32,
    pub prev: *mut f32,
}

/// The object preconditioner: `pre` `[nobj, nyo, nxo]` floats, `max_bits` its maximum as
/// `f32` bits, `count_max` the brightest frame's photons.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LsqmlPre {
    pub pre: *mut f32,
    pub max_bits: *mut u32,
    pub count_max: *const u64,
}

/// The column pass, `fused::cols`' thread and shared-memory layout without its reduction
/// slots: column FFT of the row transforms in columns `u = blockIdx.x * FUSED_COLS + c` of
/// every mode `m` of frame `fl = blockIdx.y`, then per far-field pixel `icalc = Σ_m |Psi_m|² /
/// n` in ascending `m`, written at detector pixel `j` under [`LSQML_ICALC`]. Under
/// [`LSQML_CHI`] every mode is scaled by `−a_far[f0 + fl] c / n`, `c` the cost's residual of
/// `obs` and `icalc`, 0 where masked, inverse column-FFT'd and stored; without it nothing is
/// stored, so `work` keeps its row transforms. Grid `(nx / FUSED_COLS, nf_max)`,
/// block `nmode * FUSED_COLS * ny / FFT_EPT`, dynamic shared memory `2 * FUSED_COLS * ny`
/// bytes of counts, then the larger of `nmode * FUSED_COLS` column FFTs and `4 * nmode * ny
/// * FUSED_COLS` bytes. `counts` must be 16-byte aligned; it and `a_far` are read only under
/// [`LSQML_CHI`].
///
/// # Safety
///
/// `NY == g.ny`; `p` as [`LsqmlCols`] says for `g`; `ctl` one entry; launch as above.
#[device]
unsafe fn cols<const NY: u32>(p: LsqmlCols, ctl: *const StackCtl, g: StackGeom, flags: u32) {
    let (f0, nf) = unsafe { stack_frames(ctl, g) };
    let fl = thread::blockIdx_y();
    if fl >= nf {
        return;
    }
    let (ny, nx, nmode) = (NY, g.nx, g.nmode);
    let n = ny as usize * nx as usize;
    let tpf = ny / FFT_EPT;
    let nslot = FUSED_COLS * tpf;
    let tid = thread::threadIdx_x();
    let c = tid % FUSED_COLS;
    let f = (tid / FUSED_COLS) % tpf;
    let m = tid / nslot;
    let b = m * FUSED_COLS + c;
    let bpb = nmode * FUSED_COLS;
    let u = thread::blockIdx_x() * FUSED_COLS + c;
    let plane = (fl * nmode + m) as usize * n;
    let chi = flags & LSQML_CHI != 0;
    let poisson = flags & LSQML_POISSON != 0;
    let pos = (f0 + fl) as usize;
    let mut r = [0.0f32; 2 * FFT_EPT as usize];
    unsafe {
        let tile = DynamicSharedArray::<u8, 16>::get();
        let fft_smem = tile.add(2 * (FUSED_COLS * ny) as usize);
        // `|Psi_m(v, u)|^2` at `(m * ny + v) * FUSED_COLS + c`, over the FFTs' region.
        let power = fft_smem.cast::<f32>();
        let col0 = (thread::blockIdx_x() * FUSED_COLS + nx / 2) % nx;
        if chi {
            let counts = p.obs.counts.add(*p.obs.order.add(pos) as usize * n);
            let mut row = tid;
            while row < ny {
                async_copy::cp_async_cg_16(
                    tile.add(2 * (row * FUSED_COLS) as usize).cast(),
                    counts.add((row * nx + col0) as usize).cast(),
                );
                row += nmode * nslot;
            }
            async_copy::cp_async_commit_group();
        }
        #[unroll]
        for i in 0..FFT_EPT {
            let v = load_c(p.work, plane + ((f + i * tpf) * nx + u) as usize);
            r[2 * i as usize] = v.0;
            r[2 * i as usize + 1] = v.1;
        }
        block_fft::<NY, false>(&mut r, fft_smem, b, f, bpb);
        thread::sync_threads();
        #[unroll]
        for i in 0..FFT_EPT {
            let (re, im) = (r[2 * i as usize], r[2 * i as usize + 1]);
            let v = f + i * tpf;
            *power.add(((m * ny + v) * FUSED_COLS + c) as usize) = re * re + im * im;
        }
        async_copy::cp_async_wait_all();
        thread::sync_threads();
        let inv_n = 1.0 / n as f32;
        let tile = tile.cast::<u16>();
        let a = if chi { *p.a_far.add(pos) } else { 0.0 };
        #[unroll]
        for i in 0..FFT_EPT {
            let v = f + i * tpf;
            let mut s = 0.0f32;
            for mm in 0..nmode {
                s += *power.add(((mm * ny + v) * FUSED_COLS + c) as usize);
            }
            let icalc = s * inv_n;
            let y = (v + ny / 2) % ny;
            let j = (y * nx + col0 + c) as usize;
            if m == 0 && flags & LSQML_ICALC != 0 {
                *p.icalc.add(fl as usize * n + j) = icalc;
            }
            if chi {
                let obs = *tile.add((y * FUSED_COLS + c) as usize);
                let res = if obs == MASKED {
                    0.0
                } else if poisson {
                    1.0 - obs as f32 / (icalc + EPS_POISSON)
                } else {
                    2.0 * (1.0 - (obs as f32).sqrt() / (icalc.sqrt() + EPS_AMPLITUDE))
                };
                let scale = -a * res * inv_n;
                r[2 * i as usize] *= scale;
                r[2 * i as usize + 1] *= scale;
            }
        }
        if chi {
            // Every power is read before the inverse FFT reuses the region.
            thread::sync_threads();
            block_fft::<NY, true>(&mut r, fft_smem, b, f, bpb);
            #[unroll]
            for i in 0..FFT_EPT {
                let k = 2 * i as usize;
                store_c(
                    p.work,
                    plane + ((f + i * tpf) * nx + u) as usize,
                    (r[k], r[k + 1]),
                );
            }
        }
    }
}

#[cuda_module]
pub mod lsqml_kernels {
    use super::{
        EPS_XI, LAMBDA0, LSQML_DOTS, LSQML_REC, LSQML_XI, LsqmlCols, LsqmlFar, LsqmlPre, LsqmlSpan,
        cols, store_sums,
    };
    use crate::bh::BhObs;
    use crate::grad::{GAMMA_CHUNK, ObjProbePtrs};
    use crate::pixel::*;
    use crate::*;
    use cuda_device::SharedArray;

    /// [`super::cols`] for 64-point columns.
    ///
    /// # Safety
    ///
    /// As [`super::cols`].
    #[kernel]
    #[launch_bounds(crate::fused::FUSED_COL_THREADS)]
    pub unsafe fn lsqml_cols_64(p: LsqmlCols, ctl: *const StackCtl, g: StackGeom, flags: u32) {
        unsafe { cols::<64>(p, ctl, g, flags) }
    }

    /// [`super::cols`] for 128-point columns.
    ///
    /// # Safety
    ///
    /// As [`super::cols`].
    #[kernel]
    #[launch_bounds(crate::fused::FUSED_COL_THREADS)]
    pub unsafe fn lsqml_cols_128(p: LsqmlCols, ctl: *const StackCtl, g: StackGeom, flags: u32) {
        unsafe { cols::<128>(p, ctl, g, flags) }
    }

    /// [`super::cols`] for 256-point columns.
    ///
    /// # Safety
    ///
    /// As [`super::cols`].
    #[kernel]
    #[launch_bounds(crate::fused::FUSED_COL_THREADS)]
    pub unsafe fn lsqml_cols_256(p: LsqmlCols, ctl: *const StackCtl, g: StackGeom, flags: u32) {
        unsafe { cols::<256>(p, ctl, g, flags) }
    }

    /// [`super::cols`] for 512-point columns.
    ///
    /// # Safety
    ///
    /// As [`super::cols`].
    #[kernel]
    #[launch_bounds(crate::fused::FUSED_COL_THREADS)]
    pub unsafe fn lsqml_cols_512(p: LsqmlCols, ctl: *const StackCtl, g: StackGeom, flags: u32) {
        unsafe { cols::<512>(p, ctl, g, flags) }
    }

    /// `x[0..n] = 1/2`, the `a_far` of a frame never stepped.
    ///
    /// # Safety
    ///
    /// `x` holds `n` floats.
    #[kernel]
    pub unsafe fn lsqml_half(x: *mut f32, n: u32) {
        let i = global_index();
        if i < n as usize {
            unsafe { *x.add(i) = 0.5 };
        }
    }

    /// `*out = max(*out, Σ_unmasked counts)` over frames `blockIdx.x`, the photons of the
    /// brightest frame, summed exactly.
    ///
    /// # Safety
    ///
    /// `counts` holds `[gridDim.x, n]`, `out` one `u64`; block `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn lsqml_count_max(counts: *const u16, n: u32, out: *mut u64) {
        use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU64};
        static mut RED: SharedArray<u64, 256> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as usize;
        let frame = unsafe { counts.add(thread::blockIdx_x() as usize * n as usize) };
        let mut s = 0u64;
        let mut d = t;
        while d < n as usize {
            let c = unsafe { *frame.add(d) };
            if c != MASKED {
                s += u64::from(c);
            }
            d += PIXEL_THREADS as usize;
        }
        let red = (&raw mut RED).cast::<u64>();
        unsafe {
            *red.add(t) = s;
            let mut half = PIXEL_THREADS as usize / 2;
            while half > 0 {
                thread::sync_threads();
                if t < half {
                    *red.add(t) += *red.add(t + half);
                }
                half /= 2;
            }
            if t == 0 {
                DeviceAtomicU64::from_ptr(out).fetch_max(*red, AtomicOrdering::Relaxed);
            }
        }
    }

    /// pty-chi's object preconditioner over every view of `views`: with `I = Σ_v |P[v.probe]
    /// (y − v.oy, x − v.ox)|²` over the views of object mode `blockIdx.z` whose integer
    /// window holds object pixel `(y, x)`, `pre = (I · power / (4 n²) + I) / 2` with `init !=
    /// 0` (`power = *count_max`), else `pre = (pre + I) / 2`; every pixel of the object is
    /// written, and the new maximum merged into `*max_bits`, which must start at zero. Grid
    /// `(ntx, nty, nobj)` of `OBJ_TILE` tiles, block `(OBJ_TILE, OBJ_TILE, 1)`.
    ///
    /// # Safety
    ///
    /// `probe` holds `[nprobe, ny, nx]` complex values, `views` the scan's views with runs
    /// `tiles` covering every view that touches each tile, `pre` as [`LsqmlPre`] says.
    #[kernel]
    pub unsafe fn lsqml_pre(
        probe: *const f32,
        views: *const ViewAddr,
        tiles: TilePtrs,
        pre: LsqmlPre,
        init: u32,
        g: StackGeom,
    ) {
        static mut RED: SharedArray<f32, 256> = SharedArray::UNINIT;
        let (tile, y, x) = tile_and_pixel();
        let (ny, nx, nyo, nxo) = (g.ny, g.nx, g.nyo, g.nxo);
        let n = ny as usize * nx as usize;
        let iobj = thread::blockIdx_z();
        let mut out = 0.0f32;
        if y < nyo && x < nxo {
            let mut illum = 0.0f32;
            unsafe {
                let (lo, end) = tile_run(tiles, tile);
                for k in lo..end {
                    let v = *views.add(*tiles.views.add(k as usize) as usize);
                    if v.obj == iobj && y >= v.oy && y < v.oy + ny && x >= v.ox && x < v.ox + nx {
                        let at = v.probe as usize * n + ((y - v.oy) * nx + x - v.ox) as usize;
                        let p = load_c(probe, at);
                        illum += p.0 * p.0 + p.1 * p.1;
                    }
                }
                let px = (iobj as usize * nyo as usize + y as usize) * nxo as usize + x as usize;
                let old = if init != 0 {
                    illum * (*pre.count_max as f32 / (4.0 * n as f32 * n as f32))
                } else {
                    *pre.pre.add(px)
                };
                out = (old + illum) / 2.0;
                *pre.pre.add(px) = out;
            }
        }
        let t = (thread::threadIdx_y() * OBJ_TILE + thread::threadIdx_x()) as usize;
        unsafe { block_max_into((&raw mut RED).cast::<f32>(), t, out, pre.max_bits) }
    }

    /// The preconditioned object direction: `d[i] /= √(pre[i]² + (0.1 max)²)` with `max =
    /// f32::from_bits(*max_bits)`, for `n` elements.
    ///
    /// # Safety
    ///
    /// `d` holds `n` complex values, `pre` `n` floats, `max_bits` one `u32`.
    #[kernel]
    pub unsafe fn lsqml_dir_obj(d: *mut f32, pre: *const f32, max_bits: *const u32, n: u32) {
        let i = global_index();
        if i >= n as usize {
            return;
        }
        unsafe {
            let floor = 0.1 * f32::from_bits(*max_bits);
            let p = *pre.add(i);
            let s = 1.0 / (p * p + floor * floor).sqrt();
            let v = load_c(d, i);
            store_c(d, i, (v.0 * s, v.1 * s));
        }
    }

    /// The Poisson `a_far` sums of chunk `c = blockIdx.x` of frame `f0 + fl`, `fl =
    /// blockIdx.y`, at `a′ = *prev`: over its unmasked detector pixels `d` in `c *
    /// GAMMA_CHUNK .. min((c + 1) * GAMMA_CHUNK, n)`, with `xi = 1 − obs / (icalc + 1e-5)`,
    /// `Σ xi (icalc − obs / (1 − a′ xi))` and `Σ xi² icalc`, to `out[2p .. 2p + 2]`, `p = (f0
    /// + fl) * nchunk + c`.
    ///
    /// # Safety
    ///
    /// `icalc` holds `[nf_max, ny, nx]` floats, `obs` as [`BhObs`] says, `prev` one float,
    /// `out` `2 * nframe * nchunk`, `ctl` one entry; grid `(nchunk, nf_max, 1)`, block
    /// `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn lsqml_xi(
        icalc: *const f32,
        obs: BhObs,
        prev: *const f32,
        out: *mut f32,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        static mut RED: SharedArray<f32, 512> = SharedArray::UNINIT;
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let n = g.ny as usize * g.nx as usize;
        let fl = thread::blockIdx_y() as usize;
        // The whole block leaves together, before its first barrier.
        if fl >= nf as usize {
            return;
        }
        let c = thread::blockIdx_x() as usize;
        let chunk = GAMMA_CHUNK as usize;
        let end = n.min((c + 1) * chunk);
        let pos = f0 as usize + fl;
        let mut acc = [0.0f32; LSQML_XI as usize];
        unsafe {
            let a = *prev;
            let frame = *obs.order.add(pos) as usize;
            let mut d = c * chunk + thread::threadIdx_x() as usize;
            while d < end {
                let count = *obs.counts.add(frame * n + d);
                if count != MASKED {
                    let (yt, yp) = (count as f32, *icalc.add(fl * n + d));
                    let xi = 1.0 - yt / (yp + EPS_XI);
                    acc[0] += xi * (yp - yt / (1.0 - a * xi));
                    acc[1] += xi * xi * yp;
                }
                d += PIXEL_THREADS as usize;
            }
            let o = out.add(LSQML_XI as usize * (pos * n.div_ceil(chunk) + c));
            store_sums::<2>((&raw mut RED).cast::<f32>(), acc, o);
        }
    }

    /// `*far.prev` = the mean of `far.a_scan[far.order[p]]` over the batch's positions, the
    /// first `a′` of its Poisson `a_far` passes.
    ///
    /// # Safety
    ///
    /// `far` as [`LsqmlFar`] says, `order` over the span; grid `(1, 1, 1)`, block
    /// `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn lsqml_far_mean(far: LsqmlFar, span: LsqmlSpan) {
        static mut RED: SharedArray<f32, 256> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as usize;
        let mut acc = [0.0f32; 1];
        let mut i = t;
        while i < span.nb as usize {
            acc[0] += unsafe {
                *far.a_scan
                    .add(*far.order.add(span.p0 as usize + i) as usize)
            };
            i += PIXEL_THREADS as usize;
        }
        unsafe {
            store_sums::<1>((&raw mut RED).cast::<f32>(), acc, far.prev);
            if t == 0 {
                *far.prev /= span.nb as f32;
            }
        }
    }

    /// One Poisson `a_far` pass over the batch's positions `p` from the [`lsqml_xi`] sums at
    /// `a′ = *far.prev`: `far.a_pos[p] = clamp(a′ / 2 + num / den / 2, 0, 1)` (`num / den` 0
    /// where `den <= 0`) with each frame's chunks summed in order, under `last != 0` also
    /// `far.a_scan[far.order[p]]`; then `*far.prev` = the mean of the new values.
    ///
    /// # Safety
    ///
    /// `sums` holds `2 * span.nchunk` floats per position, `far` as [`LsqmlFar`] says, `order`
    /// over the span; grid `(1, 1, 1)`, block `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn lsqml_far_step(sums: *const f32, span: LsqmlSpan, far: LsqmlFar, last: u32) {
        static mut RED: SharedArray<f32, 256> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as usize;
        let h = LSQML_XI as usize;
        let nchunk = span.nchunk as usize;
        // Read by every thread before `store_sums` lets thread 0 overwrite it.
        let a = unsafe { *far.prev };
        let mut acc = [0.0f32; 1];
        let mut i = t;
        while i < span.nb as usize {
            let p = span.p0 as usize + i;
            let (mut num, mut den) = (0.0f32, 0.0f32);
            for c in 0..nchunk {
                unsafe {
                    num += *sums.add(h * (p * nchunk + c));
                    den += *sums.add(h * (p * nchunk + c) + 1);
                }
            }
            let ratio = if den > 0.0 { num / den } else { 0.0 };
            let v = (0.5 * a + 0.5 * ratio).clamp(0.0, 1.0);
            unsafe {
                *far.a_pos.add(p) = v;
                if last != 0 {
                    *far.a_scan.add(*far.order.add(p) as usize) = v;
                }
            }
            acc[0] += v;
            i += PIXEL_THREADS as usize;
        }
        unsafe {
            store_sums::<1>((&raw mut RED).cast::<f32>(), acc, far.prev);
            if t == 0 {
                *far.prev /= span.nb as f32;
            }
        }
    }

    /// The step-size sums of chunk `c = blockIdx.x` of frame `f0 + fl`, `fl = blockIdx.y`,
    /// over its modes and the pixels `d` in `c * GAMMA_CHUNK .. min((c + 1) * GAMMA_CHUNK,
    /// n)`: with view `w = views[(f0 + fl) * nmode + m]` and `q = exp(i quad r²)`, `u = dO_w
    /// P_w q` and `v = O_w dP_w q` (`x` the object and probe, `d` their directions) and
    /// `chi` at the exit-wave index of `d`, `out[6p .. 6p + 6]` gets the [`LSQML_DOTS`] sums,
    /// `p = (f0 + fl) * nchunk + c`.
    ///
    /// # Safety
    ///
    /// `chi` holds `[nf_max * nmode, ny, nx]`, `x` and `d` as [`ObjProbePtrs`] says, `views`
    /// `nframe * nmode` records in range, `out` `6 * nframe * nchunk` floats, `ctl` one
    /// entry; grid `(nchunk, nf_max, 1)`, block `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn lsqml_dots(
        chi: *const f32,
        x: ObjProbePtrs,
        d: ObjProbePtrs,
        views: *const ViewAddr,
        out: *mut f32,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        static mut RED: SharedArray<f32, 1536> = SharedArray::UNINIT;
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
        let mut acc = [0.0f32; LSQML_DOTS as usize];
        let mut i = c * chunk + thread::threadIdx_x() as usize;
        while i < end {
            let (py, px) = ((i / nx as usize) as u32, (i % nx as usize) as u32);
            let at = psi_index(py, px, ny, nx);
            let q = quad_phase(g.quad, py, px, ny, nx);
            for m in 0..nmode {
                unsafe {
                    let w = *views.add(pos * nmode + m);
                    let pa = w.probe as usize * n + i;
                    let (p, dp) = (load_c(x.probe, pa), load_c(d.probe, pa));
                    let o = sample_obj(x.obj, w, py, px, g.nyo, g.nxo);
                    let d_o = sample_obj(d.obj, w, py, px, g.nyo, g.nxo);
                    let u = cmul(cmul(d_o, p), q);
                    let v = cmul(cmul(o, dp), q);
                    let ch = load_c(chi, (fl * nmode + m) * n + at);
                    let uv = cmul_conj(u, v);
                    acc[0] += u.0 * u.0 + u.1 * u.1;
                    acc[1] += v.0 * v.0 + v.1 * v.1;
                    acc[2] += uv.0;
                    acc[3] -= uv.1;
                    acc[4] += cmul_conj(u, ch).0;
                    acc[5] += cmul_conj(v, ch).0;
                }
            }
            i += PIXEL_THREADS as usize;
        }
        unsafe {
            let o = out.add(LSQML_DOTS as usize * (pos * n.div_ceil(chunk) + c));
            store_sums::<6>((&raw mut RED).cast::<f32>(), acc, o);
        }
    }

    /// One batch's step sizes over its positions `p` from the [`lsqml_dots`] sums, each
    /// frame's chunks summed in order (and left in its chunk 0), `Σ |u|²` and `Σ |v|²` plus
    /// `λ0 = 1.2e-7` per mode and pixel: per frame the 2×2 system `a11 = Σ |u|² + 0.1
    /// mean(Σ |u|²)`, `a22` likewise, `a12 = Σ u conj(v)`, whose solution's real part
    /// gives `a_obj = 0.9 max(x0, 0) / nmode` and `a_probe = 0.9 max(x1, 0)`;
    /// `rec[3p .. 3p + 3] = (a_far[p], a_obj, a_probe)`. Writes `step[0]`, the mean of the
    /// `a_obj` without their `trim` smallest and largest, and `step[1]`, the mean `a_probe`.
    ///
    /// # Safety
    ///
    /// `sums` holds `6 * span.nchunk` floats per position, `a_far` `nframe`, `rec` `3 *
    /// nframe`, `step` two; `2 * trim < span.nb`; grid `(1, 1, 1)`, block `(PIXEL_THREADS, 1,
    /// 1)`.
    #[kernel]
    pub unsafe fn lsqml_step(
        sums: *mut f32,
        span: LsqmlSpan,
        trim: u32,
        a_far: *const f32,
        rec: *mut f32,
        step: *mut f32,
        g: StackGeom,
    ) {
        static mut RED: SharedArray<f32, 512> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as usize;
        let seg = PIXEL_THREADS as usize;
        let k = LSQML_DOTS as usize;
        let r = LSQML_REC as usize;
        let (p0, nb) = (span.p0, span.nb);
        let (len, nchunk) = (nb as usize, span.nchunk as usize);
        let lambda = LAMBDA0 * g.nmode as f32 * g.ny as f32 * g.nx as f32;
        let red = (&raw mut RED).cast::<f32>();
        let mut i = t;
        let mut acc = [0.0f32; 2];
        while i < len {
            let s = unsafe { sums.add(k * (p0 as usize + i) * nchunk) };
            let mut tot = [0.0f32; LSQML_DOTS as usize];
            for c in 0..nchunk {
                for (j, v) in tot.iter_mut().enumerate() {
                    *v += unsafe { *s.add(k * c + j) };
                }
            }
            tot[0] += lambda;
            tot[1] += lambda;
            for (j, &v) in tot.iter().enumerate() {
                unsafe { *s.add(j) = v };
            }
            acc[0] += tot[0];
            acc[1] += tot[1];
            i += PIXEL_THREADS as usize;
        }
        unsafe {
            *red.add(t) = acc[0];
            *red.add(seg + t) = acc[1];
            block_sum::<2>(red);
        }
        let (reg11, reg22) = unsafe { (0.1 * *red / nb as f32, 0.1 * *red.add(seg) / nb as f32) };
        let nmode = g.nmode as f32;
        let mut i = t;
        while i < len {
            let p = p0 as usize + i;
            unsafe {
                let s = sums.add(k * p * nchunk);
                let (a11, a22) = (*s + reg11, *s.add(1) + reg22);
                let (re, im) = (*s.add(2), *s.add(3));
                let (b0, b1) = (*s.add(4), *s.add(5));
                let det = a11 * a22 - (re * re + im * im);
                let x0 = (a22 * b0 - re * b1) / det;
                let x1 = (a11 * b1 - re * b0) / det;
                *rec.add(r * p) = *a_far.add(p);
                *rec.add(r * p + 1) = 0.9 * x0.max(0.0) / nmode;
                *rec.add(r * p + 2) = 0.9 * x1.max(0.0);
            }
            i += PIXEL_THREADS as usize;
        }
        // Every `rec` written above, and `reg11` and `reg22` read, before `red` is reused.
        thread::sync_threads();
        let mut acc = [0.0f32; 2];
        let mut i = t;
        while i < len {
            let p = p0 as usize + i;
            let ai = unsafe { *rec.add(r * p + 1) };
            // The rank of `a_obj` in the batch, ties by position.
            let mut rank = 0u32;
            for jj in 0..len {
                let aj = unsafe { *rec.add(r * (p0 as usize + jj) + 1) };
                if aj < ai || (aj == ai && jj < i) {
                    rank += 1;
                }
            }
            if rank >= trim && rank < nb - trim {
                acc[0] += ai;
            }
            acc[1] += unsafe { *rec.add(r * p + 2) };
            i += PIXEL_THREADS as usize;
        }
        unsafe {
            *red.add(t) = acc[0];
            *red.add(seg + t) = acc[1];
            block_sum::<2>(red);
            if t == 0 {
                *step = *red / (nb - 2 * trim) as f32;
                *step.add(1) = *red.add(seg) / nb as f32;
            }
        }
    }

    /// `x += step[0] num`, and the consumed sums are zero, for `n` elements.
    ///
    /// # Safety
    ///
    /// `x`, `sums.num` hold `n` complex values, `sums.den` `n` floats, `step` one float.
    #[kernel]
    pub unsafe fn lsqml_update(x: *mut f32, sums: SumPtrs, n: u32, step: *const f32) {
        let i = global_index();
        if i >= n as usize {
            return;
        }
        unsafe {
            let s = *step;
            let (d, v) = (load_c(sums.num, i), load_c(x, i));
            store_c(x, i, (v.0 + s * d.0, v.1 + s * d.1));
            store_c(sums.num, i, (0.0, 0.0));
            *sums.den.add(i) = 0.0;
        }
    }
}

/// Thread 0 writes the block sums of each thread's `acc` to `out[0..K]`; `s` is `K`
/// `PIXEL_THREADS` segments of shared memory. Every thread of the block must call it.
///
/// # Safety
///
/// `s` holds `K * PIXEL_THREADS` shared floats, `out` `K` floats; block `(PIXEL_THREADS, 1,
/// 1)`.
#[device]
unsafe fn store_sums<const K: usize>(s: *mut f32, acc: [f32; K], out: *mut f32) {
    let t = thread::threadIdx_x() as usize;
    let seg = PIXEL_THREADS as usize;
    unsafe {
        for (j, &a) in acc.iter().enumerate() {
            *s.add(j * seg + t) = a;
        }
        crate::block_sum::<K>(s);
        if t == 0 {
            for j in 0..K {
                *out.add(j) = *s.add(j * seg);
            }
        }
    }
}
