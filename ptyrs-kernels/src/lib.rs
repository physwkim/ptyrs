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

pub mod pixel;
pub mod rpie;

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

/// Global index of this thread in a 1D launch.
#[device]
fn global_index() -> usize {
    thread::blockIdx_x() as usize * thread::blockDim_x() as usize + thread::threadIdx_x() as usize
}

/// `sum_m |psi[plane0 + m, at]|^2` over `nmode` consecutive planes of `n` values.
///
/// # Safety
///
/// `psi` must hold planes `plane0 .. plane0 + nmode`.
#[device]
unsafe fn mode_power(psi: *const f32, plane0: usize, nmode: u32, n: usize, at: usize) -> f32 {
    let mut s = 0.0f32;
    for m in 0..nmode as usize {
        let v = unsafe { pixel::load_c(psi, (plane0 + m) * n + at) };
        s += v.0 * v.0 + v.1 * v.1;
    }
    s
}

/// This thread's object tile `ty * ntx + tx` and object pixel `(y, x)`, in a
/// `(ntx, nty, nobj)` grid of `(OBJ_TILE, OBJ_TILE, 1)` blocks.
#[device]
fn tile_and_pixel() -> (usize, u32, u32) {
    let (bx, by) = (thread::blockIdx_x(), thread::blockIdx_y());
    let tile = (by * thread::gridDim_x() + bx) as usize;
    let tile_px = pixel::OBJ_TILE;
    (
        tile,
        by * tile_px + thread::threadIdx_y(),
        bx * tile_px + thread::threadIdx_x(),
    )
}

/// PyNX `ObjNorm` of element `g`: `x = (num + reg x) / max(den + reg, 1e-12)` with
/// `reg = inertia * max`, every operation rounded on its own (no FMA contraction); then
/// zeroes `num` and `den` at `g`, which this update has consumed.
///
/// # Safety
///
/// `x`, `num` hold `g + 1` complex values, `den` `g + 1` floats.
#[device]
unsafe fn normalize_at(x: *mut f32, sums: pixel::SumPtrs, g: usize, inertia: f32, max: f32) {
    use cuda_device::float::{add_rn_f32, div_rn_f32, mul_rn_f32};
    unsafe {
        let reg = mul_rn_f32(inertia, max);
        let d = add_rn_f32(*sums.den.add(g), reg).max(1e-12);
        let (a, b) = (pixel::load_c(sums.num, g), pixel::load_c(x, g));
        let part = |a: f32, b: f32| div_rn_f32(add_rn_f32(a, mul_rn_f32(reg, b)), d);
        pixel::store_c(x, g, (part(a.0, b.0), part(a.1, b.1)));
        pixel::store_c(sums.num, g, (0.0, 0.0));
        *sums.den.add(g) = 0.0;
    }
}

/// Merges the block's thread values `m`, all non-negative, into `*out` (the bits of a
/// non-negative float) as their maximum; non-negative floats order like their bits. Every
/// thread of the `PIXEL_THREADS` block must call it, `t` being its index in the block.
///
/// # Safety
///
/// `s` must point to `PIXEL_THREADS` block-shared floats, `out` to one `u32`.
#[device]
unsafe fn block_max_into(s: *mut f32, t: usize, m: f32, out: *mut u32) {
    use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
    unsafe {
        *s.add(t) = m;
        let mut half = pixel::PIXEL_THREADS as usize / 2;
        while half > 0 {
            thread::sync_threads();
            if t < half && *s.add(t + half) > *s.add(t) {
                *s.add(t) = *s.add(t + half);
            }
            half /= 2;
        }
        if t == 0 {
            DeviceAtomicU32::from_ptr(out).fetch_max((*s).to_bits(), AtomicOrdering::Relaxed);
        }
    }
}

/// Tree sum of `PIXEL_THREADS` shared values into element 0; every thread must call it.
///
/// # Safety
///
/// `s` must point to `PIXEL_THREADS` block-shared floats, each written by its thread.
#[device]
unsafe fn block_sum(s: *mut f32) {
    let t = thread::threadIdx_x();
    let mut half = pixel::PIXEL_THREADS / 2;
    while half > 0 {
        thread::sync_threads();
        if t < half {
            unsafe { *s.add(t as usize) += *s.add((t + half) as usize) };
        }
        half /= 2;
    }
    thread::sync_threads();
}

#[cuda_module]
pub mod kernels {
    use super::*;
    use crate::pixel::*;
    use cuda_device::SharedArray;

    // M1 pixel kernels. Shapes, orders and the FFT scaling convention are documented on the
    // ptyrs-gpu wrappers, which check them; `pixel` documents the index maps. A stack is
    // frames `f0 .. f0 + nf` from `stack_frames(ctl, g)`, so views `k0 .. k0 + nk` with
    // `k0 = f0 * nmode`, `nk = nf * nmode` (frame-major, `k = frame * nmode + mode`); its exit
    // waves are planes `0 .. nk` of `psi`, which holds `nf_max * nmode`. Launches are sized
    // by `nf_max`; threads and blocks past the stack exit. A stack with `nf == 0` writes
    // nothing, so a replay against an empty entry leaves every buffer as it was. 1D kernels
    // are launched with `PIXEL_THREADS` threads per block over the stated element count.

    /// `psi[kl, psi_index(y, x)] = O_v(y, x) * P_v(y, x) * exp(i quad r^2)` for every view
    /// `v = views[k0 + kl]` of the stack, one thread per element of `[nf_max * nmode, ny, nx]`.
    ///
    /// # Safety
    ///
    /// `obj` `[nobj, nyo, nxo]`, `probe` `[nprobe, ny, nx]`, `views` `nframe * nmode` records
    /// in range for those shapes, `psi` `[nf_max * nmode, ny, nx]`, `ctl` one entry.
    #[kernel]
    pub unsafe fn obj_probe_to_psi(
        obj: *const f32,
        probe: *const f32,
        views: *const ViewAddr,
        psi: *mut f32,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let n = g.ny as usize * g.nx as usize;
        let t = global_index();
        if t >= nf as usize * g.nmode as usize * n {
            return;
        }
        let (kl, i) = (t / n, t % n);
        let (y, x) = ((i / g.nx as usize) as u32, (i % g.nx as usize) as u32);
        unsafe {
            let v = *views.add(f0 as usize * g.nmode as usize + kl);
            let o = sample_obj(obj, v, y, x, g.nyo, g.nxo);
            let p = load_c(probe, v.probe as usize * n + i);
            let q = quad_phase(g.quad, y, x, g.ny, g.nx);
            store_c(
                psi,
                kl * n + psi_index(y, x, g.ny, g.nx),
                cmul(cmul(o, p), q),
            );
        }
    }

    /// `icalc[fl, d] = sum_m |psi[fl * nmode + m, psi_index(d)]|^2 / (ny * nx)` for every
    /// frame `fl < nf` and detector pixel `d`, one thread per element of `[nf_max, ny, nx]`.
    ///
    /// # Safety
    ///
    /// `psi` `[nf_max * nmode, ny, nx]`, `icalc` `[nf_max, ny, nx]`, `ctl` one entry.
    #[kernel]
    pub unsafe fn mode_intensity(
        psi: *const f32,
        icalc: *mut f32,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        let (_, nf) = unsafe { stack_frames(ctl, g) };
        let (ny, nx, nmode) = (g.ny, g.nx, g.nmode);
        let n = ny as usize * nx as usize;
        let t = global_index();
        if t >= nf as usize * n {
            return;
        }
        let (fl, d) = (t / n, t % n);
        let at = psi_index((d / nx as usize) as u32, (d % nx as usize) as u32, ny, nx);
        unsafe {
            *icalc.add(t) = mode_power(psi, fl * nmode as usize, nmode, n, at) / n as f32;
        }
    }

    /// Replaces the modulus of every mode with the observed amplitude (PyNX
    /// `ProjectionAmplitude`, no background), folding in the `1 / (ny * nx)` an unnormalized
    /// FFT round trip needs: `psi *= sqrt(obs / max(icalc, 1e-12)) / (ny * nx)`, or
    /// `psi /= ny * nx` where `obs` is `MASKED`. One thread per element of
    /// `[nf_max, ny, nx]`.
    ///
    /// # Safety
    ///
    /// `psi` `[nf_max * nmode, ny, nx]`, `counts` `[nframe, ny, nx]`, `ctl` one entry.
    #[kernel]
    pub unsafe fn apply_amplitude(
        psi: *mut f32,
        counts: *const u16,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let (ny, nx, nmode) = (g.ny, g.nx, g.nmode);
        let n = ny as usize * nx as usize;
        let t = global_index();
        if t >= nf as usize * n {
            return;
        }
        let (fl, d) = (t / n, t % n);
        let at = psi_index((d / nx as usize) as u32, (d % nx as usize) as u32, ny, nx);
        let inv_n = 1.0 / n as f32;
        let plane0 = fl * nmode as usize;
        unsafe {
            let obs = *counts.add(f0 as usize * n + t);
            let s = if obs == MASKED {
                inv_n
            } else {
                let calc = mode_power(psi, plane0, nmode, n, at) * inv_n;
                (obs as f32 / calc.max(1e-12)).sqrt() * inv_n
            };
            for m in 0..nmode as usize {
                let j = (plane0 + m) * n + at;
                let v = load_c(psi, j);
                store_c(psi, j, (v.0 * s, v.1 * s));
            }
        }
    }

    /// Per-frame log-likelihood sums (PyNX `LLKAll`, no background): over the unmasked
    /// pixels of frame `f0 + fl`, with `calc = sum_m |psi|^2 / (ny * nx)` and `obs` the
    /// count, Poisson `calc` if `obs <= 0.1` else `calc - obs + obs ln(obs / calc)`,
    /// Gaussian `(obs - calc)^2 / (obs + 1)`, Euclidean `4 (sqrt obs - sqrt calc)^2`, and
    /// `calc`. One `PIXEL_THREADS` block per frame; `out[fl]` gets the sums of frame
    /// `fl < nf`, entries past `nf` are not written.
    ///
    /// # Safety
    ///
    /// `psi` `[nf_max * nmode, ny, nx]`, `counts` `[nframe, ny, nx]`, `out` `nf_max` records,
    /// `ctl` one entry; grid `(nf_max, 1, 1)`, block `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn llk(
        psi: *const f32,
        counts: *const u16,
        out: *mut LlkPartial,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        // Four `PIXEL_THREADS` segments: Poisson, Gaussian, Euclidean, calc.
        static mut RED: SharedArray<f32, 1024> = SharedArray::UNINIT;
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let (ny, nx, nmode) = (g.ny, g.nx, g.nmode);
        let n = ny as usize * nx as usize;
        let fl = thread::blockIdx_x() as usize;
        // The whole block leaves together, before its first barrier.
        if fl >= nf as usize {
            return;
        }
        let t = thread::threadIdx_x() as usize;
        let plane0 = fl * nmode as usize;
        let inv_n = 1.0 / n as f32;
        let mut acc = [0.0f32; 4];
        let mut d = t;
        while d < n {
            let at = psi_index((d / nx as usize) as u32, (d % nx as usize) as u32, ny, nx);
            let obs = unsafe { *counts.add((f0 as usize + fl) * n + d) };
            if obs != MASKED {
                let obs = obs as f32;
                let calc = (unsafe { mode_power(psi, plane0, nmode, n, at) } * inv_n).max(0.0);
                acc[0] += if obs <= 0.1 {
                    calc
                } else {
                    calc - obs + obs * (obs / calc).ln()
                };
                acc[1] += (obs - calc) * (obs - calc) / (obs + 1.0);
                let e = obs.sqrt() - calc.sqrt();
                acc[2] += 4.0 * e * e;
                acc[3] += calc;
            }
            d += PIXEL_THREADS as usize;
        }
        let s = (&raw mut RED).cast::<f32>();
        let seg = PIXEL_THREADS as usize;
        unsafe {
            for (j, &a) in acc.iter().enumerate() {
                *s.add(j * seg + t) = a;
                block_sum(s.add(j * seg));
            }
            if t == 0 {
                *out.add(fl) = LlkPartial {
                    poisson: *s,
                    gaussian: *s.add(seg),
                    euclidean: *s.add(2 * seg),
                    icalc: *s.add(3 * seg),
                };
            }
        }
    }

    /// Appends the stack's views to the tile slabs: view `k` of the stack goes, in ascending
    /// `k`, after the run of every tile [`axis_tiles`] puts it in. A view whose tile's slab
    /// is full is not written and adds 1 to `*overflow`; a run's length is written only if
    /// the stack has a view of the tile. One warp per tile `ty * ntx + tx`, a 1D launch of
    /// `ntiles * WARP` threads.
    ///
    /// # Safety
    ///
    /// `views` holds `nframe * nmode` records, each inside the object; `slabs` as
    /// [`TileSlabs`] says for `ntiles` tiles, with `len[t] <= start[t + 1] - start[t]`;
    /// `overflow` one `u32`; `ctl` a readable entry.
    #[kernel]
    pub unsafe fn bin_views(
        views: *const ViewAddr,
        slabs: TileSlabs,
        overflow: *mut u32,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
        use cuda_device::warp;
        let ntx = g.nxo.div_ceil(OBJ_TILE);
        let t = global_index() / WARP as usize;
        // Whole warps leave together, so every lane left reaches each ballot.
        if t >= g.nyo.div_ceil(OBJ_TILE) as usize * ntx as usize {
            return;
        }
        let (ty, tx) = ((t / ntx as usize) as u32, (t % ntx as usize) as u32);
        let lane = warp::lane_id();
        unsafe {
            let (f0, nf) = stack_frames(ctl, g);
            let (k0, nk) = (f0 * g.nmode, nf * g.nmode);
            let lo = *slabs.start.add(t);
            let cap = *slabs.start.add(t + 1) - lo;
            let len0 = *slabs.len.add(t);
            let mut added = 0u32;
            let mut base = 0u32;
            while base < nk {
                let kl = base + lane;
                let hit = kl < nk && {
                    let v = *views.add((k0 + kl) as usize);
                    let (y0, y1) = axis_tiles(v.oy, g.ny, v.fy);
                    let (x0, x1) = axis_tiles(v.ox, g.nx, v.fx);
                    y0 <= ty && ty <= y1 && x0 <= tx && tx <= x1
                };
                let hits = warp::ballot(hit);
                let slot = len0 + added + (hits & warp::lanemask_lt()).count_ones();
                if hit && slot < cap {
                    *slabs.views.add((lo + slot) as usize) = k0 + kl;
                }
                added += hits.count_ones();
                base += WARP;
            }
            if lane == 0 && added > 0 {
                let want = len0 + added;
                *slabs.len.add(t) = want.min(cap);
                if want > cap {
                    DeviceAtomicU32::from_ptr(overflow)
                        .fetch_add(want - cap, AtomicOrdering::Relaxed);
                }
            }
        }
    }

    /// Adds the stack's object-update sums to `sums` `[nobj, nyo, nxo]`: the exact
    /// transpose of [`obj_probe_to_psi`] applied to `psi`, `num += sum_v B_v^T (conj(P_v q)
    /// psi_v)`, and `den += sum_v B_v^T |P_v|^2` with `B_v` the bilinear sampling of view
    /// `v`. Each object pixel is owned by one thread, which adds its views in ascending `k`
    /// from the tile's run: the result does not depend on how frames are split into stacks.
    /// A pixel's sums are read and written only if its tile's run holds a view of the stack.
    /// Grid `(ntx, nty, nobj)` of `OBJ_TILE` tiles, block `(OBJ_TILE, OBJ_TILE, 1)`.
    ///
    /// # Safety
    ///
    /// As [`obj_probe_to_psi`] with `psi` read; `sums` hold `[nobj, nyo, nxo]`; `tiles` as
    /// [`TilePtrs`] says for tile `ty * ntx + tx`, its run covering every view of the stack
    /// that samples the tile.
    #[kernel]
    pub unsafe fn psi_to_obj(
        psi: *const f32,
        probe: *const f32,
        views: *const ViewAddr,
        tiles: TilePtrs,
        sums: SumPtrs,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        let (tile, y, x) = tile_and_pixel();
        if x >= g.nxo || y >= g.nyo {
            return;
        }
        let iobj = thread::blockIdx_z();
        let n = g.ny as usize * g.nx as usize;
        let px = (iobj as usize * g.nyo as usize + y as usize) * g.nxo as usize + x as usize;
        unsafe {
            let (f0, nf) = stack_frames(ctl, g);
            let (k0, k1) = (f0 * g.nmode, (f0 + nf) * g.nmode);
            let (mut lo, end) = tile_run(tiles, tile);
            // First entry >= k0.
            let mut hi = end;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if *tiles.views.add(mid as usize) < k0 {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            if lo == end || *tiles.views.add(lo as usize) >= k1 {
                return;
            }
            let mut acc = load_c(sums.num, px);
            let mut norm = *sums.den.add(px);
            let mut j = lo;
            while j < end {
                let k = *tiles.views.add(j as usize);
                if k >= k1 {
                    break;
                }
                j += 1;
                let v = *views.add(k as usize);
                if v.obj != iobj {
                    continue;
                }
                let plane = (k - k0) as usize * n;
                let src = obj_pixel_sources(v, y, x, g.ny, g.nx);
                for &(pry, prx, w) in src.iter().flatten() {
                    let p = load_c(probe, v.probe as usize * n + (pry * g.nx + prx) as usize);
                    let q = quad_phase(g.quad, pry, prx, g.ny, g.nx);
                    let ps = load_c(psi, plane + psi_index(pry, prx, g.ny, g.nx));
                    let u = cmul_conj(cmul(p, q), ps);
                    acc = (acc.0 + w * u.0, acc.1 + w * u.1);
                    norm += w * (p.0 * p.0 + p.1 * p.1);
                }
            }
            store_c(sums.num, px, acc);
            *sums.den.add(px) = norm;
        }
    }

    /// Adds the stack's probe-update sums to `sums` `[nprobe, ny, nx]`:
    /// `num += sum_v conj(O_v q) psi_v`, `den += sum_v |O_v|^2` over the views of each probe
    /// mode, in ascending `k`; a mode without a view in the stack is not written. One thread
    /// per element of `[nprobe, ny, nx]`.
    ///
    /// # Safety
    ///
    /// As [`obj_probe_to_psi`] with `psi` read; `sums` hold `[nprobe, ny, nx]`.
    #[kernel]
    pub unsafe fn psi_to_probe(
        psi: *const f32,
        obj: *const f32,
        views: *const ViewAddr,
        sums: SumPtrs,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let n = g.ny as usize * g.nx as usize;
        let t = global_index();
        if t >= g.nprobe as usize * n {
            return;
        }
        let (mode, i) = ((t / n) as u32, t % n);
        let (y, x) = ((i / g.nx as usize) as u32, (i % g.nx as usize) as u32);
        let q = quad_phase(g.quad, y, x, g.ny, g.nx);
        let at = psi_index(y, x, g.ny, g.nx);
        unsafe {
            let mut acc = load_c(sums.num, t);
            let mut norm = *sums.den.add(t);
            let mut added = false;
            let k0 = f0 as usize * g.nmode as usize;
            for kl in 0..nf as usize * g.nmode as usize {
                let v = *views.add(k0 + kl);
                if v.probe != mode {
                    continue;
                }
                let o = sample_obj(obj, v, y, x, g.nyo, g.nxo);
                let u = cmul_conj(cmul(o, q), load_c(psi, kl * n + at));
                acc = (acc.0 + u.0, acc.1 + u.1);
                norm += o.0 * o.0 + o.1 * o.1;
                added = true;
            }
            if added {
                store_c(sums.num, t, acc);
                *sums.den.add(t) = norm;
            }
        }
    }

    /// `*out = max(*out, x[0..n] ...)` as `f32` bits, over the non-negative values of `x`
    /// (negative values and NaN are skipped). `*out` must start as the bits of a
    /// non-negative float, e.g. zero. Grid-stride over `n`.
    ///
    /// # Safety
    ///
    /// `x` holds `n` floats, `out` one `u32`; block `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn max_nonneg_f32(x: *const f32, n: u32, out: *mut u32) {
        static mut RED: SharedArray<f32, 256> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as usize;
        let stride = thread::gridDim_x() as usize * PIXEL_THREADS as usize;
        let mut m = 0.0f32;
        let mut i = global_index();
        while i < n as usize {
            let v = unsafe { *x.add(i) };
            if v > m {
                m = v;
            }
            i += stride;
        }
        unsafe { block_max_into((&raw mut RED).cast::<f32>(), t, m, out) }
    }

    /// [`max_nonneg_f32`] of `den` `[nobj, nyo, nxo]` over the pixels of the object tiles
    /// with views, merged into `*out`. A block whose tile has none returns without reading.
    /// Grid `(ntx, nty, nobj)` of `OBJ_TILE` tiles, block `(OBJ_TILE, OBJ_TILE, 1)`.
    ///
    /// # Safety
    ///
    /// `den` holds `[nobj, nyo, nxo]` floats, `tile_len` `ntx * nty` entries, `out` one `u32`.
    #[kernel]
    pub unsafe fn max_den_tiles(
        den: *const f32,
        tile_len: *const u32,
        out: *mut u32,
        nyo: u32,
        nxo: u32,
    ) {
        static mut RED: SharedArray<f32, 256> = SharedArray::UNINIT;
        let (tile, y, x) = tile_and_pixel();
        if unsafe { *tile_len.add(tile) } == 0 {
            return;
        }
        let mut m = 0.0f32;
        if x < nxo && y < nyo {
            let px = (thread::blockIdx_z() as usize * nyo as usize + y as usize) * nxo as usize
                + x as usize;
            let v = unsafe { *den.add(px) };
            if v > m {
                m = v;
            }
        }
        let t = (thread::threadIdx_y() * OBJ_TILE + thread::threadIdx_x()) as usize;
        unsafe { block_max_into((&raw mut RED).cast::<f32>(), t, m, out) }
    }

    /// PyNX `ObjNorm`: `x = (num + reg x) / max(den + reg, 1e-12)` with
    /// `reg = inertia * f32::from_bits(*max_bits)`, for `n` elements, zeroing the `n` sums it
    /// consumed. Every operation is rounded on its own (no FMA contraction), so the result is
    /// the plain f32 formula.
    ///
    /// # Safety
    ///
    /// `x`, `sums.num` hold `n` complex values, `sums.den` `n` floats, `max_bits` one `u32`.
    #[kernel]
    pub unsafe fn normalize_update(
        x: *mut f32,
        sums: SumPtrs,
        n: u32,
        max_bits: *const u32,
        inertia: f32,
    ) {
        let g = global_index();
        if g >= n as usize {
            return;
        }
        unsafe { normalize_at(x, sums, g, inertia, f32::from_bits(*max_bits)) }
    }

    /// [`normalize_update`] of `x` `[nobj, nyo, nxo]` on the pixels of the object tiles with
    /// views, zeroing their sums; a block whose tile has none returns without reading or
    /// writing. Grid `(ntx, nty, nobj)` of `OBJ_TILE` tiles, block `(OBJ_TILE, OBJ_TILE, 1)`.
    ///
    /// # Safety
    ///
    /// `x`, `sums` hold `[nobj, nyo, nxo]` values, `tile_len` `ntx * nty` entries, `max_bits`
    /// one `u32`.
    #[kernel]
    pub unsafe fn normalize_tiles(
        x: *mut f32,
        sums: SumPtrs,
        tile_len: *const u32,
        max_bits: *const u32,
        inertia: f32,
        nyo: u32,
        nxo: u32,
    ) {
        let (tile, y, xo) = tile_and_pixel();
        if xo >= nxo || y >= nyo || unsafe { *tile_len.add(tile) } == 0 {
            return;
        }
        let px = (thread::blockIdx_z() as usize * nyo as usize + y as usize) * nxo as usize
            + xo as usize;
        unsafe { normalize_at(x, sums, px, inertia, f32::from_bits(*max_bits)) }
    }

    /// DM reflection (PyNX `DM1` and the copy `DM2` reads): with `psi1 = (2 - alpha) po -
    /// (1 - alpha) psi`, sets `psi = psi1` and `po = (1 - alpha) (po - psi1)`, for `n`
    /// elements.
    ///
    /// # Safety
    ///
    /// `psi`, `po` hold `n` complex values.
    #[kernel]
    pub unsafe fn dm_reflect(psi: *mut f32, po: *mut f32, n: u32, alpha: f32) {
        let g = global_index();
        if g >= n as usize {
            return;
        }
        unsafe {
            let (p, o) = (load_c(psi, g), load_c(po, g));
            let (a, b) = (2.0 - alpha, 1.0 - alpha);
            let p1 = (a * o.0 - b * p.0, a * o.1 - b * p.1);
            store_c(psi, g, p1);
            store_c(po, g, (b * (o.0 - p1.0), b * (o.1 - p1.1)));
        }
    }

    /// `y += x` for `n` complex values.
    ///
    /// # Safety
    ///
    /// `y`, `x` hold `n` complex values.
    #[kernel]
    pub unsafe fn add_c32(y: *mut f32, x: *const f32, n: u32) {
        let g = global_index();
        if g >= n as usize {
            return;
        }
        unsafe {
            let (a, b) = (load_c(y, g), load_c(x, g));
            store_c(y, g, (a.0 + b.0, a.1 + b.1));
        }
    }

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
