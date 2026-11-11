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
    use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};

    // M1 pixel kernels. Shapes, orders and the FFT scaling convention are documented on the
    // ptyrs-gpu wrappers, which check them; `pixel` documents the index maps. A "stack" is
    // views `k0 .. k0 + nk` (frame-major, `k = frame * nmode + mode`) or frames
    // `f0 .. f0 + nf`; its exit waves are planes `0 .. nk` of `psi`. 1D kernels are launched
    // with `PIXEL_THREADS` threads per block over the stated element count.

    /// `psi[kl, psi_index(y, x)] = O_v(y, x) * P_v(y, x) * exp(i quad r^2)` for every view
    /// `v = views[k0 + kl]` of `g`, one thread per element of `[nk, ny, nx]`.
    ///
    /// # Safety
    ///
    /// `obj` `[nobj, nyo, nxo]`, `probe` `[nprobe, ny, nx]`, `views` `k0 + nk` records in
    /// range for those shapes, `psi` `[nk, ny, nx]`.
    #[kernel]
    pub unsafe fn obj_probe_to_psi(
        obj: *const f32,
        probe: *const f32,
        views: *const ViewAddr,
        psi: *mut f32,
        g: StackGeom,
    ) {
        let n = g.ny as usize * g.nx as usize;
        let t = global_index();
        if t >= g.nk as usize * n {
            return;
        }
        let (kl, i) = (t / n, t % n);
        let (y, x) = ((i / g.nx as usize) as u32, (i % g.nx as usize) as u32);
        unsafe {
            let v = *views.add(g.k0 as usize + kl);
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
    /// frame `fl < nf` and detector pixel `d`, one thread per element of `[nf, ny, nx]`.
    ///
    /// # Safety
    ///
    /// `psi` `[nf * nmode, ny, nx]`, `icalc` `[nf, ny, nx]`.
    #[kernel]
    pub unsafe fn mode_intensity(
        psi: *const f32,
        icalc: *mut f32,
        nf: u32,
        nmode: u32,
        ny: u32,
        nx: u32,
    ) {
        let n = ny as usize * nx as usize;
        let g = global_index();
        if g >= nf as usize * n {
            return;
        }
        let (fl, d) = (g / n, g % n);
        let at = psi_index((d / nx as usize) as u32, (d % nx as usize) as u32, ny, nx);
        unsafe {
            *icalc.add(g) = mode_power(psi, fl * nmode as usize, nmode, n, at) / n as f32;
        }
    }

    /// Replaces the modulus of every mode with the observed amplitude (PyNX
    /// `ProjectionAmplitude`, no background), folding in the `1 / (ny * nx)` an unnormalized
    /// FFT round trip needs: `psi *= sqrt(obs / max(icalc, 1e-12)) / (ny * nx)`, or
    /// `psi /= ny * nx` where `obs` is `MASKED`. One thread per element of `[nf, ny, nx]`.
    ///
    /// # Safety
    ///
    /// `psi` `[nf * nmode, ny, nx]`, `counts` `[f0 + nf, ny, nx]`.
    #[kernel]
    pub unsafe fn apply_amplitude(
        psi: *mut f32,
        counts: *const u16,
        f0: u32,
        nf: u32,
        nmode: u32,
        ny: u32,
        nx: u32,
    ) {
        let n = ny as usize * nx as usize;
        let g = global_index();
        if g >= nf as usize * n {
            return;
        }
        let (fl, d) = (g / n, g % n);
        let at = psi_index((d / nx as usize) as u32, (d % nx as usize) as u32, ny, nx);
        let inv_n = 1.0 / n as f32;
        let plane0 = fl * nmode as usize;
        unsafe {
            let obs = *counts.add(f0 as usize * n + g);
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
    /// `calc`. One `PIXEL_THREADS` block per frame; `out[fl]` gets the frame's sums.
    ///
    /// # Safety
    ///
    /// `psi` `[nf * nmode, ny, nx]`, `counts` `[f0 + nf, ny, nx]`, `out` `nf` records; grid
    /// `(nf, 1, 1)`, block `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn llk(
        psi: *const f32,
        counts: *const u16,
        out: *mut LlkPartial,
        f0: u32,
        nmode: u32,
        ny: u32,
        nx: u32,
    ) {
        // Four `PIXEL_THREADS` segments: Poisson, Gaussian, Euclidean, calc.
        static mut RED: SharedArray<f32, 1024> = SharedArray::UNINIT;
        let n = ny as usize * nx as usize;
        let fl = thread::blockIdx_x() as usize;
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

    /// Adds the stack's object-update sums to `sums` `[nobj, nyo, nxo]`: the exact
    /// transpose of [`obj_probe_to_psi`] applied to `psi`, `num += sum_v B_v^T (conj(P_v q)
    /// psi_v)`, and `den += sum_v B_v^T |P_v|^2` with `B_v` the bilinear sampling of view
    /// `v`. Each object pixel is owned by one thread, which adds its views in ascending `k`
    /// from the tile list: the result does not depend on how frames are split into stacks.
    /// Grid `(ceil(nxo / OBJ_TILE), ceil(nyo / OBJ_TILE), nobj)`, block
    /// `(OBJ_TILE, OBJ_TILE, 1)`.
    ///
    /// # Safety
    ///
    /// As [`obj_probe_to_psi`] with `psi` read; `sums` hold `[nobj, nyo, nxo]`;
    /// `tile_start` holds `ntiles + 1` ascending offsets into `tile_views` (tile
    /// `ty * gridDim.x + tx`), whose runs are ascending view indices covering every view
    /// that samples the tile.
    #[kernel]
    pub unsafe fn psi_to_obj(
        psi: *const f32,
        probe: *const f32,
        views: *const ViewAddr,
        tile_start: *const u32,
        tile_views: *const u32,
        sums: SumPtrs,
        g: StackGeom,
    ) {
        let x = thread::blockIdx_x() * OBJ_TILE + thread::threadIdx_x();
        let y = thread::blockIdx_y() * OBJ_TILE + thread::threadIdx_y();
        if x >= g.nxo || y >= g.nyo {
            return;
        }
        let iobj = thread::blockIdx_z();
        let n = g.ny as usize * g.nx as usize;
        let tile = (thread::blockIdx_y() * thread::gridDim_x() + thread::blockIdx_x()) as usize;
        let px = (iobj as usize * g.nyo as usize + y as usize) * g.nxo as usize + x as usize;
        unsafe {
            let (mut lo, end) = (*tile_start.add(tile), *tile_start.add(tile + 1));
            // First entry >= k0.
            let mut hi = end;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if *tile_views.add(mid as usize) < g.k0 {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            let mut acc = load_c(sums.num, px);
            let mut norm = *sums.den.add(px);
            let mut j = lo;
            while j < end {
                let k = *tile_views.add(j as usize);
                if k >= g.k0 + g.nk {
                    break;
                }
                j += 1;
                let v = *views.add(k as usize);
                if v.obj != iobj {
                    continue;
                }
                let plane = (k - g.k0) as usize * n;
                let src = obj_pixel_sources(v, y, x, g.ny, g.nx, g.nyo, g.nxo);
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
    /// mode, in ascending `k`. One thread per element of `[nprobe, ny, nx]`.
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
        g: StackGeom,
    ) {
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
            for kl in 0..g.nk as usize {
                let v = *views.add(g.k0 as usize + kl);
                if v.probe != mode {
                    continue;
                }
                let o = sample_obj(obj, v, y, x, g.nyo, g.nxo);
                let u = cmul_conj(cmul(o, q), load_c(psi, kl * n + at));
                acc = (acc.0 + u.0, acc.1 + u.1);
                norm += o.0 * o.0 + o.1 * o.1;
            }
            store_c(sums.num, t, acc);
            *sums.den.add(t) = norm;
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
        let s = (&raw mut RED).cast::<f32>();
        unsafe {
            *s.add(t) = m;
            let mut half = PIXEL_THREADS as usize / 2;
            while half > 0 {
                thread::sync_threads();
                if t < half && *s.add(t + half) > *s.add(t) {
                    *s.add(t) = *s.add(t + half);
                }
                half /= 2;
            }
            if t == 0 {
                // Non-negative floats order like their bit patterns.
                DeviceAtomicU32::from_ptr(out).fetch_max((*s).to_bits(), AtomicOrdering::Relaxed);
            }
        }
    }

    /// PyNX `ObjNorm`: `x = (num + reg x) / max(den + reg, 1e-12)` with
    /// `reg = inertia * f32::from_bits(*max_bits)`, for `n` elements.
    ///
    /// # Safety
    ///
    /// `x`, `num` hold `n` complex values, `den` `n` floats, `max_bits` one `u32`.
    #[kernel]
    pub unsafe fn normalize_update(
        x: *mut f32,
        num: *const f32,
        den: *const f32,
        n: u32,
        max_bits: *const u32,
        inertia: f32,
    ) {
        let g = global_index();
        if g >= n as usize {
            return;
        }
        unsafe {
            let reg = inertia * f32::from_bits(*max_bits);
            let d = (*den.add(g) + reg).max(1e-12);
            let (a, b) = (load_c(num, g), load_c(x, g));
            store_c(x, g, ((a.0 + reg * b.0) / d, (a.1 + reg * b.1) / d));
        }
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
