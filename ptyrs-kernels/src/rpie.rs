//! rPIE kernels (pty-chi `RPIEReconstructor`), on the layouts of [`crate::pixel`]. The
//! object and probe steps reuse `psi_to_obj` / `psi_to_probe` with weighted inputs built
//! here.

use cuda_host::cuda_module;

/// Values per block of `rpie_power_max`: a plane of `n` values is `ceil(n / POWER_CHUNK)`
/// contiguous chunks, the last one short.
pub const POWER_CHUNK: u32 = 2048;

/// Planes and regulariser of `rpie_weight`: `nk` planes of `n` complex values, plane `k`
/// against its max `mx[k]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WeightGeom {
    pub nk: u32,
    pub n: u32,
    pub alpha: f32,
}

/// The probe `rpie_overlap` gathers mode `mode` of: the one every frame shares (`per_frame`
/// 0) or each frame's own.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OverlapOf {
    pub mode: u32,
    pub per_frame: u32,
}

#[cuda_module]
pub mod rpie_kernels {
    use super::POWER_CHUNK;
    use crate::bh::BhObs;
    use crate::pixel::*;
    use crate::*;
    use cuda_device::SharedArray;

    /// pty-chi's modulus constraint on the forward-FFT'd stack, with the `1 / (ny * nx)` of
    /// the unnormalized round trip folded in: each pixel's modes are scaled by
    /// `sqrt(obs + 1e-7) / (sqrt(icalc) + 1e-7) / n` with `icalc = sum_m |psi|^2 / n`, or by
    /// `1 / n` where `obs`, the count of the frame at the plane's epoch position, is
    /// `MASKED`. One thread per element of `[nf_max, ny, nx]`.
    ///
    /// # Safety
    ///
    /// `psi` `[nf_max * nmode, ny, nx]`, `obs` as [`BhObs`] says, `ctl` one entry.
    #[kernel]
    pub unsafe fn rpie_amplitude(psi: *mut f32, obs: BhObs, ctl: *const StackCtl, g: StackGeom) {
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
            let frame = *obs.order.add(f0 as usize + fl) as usize;
            let obs = *obs.counts.add(frame * n + d);
            let s = if obs == MASKED {
                inv_n
            } else {
                let calc = (mode_power(psi, plane0, nmode, n, at) * inv_n).sqrt();
                (obs as f32 + 1e-7).sqrt() / (calc + 1e-7) * inv_n
            };
            for m in 0..nmode as usize {
                let j = (plane0 + m) * n + at;
                let v = load_c(psi, j);
                store_c(psi, j, (v.0 * s, v.1 * s));
            }
        }
    }

    /// Exit waves of each frame's own probe: `psi[kl, j] = O_v(y, x) * w[fl * nprobe +
    /// v.probe, j]` for every view `v = views[f0 * nmode + kl]` of the stack, `fl = kl /
    /// nmode`, `j = psi_index(y, x)`, `O_v` sampled as `obj_probe_to_psi` samples it. `w`
    /// holds the probe modes of each frame in the exit waves' layout, the quadratic phase
    /// included; the product is the fused shifted row pass's. One thread per element of
    /// `[nf_max * nmode, ny, nx]`, in stored order.
    ///
    /// # Safety
    ///
    /// `obj` `[nobj, nyo, nxo]`, `w` `[nf_max * nprobe, ny, nx]`, `views` `nframe * nmode`
    /// records in range for those shapes, `psi` `[nf_max * nmode, ny, nx]`, `ctl` one entry.
    #[kernel]
    pub unsafe fn rpie_obj_wave_to_psi(
        obj: *const f32,
        w: *const f32,
        views: *const ViewAddr,
        psi: *mut f32,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let (ny, nx) = (g.ny, g.nx);
        let n = ny as usize * nx as usize;
        let t = global_index();
        if t >= nf as usize * g.nmode as usize * n {
            return;
        }
        let (kl, j) = (t / n, t % n);
        // Natural pixel stored at `j` (the shift is its own inverse).
        let y = ((j / nx as usize) as u32 + ny / 2) % ny;
        let x = ((j % nx as usize) as u32 + nx / 2) % nx;
        unsafe {
            let v = *views.add(f0 as usize * g.nmode as usize + kl);
            let plane = kl / g.nmode as usize * g.nprobe as usize + v.probe as usize;
            let o = sample_obj(obj, v, y, x, g.nyo, g.nxo);
            store_c(psi, t, cmul(o, load_c(w, plane * n + j)));
        }
    }

    /// Probe mode `of.mode`'s overlap over the stack, placed as `psi_to_obj` places the
    /// object step: `d[y, x] = (sum_k sum w |p_k(pry, prx)|^2, 0)` over the stack's views `k`
    /// of that mode and the `(pry, prx, w)` of [`obj_pixel_sources`]. `p_k` is mode `v.probe`
    /// of `p`: with `of.per_frame` 0 the probe `[nprobe, ny, nx]`, natural; else each frame's
    /// own, `[nf_max * nprobe, ny, nx]` in the exit waves' layout as
    /// [`rpie_obj_wave_to_psi`] reads `w`. Overwrites the pixels of the tiles it owns, which
    /// hold every pixel a view of the stack samples, and leaves the rest. A stack-tile
    /// launch (`for_each_stack_tile`), one object mode.
    ///
    /// # Safety
    ///
    /// As `psi_to_obj` with `p` for its probe; `d` holds `[nyo, nxo]`, the views' object
    /// is one mode.
    ///
    /// [`rpie_obj_wave_to_psi`]: LoadedModule::rpie_obj_wave_to_psi
    #[kernel]
    pub unsafe fn rpie_overlap(
        p: *const f32,
        of: super::OverlapOf,
        views: *const ViewAddr,
        tiles: TilePtrs,
        d: *mut f32,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        static mut SHARED: SharedArray<u32, STACK_TILE_SHARED> = SharedArray::UNINIT;
        let shared = (&raw mut SHARED).cast::<u32>();
        let n = g.ny as usize * g.nx as usize;
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let (k0, nk) = (f0 * g.nmode, nf * g.nmode);
        let pixel = move |lo: u32, end: u32, y: u32, x: u32| unsafe {
            if x >= g.nxo || y >= g.nyo {
                return;
            }
            let mut s = 0.0f32;
            let mut j = lo;
            while j < end {
                let k = *tiles.views.add(j as usize);
                let i = k.wrapping_sub(k0);
                if i >= nk {
                    break;
                }
                j += 1;
                let v = *views.add(k as usize);
                if v.probe != of.mode {
                    continue;
                }
                let plane = (i / g.nmode * g.nprobe + v.probe) as usize * n;
                obj_pixel_sources(v, y, x, g.ny, g.nx, |pry, prx, w| {
                    let at = if of.per_frame == 0 {
                        (v.probe * g.ny + pry) as usize * g.nx as usize + prx as usize
                    } else {
                        plane + psi_index(pry, prx, g.ny, g.nx)
                    };
                    let a = load_c(p, at);
                    s += w * (a.0 * a.0 + a.1 * a.1);
                });
            }
            store_c(d, y as usize * g.nxo as usize + x as usize, (s, 0.0));
        };
        unsafe { for_each_stack_tile(views, tiles, shared, ctl, g, pixel) }
    }

    /// The overlap each view of probe mode `mode` reads back: `dens[kl, j] = max(0, Re O(y,
    /// x))` for view `v = views[f0 * nmode + kl]` of the stack, `O` the one-mode object `d`
    /// sampled as `sample_obj` samples `v`'s, `(y, x)` the natural pixel stored at `j`. The
    /// other views' entries are left. One thread per element of `[nf_max * nmode, ny, nx]`.
    ///
    /// # Safety
    ///
    /// `d` `[nyo, nxo]`, `views` `nframe * nmode` records in range for that object, `dens`
    /// `[nf_max * nmode, ny, nx]` floats, `ctl` one entry.
    #[kernel]
    pub unsafe fn rpie_overlap_read(
        d: *const f32,
        mode: u32,
        views: *const ViewAddr,
        dens: *mut f32,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let (ny, nx) = (g.ny, g.nx);
        let n = ny as usize * nx as usize;
        let t = global_index();
        if t >= nf as usize * g.nmode as usize * n {
            return;
        }
        let (kl, j) = (t / n, t % n);
        // Natural pixel stored at `j` (the shift is its own inverse).
        let y = ((j / nx as usize) as u32 + ny / 2) % ny;
        let x = ((j % nx as usize) as u32 + nx / 2) % nx;
        unsafe {
            let mut v = *views.add(f0 as usize * g.nmode as usize + kl);
            if v.probe != mode {
                return;
            }
            v.obj = 0;
            let o = sample_obj(d, v, y, x, g.nyo, g.nxo).0;
            *dens.add(t) = if o > 0.0 { o } else { 0.0 };
        }
    }

    /// `*out = max(*out, max_i sum_{m < nsum} x[(b * nsum + m) * n + i])` over `i` in chunk
    /// `c = blockIdx.x`, `c * POWER_CHUNK .. min((c + 1) * POWER_CHUNK, n)`, of group `b =
    /// blockIdx.y` of `nsum` planes of `n` non-negative floats; `out` holds the bits of a
    /// non-negative float.
    ///
    /// # Safety
    ///
    /// `x` holds `gridDim.y * nsum` planes of `n` floats, `out` one `u32`; grid `(ceil(n /
    /// POWER_CHUNK), ngroup, 1)`, block `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn rpie_sum_max(x: *const f32, nsum: u32, n: u32, out: *mut u32) {
        static mut RED: SharedArray<f32, 256> = SharedArray::UNINIT;
        let b = thread::blockIdx_y() as usize;
        let c = thread::blockIdx_x() as usize;
        let t = thread::threadIdx_x() as usize;
        let chunk = POWER_CHUNK as usize;
        let end = (n as usize).min((c + 1) * chunk);
        let mut m = 0.0f32;
        let mut i = c * chunk + t;
        while i < end {
            let mut s = 0.0f32;
            for k in 0..nsum as usize {
                s += unsafe { *x.add((b * nsum as usize + k) * n as usize + i) };
            }
            if s > m {
                m = s;
            }
            i += PIXEL_THREADS as usize;
        }
        unsafe { block_max_into((&raw mut RED).cast::<f32>(), t, m, out) }
    }

    /// The object step's weighted `Δ`: `out[i] = x[i] / ((1 - alpha) dens[i] + alpha
    /// mx[0])` for `i < len`. One thread per element.
    ///
    /// # Safety
    ///
    /// `x` and `out` hold `len` complex values, `dens` `len` floats, `mx` one.
    #[kernel]
    pub unsafe fn rpie_overlap_weight(
        x: *const f32,
        dens: *const f32,
        mx: *const f32,
        out: *mut f32,
        len: u32,
        alpha: f32,
    ) {
        let t = global_index();
        if t >= len as usize {
            return;
        }
        unsafe {
            let d = (1.0 - alpha) * *dens.add(t) + alpha * *mx;
            let v = load_c(x, t);
            store_c(out, t, (zdiv(v.0, d), zdiv(v.1, d)));
        }
    }

    /// Every frame's patch of probe mode `mode`: `out[fl, i] = (|p[mode, i]|^2, 0)` over the
    /// `len` values of `out`, planes of `n`. One thread per element.
    ///
    /// # Safety
    ///
    /// `p` holds plane `mode` of `n` complex values, `out` `len` complex values.
    #[kernel]
    pub unsafe fn rpie_mode_power(p: *const f32, mode: u32, n: u32, len: u32, out: *mut f32) {
        let t = global_index();
        if t >= len as usize {
            return;
        }
        unsafe {
            let a = load_c(p, mode as usize * n as usize + t % n as usize);
            store_c(out, t, (a.0 * a.0 + a.1 * a.1, 0.0));
        }
    }

    /// `chunks[b * nchunk + c] = max(0, max_i sum_{m < nsum} |x[(b * nsum + m) * n + i]|^2)`
    /// over `i` in chunk `c = blockIdx.x`, `c * POWER_CHUNK .. min((c + 1) * POWER_CHUNK,
    /// n)`, of group `b = blockIdx.y` of `nsum` planes, `nchunk = ceil(n / POWER_CHUNK)`;
    /// [`rpie_max_chunks`] takes each group's max over its chunks.
    ///
    /// # Safety
    ///
    /// `x` holds `gridDim.y * nsum` planes of `n` complex values, `chunks` `gridDim.y *
    /// nchunk` floats; grid `(nchunk, ngroup, 1)`, block `(PIXEL_THREADS, 1, 1)`.
    ///
    /// [`rpie_max_chunks`]: LoadedModule::rpie_max_chunks
    #[kernel]
    pub unsafe fn rpie_power_max(x: *const f32, nsum: u32, n: u32, chunks: *mut f32) {
        static mut RED: SharedArray<f32, 256> = SharedArray::UNINIT;
        let b = thread::blockIdx_y() as usize;
        let c = thread::blockIdx_x() as usize;
        let t = thread::threadIdx_x() as usize;
        let chunk = POWER_CHUNK as usize;
        let end = (n as usize).min((c + 1) * chunk);
        let mut m = 0.0f32;
        let mut i = c * chunk + t;
        while i < end {
            let p = unsafe { mode_power(x, b * nsum as usize, nsum, n as usize, i) };
            if p > m {
                m = p;
            }
            i += PIXEL_THREADS as usize;
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
                *chunks.add(b * (n as usize).div_ceil(chunk) + c) = *s;
            }
        }
    }

    /// `out[b] = max(0, max_c chunks[b * nchunk + c])` for each of `ngroup` groups, one thread
    /// per group: with [`rpie_power_max`]'s chunks, the largest summed power of each group.
    ///
    /// # Safety
    ///
    /// `chunks` holds `ngroup * nchunk` floats, `out` `ngroup`.
    ///
    /// [`rpie_power_max`]: LoadedModule::rpie_power_max
    #[kernel]
    pub unsafe fn rpie_max_chunks(chunks: *const f32, nchunk: u32, ngroup: u32, out: *mut f32) {
        let b = global_index();
        if b >= ngroup as usize {
            return;
        }
        let mut m = 0.0f32;
        for c in 0..nchunk as usize {
            let p = unsafe { *chunks.add(b * nchunk as usize + c) };
            if p > m {
                m = p;
            }
        }
        unsafe { *out.add(b) = m };
    }

    /// The probe step's weight: `x[k, i] /= (1 - alpha) |w[k, i]|^2 + alpha mx[k]` over
    /// the `[nk, n]` of `g`. One thread per element.
    ///
    /// # Safety
    ///
    /// `x`, `w` hold `nk * n` complex values, `mx` `nk` floats.
    #[kernel]
    pub unsafe fn rpie_weight(x: *mut f32, w: *const f32, mx: *const f32, g: super::WeightGeom) {
        let t = global_index();
        if t >= g.nk as usize * g.n as usize {
            return;
        }
        let k = t / g.n as usize;
        unsafe {
            let wv = load_c(w, t);
            let d = (1.0 - g.alpha) * (wv.0 * wv.0 + wv.1 * wv.1) + g.alpha * *mx.add(k);
            let v = load_c(x, t);
            store_c(x, t, (zdiv(v.0, d), zdiv(v.1, d)));
        }
    }

    /// The rPIE step, consuming its numerator: `x[i] += s * num[i]`, then `num[i] = 0`.
    /// Product and sum round separately, as `scale_c32` then `add_c32` did. One thread per
    /// element.
    ///
    /// # Safety
    ///
    /// `x` and `num` hold `n` complex values.
    #[kernel]
    pub unsafe fn rpie_step(x: *mut f32, num: *mut f32, n: u32, s: f32) {
        use cuda_device::float::{add_rn_f32, mul_rn_f32};
        let g = global_index();
        if g >= n as usize {
            return;
        }
        unsafe {
            let (a, d) = (load_c(x, g), load_c(num, g));
            let step = |a: f32, d: f32| add_rn_f32(a, mul_rn_f32(d, s));
            store_c(x, g, (step(a.0, d.0), step(a.1, d.1)));
            store_c(num, g, (0.0, 0.0));
        }
    }
}
