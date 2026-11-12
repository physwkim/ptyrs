//! rPIE kernels (pty-chi `RPIEReconstructor`), on the layouts of [`crate::pixel`]. The
//! object and probe steps reuse `psi_to_obj` / `psi_to_probe` with weighted inputs built
//! here, so these kernels need no view addresses.

use cuda_host::cuda_module;

/// Values per block of `rpie_power_max`: a plane of `n` values is `ceil(n / POWER_CHUNK)`
/// contiguous chunks, the last one short.
pub const POWER_CHUNK: u32 = 2048;

/// Planes and regulariser of `rpie_weight`: `nk` planes of `n` complex values; plane `k`
/// reads its max at `mx[k * mstride]` (`mstride` 0 or 1).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WeightGeom {
    pub nk: u32,
    pub n: u32,
    pub mstride: u32,
    pub alpha: f32,
}

#[cuda_module]
pub mod rpie_kernels {
    use super::POWER_CHUNK;
    use crate::pixel::*;
    use crate::*;
    use cuda_device::SharedArray;

    /// pty-chi's modulus constraint on the forward-FFT'd stack, with the `1 / (ny * nx)` of
    /// the unnormalized round trip folded in: each pixel's modes are scaled by
    /// `sqrt(obs + 1e-7) / (sqrt(icalc) + 1e-7) / n` with `icalc = sum_m |psi|^2 / n`, or by
    /// `1 / n` where `obs` is `MASKED`. One thread per element of `[nf_max, ny, nx]`.
    ///
    /// # Safety
    ///
    /// `psi` `[nf_max * nmode, ny, nx]`, `counts` `[nframe, ny, nx]`, `ctl` one entry.
    #[kernel]
    pub unsafe fn rpie_amplitude(
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

    /// The rPIE step weights: `out[k, i] = x[k, i] / ((1 - alpha) |w[k, i]|^2 + alpha
    /// mx[k * mstride])` over the `[nk, n]` of `g` (`mstride` 0 shares `mx[0]` between
    /// planes). `out` may be `x`. One thread per element.
    ///
    /// # Safety
    ///
    /// `x`, `w`, `out` hold `nk * n` complex values, `mx` `(nk - 1) * mstride + 1` floats.
    #[kernel]
    pub unsafe fn rpie_weight(
        x: *const f32,
        w: *const f32,
        mx: *const f32,
        out: *mut f32,
        g: super::WeightGeom,
    ) {
        let t = global_index();
        if t >= g.nk as usize * g.n as usize {
            return;
        }
        let k = t / g.n as usize;
        unsafe {
            let wv = load_c(w, t);
            let m = *mx.add(k * g.mstride as usize);
            let d = (1.0 - g.alpha) * (wv.0 * wv.0 + wv.1 * wv.1) + g.alpha * m;
            let v = load_c(x, t);
            store_c(out, t, (v.0 / d, v.1 / d));
        }
    }

    /// The rPIE step, consuming its sums: `x[i] += s * num[i]`, then `num[i] = den[i] = 0`.
    /// Product and sum round separately, as `scale_c32` then `add_c32` did. One thread per
    /// element.
    ///
    /// # Safety
    ///
    /// `x` and `sums.num` hold `n` complex values, `sums.den` `n` floats.
    #[kernel]
    pub unsafe fn rpie_step(x: *mut f32, sums: SumPtrs, n: u32, s: f32) {
        use cuda_device::float::{add_rn_f32, mul_rn_f32};
        let g = global_index();
        if g >= n as usize {
            return;
        }
        unsafe {
            let (a, d) = (load_c(x, g), load_c(sums.num, g));
            let step = |a: f32, d: f32| add_rn_f32(a, mul_rn_f32(d, s));
            store_c(x, g, (step(a.0, d.0), step(a.1, d.1)));
            store_c(sums.num, g, (0.0, 0.0));
            *sums.den.add(g) = 0.0;
        }
    }
}
