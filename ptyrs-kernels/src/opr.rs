//! pty-chi's orthogonal probe relaxation (OPR) of probe mode 0 on the device, as ptyrs
//! `engine::opr` defines it: with `k` eigenmodes `U_i`, frame `f` sees `c_f P_0 + Σ_i w_fi
//! U_i`, its other probe modes shared. Every plane is `n` pixels in the probe's own layout;
//! the eigenmodes are `[k, n]`, the weights `w_fi` at `f k + i` and `c_f` at `f`, both by scan
//! frame. A batch is epoch positions `span.p0 .. span.p0 + span.nb` through `order` (scan
//! frame by epoch position); its per-frame planes are by batch frame `b`, position `p0 + b`.
//! Per-frame sums go per [`GAMMA_CHUNK`](crate::grad::GAMMA_CHUNK) chunk and are totalled
//! in chunk order; sums over the batch in frame order.

use cuda_device::device;
use cuda_host::cuda_module;

/// Floats [`opr_kernels::LoadedModule::opr_frame_sums`] writes per frame and chunk, of a
/// shared plane `A` against the frame's planes: `Σ Re(conj(A) d)`, `Σ |A|² vv`, and `Re` and
/// `Im` of `Σ conj(A) r`.
pub const OPR_SUMS: u32 = 4;

/// A batch frame's planes, each `[nb, n]` by batch frame; a null pointer is a plane
/// [`opr_kernels::LoadedModule::opr_frame_sums`] does not read, its sums 0. `d` the probe
/// direction of mode 0, `vv` the squared object modulus at it, `r` the residues.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OprFrames {
    pub d: *const f32,
    pub vv: *const f32,
    pub r: *const f32,
}

/// The scan's weights by scan frame and the order that puts a batch's epoch positions on
/// them: `w_fi` at `weights[f k + i]`, `c_f` at `intensity[f]` (null when not refined, `c_f =
/// 1`), and `order[p]` the scan frame at epoch position `p`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OprScan {
    pub order: *const u32,
    pub weights: *mut f32,
    pub intensity: *mut f32,
    pub k: u32,
}

/// `Σ_c sums[OPR_SUMS (b nchunk + c) + j]` over the chunks in order.
///
/// # Safety
///
/// `sums` holds `OPR_SUMS nchunk` floats per frame through `b`.
#[device]
unsafe fn frame_total(sums: *const f32, b: usize, nchunk: usize, j: usize) -> f32 {
    let k = OPR_SUMS as usize;
    let mut t = 0.0f32;
    for c in 0..nchunk {
        t += unsafe { *sums.add(k * (b * nchunk + c) + j) };
    }
    t
}

/// `Σ_c part[c]` over the `nchunk` chunks in order.
///
/// # Safety
///
/// `part` holds `nchunk` floats.
#[device]
unsafe fn plane_total(part: *const f32, nchunk: usize) -> f32 {
    let mut t = 0.0f32;
    for c in 0..nchunk {
        t += unsafe { *part.add(c) };
    }
    t
}

#[cuda_module]
pub mod opr_kernels {
    use super::{OPR_SUMS, OprFrames, OprScan, frame_total, plane_total};
    use crate::bh::{CHUNK_PIXELS, store_sums};
    use crate::grad::GAMMA_CHUNK;
    use crate::pixel::*;
    use crate::position::BatchSpan;
    use crate::*;
    use cuda_device::SharedArray;

    /// The batch's per-frame mode-0 planes `out[b n + j] = c_f P_0[j] + w_f0 U_0[j] + …`, in
    /// that order, `f` the scan frame at epoch position `p0 + b`.
    ///
    /// # Safety
    ///
    /// `probe0` and `out` hold `n` and `nb n` complex values, `eigen` `k n`, `scan` as
    /// [`OprScan`] says over the span's positions; 1D grid over `nb n`.
    #[kernel]
    pub unsafe fn opr_unique(
        probe0: *const f32,
        eigen: *const f32,
        scan: OprScan,
        span: BatchSpan,
        n: u32,
        out: *mut f32,
    ) {
        let i = global_index();
        let (n, k) = (n as usize, scan.k as usize);
        if i >= span.nb as usize * n {
            return;
        }
        let (b, j) = (i / n, i % n);
        let f = unsafe { *scan.order.add(span.p0 as usize + b) } as usize;
        let c = if scan.intensity.is_null() {
            1.0
        } else {
            unsafe { *scan.intensity.add(f) }
        };
        let p = unsafe { load_c(probe0, j) };
        let mut v = (c * p.0, c * p.1);
        for m in 0..k {
            let w = unsafe { *scan.weights.add(f * k + m) };
            let u = unsafe { load_c(eigen, m * n + j) };
            v = (v.0 + w * u.0, v.1 + w * u.1);
        }
        unsafe { store_c(out, i, v) };
    }

    /// The residues in place of the moved directions `m`, `[nb, n]`: `m_b[j] −= mean_b
    /// m_b[j]`, the mean summed in frame order.
    ///
    /// # Safety
    ///
    /// `m` holds `nb n` complex values; 1D grid over `n`.
    #[kernel]
    pub unsafe fn opr_residual(m: *mut f32, nb: u32, n: u32) {
        let j = global_index();
        let n = n as usize;
        if j >= n {
            return;
        }
        let mut s = (0.0f32, 0.0f32);
        for b in 0..nb as usize {
            let v = unsafe { load_c(m, b * n + j) };
            s = (s.0 + v.0, s.1 + v.1);
        }
        let s = (s.0 / nb as f32, s.1 / nb as f32);
        for b in 0..nb as usize {
            let v = unsafe { load_c(m, b * n + j) };
            unsafe { store_c(m, b * n + j, (v.0 - s.0, v.1 - s.1)) };
        }
    }

    /// The [`OPR_SUMS`] of chunk `c = blockIdx.x` of batch frame `b = blockIdx.y` over the
    /// pixels `c GAMMA_CHUNK .. min((c + 1) GAMMA_CHUNK, n)` of the shared plane `a` against
    /// the frame's planes of `p`, into `sums[OPR_SUMS (b nchunk + c) ..]`.
    ///
    /// # Safety
    ///
    /// `a` holds `n` complex values, `p` as [`OprFrames`] says, `sums` `OPR_SUMS nchunk` floats
    /// per batch frame; grid `(nchunk, nb, 1)`, block `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn opr_frame_sums(a: *const f32, p: OprFrames, n: u32, sums: *mut f32) {
        static mut RED: SharedArray<f32, 1024> = SharedArray::UNINIT;
        let OprFrames { d, vv, r } = p;
        let n = n as usize;
        let (c, b) = (thread::blockIdx_x() as usize, thread::blockIdx_y() as usize);
        let chunk = GAMMA_CHUNK as usize;
        let end = n.min((c + 1) * chunk);
        let base = c * chunk + thread::threadIdx_x() as usize;
        let mut acc = [0.0f32; OPR_SUMS as usize];
        for g in 0..CHUNK_PIXELS {
            let j = base + g * PIXEL_THREADS as usize;
            if j < end {
                let u = unsafe { load_c(a, j) };
                let at = b * n + j;
                if !d.is_null() {
                    acc[0] += cmul_conj(u, unsafe { load_c(d, at) }).0;
                }
                if !vv.is_null() {
                    acc[1] += (u.0 * u.0 + u.1 * u.1) * unsafe { *vv.add(at) };
                }
                if !r.is_null() {
                    let z = cmul_conj(u, unsafe { load_c(r, at) });
                    acc[2] += z.0;
                    acc[3] += z.1;
                }
            }
        }
        let nchunk = n.div_ceil(chunk);
        unsafe {
            store_sums::<4>(
                (&raw mut RED).cast::<f32>(),
                acc,
                sums.add(OPR_SUMS as usize * (b * nchunk + c)),
            );
        }
    }

    /// The main mode's weights from the [`opr_frame_sums`] of `P_0` against `d` and `vv`:
    /// `c_f += 0.1 Σ Re(conj(P_0) d_b) / Σ |P_0|² vv_b`, `f` the scan frame at epoch position
    /// `p0 + b`.
    ///
    /// # Safety
    ///
    /// `sums` holds `OPR_SUMS span.nchunk` floats per batch frame, `scan` as [`OprScan`]
    /// says over the span's positions with `intensity` not null; grid `(1, 1, 1)`, block
    /// `(PIXEL_THREADS, 1, 1)`.
    ///
    /// [`opr_frame_sums`]: LoadedModule::opr_frame_sums
    #[kernel]
    pub unsafe fn opr_intensity(sums: *const f32, span: BatchSpan, scan: OprScan) {
        let nchunk = span.nchunk as usize;
        let mut b = thread::threadIdx_x() as usize;
        while b < span.nb as usize {
            unsafe {
                let (num, den) = (
                    frame_total(sums, b, nchunk, 0),
                    frame_total(sums, b, nchunk, 1),
                );
                let f = *scan.order.add(span.p0 as usize + b) as usize;
                *scan.intensity.add(f) += 0.1 * num / den;
            }
            b += PIXEL_THREADS as usize;
        }
    }

    /// Eigenmode `i`'s per-frame projections from the [`opr_frame_sums`] of `U_i` against the
    /// residues: `coef[b] = (Σ Re(conj(r_b) U_i) / n² + w_bi) / Σ_b w_bi²`, `w_bi` that of the
    /// scan frame at epoch position `p0 + b`.
    ///
    /// # Safety
    ///
    /// `sums` holds `OPR_SUMS span.nchunk` floats per batch frame, `scan` as [`OprScan`]
    /// says over the span's positions with `i < k`, `coef` `nb` floats; grid `(1, 1, 1)`,
    /// block `(PIXEL_THREADS, 1, 1)`.
    ///
    /// [`opr_frame_sums`]: LoadedModule::opr_frame_sums
    #[kernel]
    pub unsafe fn opr_coef(
        sums: *const f32,
        span: BatchSpan,
        scan: OprScan,
        i: u32,
        n: u32,
        coef: *mut f32,
    ) {
        static mut RED: SharedArray<f32, 256> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as usize;
        let (nb, nchunk) = (span.nb as usize, span.nchunk as usize);
        let w = |b: usize| unsafe {
            let f = *scan.order.add(span.p0 as usize + b) as usize;
            *scan.weights.add(f * scan.k as usize + i as usize)
        };
        let mut acc = 0.0f32;
        let mut b = t;
        while b < nb {
            acc += w(b) * w(b);
            b += PIXEL_THREADS as usize;
        }
        let w2 = unsafe {
            let s = (&raw mut RED).cast::<f32>();
            *s.add(t) = acc;
            crate::block_sum::<1>(s);
            *s
        };
        let nn = n as f32 * n as f32;
        let mut b = t;
        while b < nb {
            let s = unsafe { frame_total(sums, b, nchunk, 2) };
            unsafe { *coef.add(b) = (s / nn + w(b)) / w2 };
            b += PIXEL_THREADS as usize;
        }
    }

    /// `e = mean_b r_b coef[b]`, summed in frame order, and chunk `c = blockIdx.x`'s `Σ |e|²`
    /// into `part[c]`.
    ///
    /// # Safety
    ///
    /// `r` holds `nb n` complex values, `coef` `nb` floats, `e` `n` complex values, `part`
    /// `nchunk` floats; grid `(nchunk, 1, 1)`, block `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn opr_e(
        r: *const f32,
        coef: *const f32,
        nb: u32,
        n: u32,
        e: *mut f32,
        part: *mut f32,
    ) {
        static mut RED: SharedArray<f32, 256> = SharedArray::UNINIT;
        let n = n as usize;
        let c = thread::blockIdx_x() as usize;
        let chunk = GAMMA_CHUNK as usize;
        let end = n.min((c + 1) * chunk);
        let base = c * chunk + thread::threadIdx_x() as usize;
        let mut acc = [0.0f32; 1];
        for g in 0..CHUNK_PIXELS {
            let j = base + g * PIXEL_THREADS as usize;
            if j < end {
                let mut s = (0.0f32, 0.0f32);
                for b in 0..nb as usize {
                    let (v, p) = unsafe { (load_c(r, b * n + j), *coef.add(b)) };
                    s = (s.0 + v.0 * p, s.1 + v.1 * p);
                }
                let s = (s.0 / nb as f32, s.1 / nb as f32);
                unsafe { store_c(e, j, s) };
                acc[0] += s.0 * s.0 + s.1 * s.1;
            }
        }
        unsafe { store_sums::<1>((&raw mut RED).cast::<f32>(), acc, part.add(c)) };
    }

    /// `U′ = U + a_u n e / √Σ|e|²` in place of `u` (`a_u √n e / mnorm(e)`), `Σ|e|²` from
    /// `part[0 .. nchunk]` in chunk order, and chunk `c = blockIdx.x`'s `Σ |U′|²` into
    /// `part[nchunk + c]`.
    ///
    /// # Safety
    ///
    /// `u` and `e` hold `n` complex values, `part` `2 nchunk` floats; grid `(nchunk, 1, 1)`,
    /// block `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn opr_move(u: *mut f32, e: *const f32, a_u: f32, n: u32, part: *mut f32) {
        static mut RED: SharedArray<f32, 256> = SharedArray::UNINIT;
        let n = n as usize;
        let nchunk = n.div_ceil(GAMMA_CHUNK as usize);
        let s = a_u * n as f32 / unsafe { plane_total(part, nchunk) }.sqrt();
        let c = thread::blockIdx_x() as usize;
        let chunk = GAMMA_CHUNK as usize;
        let end = n.min((c + 1) * chunk);
        let base = c * chunk + thread::threadIdx_x() as usize;
        let mut acc = [0.0f32; 1];
        for g in 0..CHUNK_PIXELS {
            let j = base + g * PIXEL_THREADS as usize;
            if j < end {
                let (x, y) = unsafe { (load_c(u, j), load_c(e, j)) };
                let v = (x.0 + s * y.0, x.1 + s * y.1);
                unsafe { store_c(u, j, v) };
                acc[0] += v.0 * v.0 + v.1 * v.1;
            }
        }
        unsafe { store_sums::<1>((&raw mut RED).cast::<f32>(), acc, part.add(nchunk + c)) };
    }

    /// `U″ = n U′ / √Σ|U′|²` in place of `u` (`√n U′ / mnorm(U′)`), `Σ|U′|²` from
    /// `part[nchunk .. 2 nchunk]` in chunk order.
    ///
    /// # Safety
    ///
    /// `u` holds `n` complex values, `part` `2 nchunk` floats; 1D grid over `n`.
    #[kernel]
    pub unsafe fn opr_scale(u: *mut f32, n: u32, part: *const f32) {
        let j = global_index();
        let n = n as usize;
        if j >= n {
            return;
        }
        let nchunk = n.div_ceil(GAMMA_CHUNK as usize);
        let s = n as f32 / unsafe { plane_total(part.add(nchunk), nchunk) }.sqrt();
        let v = unsafe { load_c(u, j) };
        unsafe { store_c(u, j, (s * v.0, s * v.1)) };
    }

    /// Eigenmode `i`'s weights from the [`opr_frame_sums`] of `U″` against `d`, `vv` and the
    /// residues: with `den_b = Σ |U″|² vv_b`, `w_bi += Σ Re(conj(U″) d_b) / (den_b + 0.1
    /// mean_b den_b)` at the scan frame at epoch position `p0 + b`, the mean summed in frame
    /// order; under
    /// `deflate != 0` also `coef[2b .. 2b + 2] = Σ conj(U″) r_b / n²` (`Σ |U″|² = n²` by its
    /// normalization).
    ///
    /// # Safety
    ///
    /// `sums` holds `OPR_SUMS span.nchunk` floats per batch frame, `scan` as [`OprScan`]
    /// says over the span's positions with `i < k`, `coef` `2 nb` floats; grid `(1, 1, 1)`,
    /// block `(PIXEL_THREADS, 1, 1)`.
    ///
    /// [`opr_frame_sums`]: LoadedModule::opr_frame_sums
    #[kernel]
    pub unsafe fn opr_weights(
        sums: *const f32,
        span: BatchSpan,
        scan: OprScan,
        i: u32,
        n: u32,
        deflate: u32,
        coef: *mut f32,
    ) {
        static mut RED: SharedArray<f32, 256> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as usize;
        let (nb, nchunk) = (span.nb as usize, span.nchunk as usize);
        let mut acc = 0.0f32;
        let mut b = t;
        while b < nb {
            acc += unsafe { frame_total(sums, b, nchunk, 1) };
            b += PIXEL_THREADS as usize;
        }
        let reg = unsafe {
            let s = (&raw mut RED).cast::<f32>();
            *s.add(t) = acc;
            crate::block_sum::<1>(s);
            0.1 * *s / nb as f32
        };
        let nn = n as f32 * n as f32;
        let mut b = t;
        while b < nb {
            unsafe {
                let num = frame_total(sums, b, nchunk, 0);
                let den = frame_total(sums, b, nchunk, 1);
                let f = *scan.order.add(span.p0 as usize + b) as usize;
                *scan.weights.add(f * scan.k as usize + i as usize) += num / (den + reg);
                if deflate != 0 {
                    *coef.add(2 * b) = frame_total(sums, b, nchunk, 2) / nn;
                    *coef.add(2 * b + 1) = frame_total(sums, b, nchunk, 3) / nn;
                }
            }
            b += PIXEL_THREADS as usize;
        }
    }

    /// `r_b −= coef_b U″` for every batch frame `b`, `coef_b` complex at `coef[2b ..]`.
    ///
    /// # Safety
    ///
    /// `r` holds `nb n` complex values, `u` `n`, `coef` `2 nb` floats; 1D grid over `n`.
    #[kernel]
    pub unsafe fn opr_deflate(r: *mut f32, u: *const f32, coef: *const f32, nb: u32, n: u32) {
        let j = global_index();
        let n = n as usize;
        if j >= n {
            return;
        }
        let u = unsafe { load_c(u, j) };
        for b in 0..nb as usize {
            unsafe {
                let c = load_c(coef, b);
                let (v, cu) = (load_c(r, b * n + j), cmul(c, u));
                store_c(r, b * n + j, (v.0 - cu.0, v.1 - cu.1));
            }
        }
    }
}
