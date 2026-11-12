//! Poisson ML conjugate-gradient kernels (PyNX `ML`): the Polak–Ribière `β`, the search
//! direction, the line-search step `γ` and the step, on arrays of complex values (object or
//! probe). A CG coefficient lives on the device as a pair `(num, den)` of floats whose
//! quotient it is; partial pairs are reduced in a fixed order, so coefficients and iterates
//! are bitwise reproducible.

use cuda_device::{device, thread};
use cuda_host::cuda_module;

use crate::pixel::PIXEL_THREADS;

/// Blocks of every `cg_beta_parts` launch, and so the partial pairs each launch writes.
pub const CG_BLOCKS: u32 = 1024;

/// Pixels per block of `cg_gamma_frames`: a frame of `n` pixels is `ceil(n / GAMMA_CHUNK)`
/// contiguous chunks, the last one short, each writing one pair.
pub const GAMMA_CHUNK: u32 = 2048;

/// The three exit-wave stacks of the line search, each `[nf_max * nmode, ny, nx]` in
/// `obj_probe_to_psi`'s layout: `po` along nothing, `a` linear and `b` quadratic in the step.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GammaPtrs {
    pub po: *mut f32,
    pub a: *mut f32,
    pub b: *mut f32,
}

/// An object `[nobj, nyo, nxo]` and a probe `[nprobe, ny, nx]`, or a direction for both.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ObjProbePtrs {
    pub obj: *const f32,
    pub probe: *const f32,
}

/// The `(sum |x|, sum |x|^2)` pairs of the object and the probe, the calculated photons
/// over the unmasked pixels of every frame (one float), or null to keep `O P`, and the
/// background's photons over the same pixels (one float), or null for none.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NormPtrs {
    pub obj: *const f32,
    pub probe: *const f32,
    pub icalc: *const f32,
    pub background: *const f32,
}

#[cuda_module]
pub mod grad_kernels {
    use super::{GAMMA_CHUNK, GammaPtrs, NormPtrs, ObjProbePtrs};
    use crate::pixel::*;
    use crate::*;
    use cuda_device::SharedArray;

    /// One array's Polak–Ribière partial sums, then `h = g`: block `b` writes
    /// `part[2b] = sum Re(conj(g) (g - h))` and `part[2b + 1] = sum |h|^2` over the elements
    /// `i ≡ b * PIXEL_THREADS + t (mod CG_BLOCKS * PIXEL_THREADS)`. `h` holds the previous
    /// cycle's gradient on entry and this cycle's on exit.
    ///
    /// # Safety
    ///
    /// `g`, `h` hold `n` complex values, `part` `2 * CG_BLOCKS` floats; grid
    /// `(CG_BLOCKS, 1, 1)`, block `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn cg_beta_parts(g: *const f32, h: *mut f32, n: u32, part: *mut f32) {
        static mut RED: SharedArray<f32, 512> = SharedArray::UNINIT;
        let stride = thread::gridDim_x() as usize * PIXEL_THREADS as usize;
        let mut acc = [0.0f32; 2];
        let mut i = global_index();
        while i < n as usize {
            unsafe {
                let (a, b) = (load_c(g, i), load_c(h, i));
                acc[0] += a.0 * (a.0 - b.0) + a.1 * (a.1 - b.1);
                acc[1] += b.0 * b.0 + b.1 * b.1;
                store_c(h, i, a);
            }
            i += stride;
        }
        let out = unsafe { part.add(2 * thread::blockIdx_x() as usize) };
        unsafe { super::store_pair((&raw mut RED).cast::<f32>(), acc, out) }
    }

    /// `out[0..2] = sum_b part[2b .. 2b + 2]` over `npart` pairs.
    ///
    /// # Safety
    ///
    /// `part` holds `2 * npart` floats, `out` two; grid `(1, 1, 1)`, block
    /// `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn cg_sum_parts(part: *const f32, npart: u32, out: *mut f32) {
        static mut RED: SharedArray<f32, 512> = SharedArray::UNINIT;
        let mut acc = [0.0f32; 2];
        let mut b = thread::threadIdx_x() as usize;
        while b < npart as usize {
            unsafe {
                acc[0] += *part.add(2 * b);
                acc[1] += *part.add(2 * b + 1);
            }
            b += PIXEL_THREADS as usize;
        }
        unsafe { super::store_pair((&raw mut RED).cast::<f32>(), acc, out) }
    }

    /// Adds one array's block sums `(sum |x|, sum |x|^2)` to the `CG_BLOCKS` pairs of `part`:
    /// block `b` over the elements `i ≡ b * PIXEL_THREADS + t (mod CG_BLOCKS * PIXEL_THREADS)`.
    ///
    /// # Safety
    ///
    /// `x` holds `n` complex values, `part` `2 * CG_BLOCKS` floats; grid `(CG_BLOCKS, 1, 1)`,
    /// block `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn cg_norm_parts(x: *const f32, n: u32, part: *mut f32) {
        static mut RED: SharedArray<f32, 512> = SharedArray::UNINIT;
        let stride = thread::gridDim_x() as usize * PIXEL_THREADS as usize;
        let mut acc = [0.0f32; 2];
        let mut i = global_index();
        while i < n as usize {
            let v = unsafe { load_c(x, i) };
            let s = v.0 * v.0 + v.1 * v.1;
            acc[0] += s.sqrt();
            acc[1] += s;
            i += stride;
        }
        let sum = unsafe { super::block_pair((&raw mut RED).cast::<f32>(), acc) };
        if thread::threadIdx_x() == 0 {
            unsafe {
                let out = part.add(2 * thread::blockIdx_x() as usize);
                *out += sum[0];
                *out.add(1) += sum[1];
            }
        }
    }

    /// Adds the calculated photons of the stack's frames, `sum_fl s[f0 + fl] partials[fl].icalc`
    /// over `fl < nf` in a fixed order, to `out[0]`; `s` is the per-frame intensity `scale`,
    /// or 1 if it is null.
    ///
    /// # Safety
    ///
    /// `partials` holds `nf_max` records, `out` and `ctl` one entry each, `scale` is null or
    /// holds `nframe` floats; grid `(1, 1, 1)`, block `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn cg_icalc_add(
        partials: *const LlkPartial,
        scale: *const f32,
        ctl: *const StackCtl,
        g: StackGeom,
        out: *mut f32,
    ) {
        static mut RED: SharedArray<f32, 256> = SharedArray::UNINIT;
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let t = thread::threadIdx_x() as usize;
        let mut acc = 0.0f32;
        let mut fl = t;
        while fl < nf as usize {
            let icalc = unsafe { (*partials.add(fl)).icalc };
            acc += if scale.is_null() {
                icalc
            } else {
                unsafe { *scale.add(f0 as usize + fl) * icalc }
            };
            fl += PIXEL_THREADS as usize;
        }
        let s = (&raw mut RED).cast::<f32>();
        unsafe {
            *s.add(t) = acc;
            crate::block_sum::<1>(s);
            if t == 0 {
                *out += *s;
            }
        }
    }

    /// The background's photons over the unmasked pixels of every frame, `out[0] = sum_j b[j]
    /// nobs[j]` in a fixed order, `nobs[j]` the frames where detector pixel `j` is unmasked.
    ///
    /// # Safety
    ///
    /// `b` and `nobs` hold `n` values, `out` one; grid `(1, 1, 1)`, block
    /// `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn cg_background_photons(b: *const f32, nobs: *const u32, n: u32, out: *mut f32) {
        static mut RED: SharedArray<f32, 256> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as usize;
        let mut acc = 0.0f32;
        let mut j = t;
        while j < n as usize {
            acc += unsafe { *b.add(j) * *nobs.add(j) as f32 };
            j += PIXEL_THREADS as usize;
        }
        let s = (&raw mut RED).cast::<f32>();
        unsafe {
            *s.add(t) = acc;
            crate::block_sum::<1>(s);
            if t == 0 {
                *out = *s;
            }
        }
    }

    /// PyNX `ScaleObjProbe` from the pairs `(sum |x|, sum |x|^2)` of the object and the
    /// probe: with `s = sqrt((total - background) / icalc)` (`absolute=True`, `background` 0
    /// if `norms.background` is null), or 1 if `norms.icalc` is null (`absolute=False`),
    /// `obj *= sqrt(probe[0] / obj[0] * s)` and `probe *= sqrt(obj[0] / probe[0] * s)`.
    ///
    /// # Safety
    ///
    /// `obj` holds `nobj` complex values, `probe` `nprobe`, `norms.obj` and `norms.probe` two
    /// floats each, `norms.icalc` and `norms.background` one or are null; one thread per
    /// element of the longer array.
    #[kernel]
    pub unsafe fn cg_scale_obj_probe(
        obj: *mut f32,
        nobj: u32,
        probe: *mut f32,
        nprobe: u32,
        norms: NormPtrs,
        total: f32,
    ) {
        let i = global_index();
        unsafe {
            let (os, ps) = (*norms.obj, *norms.probe);
            let s = if norms.icalc.is_null() {
                1.0
            } else if norms.background.is_null() {
                (total / *norms.icalc).sqrt()
            } else {
                ((total - *norms.background) / *norms.icalc).sqrt()
            };
            for (x, len, f) in [
                (obj, nobj, (ps / os * s).sqrt()),
                (probe, nprobe, (os / ps * s).sqrt()),
            ] {
                if i < len as usize {
                    let v = load_c(x, i);
                    store_c(x, i, (v.0 * f, v.1 * f));
                }
            }
        }
    }

    /// PyNX `cu_linear_comb_fcfc_beta`: `d = beta d - g` with `beta = beta_pair[0] /
    /// beta_pair[1]`, or 0 where that is negative or not finite (a reset to `-g`); then the
    /// consumed gradient sums are zero.
    ///
    /// # Safety
    ///
    /// `d`, `g.num` hold `n` complex values, `g.den` `n` floats, `beta_pair` two.
    #[kernel]
    pub unsafe fn cg_direction(d: *mut f32, g: SumPtrs, n: u32, beta_pair: *const f32) {
        let i = global_index();
        if i >= n as usize {
            return;
        }
        unsafe {
            let b = *beta_pair / *beta_pair.add(1);
            let b = if b.is_finite() && b >= 0.0 { b } else { 0.0 };
            let (dv, gv) = (load_c(d, i), load_c(g.num, i));
            store_c(d, i, (b * dv.0 - gv.0, b * dv.1 - gv.1));
            store_c(g.num, i, (0.0, 0.0));
            *g.den.add(i) = 0.0;
        }
    }

    /// The Poisson gradient in Fourier space (PyNX `GradPoissonFourier`, no background, no
    /// Hann filter) of the forward-FFT'd (unnormalized) stack, with the `1 / (ny * nx)` of
    /// the FFT round trip folded in: `psi *= 2 (1 - obs / max(calc, 1e-12)) / (ny * nx)` with
    /// `calc = sum_m |psi|^2 / (ny * nx)`, or `psi = 0` where `obs` is `MASKED`. One thread
    /// per element of `[nf_max, ny, nx]`.
    ///
    /// # Safety
    ///
    /// `psi` `[nf_max * nmode, ny, nx]`, `counts` `[nframe, ny, nx]`, `ctl` one entry.
    #[kernel]
    pub unsafe fn cg_poisson_grad(
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
                0.0
            } else {
                let calc = mode_power(psi, plane0, nmode, n, at) * inv_n;
                2.0 * (1.0 - obs as f32 / calc.max(1e-12)) * inv_n
            };
            for m in 0..nmode as usize {
                let j = (plane0 + m) * n + at;
                let v = load_c(psi, j);
                store_c(psi, j, (v.0 * s, v.1 * s));
            }
        }
    }

    /// The line-search exit waves of `x = (O, P)` along `d = (dO, dP)`, in
    /// `obj_probe_to_psi`'s layout and order of operations: for view `v = views[f0 * nmode +
    /// kl]`, `q = exp(i quad r^2)`, `po = O_v P_v q`, `a = (dO_v P_v + O_v dP_v) q`, `b = dO_v
    /// dP_v q` at plane `kl`. One thread per element of `[nf_max * nmode, ny, nx]`.
    ///
    /// # Safety
    ///
    /// `x`, `d` as [`ObjProbePtrs`] says, `views` `nframe * nmode` records in range for
    /// those shapes, each of `out` `[nf_max * nmode, ny, nx]`, `ctl` one entry.
    #[kernel]
    pub unsafe fn cg_gamma_fields(
        x: ObjProbePtrs,
        d: ObjProbePtrs,
        views: *const ViewAddr,
        out: GammaPtrs,
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
        let (py, px) = ((i / g.nx as usize) as u32, (i % g.nx as usize) as u32);
        unsafe {
            let v = *views.add(f0 as usize * g.nmode as usize + kl);
            let o = sample_obj(x.obj, v, py, px, g.nyo, g.nxo);
            let d_o = sample_obj(d.obj, v, py, px, g.nyo, g.nxo);
            let at = v.probe as usize * n + i;
            let (p, d_p) = (load_c(x.probe, at), load_c(d.probe, at));
            let q = quad_phase(g.quad, py, px, g.ny, g.nx);
            let (a0, a1) = (cmul(d_o, p), cmul(o, d_p));
            let j = kl * n + psi_index(py, px, g.ny, g.nx);
            store_c(out.po, j, cmul(cmul(o, p), q));
            store_c(out.a, j, cmul((a0.0 + a1.0, a0.1 + a1.1), q));
            store_c(out.b, j, cmul(cmul(d_o, d_p), q));
        }
    }

    /// PyNX `CG_Poisson_Gamma` per chunk `c = blockIdx.x` of frame `f0 + fl`, `fl =
    /// blockIdx.y`, on the forward-FFT'd (unnormalized) line-search stacks: over the unmasked
    /// detector pixels `d` in `c * GAMMA_CHUNK .. min((c + 1) * GAMMA_CHUNK, n)`, with `k` the
    /// frame's intensity `scale` (1 if it is null), `b` the detector `background[d]` (0 if it
    /// is null), `calc = max(k sum_m |po|^2 / n + b, 1e-12)`,
    /// `r = k sum_m Re(conj(po) a) / n`, `s = k sum_m (|a|^2 + 2 Re(conj(po) b)) / n` and
    /// `f = 1 - obs / calc`, pair `p = (f0 + fl) * nchunk + c`,
    /// `nchunk = ceil(n / GAMMA_CHUNK)`, gets `out[2 p] = sum -2 r f` and `out[2 p + 1] =
    /// sum 2 s f + obs (2 r / calc)^2`, the numerator and denominator of `γ = -L'(0) /
    /// L''(0)` (ptyrs-engine `cg_poisson_gamma`, which has the cross term right where PyNX's
    /// has not). Pairs of frames outside the stack are not written.
    ///
    /// # Safety
    ///
    /// Each of `x` `[nf_max * nmode, ny, nx]`, `counts` `[nframe, ny, nx]`, `out`
    /// `2 * nframe * nchunk` floats, `ctl` one entry, `scale` null or `nframe` floats,
    /// `background` null or `n`; grid `(nchunk, nf_max, 1)`, block `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn cg_gamma_frames(
        x: GammaPtrs,
        counts: *const u16,
        scale: *const f32,
        background: *const f32,
        out: *mut f32,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        static mut RED: SharedArray<f32, 512> = SharedArray::UNINIT;
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
        let inv_n = 1.0 / n as f32;
        // `1 * x == x` and `x + 0 == x` for `x >= 0`: without a scale and a background every
        // pair is the plain kernel's, bit for bit.
        let k = if scale.is_null() {
            1.0
        } else {
            unsafe { *scale.add(f0 as usize + fl) }
        };
        let mut acc = [0.0f32; 2];
        let mut d = c * chunk + thread::threadIdx_x() as usize;
        while d < end {
            let obs = unsafe { *counts.add((f0 as usize + fl) * n + d) };
            if obs != MASKED {
                let at = psi_index((d / nx as usize) as u32, (d % nx as usize) as u32, ny, nx);
                let (mut calc, mut r, mut s) = (0.0f32, 0.0f32, 0.0f32);
                for m in 0..nmode {
                    let j = (fl * nmode + m) * n + at;
                    let (p, a, b) = unsafe { (load_c(x.po, j), load_c(x.a, j), load_c(x.b, j)) };
                    calc += p.0 * p.0 + p.1 * p.1;
                    r += cmul_conj(p, a).0;
                    s += a.0 * a.0 + a.1 * a.1 + 2.0 * cmul_conj(p, b).0;
                }
                let b = if background.is_null() {
                    0.0
                } else {
                    unsafe { *background.add(d) }
                };
                let calc = (k * (calc * inv_n) + b).max(1e-12);
                let (r, s, obs) = (k * (r * inv_n), k * (s * inv_n), obs as f32);
                let f = 1.0 - obs / calc;
                let rc = 2.0 * r / calc;
                acc[0] -= 2.0 * r * f;
                acc[1] += 2.0 * s * f + obs * rc * rc;
            }
            d += PIXEL_THREADS as usize;
        }
        let o = unsafe { out.add(2 * ((f0 as usize + fl) * n.div_ceil(chunk) + c)) };
        unsafe { super::store_pair((&raw mut RED).cast::<f32>(), acc, o) }
    }

    /// Poisson's fixed-point step of the per-frame intensity scale from the frames' far sums,
    /// `scale[f] *= sums[f].ratio / sums[f].intensity` where the intensity is above 0, then
    /// division by the mean over the `nframe` frames.
    ///
    /// # Safety
    ///
    /// `scale` and `sums` hold `nframe` entries; grid `(1, 1, 1)`, block
    /// `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn cg_scale_step(scale: *mut f32, sums: *const ScalePartial, nframe: u32) {
        static mut RED: SharedArray<f32, 256> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as usize;
        let mut acc = 0.0f32;
        let mut f = t;
        while f < nframe as usize {
            unsafe {
                let p = *sums.add(f);
                if p.intensity > 0.0 {
                    *scale.add(f) *= p.ratio / p.intensity;
                }
                acc += *scale.add(f);
            }
            f += PIXEL_THREADS as usize;
        }
        let s = (&raw mut RED).cast::<f32>();
        let inv = unsafe {
            *s.add(t) = acc;
            crate::block_sum::<1>(s);
            nframe as f32 / *s
        };
        let mut f = t;
        while f < nframe as usize {
            unsafe { *scale.add(f) *= inv };
            f += PIXEL_THREADS as usize;
        }
    }

    /// pty-chi's gradient position step on the offsets `[nframe, 2]`, `(dy, dx)` per frame,
    /// from each frame's position sums `num` and `den`: per axis, `s = clamp(step num / (den +
    /// max den), -limit, limit)`, 0 where that denominator is not above 0, then `d += s -
    /// mean s`.
    ///
    /// # Safety
    ///
    /// `offsets` holds `2 nframe` floats, `num` and `den` `nframe` entries; one
    /// `(PIXEL_THREADS, 1, 1)` block.
    #[kernel]
    pub unsafe fn cg_position_step(
        offsets: *mut f32,
        num: *const PositionPartial,
        den: *const PositionPartial,
        nframe: u32,
        step: f32,
        limit: f32,
    ) {
        static mut RED: SharedArray<f32, 1024> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as usize;
        let seg = PIXEL_THREADS as usize;
        let (n, s) = (nframe as usize, (&raw mut RED).cast::<f32>());
        let mut max = [0.0f32; 2];
        let mut f = t;
        while f < n {
            let d = unsafe { *den.add(f) };
            for (m, d) in max.iter_mut().zip([d.dy, d.dx]) {
                if d > *m {
                    *m = d;
                }
            }
            f += seg;
        }
        let max = unsafe {
            *s.add(t) = max[0];
            *s.add(seg + t) = max[1];
            super::block_max::<2>(s);
            [*s, *s.add(seg)]
        };
        let steps = |f: usize| {
            let (p, d) = unsafe { (*num.add(f), *den.add(f)) };
            [
                super::position_axis_step(p.dy, d.dy + max[0], step, limit),
                super::position_axis_step(p.dx, d.dx + max[1], step, limit),
            ]
        };
        let mut acc = [0.0f32; 2];
        let mut f = t;
        while f < n {
            let st = steps(f);
            acc = [acc[0] + st[0], acc[1] + st[1]];
            f += seg;
        }
        // The sums take the next two segments: the maxima just read may still be unread.
        let mean = unsafe {
            let s = s.add(2 * seg);
            *s.add(t) = acc[0];
            *s.add(seg + t) = acc[1];
            crate::block_sum::<2>(s);
            [*s / nframe as f32, *s.add(seg) / nframe as f32]
        };
        let mut f = t;
        while f < n {
            let st = steps(f);
            unsafe {
                *offsets.add(2 * f) += st[0] - mean[0];
                *offsets.add(2 * f + 1) += st[1] - mean[1];
            }
            f += seg;
        }
    }

    /// PyNX `cu_linear_comb_fcfc_gamma`: with `gamma = gamma_pair[0] / gamma_pair[1]`,
    /// `x += gamma d` if `gamma` is finite, else `x` stays and `d = -h` (with `h` this cycle's
    /// gradient, as `cg_beta_parts` leaves it).
    ///
    /// # Safety
    ///
    /// `x`, `d`, `h` hold `n` complex values, `gamma_pair` two floats.
    #[kernel]
    pub unsafe fn cg_step(x: *mut f32, d: *mut f32, h: *const f32, n: u32, gamma_pair: *const f32) {
        let i = global_index();
        if i >= n as usize {
            return;
        }
        unsafe {
            let c = *gamma_pair / *gamma_pair.add(1);
            if c.is_finite() {
                let (xv, dv) = (load_c(x, i), load_c(d, i));
                store_c(x, i, (xv.0 + c * dv.0, xv.1 + c * dv.1));
            } else {
                let hv = load_c(h, i);
                store_c(d, i, (-hv.0, -hv.1));
            }
        }
    }
}

/// `clamp(step num / den, -limit, limit)` of `cg_position_step`, 0 unless `den > 0`; NaN stays.
#[device]
fn position_axis_step(num: f32, den: f32, step: f32, limit: f32) -> f32 {
    if den > 0.0 {
        let s = step * num / den;
        if s > limit {
            limit
        } else if s < -limit {
            -limit
        } else {
            s
        }
    } else {
        0.0
    }
}

/// Tree maxima of `K` segments of `PIXEL_THREADS` shared values, each into its element 0, as
/// [`crate::block_sum`] sums them; every thread must call it.
///
/// # Safety
///
/// `s` must point to `K * PIXEL_THREADS` block-shared floats, each written by its thread.
#[device]
unsafe fn block_max<const K: usize>(s: *mut f32) {
    let t = thread::threadIdx_x() as usize;
    let seg = PIXEL_THREADS as usize;
    let mut half = seg / 2;
    while half > 0 {
        thread::sync_threads();
        if t < half {
            for j in 0..K {
                unsafe {
                    let (a, b) = (s.add(j * seg + t), *s.add(j * seg + t + half));
                    if b > *a {
                        *a = b;
                    }
                }
            }
        }
        half /= 2;
    }
    thread::sync_threads();
}

/// Block sums of each thread's `acc`, returned to every thread; `s` is two `PIXEL_THREADS`
/// segments of shared memory. Every thread of the block must call it.
///
/// # Safety
///
/// `s` holds `2 * PIXEL_THREADS` shared floats; block `(PIXEL_THREADS, 1, 1)`.
#[device]
unsafe fn block_pair(s: *mut f32, acc: [f32; 2]) -> [f32; 2] {
    let t = thread::threadIdx_x() as usize;
    let seg = PIXEL_THREADS as usize;
    unsafe {
        for (j, &a) in acc.iter().enumerate() {
            *s.add(j * seg + t) = a;
        }
        crate::block_sum::<2>(s);
        [*s, *s.add(seg)]
    }
}

/// [`block_pair`] of `acc`, which thread 0 writes to `out[0..2]`.
///
/// # Safety
///
/// As [`block_pair`]; `out` holds two floats.
#[device]
unsafe fn store_pair(s: *mut f32, acc: [f32; 2], out: *mut f32) {
    let sum = unsafe { block_pair(s, acc) };
    if thread::threadIdx_x() == 0 {
        unsafe {
            *out = sum[0];
            *out.add(1) = sum[1];
        }
    }
}
