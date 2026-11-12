//! pty-chi BH (bilinear-Hessian) kernels on the amplitude cost `F = Σ_frames Σ_unmasked
//! (A − √obs)²`, `A = √(Σ_m |psi_m|² / n)` the modelled amplitude of the unnormalized far
//! field `psi` (ptyrs-engine `bh`): the residual whose back-projection is the gradient, the
//! dot products and the Hessian sums of a step, the step coefficients and the update. The
//! step's scalars stay on the device; partial sums are reduced in a fixed order.
//!
//! Frames: a stack's frame `fl` is epoch position `f0 + fl`; its views are at that position
//! of `views`, its counts at frame `order[f0 + fl]` of `counts`, the scan's natural order.

use cuda_device::{device, thread};
use cuda_host::cuda_module;

use crate::grad::ObjProbePtrs;
use crate::pixel::PIXEL_THREADS;

/// pty-chi's offset on the modelled amplitude `A`.
pub const BH_EPS: f32 = 1e-8;

/// The stacks of one BH stack the Hessian reads, each `[nf_max * nmode, ny, nx]` in
/// `obj_probe_to_psi`'s layout: `psi` its forward-FFT'd exit waves, `r` the residual after
/// its inverse FFT, `lu` and `lv` the forward-FFT'd derivatives of `psi` along `u` and `v`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BhFields {
    pub psi: *const f32,
    pub r: *const f32,
    pub lu: *const f32,
    pub lv: *const f32,
}

/// The observations of a scan in epoch order: `counts` `[nframe, ny, nx]` in the scan's own
/// order and `order`, the scan frame at each epoch position (`nframe` ids `< nframe`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BhObs {
    pub counts: *const u16,
    pub order: *const u32,
}

/// The displacements of a Hessian: `u = (u.obj, u_scale * u.probe)` and, with `cg != 0`,
/// `v`; without, `v` is never read.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BhDirs {
    pub u: ObjProbePtrs,
    pub v: ObjProbePtrs,
    pub u_scale: f32,
    pub cg: u32,
}

/// Floats a [`bh_kernels::LoadedModule::bh_hess_frames`] block writes: the partial `H(u, u)`,
/// `H(u, v)`, `H(v, v)`.
pub const BH_HESS: u32 = 3;
/// Floats of one step record of [`bh_kernels::LoadedModule::bh_step`]: `β, α, top, bottom`.
pub const BH_STEP: u32 = 4;

#[cuda_module]
pub mod bh_kernels {
    use super::{BH_EPS, BH_HESS, BH_STEP, BhDirs, BhFields, BhObs};
    use crate::grad::{GAMMA_CHUNK, ObjProbePtrs};
    use crate::pixel::*;
    use crate::*;
    use cuda_device::SharedArray;

    /// The residual of the stack's forward-FFT'd (unnormalized) exit waves `psi`, ready for
    /// the unnormalized inverse FFT: `r = psi · 2 (1 − √obs / a) / n` with `a = √(Σ_m |psi|²
    /// / n) + BH_EPS`, `obs` the frame's count, or `r = 0` where it is `MASKED`. Its inverse FFT, back-projected by
    /// `psi_to_obj` and `psi_to_probe`, is the gradient `2 ∂F/∂x*`. One thread per element of
    /// `[nf_max, ny, nx]`.
    ///
    /// # Safety
    ///
    /// `psi`, `r` `[nf_max * nmode, ny, nx]`, `obs` as [`BhObs`] says, `ctl` one entry.
    #[kernel]
    pub unsafe fn bh_residual(
        psi: *const f32,
        r: *mut f32,
        obs: BhObs,
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
            let frame = *obs.order.add(f0 as usize + fl) as usize;
            let count = *obs.counts.add(frame * n + d);
            let s = if count == MASKED {
                0.0
            } else {
                let a = (mode_power(psi, plane0, nmode, n, at) * inv_n).sqrt() + BH_EPS;
                2.0 * (1.0 - (count as f32).sqrt() / a) * inv_n
            };
            for m in 0..nmode as usize {
                let j = (plane0 + m) * n + at;
                let v = load_c(psi, j);
                store_c(r, j, (v.0 * s, v.1 * s));
            }
        }
    }

    /// The stack's far-field derivatives before their forward FFT, in `obj_probe_to_psi`'s
    /// layout: for view `w = views[f0 * nmode + kl]` and `q = exp(i quad r²)`, `lu = q (u_P
    /// O_w + P u_O,w)` with `u_P = u_scale * u.probe`, and with `cg`, `lv = q (v_P O_w + P
    /// v_O,w)` (else `lv` is not written). One thread per element of `[nf_max * nmode, ny,
    /// nx]`.
    ///
    /// # Safety
    ///
    /// `x`, `dirs` as [`ObjProbePtrs`] says, `views` `nframe * nmode` records in range for
    /// those shapes, `lu`, `lv` `[nf_max * nmode, ny, nx]`, `ctl` one entry.
    #[kernel]
    pub unsafe fn bh_far(
        x: ObjProbePtrs,
        dirs: BhDirs,
        views: *const ViewAddr,
        lu: *mut f32,
        lv: *mut f32,
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
            let w = *views.add(f0 as usize * g.nmode as usize + kl);
            let o = sample_obj(x.obj, w, py, px, g.nyo, g.nxo);
            let at = w.probe as usize * n + i;
            let p = load_c(x.probe, at);
            let q = quad_phase(g.quad, py, px, g.ny, g.nx);
            let j = kl * n + psi_index(py, px, g.ny, g.nx);
            let derivative = |d: ObjProbePtrs, scale: f32| {
                let (dp, d_o) = (
                    load_c(d.probe, at),
                    sample_obj(d.obj, w, py, px, g.nyo, g.nxo),
                );
                let (a, b) = (cmul((scale * dp.0, scale * dp.1), o), cmul(p, d_o));
                cmul((a.0 + b.0, a.1 + b.1), q)
            };
            store_c(lu, j, derivative(dirs.u, dirs.u_scale));
            if dirs.cg != 0 {
                store_c(lv, j, derivative(dirs.v, 1.0));
            }
        }
    }

    /// The Hessian partials of chunk `c = blockIdx.x` of frame `f0 + fl`, `fl = blockIdx.y`,
    /// with `F`'s second derivative along `u` and `v` (pty-chi's bilinear Hessian, `A +
    /// BH_EPS` for `A`) `H(u, v) = Σ Re(conj(r) q (u_P v_O + v_P u_O)) + 2 Σ_unmasked [(1 −
    /// √obs/a) Σ_m Re(conj(lu) lv) / n + √obs (c_u / n) (c_v / n) / a³]`, `c_u = Σ_m
    /// Re(conj(psi) lu)`, over the pixels `d` in `c * GAMMA_CHUNK .. min((c + 1) * GAMMA_CHUNK,
    /// n)`: the near term at probe pixel `d`, the far term at detector pixel `d`. Triple
    /// `p = (f0 + fl) * nchunk + c`, `nchunk = ceil(n / GAMMA_CHUNK)`, gets `out[3p ..
    /// 3p + 3] = (H(u, u), H(u, v), H(v, v))`, the last two 0 without `cg`. Triples of frames
    /// outside the stack are not written.
    ///
    /// # Safety
    ///
    /// Each of `f` `[nf_max * nmode, ny, nx]`, `dirs` as [`BhDirs`] says, `views` `nframe *
    /// nmode` records in range, `obs` as [`BhObs`] says, `out` `3 * nframe * nchunk` floats,
    /// `ctl` one entry; grid `(nchunk, nf_max, 1)`, block `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn bh_hess_frames(
        f: BhFields,
        dirs: BhDirs,
        views: *const ViewAddr,
        obs: BhObs,
        out: *mut f32,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        static mut RED: SharedArray<f32, 768> = SharedArray::UNINIT;
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let (ny, nx, nmode) = (g.ny, g.nx, g.nmode as usize);
        let n = ny as usize * nx as usize;
        let fl = thread::blockIdx_y() as usize;
        // The whole block leaves together, before its first barrier.
        if fl >= nf as usize {
            return;
        }
        let cg = dirs.cg != 0;
        let c = thread::blockIdx_x() as usize;
        let chunk = GAMMA_CHUNK as usize;
        let end = n.min((c + 1) * chunk);
        let inv_n = 1.0 / n as f32;
        let pos = f0 as usize + fl;
        let frame = unsafe { *obs.order.add(pos) } as usize;
        let mut acc = [0.0f32; BH_HESS as usize];
        let mut d = c * chunk + thread::threadIdx_x() as usize;
        while d < end {
            let (py, px) = ((d / nx as usize) as u32, (d % nx as usize) as u32);
            let at = psi_index(py, px, ny, nx);
            let q = quad_phase(g.quad, py, px, ny, nx);
            let (mut calc, mut cu, mut cv) = (0.0f32, 0.0f32, 0.0f32);
            let (mut suu, mut suv, mut svv) = (0.0f32, 0.0f32, 0.0f32);
            for m in 0..nmode {
                let j = (fl * nmode + m) * n + at;
                unsafe {
                    let w = *views.add(pos * nmode + m);
                    let rq = cmul_conj(load_c(f.r, j), q);
                    let pa = w.probe as usize * n + d;
                    let up = load_c(dirs.u.probe, pa);
                    let up = (dirs.u_scale * up.0, dirs.u_scale * up.1);
                    let uo = sample_obj(dirs.u.obj, w, py, px, g.nyo, g.nxo);
                    acc[0] += 2.0 * cmul(rq, cmul(up, uo)).0;
                    let (p, lu) = (load_c(f.psi, j), load_c(f.lu, j));
                    calc += p.0 * p.0 + p.1 * p.1;
                    cu += cmul_conj(p, lu).0;
                    suu += lu.0 * lu.0 + lu.1 * lu.1;
                    if cg {
                        let vp = load_c(dirs.v.probe, pa);
                        let vo = sample_obj(dirs.v.obj, w, py, px, g.nyo, g.nxo);
                        let (a, b) = (cmul(up, vo), cmul(vp, uo));
                        acc[1] += cmul(rq, (a.0 + b.0, a.1 + b.1)).0;
                        acc[2] += 2.0 * cmul(rq, cmul(vp, vo)).0;
                        let lv = load_c(f.lv, j);
                        cv += cmul_conj(p, lv).0;
                        suv += cmul_conj(lu, lv).0;
                        svv += lv.0 * lv.0 + lv.1 * lv.1;
                    }
                }
            }
            let count = unsafe { *obs.counts.add(frame * n + d) };
            if count != MASKED {
                let a = (calc * inv_n).sqrt() + BH_EPS;
                let root = (count as f32).sqrt();
                let fa = 1.0 - root / a;
                let k = root / (a * a * a);
                let (cu, cv) = (cu * inv_n, cv * inv_n);
                acc[0] += 2.0 * (fa * suu * inv_n + k * cu * cu);
                acc[1] += 2.0 * (fa * suv * inv_n + k * cu * cv);
                acc[2] += 2.0 * (fa * svv * inv_n + k * cv * cv);
            }
            d += PIXEL_THREADS as usize;
        }
        let o = unsafe { out.add(BH_HESS as usize * (pos * n.div_ceil(chunk) + c)) };
        unsafe { super::store_sums::<3>((&raw mut RED).cast::<f32>(), acc, o) }
    }

    /// One batch's step from the `ndots` pairs of `dots` (`ptyrs_tile::TileDots::dots`: `(⟨g,
    /// Pg⟩, ⟨g, D′⟩)`) and the `nhess` triples of `hess` ([`bh_hess_frames`] with `u = Pg`, `v =
    /// D′`), each summed in a fixed order: `β = H(u, v) / H(v, v)` with `cg`, else 0; `top =
    /// ⟨g, Pg⟩ − β ⟨g, D′⟩ = −⟨g, D⟩` and `bottom = H(u, u) − 2β H(u, v) + β² H(v, v) = H(D,
    /// D)` for `D = β D′ − Pg`; `α = top / bottom * frac`. Writes `out[0..4] = (β, α, top,
    /// bottom)`.
    ///
    /// # Safety
    ///
    /// `dots` holds `2 * ndots` floats, `hess` `3 * nhess`, `out` four; grid `(1, 1, 1)`,
    /// block `(PIXEL_THREADS, 1, 1)`.
    ///
    /// [`bh_hess_frames`]: LoadedModule::bh_hess_frames
    #[kernel]
    pub unsafe fn bh_step(
        dots: *const f32,
        ndots: u32,
        hess: *const f32,
        nhess: u32,
        cg: u32,
        frac: f32,
        out: *mut f32,
    ) {
        static mut RED: SharedArray<f32, 1280> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as usize;
        let mut acc = [0.0f32; 5];
        let mut b = t;
        while b < ndots as usize {
            unsafe {
                acc[0] += *dots.add(2 * b);
                acc[1] += *dots.add(2 * b + 1);
            }
            b += PIXEL_THREADS as usize;
        }
        let h = BH_HESS as usize;
        let mut b = t;
        while b < nhess as usize {
            for k in 0..h {
                acc[2 + k] += unsafe { *hess.add(h * b + k) };
            }
            b += PIXEL_THREADS as usize;
        }
        let s = (&raw mut RED).cast::<f32>();
        let seg = PIXEL_THREADS as usize;
        unsafe {
            for (j, &a) in acc.iter().enumerate() {
                *s.add(j * seg + t) = a;
            }
            crate::block_sum::<5>(s);
        }
        if t == 0 {
            unsafe {
                let (gg, gd) = (*s, *s.add(seg));
                let (huu, huv, hvv) = (*s.add(2 * seg), *s.add(3 * seg), *s.add(4 * seg));
                let beta = if cg != 0 { huv / hvv } else { 0.0 };
                let top = gg - beta * gd;
                let bottom = huu - 2.0 * beta * huv + beta * beta * hvv;
                let rec: [f32; BH_STEP as usize] = [beta, top / bottom * frac, top, bottom];
                for (k, v) in rec.into_iter().enumerate() {
                    *out.add(k) = v;
                }
            }
        }
    }

    /// One array's step from `step = (β, α, ..)` of [`bh_step`]: `D = β prev − scale g`,
    /// `x += α D`, `prev = D`, and the consumed gradient sums are zero. Outside CG `β = 0`,
    /// so a finite `prev` drops out exactly.
    ///
    /// # Safety
    ///
    /// `x`, `prev`, `g.num` hold `n` complex values, `g.den` `n` floats, `step` two.
    ///
    /// [`bh_step`]: LoadedModule::bh_step
    #[kernel]
    pub unsafe fn bh_update(
        x: *mut f32,
        prev: *mut f32,
        g: SumPtrs,
        n: u32,
        scale: f32,
        step: *const f32,
    ) {
        let i = global_index();
        if i >= n as usize {
            return;
        }
        unsafe {
            let (beta, alpha) = (*step, *step.add(1));
            let (gv, pv) = (load_c(g.num, i), load_c(prev, i));
            let dv = (beta * pv.0 - scale * gv.0, beta * pv.1 - scale * gv.1);
            let xv = load_c(x, i);
            store_c(x, i, (xv.0 + alpha * dv.0, xv.1 + alpha * dv.1));
            store_c(prev, i, dv);
            store_c(g.num, i, (0.0, 0.0));
            *g.den.add(i) = 0.0;
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
pub(crate) unsafe fn store_sums<const K: usize>(s: *mut f32, acc: [f32; K], out: *mut f32) {
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
