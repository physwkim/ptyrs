//! pty-chi BH (bilinear-Hessian) kernels on the amplitude cost `F = Σ_frames Σ_unmasked
//! (A − √obs)²`, `A = √(Σ_m |psi_m|² / n)` the modelled amplitude of the unnormalized far
//! field `psi` (ptyrs-engine `bh`): the residual whose back-projection is the gradient, the
//! dot products and the Hessian sums of a step, the step coefficients and the update. The
//! step's scalars stay on the device; partial sums are reduced in a fixed order.
//!
//! Frames: a stack's frame `fl` is epoch position `f0 + fl`; its views are at that position
//! of `views`, its counts at frame `order[f0 + fl]` of `counts`, the scan's natural order.
//!
//! Under the position model (`ptyrs::engine::Features::position`) the exit waves are `O W`,
//! `W` the probe moved by each frame's offsets (`ptyrs::gpu::Shift`): `W = IDFT(A e)` with
//! `A` the probe's spectra and `e` the frame's ramp, so along a probe change `Au` and an
//! offsets change `u_d` it changes by `δ_u W = IDFT(e (Au + i a_u A))`, `a_u = −2π (u_y fy +
//! u_x fx)`. The `bh_pos_*` kernels take the offsets as a third variable of the step.

use cuda_device::{device, thread};
use cuda_host::cuda_module;

use crate::grad::ObjProbePtrs;
use crate::pixel::{PIXEL_THREADS, cmul, cmul_conj, freq, load_c, ramp};

/// pty-chi's offset on the modelled amplitude `A`.
pub const BH_EPS: f32 = 1e-8;

/// The far fields of one BH stack the Hessian's far term reads, each `[nf_max * nmode, ny,
/// nx]` in `obj_probe_to_psi`'s layout: `psi` its forward-FFT'd exit waves, `lu` and `lv`
/// the forward-FFT'd derivatives of the exit waves along `u` and `v`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BhFar {
    pub psi: *const f32,
    pub lu: *const f32,
    pub lv: *const f32,
}

/// The fields of one BH stack [`bh_kernels::LoadedModule::bh_hess_frames`] reads: the far
/// ones and `r`, the residual after its inverse FFT, as they are.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BhFields {
    pub far: BhFar,
    pub r: *const f32,
}

/// The fields of one BH stack [`bh_kernels::LoadedModule::bh_pos_far`] reads: the far ones
/// and `y`, the forward-FFT'd `conj(O) r` of each view ([`bh_kernels::LoadedModule::bh_obj_conj`]),
/// as they are.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BhMovedFar {
    pub far: BhFar,
    pub y: *const f32,
}

/// One BH stack under the position model as [`bh_kernels::LoadedModule::bh_pos_near`] reads
/// it, in `obj_probe_to_psi`'s layout: the object `obj`, the moved probes `w` `[nf_max *
/// nprobe, ny, nx]` (plane `fl * nprobe + p`), the residual `r` after its inverse FFT, and
/// `lu`, `lv` `[nf_max * nmode, ny, nx]`, `δ_u W` and `δ_v W` of each view on entry.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BhNear {
    pub obj: *const f32,
    pub w: *const f32,
    pub r: *const f32,
    pub lu: *mut f32,
    pub lv: *mut f32,
}

/// The offsets' part of a step's displacements: `ud`, `u`'s, `[nframe, 2]` by epoch position;
/// `prev`, the last direction's, `[nframe, 2]` by scan frame, which with `cg != 0` is `v`'s;
/// `order` the scan frame at each epoch position.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BhPosDirs {
    pub ud: *mut f32,
    pub prev: *mut f32,
    pub order: *const u32,
    pub cg: u32,
}

/// What the Hessian of a step under the position model reads of the moved probes, each
/// spectrum `[nprobe, ny, nx]` in FFT order: `a` the estimate's `A`, `au` `u`'s probe part
/// `Au` unscaled and, with `dirs.cg`, `av` `v`'s (else unread); `offsets` `[nframe, 2]` by
/// epoch position, those of the exit waves; `probe_scale` `u`'s probe part over `Au`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BhMoves {
    pub a: *const f32,
    pub au: *const f32,
    pub av: *const f32,
    pub offsets: *const f32,
    pub probe_scale: f32,
    pub dirs: BhPosDirs,
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
    use super::{
        BH_EPS, BH_HESS, BH_STEP, BhDirs, BhFields, BhMovedFar, BhMoves, BhNear, BhObs, BhPosDirs,
        far_term, ramp_at, rates,
    };
    use crate::grad::{GAMMA_CHUNK, ObjProbePtrs};
    use crate::pixel::*;
    use crate::position::BatchSpan;
    use crate::*;
    use core::f32::consts::PI;
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
        let pos = f0 as usize + fl;
        let frame = unsafe { *obs.order.add(pos) } as usize;
        let mut acc = [0.0f32; BH_HESS as usize];
        let mut d = c * chunk + thread::threadIdx_x() as usize;
        while d < end {
            let (py, px) = ((d / nx as usize) as u32, (d % nx as usize) as u32);
            let at = psi_index(py, px, ny, nx);
            let q = quad_phase(g.quad, py, px, ny, nx);
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
                    if cg {
                        let vp = load_c(dirs.v.probe, pa);
                        let vo = sample_obj(dirs.v.obj, w, py, px, g.nyo, g.nxo);
                        let (a, b) = (cmul(up, vo), cmul(vp, uo));
                        acc[1] += cmul(rq, (a.0 + b.0, a.1 + b.1)).0;
                        acc[2] += 2.0 * cmul(rq, cmul(vp, vo)).0;
                    }
                }
            }
            let count = unsafe { *obs.counts.add(frame * n + d) };
            if count != MASKED {
                let root = (count as f32).sqrt();
                let t = unsafe { far_term(f.far, fl * nmode * n + at, n, nmode, cg, root) };
                // Unrolled, as every loop over `acc`, so it stays in registers.
                #[unroll]
                for h in 0..BH_HESS {
                    let h = h as usize;
                    acc[h] += t[h];
                }
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
            // Unrolled, as every loop over `acc`, so it stays in registers.
            #[unroll]
            for k in 0..BH_HESS {
                let k = k as usize;
                acc[2 + k] += unsafe { *hess.add(h * b + k) };
            }
            b += PIXEL_THREADS as usize;
        }
        let s = (&raw mut RED).cast::<f32>();
        let seg = PIXEL_THREADS as usize;
        unsafe {
            #[unroll]
            for j in 0..5u32 {
                let j = j as usize;
                *s.add(j * seg + t) = acc[j];
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
                #[unroll]
                for k in 0..BH_STEP {
                    let k = k as usize;
                    *out.add(k) = rec[k];
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

    /// `y = conj(O_v) r` of every view `v` of the stack in `obj_probe_to_psi`'s layout, `O_v`
    /// the object `v` samples at the natural pixel stored at each index (the probe-side
    /// gradient of a moved probe before its forward FFT). One thread per element of `[nf_max *
    /// nmode, ny, nx]`.
    ///
    /// # Safety
    ///
    /// `obj` holds `[nobj, nyo, nxo]` with every view's window inside, `views` the epoch's
    /// views, `r` and `y` `[nf_max * nmode, ny, nx]`, `ctl` one entry.
    #[kernel]
    pub unsafe fn bh_obj_conj(
        obj: *const f32,
        views: *const ViewAddr,
        r: *const f32,
        y: *mut f32,
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
        let py = ((j / nx as usize) as u32 + ny / 2) % ny;
        let px = ((j % nx as usize) as u32 + nx / 2) % nx;
        unsafe {
            let v = *views.add(f0 as usize * g.nmode as usize + kl);
            let o = sample_obj(obj, v, py, px, g.nyo, g.nxo);
            store_c(y, t, cmul_conj(o, load_c(r, t)));
        }
    }

    /// The offsets' gradient partials of chunk `c = blockIdx.x` of frame `f0 + fl`, `fl =
    /// blockIdx.y`: with `Y` the forward-FFT'd [`bh_obj_conj`] of the stack, `e` and `(fy,
    /// fx)` the frame's ramp and frequencies at FFT index `k` (`ramp_at`) and `A_p`
    /// view `v`'s probe spectrum, `im = Σ_v Im(conj(Y_v) e A_p)` over the frame's views and
    /// `out[2p .. 2p + 2] = Σ_k (fy im, fx im)` over `k` in `c * GAMMA_CHUNK .. min((c + 1) *
    /// GAMMA_CHUNK, n)`, `p = (f0 + fl) * nchunk + c`. `2π` times their sum over `c` is
    /// `∂F/∂d` ([`bh_pos_dots`]).
    ///
    /// # Safety
    ///
    /// `y` `[nf_max * nmode, ny, nx]`, `a` `[nprobe, ny, nx]`, `offsets` `[nframe, 2]` by epoch
    /// position, `views` the epoch's, `out` `2 * nframe * nchunk` floats, `ctl` one entry; grid
    /// `(nchunk, nf_max, 1)`, block `(PIXEL_THREADS, 1, 1)`.
    ///
    /// [`bh_obj_conj`]: LoadedModule::bh_obj_conj
    /// [`bh_pos_dots`]: LoadedModule::bh_pos_dots
    #[kernel]
    pub unsafe fn bh_pos_grad(
        y: *const f32,
        a: *const f32,
        offsets: *const f32,
        views: *const ViewAddr,
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
        let pos = f0 as usize + fl;
        let mut acc = [0.0f32; 2];
        let mut k = c * chunk + thread::threadIdx_x() as usize;
        while k < end {
            let (e, fr) = unsafe { ramp_at(offsets, pos, k, ny, nx) };
            let mut im = 0.0f32;
            for m in 0..nmode {
                unsafe {
                    let p = (*views.add(pos * nmode + m)).probe as usize;
                    let z = cmul_conj(load_c(y, (fl * nmode + m) * n + k), e);
                    im += cmul(z, load_c(a, p * n + k)).1;
                }
            }
            acc[0] += fr.0 * im;
            acc[1] += fr.1 * im;
            k += PIXEL_THREADS as usize;
        }
        let o = unsafe { out.add(2 * (pos * n.div_ceil(chunk) + c)) };
        unsafe { super::store_sums::<2>((&raw mut RED).cast::<f32>(), acc, o) }
    }

    /// The offsets' part of one batch's displacement and dots: per position `p` of `span` and
    /// axis `a`, `g = 2π Σ_c grads[2 (p nchunk + c) + a]` ([`bh_pos_grad`]), `d.ud[2p + a] =
    /// scale g` and the pair `out[0..2] = (Σ g ud, Σ g prev[2 order[p] + a])`, the second 0
    /// without `d.cg`: the offsets' terms of `(⟨g, Pg⟩, ⟨g, D′⟩)`.
    ///
    /// # Safety
    ///
    /// `grads` holds `2 * span.nchunk` floats per position, `d` as [`BhPosDirs`] says, `out`
    /// two; grid `(1, 1, 1)`, block `(PIXEL_THREADS, 1, 1)`.
    ///
    /// [`bh_pos_grad`]: LoadedModule::bh_pos_grad
    #[kernel]
    pub unsafe fn bh_pos_dots(
        grads: *const f32,
        span: BatchSpan,
        d: BhPosDirs,
        scale: f32,
        out: *mut f32,
    ) {
        static mut RED: SharedArray<f32, 512> = SharedArray::UNINIT;
        let nchunk = span.nchunk as usize;
        let mut acc = [0.0f32; 2];
        let mut i = thread::threadIdx_x() as usize;
        while i < 2 * span.nb as usize {
            let (p, a) = (span.p0 as usize + i / 2, i % 2);
            let mut gd = 0.0f32;
            for c in 0..nchunk {
                gd += unsafe { *grads.add(2 * (p * nchunk + c) + a) };
            }
            gd *= 2.0 * PI;
            let ud = scale * gd;
            unsafe {
                *d.ud.add(2 * p + a) = ud;
                if d.cg != 0 {
                    acc[1] += gd * *d.prev.add(2 * *d.order.add(p) as usize + a);
                }
            }
            acc[0] += gd * ud;
            i += PIXEL_THREADS as usize;
        }
        unsafe { super::store_sums::<2>((&raw mut RED).cast::<f32>(), acc, out) }
    }

    /// The spectra of the moved probes' first-order changes along `u` and `v`, before their
    /// inverse FFT, one per view in `obj_probe_to_psi`'s layout: for view `v` of frame `f0 +
    /// fl` with probe `p`, at FFT index `k`, `lu = e (s Au_p + i a_u A_p)` and with `cg`, `lv =
    /// e (Av_p + i a_v A_p)` (else `lv` is not written), `e` the frame's ramp, `s =
    /// mv.probe_scale` and `a_u`, `a_v` the frame's rates (`rates`). One thread per
    /// element of `[nf_max * nmode, ny, nx]`.
    ///
    /// # Safety
    ///
    /// `mv` as [`BhMoves`] says, `views` the epoch's, `lu`, `lv` `[nf_max * nmode, ny, nx]`,
    /// `ctl` one entry.
    #[kernel]
    pub unsafe fn bh_pos_spec(
        mv: BhMoves,
        views: *const ViewAddr,
        lu: *mut f32,
        lv: *mut f32,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let (ny, nx, nmode) = (g.ny, g.nx, g.nmode as usize);
        let n = ny as usize * nx as usize;
        let t = global_index();
        if t >= nf as usize * nmode * n {
            return;
        }
        let (kl, k) = (t / n, t % n);
        let pos = f0 as usize + kl / nmode;
        unsafe {
            let p = (*views.add(f0 as usize * nmode + kl)).probe as usize * n + k;
            let (e, fr) = ramp_at(mv.offsets, pos, k, ny, nx);
            let (a_u, a_v) = rates(mv.dirs, pos, fr);
            let (a, au) = (load_c(mv.a, p), load_c(mv.au, p));
            let s = mv.probe_scale;
            store_c(lu, t, cmul(e, (s * au.0 - a_u * a.1, s * au.1 + a_u * a.0)));
            if mv.dirs.cg != 0 {
                let av = load_c(mv.av, p);
                store_c(lv, t, cmul(e, (av.0 - a_v * a.1, av.1 + a_v * a.0)));
            }
        }
    }

    /// The near Hessian partials of chunk `c = blockIdx.x` of frame `f0 + fl`, `fl =
    /// blockIdx.y`, under the position model, and the exit waves' first-order changes: with
    /// `f.lu`, `f.lv` the inverse-FFT'd [`bh_pos_spec`] (`δ_u W`, `δ_v W`), over the probe
    /// pixels `d` of the chunk and the frame's views, `H(u, u) = 2 Σ Re(conj(r) u_O δ_u W)`,
    /// `H(u, v) = Σ Re(conj(r) (u_O δ_v W + v_O δ_u W))`, `H(v, v) = 2 Σ Re(conj(r) v_O δ_v
    /// W)` into `out[3p .. 3p + 3]`, `p = (f0 + fl) * nchunk + c`, the last two 0 without
    /// `cg`; then `lu = u_O W + O δ_u W` and with `cg` `lv = v_O W + O δ_v W` in place. Of
    /// `dirs` only `u.obj`, `v.obj` and `cg` are read.
    ///
    /// # Safety
    ///
    /// `f` as [`BhNear`] says, `dirs.u.obj` and with `cg` `dirs.v.obj` `[nobj, nyo, nxo]`,
    /// `views` the epoch's, `out` `3 * nframe * nchunk` floats, `ctl` one entry; grid
    /// `(nchunk, nf_max, 1)`, block `(PIXEL_THREADS, 1, 1)`.
    ///
    /// [`bh_pos_spec`]: LoadedModule::bh_pos_spec
    #[kernel]
    pub unsafe fn bh_pos_near(
        f: BhNear,
        dirs: BhDirs,
        views: *const ViewAddr,
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
        let pos = f0 as usize + fl;
        let mut acc = [0.0f32; BH_HESS as usize];
        let mut d = c * chunk + thread::threadIdx_x() as usize;
        while d < end {
            let (py, px) = ((d / nx as usize) as u32, (d % nx as usize) as u32);
            let at = psi_index(py, px, ny, nx);
            for m in 0..nmode {
                let j = (fl * nmode + m) * n + at;
                unsafe {
                    let v = *views.add(pos * nmode + m);
                    let w = load_c(f.w, (fl * g.nprobe as usize + v.probe as usize) * n + at);
                    let o = sample_obj(f.obj, v, py, px, g.nyo, g.nxo);
                    let r = load_c(f.r, j);
                    let uo = sample_obj(dirs.u.obj, v, py, px, g.nyo, g.nxo);
                    let du = load_c(f.lu, j);
                    acc[0] += 2.0 * cmul_conj(r, cmul(uo, du)).0;
                    let (a, b) = (cmul(uo, w), cmul(o, du));
                    store_c(f.lu, j, (a.0 + b.0, a.1 + b.1));
                    if cg {
                        let vo = sample_obj(dirs.v.obj, v, py, px, g.nyo, g.nxo);
                        let dv = load_c(f.lv, j);
                        let (a, b) = (cmul(uo, dv), cmul(vo, du));
                        acc[1] += cmul_conj(r, (a.0 + b.0, a.1 + b.1)).0;
                        acc[2] += 2.0 * cmul_conj(r, cmul(vo, dv)).0;
                        let (a, b) = (cmul(vo, w), cmul(o, dv));
                        store_c(f.lv, j, (a.0 + b.0, a.1 + b.1));
                    }
                }
            }
            d += PIXEL_THREADS as usize;
        }
        let o = unsafe { out.add(BH_HESS as usize * (pos * n.div_ceil(chunk) + c)) };
        unsafe { super::store_sums::<3>((&raw mut RED).cast::<f32>(), acc, o) }
    }

    /// Adds the far and the moved-probe Hessian partials of chunk `c = blockIdx.x` of frame
    /// `f0 + fl`, `fl = blockIdx.y`, to the triple [`bh_pos_near`] wrote, over the detector
    /// pixels `d` of the chunk at FFT index `k`: the far term of [`bh_hess_frames`] at the
    /// unmasked ones and, with `z = conj(Y_v) e` of each view (`Y` the forward-FFT'd
    /// [`bh_obj_conj`], `e` the frame's ramp), `ra = Σ_v Re(z A_p)`, `iu = Σ_v Im(z Au_p)`, `iv
    /// = Σ_v Im(z Av_p)` and `s = mv.probe_scale`, the terms `Σ Re(conj(r) O δ²W)` of every
    /// `d`: `H(u, u) −= 2 a_u s iu + a_u² ra`, and with `cg` `H(u, v) −= a_v s iu + a_u iv +
    /// a_u a_v ra`, `H(v, v) −= 2 a_v iv + a_v² ra`.
    ///
    /// # Safety
    ///
    /// Each of `f` `[nf_max * nmode, ny, nx]`, `mv` as [`BhMoves`] says, `views` the epoch's,
    /// `counts` `[nframe, ny, nx]` in the scan's order, `out` `3 * nframe * nchunk` floats,
    /// `ctl` one entry; grid `(nchunk, nf_max, 1)`, block `(PIXEL_THREADS, 1, 1)`.
    ///
    /// [`bh_pos_near`]: LoadedModule::bh_pos_near
    /// [`bh_hess_frames`]: LoadedModule::bh_hess_frames
    /// [`bh_obj_conj`]: LoadedModule::bh_obj_conj
    #[kernel]
    pub unsafe fn bh_pos_far(
        f: BhMovedFar,
        mv: BhMoves,
        views: *const ViewAddr,
        counts: *const u16,
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
        let cg = mv.dirs.cg != 0;
        let c = thread::blockIdx_x() as usize;
        let chunk = GAMMA_CHUNK as usize;
        let end = n.min((c + 1) * chunk);
        let pos = f0 as usize + fl;
        let frame = unsafe { *mv.dirs.order.add(pos) } as usize;
        let s = mv.probe_scale;
        let mut acc = [0.0f32; BH_HESS as usize];
        let mut d = c * chunk + thread::threadIdx_x() as usize;
        while d < end {
            let k = psi_index((d / nx as usize) as u32, (d % nx as usize) as u32, ny, nx);
            let count = unsafe { *counts.add(frame * n + d) };
            if count != MASKED {
                let root = (count as f32).sqrt();
                let t = unsafe { far_term(f.far, fl * nmode * n + k, n, nmode, cg, root) };
                // Unrolled, as every loop over `acc`, so it stays in registers.
                #[unroll]
                for h in 0..BH_HESS {
                    let h = h as usize;
                    acc[h] += t[h];
                }
            }
            let (e, fr) = unsafe { ramp_at(mv.offsets, pos, k, ny, nx) };
            let (a_u, a_v) = unsafe { rates(mv.dirs, pos, fr) };
            let (mut ra, mut iu, mut iv) = (0.0f32, 0.0f32, 0.0f32);
            for m in 0..nmode {
                unsafe {
                    let p = (*views.add(pos * nmode + m)).probe as usize * n + k;
                    let z = cmul_conj(load_c(f.y, (fl * nmode + m) * n + k), e);
                    ra += cmul(z, load_c(mv.a, p)).0;
                    iu += cmul(z, load_c(mv.au, p)).1;
                    if cg {
                        iv += cmul(z, load_c(mv.av, p)).1;
                    }
                }
            }
            acc[0] -= 2.0 * a_u * s * iu + a_u * a_u * ra;
            if cg {
                acc[1] -= a_v * s * iu + a_u * iv + a_u * a_v * ra;
                acc[2] -= 2.0 * a_v * iv + a_v * a_v * ra;
            }
            d += PIXEL_THREADS as usize;
        }
        let o = unsafe { out.add(BH_HESS as usize * (pos * n.div_ceil(chunk) + c)) };
        unsafe { super::add_sums::<3>((&raw mut RED).cast::<f32>(), acc, o) }
    }

    /// The offsets' update of one batch from `step = (β, α, ..)` of [`bh_step`]: per position
    /// `p` of `span`, frame `f = order[p]` and axis `a`, `D = β prev[2f + a] − ud[2p + a]`,
    /// `offsets[2f + a] += step_size α D` (pty-chi's position step size, unclipped) and
    /// `prev[2f + a] = D`. Outside CG `β = 0`, so a finite `prev` drops out exactly.
    ///
    /// # Safety
    ///
    /// `d` as [`BhPosDirs`] says, `offsets` `[nframe, 2]` by frame, `step` two floats.
    ///
    /// [`bh_step`]: LoadedModule::bh_step
    #[kernel]
    pub unsafe fn bh_pos_update(
        span: BatchSpan,
        d: BhPosDirs,
        offsets: *mut f32,
        step: *const f32,
        step_size: f32,
    ) {
        let i = global_index();
        if i >= 2 * span.nb as usize {
            return;
        }
        let (p, a) = (span.p0 as usize + i / 2, i % 2);
        unsafe {
            let (beta, alpha) = (*step, *step.add(1));
            let at = 2 * *d.order.add(p) as usize + a;
            let dv = beta * *d.prev.add(at) - *d.ud.add(2 * p + a);
            *offsets.add(at) += step_size * alpha * dv;
            *d.prev.add(at) = dv;
        }
    }
}

/// The far term of [`bh_kernels::LoadedModule::bh_hess_frames`] at one unmasked detector
/// pixel, `[H(u, u), H(u, v), H(v, v)]` (the last two 0 without `cg`): `2 [(1 − root / a)
/// Σ_m Re(conj(lu) lv) / n + root (c_u / n) (c_v / n) / a³]` with `a = √(Σ_m |psi|² / n) +
/// BH_EPS`, `c_u = Σ_m Re(conj(psi) lu)`, the modes at `j + m n`.
///
/// # Safety
///
/// Each of `f` holds `j + nmode * n` complex values (`lv` read only with `cg`).
#[device]
pub(crate) unsafe fn far_term(
    f: BhFar,
    j: usize,
    n: usize,
    nmode: usize,
    cg: bool,
    root: f32,
) -> [f32; 3] {
    let inv_n = 1.0 / n as f32;
    let (mut calc, mut cu, mut cv) = (0.0f32, 0.0f32, 0.0f32);
    let (mut suu, mut suv, mut svv) = (0.0f32, 0.0f32, 0.0f32);
    for m in 0..nmode {
        let j = j + m * n;
        unsafe {
            let (p, lu) = (load_c(f.psi, j), load_c(f.lu, j));
            calc += p.0 * p.0 + p.1 * p.1;
            cu += cmul_conj(p, lu).0;
            suu += lu.0 * lu.0 + lu.1 * lu.1;
            if cg {
                let lv = load_c(f.lv, j);
                cv += cmul_conj(p, lv).0;
                suv += cmul_conj(lu, lv).0;
                svv += lv.0 * lv.0 + lv.1 * lv.1;
            }
        }
    }
    let a = (calc * inv_n).sqrt() + BH_EPS;
    let fa = 1.0 - root / a;
    let k = root / (a * a * a);
    let (cu, cv) = (cu * inv_n, cv * inv_n);
    [
        2.0 * (fa * suu * inv_n + k * cu * cu),
        2.0 * (fa * suv * inv_n + k * cu * cv),
        2.0 * (fa * svv * inv_n + k * cv * cv),
    ]
}

/// Frame `pos`'s ramp `e` at FFT index `k` of an `[ny, nx]` plane and its frequencies `(fy,
/// fx)`, `offsets[2 pos ..]` its `(dy, dx)` (as `crate::fused`'s shift computes it).
///
/// # Safety
///
/// `offsets` holds `2 pos + 2` floats.
#[device]
pub(crate) unsafe fn ramp_at(
    offsets: *const f32,
    pos: usize,
    k: usize,
    ny: u32,
    nx: u32,
) -> ((f32, f32), (f32, f32)) {
    let (ky, kx) = ((k / nx as usize) as u32, (k % nx as usize) as u32);
    let (dy, dx) = unsafe { (*offsets.add(2 * pos), *offsets.add(2 * pos + 1)) };
    let e = cmul(ramp(dy, ky, ny), ramp(dx, kx, nx));
    (e, (freq(ky, ny), freq(kx, nx)))
}

/// The rates `(a_u, a_v)` of position `pos` at frequencies `(fy, fx)`: `a = −2π (d_y fy +
/// d_x fx)` of `u`'s offsets `ud` and, with `cg`, `v`'s `prev` (else 0).
///
/// # Safety
///
/// `d` as [`BhPosDirs`] says, `pos` an epoch position.
#[device]
pub(crate) unsafe fn rates(d: BhPosDirs, pos: usize, fr: (f32, f32)) -> (f32, f32) {
    let tau = -2.0 * core::f32::consts::PI;
    let rate = |o: *const f32| unsafe { tau * (*o * fr.0 + *o.add(1) * fr.1) };
    let a_u = unsafe { rate(d.ud.add(2 * pos)) };
    let a_v = if d.cg != 0 {
        unsafe { rate(d.prev.add(2 * *d.order.add(pos) as usize)) }
    } else {
        0.0
    };
    (a_u, a_v)
}

/// Leaves the block sums of each thread's `acc` in element 0 of `K` `PIXEL_THREADS` segments
/// of `s`. Every thread of the block must call it.
///
/// # Safety
///
/// `s` holds `K * PIXEL_THREADS` shared floats; block `(PIXEL_THREADS, 1, 1)`.
#[device]
unsafe fn reduce<const K: usize>(s: *mut f32, acc: [f32; K]) {
    let t = thread::threadIdx_x() as usize;
    let seg = PIXEL_THREADS as usize;
    unsafe {
        // Unrolled, so `acc` stays in registers rather than local memory.
        #[unroll]
        for j in 0..K as u32 {
            let j = j as usize;
            *s.add(j * seg + t) = acc[j];
        }
        crate::block_sum::<K>(s);
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
    unsafe { reduce::<K>(s, acc) };
    if thread::threadIdx_x() == 0 {
        for j in 0..K {
            unsafe { *out.add(j) = *s.add(j * PIXEL_THREADS as usize) };
        }
    }
}

/// [`store_sums`], adding the sums to `out[0..K]`.
///
/// # Safety
///
/// As [`store_sums`].
#[device]
pub(crate) unsafe fn add_sums<const K: usize>(s: *mut f32, acc: [f32; K], out: *mut f32) {
    unsafe { reduce::<K>(s, acc) };
    if thread::threadIdx_x() == 0 {
        for j in 0..K {
            unsafe { *out.add(j) += *s.add(j * PIXEL_THREADS as usize) };
        }
    }
}
