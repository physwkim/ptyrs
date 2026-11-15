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
//! u_x fx)`. The `bh_pos_*` kernels take the offsets as a third variable of the step; with
//! the offsets held (moved, not stepped) `bh_moved_spec` and `bh_moved_far` stand in for
//! `bh_pos_spec` and `bh_pos_far`, their terms at `a_u = a_v = 0`.
//!
//! Through a multislice object under the position model the moved probes enter the first
//! slice (`BH_ENTRY_MOVED`), changing to second order by `IDFT` of `bh_pos_second`, and the
//! last slice's plane phase moves with the offsets: `bh_phase_grad` adds its gradient to the
//! offsets' as a second segment of each frame's partials, `bh_phase_tangent` its terms to
//! the exit waves' derivatives.
//!
//! Under full-object sampling (`ptyrs::engine::Sampling::FullObject`) the exit waves are
//! `W(O) P`, `W` a view's window of the object at its offsets (`crate::sampling`), read as
//! blocks `[nobj, nf_max, ny, nx]`, plane `o * nf_max + fl`, through views whose mode `m`
//! reads probe `m % nprobe` and object mode `m / nprobe` (`ptyrs::gpu::window_views`). Along
//! `u` the windows change by `S(u_O) + Σ_a u_da ∂_a S(O)` and to second order by `Σ_a (v_da
//! ∂_a S(u_O) + u_da ∂_a S(v_O)) + Σ_ac u_da v_dc ∂_a ∂_c S(O)`; the `bh_win_*` kernels
//! take the offsets on these, and with the offsets held `bh_win_obj_grad` is `bh_win_grad`'s
//! object part alone.

use cuda_device::{device, thread};
use cuda_host::cuda_module;

use crate::grad::{GAMMA_CHUNK, ObjProbePtrs};
use crate::pixel::{PIXEL_THREADS, StackGeom, ViewAddr, cmul, fma, freq, load_c, ramp, zdiv};

/// pty-chi's offset on the modelled amplitude `A`.
pub const BH_EPS: f32 = 1e-8;

/// The pixels each thread of a `GAMMA_CHUNK` chunk sums, `PIXEL_THREADS` apart.
pub(crate) const CHUNK_PIXELS: usize = (GAMMA_CHUNK / PIXEL_THREADS) as usize;

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

/// What [`bh_kernels::LoadedModule::bh_moved_spec`] reads of the moved probes with the
/// offsets held, each spectrum `[nprobe, ny, nx]` in FFT order: `au` `u`'s probe part `Au`
/// unscaled and, with `cg != 0`, `av` `v`'s (else unread); `offsets` `[nframe, 2]` by epoch
/// position, those of the exit waves; `probe_scale` `u`'s probe part over `Au`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BhMovedProbes {
    pub au: *const f32,
    pub av: *const f32,
    pub offsets: *const f32,
    pub probe_scale: f32,
    pub cg: u32,
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

/// One stack under full-object sampling as the `bh_win_*` kernels read it: `r` the residual
/// after its inverse FFT, `[nf_max * nmode, ny, nx]` in `obj_probe_to_psi`'s layout, and
/// `probe` `[nprobe, ny, nx]`, the estimate's.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BhWinFields {
    pub r: *const f32,
    pub probe: *const f32,
}

/// The offsets' gradient partials of one segment: `out` `2 * nframe * nseg * nchunk` floats,
/// a frame's row `nseg` segments of `nchunk` pairs, of which `seg < nseg` writes its own.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BhPartials {
    pub out: *mut f32,
    pub seg: u32,
    pub nseg: u32,
}

/// The windows [`bh_kernels::LoadedModule::bh_win_hess`] reads, each three blocks `[nobj,
/// nf_max, ny, nx]`: `second` the estimate's second derivatives over `(y, y)`, `(y, x)` and
/// `(x, x)`; `du` and with `cg` `dv` those of `u_O` and `v_O`, blocks 1 and 2 their
/// derivatives over `y` and `x` (block 0 unread; `dv` unread without `cg`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BhWinDerivs {
    pub second: *const f32,
    pub du: *const f32,
    pub dv: *const f32,
}

/// A stack's derivatives through a multislice object, each `[nf_max * nmode, ny, nx]` in
/// `obj_probe_to_psi`'s layout: `lu` and `lv` the waves' along `u` and `v`, `muu`, `muv` and
/// `mvv` their second derivatives along `(u, u)`, `(u, v)` and `(v, v)`. Without `cg` only
/// `lu` and `muu` are read or written.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BhTangents {
    pub lu: *mut f32,
    pub lv: *mut f32,
    pub muu: *mut f32,
    pub muv: *mut f32,
    pub mvv: *mut f32,
}

/// A slice of a multislice object and the waves `entry` entering it, laid out as `kind`
/// says: [`BH_ENTRY_WAVES`], [`BH_ENTRY_PROBE`] or [`BH_ENTRY_MOVED`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BhSliceIn {
    pub slice: *const f32,
    pub entry: *const f32,
    pub kind: u32,
}

/// [`BhSliceIn::kind`]: one wave per view, `[nf_max * nmode, ny, nx]` as [`BhTangents`]'.
pub const BH_ENTRY_WAVES: u32 = 0;
/// [`BhSliceIn::kind`]: the probe `[nprobe, ny, nx]` in the natural layout, each view's wave
/// its probe.
pub const BH_ENTRY_PROBE: u32 = 1;
/// [`BhSliceIn::kind`]: the moved probes `[nf_max * nprobe, ny, nx]` in `obj_probe_to_psi`'s
/// layout, plane `fl * nprobe + p`, each view's wave its probe moved with its frame.
pub const BH_ENTRY_MOVED: u32 = 2;

/// The exit waves' second derivatives along `(u, u)`, `(u, v)` and `(v, v)` that
/// [`bh_kernels::LoadedModule::bh_sliced_hess_frames`] reads, laid out as [`BhTangents`]';
/// without `cg` only `uu`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BhSecond {
    pub uu: *const f32,
    pub uv: *const f32,
    pub vv: *const f32,
}

/// Floats a [`bh_kernels::LoadedModule::bh_hess_frames`] block writes: the partial `H(u, u)`,
/// `H(u, v)`, `H(v, v)`.
pub const BH_HESS: u32 = 3;
/// Floats of one step record of [`bh_kernels::LoadedModule::bh_step`]: `β, α, top, bottom`.
pub const BH_STEP: u32 = 4;

#[cuda_module]
pub mod bh_kernels {
    use super::{
        BH_ENTRY_PROBE, BH_EPS, BH_HESS, BH_STEP, BhDirs, BhFar, BhFields, BhMovedFar,
        BhMovedProbes, BhMoves, BhNear, BhObs, BhPartials, BhPosDirs, BhSecond, BhSliceIn,
        BhTangents, BhWinDerivs, BhWinFields, CHUNK_PIXELS, add_im, add_patch, add_re, entry_wave,
        far_add, far_tail, near_prod, near_sum, partials_at, ramp_at, rates, re_conj, unramp,
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
                2.0 * (1.0 - zdiv(zsqrt(count as f32), a)) * inv_n
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
        // A pixel's loads of a mode, the far term's among them, are issued before its sums,
        // which keep the mode order; masked pixels load the far term's too.
        while d < end {
            let count = unsafe { *obs.counts.add(frame * n + d) };
            let (py, px) = ((d / nx as usize) as u32, (d % nx as usize) as u32);
            let at = psi_index(py, px, ny, nx);
            let q = quad_phase(g.quad, py, px, ny, nx);
            let mut far = [0.0f32; 6];
            for m in 0..nmode {
                let j = (fl * nmode + m) * n + at;
                let w = unsafe { *views.add(pos * nmode + m) };
                let pa = w.probe as usize * n + d;
                let (r, up, uo, p, lu) = unsafe {
                    (
                        load_c(f.r, j),
                        load_c(dirs.u.probe, pa),
                        obj_corners(dirs.u.obj, w, py, px, g.nyo, g.nxo),
                        load_c(f.far.psi, j),
                        load_c(f.far.lu, j),
                    )
                };
                let (mut vp, mut vo, mut lv) = ((0.0, 0.0), [(0.0, 0.0); 4], (0.0, 0.0));
                if cg {
                    unsafe {
                        vp = load_c(dirs.v.probe, pa);
                        vo = obj_corners(dirs.v.obj, w, py, px, g.nyo, g.nxo);
                        lv = load_c(f.far.lv, j);
                    }
                }
                let rq = cmul_conj(r, q);
                let up = (dirs.u_scale * up.0, dirs.u_scale * up.1);
                let uo = bilinear(uo, w);
                acc[0] += 2.0 * cmul(rq, cmul(up, uo)).0;
                if cg {
                    let vo = bilinear(vo, w);
                    let (a, b) = (cmul(up, vo), cmul(vp, uo));
                    acc[1] += cmul(rq, (a.0 + b.0, a.1 + b.1)).0;
                    acc[2] += 2.0 * cmul(rq, cmul(vp, vo)).0;
                }
                far_add(&mut far, p, lu, lv, cg);
            }
            if count != MASKED {
                let t = far_tail(far, n, zsqrt(count as f32));
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

    /// The stack's derivatives `t` through one slice `S` of a multislice object, in place:
    /// the slice's exit `S w` of the wave `w` entering it changes along `u` by `S l_u + u_S w`
    /// and to second order along `(u, v)` by `S m_uv + u_S l_v + v_S l_u`, with `l_u`, `l_v`,
    /// `m_uv` those of `w` and `u_S`, `v_S` the bilinear windows of `dirs.u.obj`, `dirs.v.obj`
    /// for view `views[f0 * nmode + kl]`, `S` `s.slice` and `w` the view's wave of `s.entry`.
    /// Entered by the probe (`BH_ENTRY_PROBE`) the slice is the first, `l_u = u_scale *
    /// dirs.u.probe`, `l_v = dirs.v.probe` and `m = 0`, and `t` is only written; else the rest
    /// are `t`'s. One thread per element of `[nf_max * nmode, ny, nx]`.
    ///
    /// # Safety
    ///
    /// `s.slice` `[nobj, nyo, nxo]`, `s.entry` as its kind says, `dirs` as [`BhDirs`] says (its
    /// probe parts unread unless entered by the probe), `t` as [`BhTangents`] says, `views`
    /// `nframe * nmode` records in range for those shapes, `ctl` one entry.
    #[kernel]
    pub unsafe fn bh_slice_tangent(
        s: BhSliceIn,
        dirs: BhDirs,
        t: BhTangents,
        views: *const ViewAddr,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let n = g.ny as usize * g.nx as usize;
        let i = global_index();
        if i >= nf as usize * g.nmode as usize * n {
            return;
        }
        let (kl, d) = (i / n, i % n);
        let (py, px) = ((d / g.nx as usize) as u32, (d % g.nx as usize) as u32);
        let at = psi_index(py, px, g.ny, g.nx);
        let j = kl * n + at;
        let cg = dirs.cg != 0;
        let add = |a: (f32, f32), b: (f32, f32)| (a.0 + b.0, a.1 + b.1);
        let zero = (0.0, 0.0);
        unsafe {
            let w = *views.add(f0 as usize * g.nmode as usize + kl);
            let x = sample_obj(s.slice, w, py, px, g.nyo, g.nxo);
            let us = sample_obj(dirs.u.obj, w, py, px, g.nyo, g.nxo);
            let vs = if cg {
                sample_obj(dirs.v.obj, w, py, px, g.nyo, g.nxo)
            } else {
                zero
            };
            let e = entry_wave(s, w, kl, d, at, g);
            let (lu, lv, [muu, muv, mvv]) = if s.kind == BH_ENTRY_PROBE {
                let pa = w.probe as usize * n + d;
                let up = load_c(dirs.u.probe, pa);
                let lv = if cg { load_c(dirs.v.probe, pa) } else { zero };
                let lu = (dirs.u_scale * up.0, dirs.u_scale * up.1);
                (lu, lv, [zero; 3])
            } else if cg {
                let m = [load_c(t.muu, j), load_c(t.muv, j), load_c(t.mvv, j)];
                (load_c(t.lu, j), load_c(t.lv, j), m)
            } else {
                let m = [load_c(t.muu, j), zero, zero];
                (load_c(t.lu, j), zero, m)
            };
            let twice = |a: (f32, f32)| (2.0 * a.0, 2.0 * a.1);
            store_c(t.lu, j, add(cmul(x, lu), cmul(us, e)));
            store_c(t.muu, j, add(cmul(x, muu), twice(cmul(us, lu))));
            if cg {
                store_c(t.lv, j, add(cmul(x, lv), cmul(vs, e)));
                let cross = add(cmul(us, lv), cmul(vs, lu));
                store_c(t.muv, j, add(cmul(x, muv), cross));
                store_c(t.mvv, j, add(cmul(x, mvv), twice(cmul(vs, lv))));
            }
        }
    }

    /// [`bh_hess_frames`] through a multislice object: the near term is `Σ Re(conj(r) m)`, `m`
    /// the exit waves' second derivatives of [`bh_slice_tangent`] and `r` the residual after
    /// its inverse FFT and the adjoint of the last slice's plane phase, which the exit waves'
    /// derivatives in `f.far` carry; `g.quad` is unread.
    ///
    /// # Safety
    ///
    /// Each of `f` and `m` `[nf_max * nmode, ny, nx]` (`f.far.lv`, `m.uv`, `m.vv` unread
    /// without `cg`), `obs` as [`BhObs`] says, `out` `3 * nframe * nchunk` floats, `ctl` one
    /// entry; grid `(nchunk, nf_max, 1)`, block `(PIXEL_THREADS, 1, 1)`.
    ///
    /// [`bh_hess_frames`]: LoadedModule::bh_hess_frames
    /// [`bh_slice_tangent`]: LoadedModule::bh_slice_tangent
    #[kernel]
    pub unsafe fn bh_sliced_hess_frames(
        f: BhFields,
        m: BhSecond,
        cg: u32,
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
        let cg = cg != 0;
        let c = thread::blockIdx_x() as usize;
        let chunk = GAMMA_CHUNK as usize;
        let end = n.min((c + 1) * chunk);
        let pos = f0 as usize + fl;
        let frame = unsafe { *obs.order.add(pos) } as usize;
        let mut acc = [0.0f32; BH_HESS as usize];
        let mut d = c * chunk + thread::threadIdx_x() as usize;
        while d < end {
            let count = unsafe { *obs.counts.add(frame * n + d) };
            let at = psi_index((d / nx as usize) as u32, (d % nx as usize) as u32, ny, nx);
            let mut far = [0.0f32; 6];
            for md in 0..nmode {
                let j = (fl * nmode + md) * n + at;
                let (r, uu, p, lu) = unsafe {
                    (
                        load_c(f.r, j),
                        load_c(m.uu, j),
                        load_c(f.far.psi, j),
                        load_c(f.far.lu, j),
                    )
                };
                acc[0] += re_conj(r, uu);
                let mut lv = (0.0, 0.0);
                if cg {
                    let (uv, vv) = unsafe {
                        lv = load_c(f.far.lv, j);
                        (load_c(m.uv, j), load_c(m.vv, j))
                    };
                    acc[1] += re_conj(r, uv);
                    acc[2] += re_conj(r, vv);
                }
                far_add(&mut far, p, lu, lv, cg);
            }
            if count != MASKED {
                let t = far_tail(far, n, zsqrt(count as f32));
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

    /// The offsets' gradient partials of chunk `c = blockIdx.x` of frame `f0 + fl`, `fl =
    /// blockIdx.y`, through the plane phase of a multislice object's last slice displaced
    /// with the probe (`optics_slice_phase` under `SLICE_OFFSET`), whose angle at stored pixel
    /// `(ky, kx)` moves by `−2 quad ρ_a` per pixel of offset `a`, `ρ = (ny freq(ky, ny), nx
    /// freq(kx, nx))`: with `X = S w` the slice's exit before the phase, `S` `s.slice` and
    /// `w` the view's wave of `s.entry`, and `r` the residual after its inverse FFT and the
    /// phase's adjoint, chunk `c`'s pair of the frame's segment `part.seg` `= Σ (quad / π)
    /// ρ_a Im(conj(r) X)` over the chunk's probe pixels and the frame's views. `2π` times it
    /// is the phase's part of `∂F/∂d` ([`bh_pos_dots`]).
    ///
    /// # Safety
    ///
    /// `s.slice` `[nobj, nyo, nxo]`, `s.entry` as its kind says, `r` `[nf_max * nmode, ny,
    /// nx]`, `views` `nframe * nmode` records in range for those shapes, `part` as
    /// [`BhPartials`] says, `ctl` one entry; grid `(nchunk, nf_max, 1)`, block
    /// `(PIXEL_THREADS, 1, 1)`.
    ///
    /// [`bh_pos_dots`]: LoadedModule::bh_pos_dots
    #[kernel]
    pub unsafe fn bh_phase_grad(
        s: BhSliceIn,
        r: *const f32,
        views: *const ViewAddr,
        quad: f32,
        part: BhPartials,
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
        let mut d = c * chunk + thread::threadIdx_x() as usize;
        while d < end {
            let (py, px) = ((d / nx as usize) as u32, (d % nx as usize) as u32);
            let (ky, kx) = ((py + ny / 2) % ny, (px + nx / 2) % nx);
            let at = psi_index(py, px, ny, nx);
            let mut im = 0.0f32;
            for m in 0..nmode {
                let kl = fl * nmode + m;
                unsafe {
                    let v = *views.add(pos * nmode + m);
                    let o = sample_obj(s.slice, v, py, px, g.nyo, g.nxo);
                    let x = cmul(o, entry_wave(s, v, kl, d, at, g));
                    im += cmul_conj(load_c(r, kl * n + at), x).1;
                }
            }
            acc[0] += ny as f32 * freq(ky, ny) * im;
            acc[1] += nx as f32 * freq(kx, nx) * im;
            d += PIXEL_THREADS as usize;
        }
        let q = quad * core::f32::consts::FRAC_1_PI;
        let o = unsafe { partials_at(part, pos, n.div_ceil(chunk), c) };
        let acc = [q * acc[0], q * acc[1]];
        unsafe { super::store_sums::<2>((&raw mut RED).cast::<f32>(), acc, o) }
    }

    /// The exit waves' derivatives `t` through the plane phase `Φ` of a multislice object's
    /// last slice displaced with the probe, in place and with `Φ` itself left out: of `ψ =
    /// Φ X`, `X = S w` the slice's exit before it ([`bh_phase_grad`]'s) with `t` its
    /// derivatives ([`bh_slice_tangent`]'s), the angle of `Φ` moving along `u` by `S_u = −2
    /// quad Σ_a ρ_a dirs.ud[2 pos + a]` and with `dirs.cg` along `v` by `S_v`, likewise of
    /// `dirs.prev[2 order[pos] + a]`, `pos = f0 + kl / nmode`: `m_uu += 2i S_u l_u − S_u² X`,
    /// with `cg` `m_uv += i S_v l_u + i S_u l_v − S_u S_v X` and `m_vv += 2i S_v l_v − S_v²
    /// X`, then `l_u += i S_u X` and with `cg` `l_v += i S_v X`. `Φ` comes back with the
    /// phase on `l_u`, `l_v` and its adjoint on `r`. One thread per element of `[nf_max *
    /// nmode, ny, nx]`.
    ///
    /// # Safety
    ///
    /// As [`bh_phase_grad`] for `s` and `views`, `dirs` as [`BhPosDirs`] says, `t` as
    /// [`BhTangents`] says (`lv`, `muv`, `mvv` unread without `cg`), `ctl` one entry.
    ///
    /// [`bh_phase_grad`]: LoadedModule::bh_phase_grad
    /// [`bh_slice_tangent`]: LoadedModule::bh_slice_tangent
    #[kernel]
    pub unsafe fn bh_phase_tangent(
        s: BhSliceIn,
        dirs: BhPosDirs,
        t: BhTangents,
        views: *const ViewAddr,
        quad: f32,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let (ny, nx, nmode) = (g.ny, g.nx, g.nmode as usize);
        let n = ny as usize * nx as usize;
        let i = global_index();
        if i >= nf as usize * nmode * n {
            return;
        }
        let (kl, d) = (i / n, i % n);
        let (py, px) = ((d / nx as usize) as u32, (d % nx as usize) as u32);
        let (ky, kx) = ((py + ny / 2) % ny, (px + nx / 2) % nx);
        let at = psi_index(py, px, ny, nx);
        let j = kl * n + at;
        let pos = f0 as usize + kl / nmode;
        let cg = dirs.cg != 0;
        // The angle's rates per pixel of offset.
        let (sy, sx) = (
            -2.0 * quad * (ny as f32 * freq(ky, ny)),
            -2.0 * quad * (nx as f32 * freq(kx, nx)),
        );
        let add = |a: (f32, f32), b: (f32, f32)| (a.0 + b.0, a.1 + b.1);
        // `i a z + b x` of the rates `a`, `b`.
        let turn =
            |a: f32, z: (f32, f32), b: f32, x: (f32, f32)| (b * x.0 - a * z.1, b * x.1 + a * z.0);
        unsafe {
            let v = *views.add(f0 as usize * nmode + kl);
            let o = sample_obj(s.slice, v, py, px, g.nyo, g.nxo);
            let x = cmul(o, entry_wave(s, v, kl, d, at, g));
            let ud = dirs.ud.add(2 * pos);
            let su = *ud * sy + *ud.add(1) * sx;
            let lu = load_c(t.lu, j);
            store_c(
                t.muu,
                j,
                add(load_c(t.muu, j), turn(2.0 * su, lu, -su * su, x)),
            );
            if cg {
                let prev = dirs.prev.add(2 * *dirs.order.add(pos) as usize);
                let sv = *prev * sy + *prev.add(1) * sx;
                let lv = load_c(t.lv, j);
                let cross = add(turn(sv, lu, -su * sv, x), turn(su, lv, 0.0, x));
                store_c(t.muv, j, add(load_c(t.muv, j), cross));
                store_c(
                    t.mvv,
                    j,
                    add(load_c(t.mvv, j), turn(2.0 * sv, lv, -sv * sv, x)),
                );
                store_c(t.lv, j, add(lv, turn(sv, x, 0.0, x)));
            }
            store_c(t.lu, j, add(lu, turn(su, x, 0.0, x)));
        }
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
    /// `x += α D`, `prev = D`, and the consumed gradient `g` is zero. Outside CG `β = 0`,
    /// so a finite `prev` drops out exactly.
    ///
    /// # Safety
    ///
    /// `x`, `prev`, `g` hold `n` complex values, `step` two.
    ///
    /// [`bh_step`]: LoadedModule::bh_step
    #[kernel]
    pub unsafe fn bh_update(
        x: *mut f32,
        prev: *mut f32,
        g: *mut f32,
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
            let (gv, pv) = (load_c(g, i), load_c(prev, i));
            let dv = (beta * pv.0 - scale * gv.0, beta * pv.1 - scale * gv.1);
            let xv = load_c(x, i);
            store_c(x, i, (xv.0 + alpha * dv.0, xv.1 + alpha * dv.1));
            store_c(prev, i, dv);
            store_c(g, i, (0.0, 0.0));
        }
    }

    /// `buf *= scale T`, or with `conj != 0` `buf *= scale conj(T)`, on every plane of the
    /// stack: an angular-spectrum propagator between its two transforms. One thread per
    /// element of `[nf_max * nmode, ny, nx]`.
    ///
    /// # Safety
    ///
    /// `buf` `[nf_max * nmode, ny, nx]`, `tf` `[ny, nx]`, `ctl` one entry.
    #[kernel]
    pub unsafe fn bh_transfer(
        buf: *mut f32,
        tf: *const f32,
        scale: f32,
        conj: u32,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        let (_, nf) = unsafe { stack_frames(ctl, g) };
        let n = g.ny as usize * g.nx as usize;
        let t = global_index();
        if t >= nf as usize * g.nmode as usize * n {
            return;
        }
        unsafe {
            let (re, im) = load_c(tf, t % n);
            let im = if conj != 0 { -im } else { im };
            store_c(buf, t, cmul(load_c(buf, t), (scale * re, scale * im)));
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
    /// chunk `c`'s pair of the frame's segment `part.seg` `= Σ_k (fy im, fx im)` over `k` in
    /// `c * GAMMA_CHUNK .. min((c + 1) * GAMMA_CHUNK, n)`. `2π` times the sum of the frame's
    /// row is `∂F/∂d` ([`bh_pos_dots`]).
    ///
    /// # Safety
    ///
    /// `y` `[nf_max * nmode, ny, nx]`, `a` `[nprobe, ny, nx]`, `offsets` `[nframe, 2]` by epoch
    /// position, `views` the epoch's, `part` as [`BhPartials`] says, `ctl` one entry; grid
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
        part: BhPartials,
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
        let base = c * chunk + thread::threadIdx_x() as usize;
        // A group's loads of a mode are issued before its sums; each pixel's `im` still adds
        // its modes in order, and `acc` its pixels.
        const G: usize = 4;
        #[unroll]
        for grp in 0..CHUNK_PIXELS / G {
            let k0 = base + grp * G * PIXEL_THREADS as usize;
            let (mut e, mut fr, mut im) =
                ([(0.0f32, 0.0f32); G], [(0.0f32, 0.0f32); G], [0.0f32; G]);
            #[unroll]
            for j in 0..G {
                let k = k0 + j * PIXEL_THREADS as usize;
                if k < end {
                    (e[j], fr[j]) = unsafe { ramp_at(offsets, pos, k, ny, nx) };
                }
            }
            for m in 0..nmode {
                let p = unsafe { (*views.add(pos * nmode + m)).probe } as usize;
                let (mut yv, mut av) = ([(0.0f32, 0.0f32); G], [(0.0f32, 0.0f32); G]);
                #[unroll]
                for j in 0..G {
                    let k = k0 + j * PIXEL_THREADS as usize;
                    if k < end {
                        unsafe {
                            yv[j] = load_c(y, (fl * nmode + m) * n + k);
                            av[j] = load_c(a, p * n + k);
                        }
                    }
                }
                #[unroll]
                for j in 0..G {
                    let z = unramp(yv[j], e[j]);
                    im[j] = add_im(im[j], z, av[j]);
                }
            }
            #[unroll]
            for j in 0..G {
                let k = k0 + j * PIXEL_THREADS as usize;
                if k < end {
                    acc[0] += fr[j].0 * im[j];
                    acc[1] += fr[j].1 * im[j];
                }
            }
        }
        let o = unsafe { partials_at(part, pos, n.div_ceil(chunk), c) };
        unsafe { super::store_sums::<2>((&raw mut RED).cast::<f32>(), acc, o) }
    }

    /// The offsets' part of one batch's displacement and dots: per position `p` of `span` and
    /// axis `a`, `g = 2π Σ_c grads[2 (p nchunk + c) + a]` ([`bh_pos_grad`]; `span.nchunk` the
    /// chunks of all of a frame's segments), `d.ud[2p + a] =
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
            store_c(
                lu,
                t,
                near_prod(e, (s * au.0 - a_u * a.1, s * au.1 + a_u * a.0)),
            );
            if mv.dirs.cg != 0 {
                let av = load_c(mv.av, p);
                store_c(lv, t, near_prod(e, (av.0 - a_v * a.1, av.1 + a_v * a.0)));
            }
        }
    }

    /// The spectra of the moved probes' second-order changes along `(u, u)` and with `cg`
    /// `(u, v)` and `(v, v)`, as [`bh_pos_spec`] lays out and names the first-order ones:
    /// `muu = e (2i a_u s Au_p − a_u² A_p)`, `muv = e (i a_v s Au_p + i a_u Av_p − a_u a_v
    /// A_p)` and `mvv = e (2i a_v Av_p − a_v² A_p)`, the last two only with `cg`. One thread
    /// per element of `[nf_max * nmode, ny, nx]`.
    ///
    /// # Safety
    ///
    /// As [`bh_pos_spec`], `muu`, `muv`, `mvv` as its `lu`.
    ///
    /// [`bh_pos_spec`]: LoadedModule::bh_pos_spec
    #[kernel]
    pub unsafe fn bh_pos_second(
        mv: BhMoves,
        views: *const ViewAddr,
        muu: *mut f32,
        muv: *mut f32,
        mvv: *mut f32,
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
        // `i a z + b x` of the rates `a`, `b`.
        let turn =
            |a: f32, z: (f32, f32), b: f32, x: (f32, f32)| (b * x.0 - a * z.1, b * x.1 + a * z.0);
        unsafe {
            let p = (*views.add(f0 as usize * nmode + kl)).probe as usize * n + k;
            let (e, fr) = ramp_at(mv.offsets, pos, k, ny, nx);
            let (a_u, a_v) = rates(mv.dirs, pos, fr);
            let (a, au) = (load_c(mv.a, p), load_c(mv.au, p));
            let s = mv.probe_scale;
            let su = (s * au.0, s * au.1);
            store_c(muu, t, cmul(e, turn(2.0 * a_u, su, -a_u * a_u, a)));
            if mv.dirs.cg != 0 {
                let av = load_c(mv.av, p);
                let z = turn(a_v, su, -a_u * a_v, a);
                let z = (z.0 - a_u * av.1, z.1 + a_u * av.0);
                store_c(muv, t, cmul(e, z));
                store_c(mvv, t, cmul(e, turn(2.0 * a_v, av, -a_v * a_v, a)));
            }
        }
    }

    /// [`bh_pos_spec`] with the offsets held: `lu = e s Au_p` and with `mv.cg` `lv = e Av_p`
    /// (else `lv` is not written), reading no rates and no `A_p`. One thread per element of
    /// `[nf_max * nmode, ny, nx]`.
    ///
    /// # Safety
    ///
    /// `mv` as [`BhMovedProbes`] says, `views` the epoch's, `lu`, `lv` `[nf_max * nmode, ny,
    /// nx]`, `ctl` one entry.
    ///
    /// [`bh_pos_spec`]: LoadedModule::bh_pos_spec
    #[kernel]
    pub unsafe fn bh_moved_spec(
        mv: BhMovedProbes,
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
            let (e, _) = ramp_at(mv.offsets, pos, k, ny, nx);
            let (au, s) = (load_c(mv.au, p), mv.probe_scale);
            store_c(lu, t, near_prod(e, (s * au.0, s * au.1)));
            if mv.cg != 0 {
                store_c(lv, t, near_prod(e, load_c(mv.av, p)));
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
        // A pixel's loads of a mode are issued before its sums and stores, which keep the
        // mode order.
        while d < end {
            let (py, px) = ((d / nx as usize) as u32, (d % nx as usize) as u32);
            let at = psi_index(py, px, ny, nx);
            for m in 0..nmode {
                let j = (fl * nmode + m) * n + at;
                let v = unsafe { *views.add(pos * nmode + m) };
                let wa = (fl * g.nprobe as usize + v.probe as usize) * n + at;
                let (w, o, r, uo, du) = unsafe {
                    (
                        load_c(f.w, wa),
                        obj_corners(f.obj, v, py, px, g.nyo, g.nxo),
                        load_c(f.r, j),
                        obj_corners(dirs.u.obj, v, py, px, g.nyo, g.nxo),
                        load_c(f.lu, j),
                    )
                };
                let (mut vo, mut dv) = ([(0.0, 0.0); 4], (0.0, 0.0));
                if cg {
                    unsafe {
                        vo = obj_corners(dirs.v.obj, v, py, px, g.nyo, g.nxo);
                        dv = load_c(f.lv, j);
                    }
                }
                let (o, uo) = (bilinear(o, v), bilinear(uo, v));
                acc[0] = fma(re_conj(r, near_prod(uo, du)), 2.0, acc[0]);
                unsafe { store_c(f.lu, j, near_sum(w, uo, du, o)) };
                if cg {
                    let vo = bilinear(vo, v);
                    acc[1] = add_re(acc[1], r, near_sum(uo, dv, vo, du));
                    acc[2] = fma(re_conj(r, near_prod(vo, dv)), 2.0, acc[2]);
                    unsafe { store_c(f.lv, j, near_sum(w, vo, dv, o)) };
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
        let base = c * chunk + thread::threadIdx_x() as usize;
        // A group's loads of a mode are issued before its sums; each pixel's sums still add
        // its modes in order, and `acc` its pixels.
        const G: usize = 1;
        #[unroll]
        for grp in 0..CHUNK_PIXELS / G {
            let d0 = base + grp * G * PIXEL_THREADS as usize;
            let mut count = [MASKED; G];
            let (mut k, mut e, mut rate) =
                ([0usize; G], [(0.0f32, 0.0f32); G], [(0.0f32, 0.0f32); G]);
            #[unroll]
            for j in 0..G {
                let d = d0 + j * PIXEL_THREADS as usize;
                if d < end {
                    count[j] = unsafe { *counts.add(frame * n + d) };
                    k[j] = psi_index(d as u32 / nx, d as u32 % nx, ny, nx);
                    let fr;
                    (e[j], fr) = unsafe { ramp_at(mv.offsets, pos, k[j], ny, nx) };
                    rate[j] = unsafe { rates(mv.dirs, pos, fr) };
                }
            }
            // Per pixel the `far_add` sums, then `ra`, `iu`, `iv`.
            let (mut far, mut zs) = ([[0.0f32; 6]; G], [[0.0f32; 3]; G]);
            for m in 0..nmode {
                let pm = unsafe { (*views.add(pos * nmode + m)).probe } as usize * n;
                let jm = (fl * nmode + m) * n;
                let mut ld = [[(0.0f32, 0.0f32); 7]; G];
                #[unroll]
                for j in 0..G {
                    if d0 + j * PIXEL_THREADS as usize >= end {
                        continue;
                    }
                    let (jk, pk) = (jm + k[j], pm + k[j]);
                    unsafe {
                        ld[j][0] = load_c(f.far.psi, jk);
                        ld[j][1] = load_c(f.far.lu, jk);
                        ld[j][3] = load_c(f.y, jk);
                        ld[j][4] = load_c(mv.a, pk);
                        ld[j][5] = load_c(mv.au, pk);
                        if cg {
                            ld[j][2] = load_c(f.far.lv, jk);
                            ld[j][6] = load_c(mv.av, pk);
                        }
                    }
                }
                #[unroll]
                for j in 0..G {
                    let [p, lu, lv, y, a, au, av] = ld[j];
                    far_add(&mut far[j], p, lu, lv, cg);
                    let z = unramp(y, e[j]);
                    zs[j][0] += cmul(z, a).0;
                    zs[j][1] = add_im(zs[j][1], z, au);
                    if cg {
                        zs[j][2] = add_im(zs[j][2], z, av);
                    }
                }
            }
            #[unroll]
            for j in 0..G {
                if d0 + j * PIXEL_THREADS as usize >= end {
                    continue;
                }
                if count[j] != MASKED {
                    let t = far_tail(far[j], n, zsqrt(count[j] as f32));
                    // Unrolled, as every loop over `acc`, so it stays in registers.
                    #[unroll]
                    for h in 0..BH_HESS {
                        let h = h as usize;
                        acc[h] += t[h];
                    }
                }
                let ((a_u, a_v), [ra, iu, iv]) = (rate[j], zs[j]);
                // The fused forms of `acc[0] -= 2 a_u s iu + a_u² ra` and its two `cg` terms.
                acc[0] = fma(-(a_u * a_u), ra, fma(-(2.0 * a_u * s), iu, acc[0]));
                if cg {
                    acc[1] -= fma(a_u * a_v, ra, fma(a_v * s, iu, a_u * iv));
                    acc[2] = fma(-(a_v * a_v), ra, fma(-(2.0 * a_v), iv, acc[2]));
                }
            }
        }
        let o = unsafe { out.add(BH_HESS as usize * (pos * n.div_ceil(chunk) + c)) };
        unsafe { super::add_sums::<3>((&raw mut RED).cast::<f32>(), acc, o) }
    }

    /// [`bh_pos_far`] with the offsets held: adds to the triple [`bh_pos_near`] wrote of chunk
    /// `c = blockIdx.x` of frame `f0 + fl`, `fl = blockIdx.y`, only the far term of
    /// [`bh_hess_frames`] at the chunk's unmasked detector pixels, its last two with `cg != 0`
    /// (else 0).
    ///
    /// # Safety
    ///
    /// Each of `f` `[nf_max * nmode, ny, nx]` (`lv` unread without `cg`), `obs` as [`BhObs`]
    /// says, `out` `3 * nframe * nchunk` floats, `ctl` one entry; grid `(nchunk, nf_max, 1)`,
    /// block `(PIXEL_THREADS, 1, 1)`.
    ///
    /// [`bh_pos_far`]: LoadedModule::bh_pos_far
    /// [`bh_pos_near`]: LoadedModule::bh_pos_near
    /// [`bh_hess_frames`]: LoadedModule::bh_hess_frames
    #[kernel]
    pub unsafe fn bh_moved_far(
        f: BhFar,
        cg: u32,
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
        let cg = cg != 0;
        let c = thread::blockIdx_x() as usize;
        let chunk = GAMMA_CHUNK as usize;
        let end = n.min((c + 1) * chunk);
        let pos = f0 as usize + fl;
        let frame = unsafe { *obs.order.add(pos) } as usize;
        let mut acc = [0.0f32; BH_HESS as usize];
        let mut d = c * chunk + thread::threadIdx_x() as usize;
        while d < end {
            let count = unsafe { *obs.counts.add(frame * n + d) };
            if count != MASKED {
                let k = psi_index(d as u32 / nx, d as u32 % nx, ny, nx);
                let mut far = [0.0f32; 6];
                for m in 0..nmode {
                    let j = (fl * nmode + m) * n + k;
                    let (p, lu) = unsafe { (load_c(f.psi, j), load_c(f.lu, j)) };
                    let lv = if cg {
                        unsafe { load_c(f.lv, j) }
                    } else {
                        (0.0, 0.0)
                    };
                    far_add(&mut far, p, lu, lv, cg);
                }
                let t = far_tail(far, n, zsqrt(count as f32));
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

    /// The object gradient's patches and the offsets' gradient partials of chunk `c =
    /// blockIdx.x` of frame `f0 + fl`, `fl = blockIdx.y`, under full-object sampling, for a
    /// slice whose segment is `part.seg`: at probe pixel `d` of the chunk and per object mode
    /// `o`, `G = Σ_p conj(q P_p) r_m`, `m = o nprobe + p`, into `patch[k]`, `k = (o nf_max +
    /// fl) n + d`, and chunk `c`'s pair of the frame's segment `= Σ Re(conj(G) win[(1 + a) blk
    /// + k]) / 2π` over the chunk's pixels and the object modes, `blk` a block's values, which
    /// [`bh_pos_dots`] sums with the other segments' to `∂F/∂d`. Patches of frames outside the
    /// stack are not written.
    ///
    /// # Safety
    ///
    /// `f` as [`BhWinFields`] says, `win` three blocks `[nobj, nf_max, ny, nx]` (block 0
    /// unread), `patch` one, `part` as [`BhPartials`] says, `ctl` one entry; grid `(nchunk,
    /// nf_max, 1)`, block `(PIXEL_THREADS, 1, 1)`.
    ///
    /// [`bh_pos_dots`]: LoadedModule::bh_pos_dots
    #[kernel]
    pub unsafe fn bh_win_grad(
        f: BhWinFields,
        win: *const f32,
        patch: *mut f32,
        part: BhPartials,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        static mut RED: SharedArray<f32, 512> = SharedArray::UNINIT;
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let (ny, nx, nmode, nprobe) = (g.ny, g.nx, g.nmode as usize, g.nprobe as usize);
        let n = ny as usize * nx as usize;
        let fl = thread::blockIdx_y() as usize;
        // The whole block leaves together, before its first barrier.
        if fl >= nf as usize {
            return;
        }
        let blk = nmode / nprobe * g.nf_max as usize * n;
        let c = thread::blockIdx_x() as usize;
        let chunk = GAMMA_CHUNK as usize;
        let end = n.min((c + 1) * chunk);
        let pos = f0 as usize + fl;
        let mut acc = [0.0f32; 2];
        let mut d = c * chunk + thread::threadIdx_x() as usize;
        while d < end {
            let (py, px) = ((d / nx as usize) as u32, (d % nx as usize) as u32);
            let at = psi_index(py, px, ny, nx);
            let q = quad_phase(g.quad, py, px, ny, nx);
            for o in 0..nmode / nprobe {
                let mut gs = (0.0f32, 0.0f32);
                for p in 0..nprobe {
                    let j = (fl * nmode + o * nprobe + p) * n + at;
                    let (pv, r) = unsafe { (load_c(f.probe, p * n + d), load_c(f.r, j)) };
                    gs = add_patch(gs, q, pv, r);
                }
                let k = (o * g.nf_max as usize + fl) * n + d;
                unsafe {
                    store_c(patch, k, gs);
                    acc[0] += re_conj(gs, load_c(win, blk + k));
                    acc[1] += re_conj(gs, load_c(win, 2 * blk + k));
                }
            }
            d += PIXEL_THREADS as usize;
        }
        let s = 0.5 * core::f32::consts::FRAC_1_PI;
        let o = unsafe { partials_at(part, pos, n.div_ceil(chunk), c) };
        let acc = [s * acc[0], s * acc[1]];
        unsafe { super::store_sums::<2>((&raw mut RED).cast::<f32>(), acc, o) }
    }

    /// [`bh_win_grad`]'s object gradient patches alone, for the offsets held: per value `k =
    /// (o nf_max + fl) n + d` of a block `[nobj, nf_max, ny, nx]` of a frame `f0 + fl` of the
    /// stack, `patch[k] = Σ_p conj(q P_p) r_m`, `m = o nprobe + p`. Patches of frames outside
    /// the stack are not written. One thread per value of a block.
    ///
    /// # Safety
    ///
    /// `f` as [`BhWinFields`] says, `patch` one block `[nobj, nf_max, ny, nx]`, `ctl` one entry.
    ///
    /// [`bh_win_grad`]: LoadedModule::bh_win_grad
    #[kernel]
    pub unsafe fn bh_win_obj_grad(
        f: BhWinFields,
        patch: *mut f32,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        let (_, nf) = unsafe { stack_frames(ctl, g) };
        let (ny, nx, nmode, nprobe) = (g.ny, g.nx, g.nmode as usize, g.nprobe as usize);
        let (n, nf_max) = (ny as usize * nx as usize, g.nf_max as usize);
        let k = global_index();
        if k >= nmode / nprobe * nf_max * n {
            return;
        }
        let (o, fl, d) = (k / n / nf_max, k / n % nf_max, k % n);
        if fl >= nf as usize {
            return;
        }
        let (py, px) = ((d / nx as usize) as u32, (d % nx as usize) as u32);
        let at = psi_index(py, px, ny, nx);
        let q = quad_phase(g.quad, py, px, ny, nx);
        let mut gs = (0.0f32, 0.0f32);
        for p in 0..nprobe {
            let j = (fl * nmode + o * nprobe + p) * n + at;
            let (pv, r) = unsafe { (load_c(f.probe, p * n + d), load_c(f.r, j)) };
            gs = add_patch(gs, q, pv, r);
        }
        unsafe { store_c(patch, k, gs) };
    }

    /// The windows' first-order changes under full-object sampling, from `S(u_O)` in block 0
    /// of `hu` and with `dirs.cg` `S(v_O)` in block 0 of `hv`: per value `t` of a block
    /// `[nobj, nf_max, ny, nx]` of a frame `f0 + fl` of the stack, `hu[t] += Σ_a ud[2 (f0 +
    /// fl) + a] win[(1 + a) blk + t]` and with `cg` `hv[t] += Σ_a prev[2 order[f0 + fl] + a]
    /// win[(1 + a) blk + t]` (else `hv` is not written), `blk` a block's values. One thread
    /// per value of a block.
    ///
    /// # Safety
    ///
    /// `win` three blocks `[nobj, nf_max, ny, nx]` (block 0 unread), `hu` and with `cg` `hv`
    /// one, `dirs` as [`BhPosDirs`] says, `ctl` one entry.
    #[kernel]
    pub unsafe fn bh_win_lift(
        win: *const f32,
        hu: *mut f32,
        hv: *mut f32,
        dirs: BhPosDirs,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let n = g.ny as usize * g.nx as usize;
        let blk = g.nmode as usize / g.nprobe as usize * g.nf_max as usize * n;
        let t = global_index();
        if t >= blk {
            return;
        }
        let fl = t / n % g.nf_max as usize;
        if fl >= nf as usize {
            return;
        }
        let pos = f0 as usize + fl;
        unsafe {
            let (wy, wx) = (load_c(win, blk + t), load_c(win, 2 * blk + t));
            let add = |h: *mut f32, d: *const f32| {
                let (dy, dx) = (*d, *d.add(1));
                let v = load_c(h, t);
                let v = (v.0 + dy * wy.0 + dx * wx.0, v.1 + dy * wy.1 + dx * wx.1);
                store_c(h, t, v);
            };
            add(hu, dirs.ud.add(2 * pos));
            if dirs.cg != 0 {
                add(hv, dirs.prev.add(2 * *dirs.order.add(pos) as usize));
            }
        }
    }

    /// The windows' second-order Hessian partials of chunk `c = blockIdx.x` of frame `f0 +
    /// fl`, `fl = blockIdx.y`, under full-object sampling, added to `out[3p .. 3p + 3]`, `p =
    /// (f0 + fl) * nchunk + c`: with `s = Σ_p conj(r_m) q P_p` at probe pixel `d` per object
    /// mode `o` (`m = o nprobe + p`) and `o″(u, v) = Σ_a (v_da ∂_a S(u_O) + u_da ∂_a S(v_O))
    /// + Σ_ac u_da v_dc ∂_a ∂_c S(O)` there, `(Σ Re(s o″(u, u)), Σ Re(s o″(u, v)), Σ Re(s
    /// o″(v, v)))` over the chunk's pixels and the object modes, the last two not added
    /// without `dirs.cg`; `u_d` from `dirs.ud`, `v_d` from `dirs.prev`.
    ///
    /// # Safety
    ///
    /// `f` as [`BhWinFields`] says, `w` as [`BhWinDerivs`] says, `dirs` as [`BhPosDirs`]
    /// says, `out` `3 * nframe * nchunk` floats, `ctl` one entry; grid `(nchunk, nf_max, 1)`,
    /// block `(PIXEL_THREADS, 1, 1)`.
    #[kernel]
    pub unsafe fn bh_win_hess(
        f: BhWinFields,
        w: BhWinDerivs,
        dirs: BhPosDirs,
        out: *mut f32,
        ctl: *const StackCtl,
        g: StackGeom,
    ) {
        static mut RED: SharedArray<f32, 768> = SharedArray::UNINIT;
        let (f0, nf) = unsafe { stack_frames(ctl, g) };
        let (ny, nx, nmode, nprobe) = (g.ny, g.nx, g.nmode as usize, g.nprobe as usize);
        let n = ny as usize * nx as usize;
        let fl = thread::blockIdx_y() as usize;
        // The whole block leaves together, before its first barrier.
        if fl >= nf as usize {
            return;
        }
        let cg = dirs.cg != 0;
        let blk = nmode / nprobe * g.nf_max as usize * n;
        let c = thread::blockIdx_x() as usize;
        let chunk = GAMMA_CHUNK as usize;
        let end = n.min((c + 1) * chunk);
        let pos = f0 as usize + fl;
        let (uy, ux) = unsafe { (*dirs.ud.add(2 * pos), *dirs.ud.add(2 * pos + 1)) };
        let (mut vy, mut vx) = (0.0f32, 0.0f32);
        if cg {
            unsafe {
                let at = 2 * *dirs.order.add(pos) as usize;
                (vy, vx) = (*dirs.prev.add(at), *dirs.prev.add(at + 1));
            }
        }
        let mut acc = [0.0f32; BH_HESS as usize];
        let mut d = c * chunk + thread::threadIdx_x() as usize;
        while d < end {
            let (py, px) = ((d / nx as usize) as u32, (d % nx as usize) as u32);
            let at = psi_index(py, px, ny, nx);
            let q = quad_phase(g.quad, py, px, ny, nx);
            for o in 0..nmode / nprobe {
                let mut s = (0.0f32, 0.0f32);
                for p in 0..nprobe {
                    let j = (fl * nmode + o * nprobe + p) * n + at;
                    let (pv, r) = unsafe { (load_c(f.probe, p * n + d), load_c(f.r, j)) };
                    let t = cmul(cmul_conj(r, q), pv);
                    s = (s.0 + t.0, s.1 + t.1);
                }
                let k = (o * g.nf_max as usize + fl) * n + d;
                let re = |x: *const f32, b: usize| cmul(s, unsafe { load_c(x, b * blk + k) }).0;
                let (wyy, wxy, wxx) = (re(w.second, 0), re(w.second, 1), re(w.second, 2));
                let (ry, rx) = (re(w.du, 1), re(w.du, 2));
                acc[0] +=
                    2.0 * (uy * ry + ux * rx) + uy * uy * wyy + 2.0 * uy * ux * wxy + ux * ux * wxx;
                if cg {
                    let (sy, sx) = (re(w.dv, 1), re(w.dv, 2));
                    acc[1] += vy * ry
                        + vx * rx
                        + uy * sy
                        + ux * sx
                        + uy * vy * wyy
                        + (uy * vx + ux * vy) * wxy
                        + ux * vx * wxx;
                    acc[2] += 2.0 * (vy * sy + vx * sx)
                        + vy * vy * wyy
                        + 2.0 * vy * vx * wxy
                        + vx * vx * wxx;
                }
            }
            d += PIXEL_THREADS as usize;
        }
        let o = unsafe { out.add(BH_HESS as usize * (pos * n.div_ceil(chunk) + c)) };
        unsafe { super::add_sums::<3>((&raw mut RED).cast::<f32>(), acc, o) }
    }
}

/// `y conj(e)`, fused as [`add_im`].
#[device]
pub(crate) fn unramp(y: (f32, f32), e: (f32, f32)) -> (f32, f32) {
    (fma(y.0, e.0, y.1 * e.1), fma(y.0, e.1, -(y.1 * e.0)))
}

/// `s + Re(conj(a) b)`, fused as [`add_im`].
#[device]
pub(crate) fn add_re(s: f32, a: (f32, f32), b: (f32, f32)) -> f32 {
    fma(a.0, b.0, fma(a.1, b.1, s))
}

/// `s + Im(z a)`, each product fused into the sum in turn: the contraction of the per-pixel
/// loop this replaced, which a grouped loop no longer gets by itself.
#[device]
pub(crate) fn add_im(s: f32, z: (f32, f32), a: (f32, f32)) -> f32 {
    fma(z.0, a.1, fma(z.1, a.0, s))
}

/// `x y` as NVVM contracted [`cmul`] in [`bh_kernels::LoadedModule::bh_pos_near`] while its
/// sums sat between its loads; with the loads first it pairs them otherwise. It also pins
/// `bh_pos_spec`'s products as NVVM contracted them, which `bh_moved_spec` must match.
#[device]
fn near_prod(x: (f32, f32), y: (f32, f32)) -> (f32, f32) {
    (fma(x.0, y.0, -(x.1 * y.1)), fma(x.1, y.0, x.0 * y.1))
}

/// `x y + p q`, `p q`'s imaginary part fused into `x y`'s as [`add_im`], pinned as
/// [`near_prod`].
#[device]
fn near_sum(x: (f32, f32), y: (f32, f32), p: (f32, f32), q: (f32, f32)) -> (f32, f32) {
    let (a, b) = (near_prod(x, y), near_prod(p, q));
    (a.0 + b.0, add_im(a.1, p, q))
}

/// `s + conj(q p) r`, `q p` as [`near_prod`] and the real part as [`add_re`]: the forms NVVM
/// contracted [`bh_kernels::LoadedModule::bh_win_grad`]'s loop to, which
/// `bh_win_obj_grad` must match bit for bit.
#[device]
fn add_patch(s: (f32, f32), q: (f32, f32), p: (f32, f32), r: (f32, f32)) -> (f32, f32) {
    let w = near_prod(q, p);
    (add_re(s.0, w, r), s.1 + fma(w.0, r.1, -(w.1 * r.0)))
}

/// `Re(conj(a) b)`, pinned as [`near_prod`].
#[device]
fn re_conj(a: (f32, f32), b: (f32, f32)) -> f32 {
    fma(a.0, b.0, a.1 * b.1)
}

/// Adds one mode's `psi`, `lu` and `lv` to the [`far_tail`] sums `s = [Σ |psi|², c_u, c_v,
/// Σ |lu|², Σ Re(conj(lu) lv), Σ |lv|²]`, the `lv` ones only with `cg`.
#[device]
pub(crate) fn far_add(s: &mut [f32; 6], p: (f32, f32), lu: (f32, f32), lv: (f32, f32), cg: bool) {
    s[0] = add_re(s[0], p, p);
    s[1] = add_re(s[1], p, lu);
    s[3] = add_re(s[3], lu, lu);
    if cg {
        s[2] = add_re(s[2], p, lv);
        s[4] = add_re(s[4], lu, lv);
        s[5] = add_re(s[5], lv, lv);
    }
}

/// The far term of [`bh_kernels::LoadedModule::bh_hess_frames`] at one unmasked detector
/// pixel, `[H(u, u), H(u, v), H(v, v)]` (the last two 0 without `cg`), from the sums `s` of
/// [`far_add`] over its modes: `2 [(1 − root / a) Σ_m Re(conj(lu) lv) / n + root (c_u / n)
/// (c_v / n) / a³]` with `a = √(Σ_m |psi|² / n) + BH_EPS`, `c_u = Σ_m Re(conj(psi) lu)`.
#[device]
pub(crate) fn far_tail(s: [f32; 6], n: usize, root: f32) -> [f32; 3] {
    let inv_n = 1.0 / n as f32;
    let a = (s[0] * inv_n).sqrt() + BH_EPS;
    let fa = 1.0 - zdiv(root, a);
    let k = zdiv(root, a * a * a);
    let (cu, cv) = (s[1] * inv_n, s[2] * inv_n);
    // Fused as NVVM contracted them, so `bh_moved_far` stays `bh_pos_far` bit for bit.
    [
        2.0 * fma(fa * s[3], inv_n, k * cu * cu),
        2.0 * fma(fa * s[4], inv_n, k * cu * cv),
        2.0 * fma(fa * s[5], inv_n, k * cv * cv),
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

/// Chunk `c`'s pair in `part.out` of position `pos`'s segment `part.seg`, of `nchunk` pairs.
///
/// # Safety
///
/// `part` as [`BhPartials`] says, `pos` an epoch position, `c < nchunk`.
#[device]
unsafe fn partials_at(part: BhPartials, pos: usize, nchunk: usize, c: usize) -> *mut f32 {
    let row = pos * part.nseg as usize + part.seg as usize;
    unsafe { part.out.add(2 * (row * nchunk + c)) }
}

/// The wave of view `v`, plane `kl` of the stack, entering slice `s` at natural pixel `d`,
/// stored at `at`, its `psi_index`.
///
/// # Safety
///
/// `s.entry` as its kind says for `g`, `v` in range for it.
#[device]
unsafe fn entry_wave(
    s: BhSliceIn,
    v: ViewAddr,
    kl: usize,
    d: usize,
    at: usize,
    g: StackGeom,
) -> (f32, f32) {
    let n = g.ny as usize * g.nx as usize;
    let i = match s.kind {
        BH_ENTRY_PROBE => v.probe as usize * n + d,
        BH_ENTRY_MOVED => (kl / g.nmode as usize * g.nprobe as usize + v.probe as usize) * n + at,
        _ => kl * n + at,
    };
    unsafe { load_c(s.entry, i) }
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
