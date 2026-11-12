//! Device bodies of the M2 fused stack passes; the `#[kernel]` entry points in
//! [`crate::kernels`] call these, and `ptyrs-gpu` documents the passes they make up.
//!
//! A stack's projection runs in three FFT passes over a `work` buffer of `nf_max * nmode`
//! planes: [`rows_fwd`] builds each view's exit wave row by row and row-FFTs it, [`cols`]
//! column-FFTs every mode of a frame's column group together, takes the LLK partials and
//! the modulus projection, and inverse column-FFTs, and [`rows_inv`] inverse row-FFTs,
//! applies the DM update and stores the new exit waves. Object sums are written only by
//! [`gather`], which owns its object pixel, adds the stack's frames in ascending order and
//! reads the stored exit waves themselves: a per-frame contribution plane written by
//! [`rows_inv`] would push the object windows the probe sums read out of L2. Index
//! conventions are those of [`crate::pixel`].
//!
//! Every pass assumes canonical views: view `f * nmode + m` multiplies probe mode
//! `m % nprobe` and object mode `m / nprobe` (`nmode == nprobe * nobj`), and all views of a
//! frame sample the object at one position. A stack with `nf == 0` writes nothing; every
//! block past the stack leaves whole, before its first barrier.

use cuda_device::{async_copy, device, shared::DynamicSharedArray, thread};

use crate::FFT_EPT;
use crate::pixel::{
    LlkPartial, MASKED, StackCtl, StackGeom, SumPtrs, TilePtrs, ViewAddr, cmul, cmul_conj, load_c,
    obj_pixel_sources, psi_index, quad_phase, sample_obj, stack_frames, store_c,
};

/// [`rows_fwd`] builds the DM reflection `R = (2 - alpha) PO - (1 - alpha) psi` instead of
/// `PO`, and [`rows_inv`] stores `P(R) + (1 - alpha) (PO - R)` instead of `P(R)`.
pub const FUSED_DM: u32 = 1;
/// [`rows_inv`] stores the new exit waves to `psi` instead of back to `work`.
pub const FUSED_TO_PSI: u32 = 2;
/// [`cols`] writes the LLK partials of the forward far field.
pub const FUSED_LLK: u32 = 4;
/// [`cols`] applies the modulus projection and the inverse column FFT.
pub const FUSED_PROJECT: u32 = 8;
/// [`cols`] applies the Poisson LLK gradient scale and the inverse column FFT; never with
/// `FUSED_PROJECT`.
pub const FUSED_GRAD: u32 = 32;

/// Columns of one frame a [`cols`] block transforms, for every mode.
pub const FUSED_COLS: u32 = 8;
const _: () = assert!(
    2 * FUSED_COLS == 16,
    "a cols counts row is one 16-byte async copy"
);
/// Most threads of a [`cols`] block, `nmode * FUSED_COLS * ny / FFT_EPT`. Its kernels are
/// built with this launch bound, so their registers cannot cap a block below it.
pub const FUSED_COL_THREADS: u32 = 512;
/// Threads of a [`rows_fwd`] or [`rows_inv`] block: `FUSED_ROW_THREADS / (nx / FFT_EPT)`
/// rows of `nx / FFT_EPT` threads each.
pub const FUSED_ROW_THREADS: u32 = 128;

/// What a pass does, passed by value: `FUSED_*` bits and the DM `alpha`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FusedCfg {
    pub flags: u32,
    pub alpha: f32,
}

/// Device addresses of the fused passes' buffers; a pass reads and writes only those its doc
/// names. `n = ny * nx`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FusedPtrs {
    /// `[nobj, nyo, nxo]`, natural order.
    pub obj: *const f32,
    /// `[nprobe, ny, nx]`, natural order.
    pub probe: *const f32,
    /// The scan's `nframe * nmode` views.
    pub views: *const ViewAddr,
    /// `[nframe, ny, nx]`, detector order.
    pub counts: *const u16,
    /// The stack's DM exit waves, `[nf_max * nmode, ny, nx]`, origin at `[0, 0]`.
    pub psi: *mut f32,
    /// The stack's transforms, `[nf_max * nmode, ny, nx]`.
    pub work: *mut f32,
    /// `P_p q`, `[nprobe, ny, nx]`, natural order.
    pub probe_q: *mut f32,
    /// `sum_p |P_p|^2`, `[ny, nx]`.
    pub probe_power: *mut f32,
    /// LLK partials of each frame's column groups, `[nf_max, nx / FUSED_COLS]`.
    pub groups: *mut LlkPartial,
}

/// Unnormalized in-register FFT of `N` points (64, 128, 256 or 512), inverse under `INV`, as
/// FFT `b` of `bpb` in the block, this thread being its thread `f`. Every thread of the block
/// must call it with the same `bpb`: it synchronizes the block. Each pass is generic in its
/// FFT side so that this resolves to one direct call and `r` stays in registers.
///
/// # Safety
///
/// `s` is block shared memory holding `bpb` FFTs of this size; the `(b, f)` pairs of the
/// block's threads are distinct and cover every thread of every FFT.
#[device]
pub(crate) unsafe fn block_fft<const N: u32, const INV: bool>(
    r: &mut [f32; 2 * FFT_EPT as usize],
    s: *mut u8,
    b: u32,
    f: u32,
    bpb: u32,
) {
    const { assert!(N == 64 || N == 128 || N == 256 || N == 512) };
    let r = r.as_mut_ptr();
    unsafe {
        match (N, INV) {
            (64, false) => crate::ptyrs_cufft_c2c_f32_n64_e16_fwd(r, s, b, f, bpb),
            (64, true) => crate::ptyrs_cufft_c2c_f32_n64_e16_inv(r, s, b, f, bpb),
            (128, false) => crate::ptyrs_cufft_c2c_f32_n128_e16_fwd(r, s, b, f, bpb),
            (128, true) => crate::ptyrs_cufft_c2c_f32_n128_e16_inv(r, s, b, f, bpb),
            (256, false) => crate::ptyrs_cufft_c2c_f32_n256_e16_fwd(r, s, b, f, bpb),
            (256, true) => crate::ptyrs_cufft_c2c_f32_n256_e16_inv(r, s, b, f, bpb),
            (_, false) => crate::ptyrs_cufft_c2c_f32_n512_e16_fwd(r, s, b, f, bpb),
            (_, true) => crate::ptyrs_cufft_c2c_f32_n512_e16_inv(r, s, b, f, bpb),
        }
    }
}

/// [`block_fft`] of a row pass: blockDim `(NX / FFT_EPT, rows)`, one row per FFT.
#[device]
pub(crate) unsafe fn row_fft<const NX: u32, const INV: bool>(r: &mut [f32; 2 * FFT_EPT as usize]) {
    let s = DynamicSharedArray::<u8, 16>::get();
    let (b, f, bpb) = (
        thread::threadIdx_y(),
        thread::threadIdx_x(),
        thread::blockDim_y(),
    );
    unsafe { block_fft::<NX, INV>(r, s, b, f, bpb) }
}

/// Pass A: `work[kl, ys, :] = FFT_x(w[ys, :])` for view `k0 + kl` of the stack and stored
/// row `ys`, with `w = PO` as `kernels::obj_probe_to_psi` stores it or, under `FUSED_DM`,
/// `w = (2 - a) PO - (1 - a) psi` as `kernels::dm_reflect` computes it. Reads `obj`,
/// `probe`, `views` and under `FUSED_DM` `psi`; writes `work`. Grid `(ny / rows, nf_max *
/// nmode)`, block `(nx / FFT_EPT, rows)`, dynamic shared memory `rows` row FFTs.
///
/// # Safety
///
/// `NX == g.nx`; the buffers it reads and writes hold their shapes for `g`; `ctl` one entry;
/// launch as above.
#[device]
pub unsafe fn rows_fwd<const NX: u32>(
    p: FusedPtrs,
    ctl: *const StackCtl,
    g: StackGeom,
    cfg: FusedCfg,
) {
    let (f0, nf) = unsafe { stack_frames(ctl, g) };
    let kl = thread::blockIdx_y();
    if kl >= nf * g.nmode {
        return;
    }
    let (ny, nx) = (g.ny, NX);
    let n = ny as usize * nx as usize;
    let ys = thread::blockIdx_x() * thread::blockDim_y() + thread::threadIdx_y();
    let t = thread::threadIdx_x();
    let tpf = nx / FFT_EPT;
    // Natural row stored at row `ys` (the shift is its own inverse).
    let y = (ys + ny / 2) % ny;
    let row = kl as usize * n + (ys * nx) as usize;
    let dm = cfg.flags & FUSED_DM != 0;
    let (a, b) = (2.0 - cfg.alpha, 1.0 - cfg.alpha);
    let mut r = [0.0f32; 2 * FFT_EPT as usize];
    unsafe {
        let v = *p.views.add((f0 * g.nmode + kl) as usize);
        #[unroll]
        for i in 0..FFT_EPT {
            let xs = t + i * tpf;
            let x = (xs + nx / 2) % nx;
            let o = sample_obj(p.obj, v, y, x, g.nyo, g.nxo);
            let pr = load_c(p.probe, v.probe as usize * n + (y * nx + x) as usize);
            let mut w = cmul(cmul(o, pr), quad_phase(g.quad, y, x, ny, nx));
            if dm {
                let s = load_c(p.psi, row + xs as usize);
                w = (a * w.0 - b * s.0, a * w.1 - b * s.1);
            }
            r[2 * i as usize] = w.0;
            r[2 * i as usize + 1] = w.1;
        }
        row_fft::<NX, false>(&mut r);
        #[unroll]
        for i in 0..FFT_EPT {
            let k = 2 * i as usize;
            store_c(p.work, row + (t + i * tpf) as usize, (r[k], r[k + 1]));
        }
    }
}

/// Pass B: column FFT of columns `u = blockIdx.x * FUSED_COLS + c` of every mode `m` of
/// frame `fl = blockIdx.y`, then per far-field pixel `calc = sum_m |Psi_m|^2 / n` in
/// ascending `m`. Under `FUSED_LLK`, `groups[fl, blockIdx.x]` gets the LLK sums of
/// `kernels::llk` over the block's pixels; under `FUSED_PROJECT`, every mode is scaled as
/// `kernels::apply_amplitude` scales it, inverse column-FFT'd and stored back. `FUSED_GRAD`
/// does the same with the scale `2 (1 - obs / max(calc, 1e-12)) / n`, 0 where masked, which
/// makes the stored waves the gradient of the Poisson LLK sum over `(Re psi, Im psi)`.
/// Reads `counts`; writes `work` and under `FUSED_LLK` `groups`.
///
/// One 1D block of `nmode * FUSED_COLS * tpf` threads (`tpf = ny / FFT_EPT`): thread `tid`
/// is thread `f = (tid / FUSED_COLS) % tpf` of FFT `m * FUSED_COLS + c`, `c = tid %
/// FUSED_COLS`, `m = tid / (FUSED_COLS * tpf)`, so a warp's loads and stores cover whole
/// row segments. Grid `(nx / FUSED_COLS, nf_max)`; dynamic shared memory `16 * FUSED_COLS *
/// tpf` bytes of LLK slots, `2 * FUSED_COLS * ny` bytes of counts, then the larger of
/// `nmode * FUSED_COLS` column FFTs and `4 * nmode * ny * FUSED_COLS` bytes.
/// `counts` must be 16-byte aligned.
///
/// # Safety
///
/// `NY == g.ny`; `work`, `counts` and under `FUSED_LLK` `groups` hold their shapes for `g`;
/// `ctl` one entry; launch as above.
#[device]
pub unsafe fn cols<const NY: u32>(p: FusedPtrs, ctl: *const StackCtl, g: StackGeom, cfg: FusedCfg) {
    let (f0, nf) = unsafe { stack_frames(ctl, g) };
    let fl = thread::blockIdx_y();
    if fl >= nf {
        return;
    }
    let (ny, nx, nmode) = (NY, g.nx, g.nmode);
    let n = ny as usize * nx as usize;
    let tpf = ny / FFT_EPT;
    // Threads per mode, and LLK slots per term.
    let nslot = FUSED_COLS * tpf;
    let tid = thread::threadIdx_x();
    let c = tid % FUSED_COLS;
    let f = (tid / FUSED_COLS) % tpf;
    let m = tid / nslot;
    let b = m * FUSED_COLS + c;
    let bpb = nmode * FUSED_COLS;
    let u = thread::blockIdx_x() * FUSED_COLS + c;
    let plane = (fl * nmode + m) as usize * n;
    let llk = cfg.flags & FUSED_LLK != 0;
    let project = cfg.flags & FUSED_PROJECT != 0;
    let grad = cfg.flags & FUSED_GRAD != 0;
    let mut r = [0.0f32; 2 * FFT_EPT as usize];
    unsafe {
        let smem = DynamicSharedArray::<u8, 16>::get();
        let slots = smem.cast::<f32>();
        // The frame's counts under this block's columns, by stored row, copied while the
        // forward FFT runs: read per pixel after it, they stalled the pass.
        let tile = smem.add(16 * nslot as usize);
        let fft_smem = tile.add(2 * (FUSED_COLS * ny) as usize);
        // `|Psi_m(v, u)|^2` at `(m * ny + v) * FUSED_COLS + c`, over the FFTs' region.
        let power = fft_smem.cast::<f32>();
        let counts = p.counts.add((f0 + fl) as usize * n);
        // The block's columns stay contiguous under the shift: `nx / 2` is a multiple of 8.
        let col0 = (thread::blockIdx_x() * FUSED_COLS + nx / 2) % nx;
        let mut row = tid;
        while row < ny {
            async_copy::cp_async_cg_16(
                tile.add(2 * (row * FUSED_COLS) as usize).cast(),
                counts.add((row * nx + col0) as usize).cast(),
            );
            row += nmode * nslot;
        }
        async_copy::cp_async_commit_group();
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
        let mut acc = [0.0f32; 4];
        #[unroll]
        for i in 0..FFT_EPT {
            let v = f + i * tpf;
            let mut s = 0.0f32;
            for mm in 0..nmode {
                s += *power.add(((mm * ny + v) * FUSED_COLS + c) as usize);
            }
            let obs = *tile.add((((v + ny / 2) % ny) * FUSED_COLS + c) as usize);
            if llk && m == 0 && obs != MASKED {
                let obs = obs as f32;
                let calc = (s * inv_n).max(0.0);
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
            if project {
                let scale = if obs == MASKED {
                    inv_n
                } else {
                    (obs as f32 / (s * inv_n).max(1e-12)).sqrt() * inv_n
                };
                r[2 * i as usize] *= scale;
                r[2 * i as usize + 1] *= scale;
            }
            if grad {
                let scale = if obs == MASKED {
                    0.0
                } else {
                    2.0 * (1.0 - obs as f32 / (s * inv_n).max(1e-12)) * inv_n
                };
                r[2 * i as usize] *= scale;
                r[2 * i as usize + 1] *= scale;
            }
        }
        if llk {
            // Mode 0's threads are `tid < nslot`; a fixed tree over them per term.
            if m == 0 {
                for (j, &a) in acc.iter().enumerate() {
                    *slots.add(j * nslot as usize + tid as usize) = a;
                }
            }
            let mut half = nslot / 2;
            while half > 0 {
                thread::sync_threads();
                if tid < half {
                    for j in 0..4 {
                        let at = j * nslot as usize + tid as usize;
                        *slots.add(at) += *slots.add(at + half as usize);
                    }
                }
                half /= 2;
            }
            thread::sync_threads();
            if tid == 0 {
                let at = fl * (nx / FUSED_COLS) + thread::blockIdx_x();
                *p.groups.add(at as usize) = LlkPartial {
                    poisson: *slots,
                    gaussian: *slots.add(nslot as usize),
                    euclidean: *slots.add(2 * nslot as usize),
                    icalc: *slots.add(3 * nslot as usize),
                };
            }
        }
        if project || grad {
            // Every power is read before the inverse FFT reuses the region.
            thread::sync_threads();
            block_fft::<NY, true>(&mut r, fft_smem, b, f, bpb);
            #[unroll]
            for i in 0..FFT_EPT {
                let k = 2 * i as usize;
                let at = plane + ((f + i * tpf) * nx + u) as usize;
                store_c(p.work, at, (r[k], r[k + 1]));
            }
        }
    }
}

/// Where [`rows_inv`] stores the new exit waves and [`gather`] reads them: `psi` under
/// `FUSED_TO_PSI`, else `work`.
#[device]
fn new_waves(p: FusedPtrs, cfg: FusedCfg) -> *mut f32 {
    if cfg.flags & FUSED_TO_PSI != 0 {
        p.psi
    } else {
        p.work
    }
}

/// Pass C: for frame `fl = blockIdx.y` of the stack and stored row `ys`, the inverse row FFT
/// of every mode `m` gives the projected exit wave `proj`. The new exit wave is `proj`, or
/// under `FUSED_DM` `proj + (1 - a) (PO - R)` with `R = (2 - a) PO - (1 - a) psi` (the
/// arithmetic of `kernels::dm_reflect` then `kernels::add_c32`); it goes to
/// [`new_waves`]. Reads `work`, `views`, and under `FUSED_DM` `obj`, `probe` and `psi`;
/// writes `work` or `psi`. Grid `(ny / rows, nf_max)`, block `(nx / FFT_EPT, rows)`, dynamic
/// shared memory `rows` row FFTs.
///
/// # Safety
///
/// `NX == g.nx`; the buffers it reads and writes hold their shapes for `g`; `ctl` one entry;
/// launch as above.
#[device]
pub unsafe fn rows_inv<const NX: u32>(
    p: FusedPtrs,
    ctl: *const StackCtl,
    g: StackGeom,
    cfg: FusedCfg,
) {
    let (f0, nf) = unsafe { stack_frames(ctl, g) };
    let fl = thread::blockIdx_y();
    if fl >= nf {
        return;
    }
    let (ny, nx) = (g.ny, NX);
    let n = ny as usize * nx as usize;
    let ys = thread::blockIdx_x() * thread::blockDim_y() + thread::threadIdx_y();
    let t = thread::threadIdx_x();
    let tpf = nx / FFT_EPT;
    let y = (ys + ny / 2) % ny;
    let dm = cfg.flags & FUSED_DM != 0;
    let out = new_waves(p, cfg);
    let (a, b) = (2.0 - cfg.alpha, 1.0 - cfg.alpha);
    let k0 = (f0 + fl) * g.nmode;
    let mut r = [0.0f32; 2 * FFT_EPT as usize];
    unsafe {
        for m in 0..g.nmode {
            let row = (fl * g.nmode + m) as usize * n + (ys * nx) as usize;
            #[unroll]
            for i in 0..FFT_EPT {
                let v = load_c(p.work, row + (t + i * tpf) as usize);
                r[2 * i as usize] = v.0;
                r[2 * i as usize + 1] = v.1;
            }
            row_fft::<NX, true>(&mut r);
            let v = *p.views.add((k0 + m) as usize);
            #[unroll]
            for i in 0..FFT_EPT {
                let k = 2 * i as usize;
                let xs = t + i * tpf;
                let mut s = (r[k], r[k + 1]);
                let at = row + xs as usize;
                if dm {
                    let x = (xs + nx / 2) % nx;
                    let pr = load_c(p.probe, v.probe as usize * n + (y * nx + x) as usize);
                    let q = quad_phase(g.quad, y, x, ny, nx);
                    let po = cmul(cmul(sample_obj(p.obj, v, y, x, g.nyo, g.nxo), pr), q);
                    let old = load_c(p.psi, at);
                    let refl = (a * po.0 - b * old.0, a * po.1 - b * old.1);
                    let d = (b * (po.0 - refl.0), b * (po.1 - refl.1));
                    s = (s.0 + d.0, s.1 + d.1);
                }
                store_c(out, at, s);
            }
        }
    }
}

/// The probe terms of [`gather`], one thread per natural pixel `(y, x)` of the frame:
/// `probe_power = sum_p |P_p|^2` in ascending `p`, and `probe_q[p] = P_p q`. Reads `probe`;
/// writes `probe_q` and `probe_power`.
///
/// # Safety
///
/// `probe`, `probe_q` and `probe_power` hold their shapes for `g`.
#[device]
pub unsafe fn probe_terms(p: FusedPtrs, g: StackGeom) {
    let (ny, nx) = (g.ny, g.nx);
    let n = ny as usize * nx as usize;
    let i = crate::global_index();
    if i >= n {
        return;
    }
    let (y, x) = ((i / nx as usize) as u32, (i % nx as usize) as u32);
    let q = quad_phase(g.quad, y, x, ny, nx);
    let mut s = 0.0f32;
    for pm in 0..g.nprobe as usize {
        let v = unsafe { load_c(p.probe, pm * n + i) };
        s += v.0 * v.0 + v.1 * v.1;
        unsafe { store_c(p.probe_q, pm * n + i, cmul(v, q)) };
    }
    unsafe { *p.probe_power.add(i) = s };
}

/// Adds the stack's object-update sums from the exit waves pass C left in [`new_waves`]:
/// `num += sum_f B_f^T c_(f, o)` with `c_(f, o) = sum_pm conj(probe_q[pm]) psi_(f, o, pm)`
/// in ascending `pm`, and `den += sum_f B_f^T probe_power`, over the stack's frames in
/// ascending order, `B_f` the bilinear sampling of frame `f` (read from its view of mode
/// `o * nprobe`). Each object pixel is owned by one thread, which reads and writes its sums
/// only if its tile's run holds a view of the stack, as `kernels::psi_to_obj`; so the
/// result does not depend on how frames are split into stacks. Reads the new waves,
/// `probe_q`, `probe_power`, `views`; writes `sums`. A stack-tile launch, as
/// `kernels::psi_to_obj`.
///
/// # Safety
///
/// As `kernels::psi_to_obj`, with the new waves in place of `psi`, and `probe_q` and
/// `probe_power` as [`probe_terms`] wrote them for this probe in place of `probe`; `shared`
/// as `for_each_stack_tile` needs.
#[device]
pub unsafe fn gather(
    p: FusedPtrs,
    tiles: TilePtrs,
    sums: SumPtrs,
    shared: *mut u32,
    ctl: *const StackCtl,
    g: StackGeom,
    cfg: FusedCfg,
) {
    let iobj = thread::blockIdx_z();
    let n = g.ny as usize * g.nx as usize;
    let waves = new_waves(p, cfg);
    let (f0, nf) = unsafe { stack_frames(ctl, g) };
    let (k0, k1) = (f0 * g.nmode, (f0 + nf) * g.nmode);
    let pixel = |lo: u32, end: u32, y: u32, x: u32| unsafe {
        if x >= g.nxo || y >= g.nyo {
            return;
        }
        let px = (iobj as usize * g.nyo as usize + y as usize) * g.nxo as usize + x as usize;
        let mut acc = load_c(sums.num, px);
        let mut norm = *sums.den.add(px);
        for j in lo..end {
            let k = *tiles.views.add(j as usize);
            if k >= k1 {
                break;
            }
            let v = *p.views.add(k as usize);
            if v.obj != iobj || v.probe != 0 {
                continue;
            }
            // Mode `o * nprobe + pm` of the frame is view `k + pm`, plane `k - k0 + pm`.
            let plane = (k - k0) as usize * n;
            obj_pixel_sources(v, y, x, g.ny, g.nx, |pry, prx, w| {
                let at = (pry * g.nx + prx) as usize;
                let ps = psi_index(pry, prx, g.ny, g.nx);
                let mut c = (0.0f32, 0.0f32);
                for pm in 0..g.nprobe as usize {
                    let u = cmul_conj(
                        load_c(p.probe_q, pm * n + at),
                        load_c(waves, plane + pm * n + ps),
                    );
                    c = (c.0 + u.0, c.1 + u.1);
                }
                acc = (acc.0 + w * c.0, acc.1 + w * c.1);
                norm += w * *p.probe_power.add(at);
            });
        }
        store_c(sums.num, px, acc);
        *sums.den.add(px) = norm;
    };
    unsafe { crate::for_each_stack_tile(p.views, tiles, shared, ctl, g, pixel) }
}
