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
//! The position model shifts the probe of frame `f` by `e_f` in Fourier space: [`shift_pre`]
//! writes the column half `T = IDFT_y(A e_f)` of the probe spectrum `A`, [`shift_rows_fwd`]
//! and [`shift_rows_inv`] stand in for [`rows_fwd`] and [`rows_inv`] with `W = IDFT_x(T)` for
//! `P q`, [`shift_rows_inv`] also taking the position partials, and [`shift_post`]
//! column-FFTs the `X` that [`shift_rows_inv`] leaves in `shift_adj` into the probe's `G`.
//! [`gather_shifted`] and [`shift_probe_sums`] then take the object and probe sums.
//!
//! Every pass assumes canonical views: view `f * nmode + m` multiplies probe mode
//! `m % nprobe` and object mode `m / nprobe` (`nmode == nprobe * nobj`), and all views of a
//! frame sample the object at one position. A stack with `nf == 0` writes nothing; every
//! block past the stack leaves whole, before its first barrier.

use cuda_device::{async_copy, device, shared::DynamicSharedArray, thread};

use crate::pixel::{
    LlkPartial, MASKED, PositionPartial, ScalePartial, StackCtl, StackGeom, SumPtrs, TilePtrs,
    ViewAddr, WARP, cmul, cmul_conj, freq, llk_add, load_c, obj_pixel_sources, psi_index,
    quad_phase, ramp, sample_obj, stack_frames, store_c,
};
use crate::{FFT_EPT_MAX, fft_ept};

/// [`rows_fwd`] builds the DM reflection `R = (2 - alpha) PO - (1 - alpha) psi` instead of
/// `PO`, and [`rows_inv`] stores `P(R) + (1 - alpha) (PO - R)` instead of `P(R)`.
pub const FUSED_DM: u32 = 1;
/// [`rows_inv`] stores the new exit waves to `psi` instead of back to `work`.
pub const FUSED_TO_PSI: u32 = 2;
/// [`cols`] writes the LLK partials of the forward far field.
pub const FUSED_LLK: u32 = 4;
/// [`cols`] applies the modulus projection and the inverse column FFT.
pub const FUSED_PROJECT: u32 = 8;
/// A frame pass walks the stack last to first: grid row `blockIdx.y` takes stack slot `count -
/// 1 - blockIdx.y` of the pass's `count` frames (views for [`rows_fwd`]), not `blockIdx.y`.
/// Blocks start in grid order, so a pass walking opposite to the one whose plane it reads
/// first reads what that one touched last, which L2 still holds when the stack's planes
/// outgrow it.
pub const FUSED_DOWN: u32 = 16;
/// [`cols`] applies the Poisson LLK gradient scale and the inverse column FFT; never with
/// `FUSED_PROJECT`. [`shift_rows_inv`]'s position sums then take the new waves as the
/// exit-wave correction, in place of `x - O W`.
pub const FUSED_GRAD: u32 = 32;
/// [`cols`] compares the counts with `calc = s[f] I` instead of `I = sum_m |Psi_m|^2 / n`.
pub const FUSED_SCALE: u32 = 64;
/// [`cols`] writes the [`ScalePartial`] sums of the forward far field.
pub const FUSED_SCALE_SUMS: u32 = 128;
/// [`cols`] adds the detector background `b[j]` to `calc`.
pub const FUSED_BACKGROUND: u32 = 256;
/// [`cols`] writes each frame's background ratio `obs / max(calc, 1e-12)` of the forward far
/// field.
pub const FUSED_BACKGROUND_SUMS: u32 = 512;
/// The model terms of [`cols`]: a build compiles in those of its `TERMS`, so that a pass
/// without them runs a build without their per-pixel work.
pub const FUSED_MODEL_TERMS: u32 =
    FUSED_SCALE | FUSED_SCALE_SUMS | FUSED_BACKGROUND | FUSED_BACKGROUND_SUMS;
/// [`shift_rows_inv`] writes each frame's object terms of the shifted probe, `shift_obj` and
/// `shift_power`, for [`gather_shifted`].
pub const FUSED_SHIFT_OBJ: u32 = 1024;
/// [`shift_rows_inv`] writes `X` and [`shift_post`] the probe adjoint's `G`.
pub const FUSED_SHIFT_PROBE: u32 = 2048;
/// [`shift_pre`] writes `T_y` and [`shift_rows_inv`] the position sums `shift_grad`.
pub const FUSED_SHIFT_POSITION: u32 = 4096;
/// [`shift_pre`] writes `T_y` and [`shift_rows_inv`] the denominator sums `shift_den`.
pub const FUSED_SHIFT_DEN: u32 = 8192;

/// Columns of one frame a [`cols`] block transforms, for every mode.
pub const FUSED_COLS: u32 = 8;
const _: () = assert!(
    2 * FUSED_COLS == 16,
    "a cols counts row is one 16-byte async copy"
);
/// Most threads of a [`cols`], [`shift_pre`] or [`shift_post`] block over columns of `ny`,
/// `nmode * FUSED_COLS * ny / fft_ept(ny)`, and their kernels' launch bound, so registers cannot
/// cap a block below it: 1024 needs a thread in 64 registers, which holds at most 8 elements.
#[device]
pub const fn fused_col_threads(ny: u32) -> u32 {
    if fft_ept(ny) <= 8 { 1024 } else { 512 }
}
/// Threads of a [`rows_fwd`] or [`rows_inv`] block: `FUSED_ROW_THREADS / tpf` rows of `tpf =
/// nx / fft_ept(nx)` threads each.
pub const FUSED_ROW_THREADS: u32 = 128;

/// Sets `fused_rows_inv_512`'s preferred shared-memory carveout in `module` to 32% (32 KB of
/// sm_89's 100, three of its blocks). Left to the driver, the fourth block its registers allow
/// takes a 64 KB carveout, and the L1 that leaves costs the kernel more than the block gains.
pub fn pin_rows_inv_carveout(
    module: &crate::kernels::LoadedModule,
) -> Result<(), cuda_core::DriverError> {
    use cuda_core::{IntoResult, sys};
    let f = module
        .as_cuda_module()
        .load_function("fused_rows_inv_512")?;
    // SAFETY: `f` keeps the module, and so the handle, alive through the call.
    unsafe {
        sys::cuFuncSetAttribute(
            f.cu_function(),
            sys::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
            32,
        )
    }
    .result()
}

/// What a pass does, passed by value: `FUSED_*` bits and the DM `alpha`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FusedCfg {
    pub flags: u32,
    pub alpha: f32,
}

/// The stack slot, of `count`, of a frame pass block: its grid row `blockIdx.y`, or `count -
/// 1 - blockIdx.y` under [`FUSED_DOWN`] in `flags`; `None` past `count`, as in a stack short
/// of `nf_max`.
#[device]
pub(crate) fn walk_slot(flags: u32, count: u32) -> Option<u32> {
    let by = thread::blockIdx_y();
    if by >= count {
        None
    } else if flags & FUSED_DOWN != 0 {
        Some(count - 1 - by)
    } else {
        Some(by)
    }
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
    /// The intensity scale `s`, `[nframe]`.
    pub scale: *const f32,
    /// Scale sums of each frame's column groups, `[nf_max, nx / FUSED_COLS]`.
    pub scale_groups: *mut ScalePartial,
    /// The background `b`, `[ny, nx]`, detector order.
    pub background: *const f32,
    /// Each frame's background ratio, `[nf_max, ny, nx]`, detector order.
    pub background_ratio: *mut f32,
    /// Each frame's probe shift `(dy, dx)` in pixels, `[nframe, 2]`.
    pub offsets: *const f32,
    /// The probe spectra `A_p = DFT(stored(P_p q)) / n`, `[nprobe, ny, nx]`, FFT order.
    pub spectrum: *const f32,
    /// One plane per frame and probe mode, `fl * nprobe + p`, `[nf_max * nprobe, ny, nx]`:
    /// `T` of the shift passes.
    pub shift: *mut f32,
    /// `T_y` of the shift passes until [`shift_rows_inv`] has read a row, which then holds the
    /// probe adjoint's `X`, then its `G`, as `shift`.
    pub shift_adj: *mut f32,
    /// `c = sum_p conj(W_p) psi_(o, p)` of each frame and object mode, `fl * nobj + o`,
    /// `[nf_max * nobj, ny, nx]`, origin at `[0, 0]`.
    pub shift_obj: *mut f32,
    /// `sum_p |W_p|^2` of each frame, `[nf_max, ny, nx]`, origin at `[0, 0]`.
    pub shift_power: *mut f32,
    /// Position sums of each frame's row blocks, `[nf_max, ny / rows]`.
    pub shift_grad: *mut PositionPartial,
    /// Denominator sums of each frame's row blocks, `[nf_max, ny / rows]`.
    pub shift_den: *mut PositionPartial,
}

/// Bytes of [`cols`]' reduction slots under `flags` for columns of `ny`: a row of
/// `FUSED_COLS * ny / fft_ept(ny)` f32 per LLK term, and per scale sum under `FUSED_SCALE_SUMS`.
#[device]
pub fn col_slot_bytes(flags: u32, ny: u32) -> u32 {
    let rows = if flags & FUSED_SCALE_SUMS != 0 { 7 } else { 4 };
    4 * rows * FUSED_COLS * (ny / fft_ept(ny))
}

/// Bytes of [`shift_rows_inv`]'s `T_y` stage under `flags` for rows of `nx`: the block's rows
/// under `FUSED_SHIFT_DEN` or `FUSED_SHIFT_POSITION`, none otherwise.
#[device]
pub fn shift_stage_bytes(flags: u32, nx: u32) -> u32 {
    if flags & (FUSED_SHIFT_DEN | FUSED_SHIFT_POSITION) != 0 {
        8 * FUSED_ROW_THREADS * fft_ept(nx)
    } else {
        0
    }
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
    r: &mut [f32; 2 * FFT_EPT_MAX as usize],
    s: *mut u8,
    b: u32,
    f: u32,
    bpb: u32,
) {
    const { assert!(N == 64 || N == 128 || N == 256 || N == 512) };
    let r = r.as_mut_ptr();
    unsafe {
        match (N, INV) {
            (64, false) => crate::ptyrs_cufft_c2c_f32_n64_e8_fwd(r, s, b, f, bpb),
            (64, true) => crate::ptyrs_cufft_c2c_f32_n64_e8_inv(r, s, b, f, bpb),
            (128, false) => crate::ptyrs_cufft_c2c_f32_n128_e8_fwd(r, s, b, f, bpb),
            (128, true) => crate::ptyrs_cufft_c2c_f32_n128_e8_inv(r, s, b, f, bpb),
            (256, false) => crate::ptyrs_cufft_c2c_f32_n256_e8_fwd(r, s, b, f, bpb),
            (256, true) => crate::ptyrs_cufft_c2c_f32_n256_e8_inv(r, s, b, f, bpb),
            (_, false) => crate::ptyrs_cufft_c2c_f32_n512_e16_fwd(r, s, b, f, bpb),
            (_, true) => crate::ptyrs_cufft_c2c_f32_n512_e16_inv(r, s, b, f, bpb),
        }
    }
}

/// Bytes a column or row pass gives each FFT past the FFT's own `smem_per_fft`: unskewed, the
/// exchange addresses of FFTs that consecutive lanes hold at one thread index repeat at the
/// FFT stride, a multiple of 128 bytes, and collide in one bank.
pub const FFT_SKEW: u32 = 16;

/// [`block_fft`] of a column pass, FFT `b` at `b * (smem_per_fft + FFT_SKEW)` bytes into `s`.
///
/// # Safety
///
/// As [`block_fft`], `s` holding `bpb` such column FFTs.
#[device]
pub(crate) unsafe fn col_fft<const N: u32, const INV: bool>(
    r: &mut [f32; 2 * FFT_EPT_MAX as usize],
    s: *mut u8,
    b: u32,
    f: u32,
    bpb: u32,
) {
    // `block_fft` places FFT `b` at `b * smem_per_fft`; the shift adds the skew.
    unsafe { block_fft::<N, INV>(r, s.add((b * FFT_SKEW) as usize), b, f, bpb) }
}

/// This thread of a row pass of `NX`: thread `f` of block row `b`, stored row `ys`. The 1D
/// block of `FUSED_ROW_THREADS` threads holds `FUSED_ROW_THREADS / tpf` rows (`tpf = NX /
/// fft_ept(NX)`, at most a warp), each warp `group = WARP / tpf` whole rows with lane `l`
/// thread `l / group` of row `l % group`: the warp touches global memory as in row-major lane
/// order, while its lanes' exchange addresses spread over the rows' [`FFT_SKEW`].
#[derive(Clone, Copy)]
struct RowLane {
    ys: u32,
    b: u32,
    f: u32,
}

#[device]
fn row_lane<const NX: u32>() -> RowLane {
    let tpf = const {
        let tpf = NX / fft_ept(NX);
        assert!(tpf <= WARP, "a row's FFT threads must fit one warp");
        tpf
    };
    let group = WARP / tpf;
    let tid = thread::threadIdx_x();
    let (w, l) = (tid / WARP, tid % WARP);
    let b = w * group + l % group;
    RowLane {
        ys: thread::blockIdx_x() * (FUSED_ROW_THREADS / tpf) + b,
        b,
        f: l / group,
    }
}

/// [`block_fft`] of a row pass, one row per FFT at `lane`.
#[device]
unsafe fn row_fft<const NX: u32, const INV: bool>(
    r: &mut [f32; 2 * FFT_EPT_MAX as usize],
    lane: RowLane,
) {
    unsafe { row_fft_at::<NX, INV>(r, DynamicSharedArray::<u8, 16>::get(), lane) }
}

/// [`row_fft`] with the block's row FFTs at `s` in shared memory.
#[device]
unsafe fn row_fft_at<const NX: u32, const INV: bool>(
    r: &mut [f32; 2 * FFT_EPT_MAX as usize],
    s: *mut u8,
    lane: RowLane,
) {
    let skew = lane.b * FFT_SKEW;
    let rows = FUSED_ROW_THREADS / (NX / const { fft_ept(NX) });
    unsafe { block_fft::<NX, INV>(r, s.add(skew as usize), lane.b, lane.f, rows) }
}

/// Pass A: `work[kl, ys, :] = FFT_x(w[ys, :])` for view `k0 + kl` of the stack, `kl` of
/// grid row `blockIdx.y` ([`FUSED_DOWN`]), and stored row `ys`, with `w = PO` as
/// `kernels::obj_probe_to_psi` stores it or, under `FUSED_DM`, `w = (2 - a) PO - (1 - a)
/// psi` as `kernels::dm_reflect` computes it. Reads `obj`, `probe`, `views` and under
/// `FUSED_DM` `psi`; writes `work`. Grid `(ny / rows, nf_max * nmode)`, block
/// `FUSED_ROW_THREADS`, dynamic shared memory `rows` row FFTs ([`FFT_SKEW`]).
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
    unsafe { rows_fwd_with::<NX, false>(p, ctl, g, cfg) }
}

/// [`rows_fwd`] with the shifted probe: `W_p = IDFT_x(T)` of the stored row of plane `fl *
/// nprobe + p` of `shift` ([`shift_pre`]) in place of `P_p q`, so `w = O W` (`R = (2 - a) O
/// W - (1 - a) psi` under `FUSED_DM`). Reads `shift` in place of `probe`; launch as
/// [`rows_fwd`].
///
/// # Safety
///
/// As [`rows_fwd`], and `shift` holds [`shift_pre`]'s `T` of this stack.
#[device]
pub unsafe fn shift_rows_fwd<const NX: u32>(
    p: FusedPtrs,
    ctl: *const StackCtl,
    g: StackGeom,
    cfg: FusedCfg,
) {
    unsafe { rows_fwd_with::<NX, true>(p, ctl, g, cfg) }
}

#[device]
unsafe fn rows_fwd_with<const NX: u32, const SHIFT: bool>(
    p: FusedPtrs,
    ctl: *const StackCtl,
    g: StackGeom,
    cfg: FusedCfg,
) {
    let (f0, nf) = unsafe { stack_frames(ctl, g) };
    let Some(kl) = walk_slot(cfg.flags, nf * g.nmode) else {
        return;
    };
    let (ny, nx) = (g.ny, NX);
    let n = ny as usize * nx as usize;
    let lane = row_lane::<NX>();
    let (ys, t) = (lane.ys, lane.f);
    let ept = const { fft_ept(NX) };
    let tpf = nx / ept;
    // Natural row stored at row `ys` (the shift is its own inverse).
    let y = (ys + ny / 2) % ny;
    let row = kl as usize * n + (ys * nx) as usize;
    let dm = cfg.flags & FUSED_DM != 0;
    let (a, b) = (2.0 - cfg.alpha, 1.0 - cfg.alpha);
    let mut r = [0.0f32; 2 * FFT_EPT_MAX as usize];
    unsafe {
        let v = *p.views.add((f0 * g.nmode + kl) as usize);
        if SHIFT {
            let plane = (kl / g.nmode * g.nprobe + v.probe) as usize * n;
            #[unroll]
            for i in 0..ept {
                let w = load_c(p.shift, plane + (ys * nx + t + i * tpf) as usize);
                r[2 * i as usize] = w.0;
                r[2 * i as usize + 1] = w.1;
            }
            row_fft::<NX, true>(&mut r, lane);
            thread::sync_threads();
        }
        #[unroll]
        for i in 0..ept {
            let xs = t + i * tpf;
            let x = (xs + nx / 2) % nx;
            let o = sample_obj(p.obj, v, y, x, g.nyo, g.nxo);
            let mut w = if SHIFT {
                cmul(o, (r[2 * i as usize], r[2 * i as usize + 1]))
            } else {
                let pr = load_c(p.probe, v.probe as usize * n + (y * nx + x) as usize);
                cmul(cmul(o, pr), quad_phase(g.quad, y, x, ny, nx))
            };
            if dm {
                let s = load_c(p.psi, row + xs as usize);
                w = (a * w.0 - b * s.0, a * w.1 - b * s.1);
            }
            r[2 * i as usize] = w.0;
            r[2 * i as usize + 1] = w.1;
        }
        row_fft::<NX, false>(&mut r, lane);
        #[unroll]
        for i in 0..ept {
            let k = 2 * i as usize;
            store_c(p.work, row + (t + i * tpf) as usize, (r[k], r[k + 1]));
        }
    }
}

/// Pass B: column FFT of columns `u = blockIdx.x * FUSED_COLS + c` of every mode `m` of
/// frame `fl` of grid row `blockIdx.y` ([`FUSED_DOWN`]), then per far-field pixel the model
/// intensity `I = sum_m |Psi_m|^2 / n` in ascending `m` and `calc = I`, under `FUSED_SCALE`
/// times `s = scale[f0 + fl]` and under `FUSED_BACKGROUND` plus `background[j]` at detector
/// pixel `j`. Under `FUSED_LLK`, `groups[fl, blockIdx.x]` gets the LLK sums of
/// `kernels::llk` of `calc` over the block's pixels, under `FUSED_SCALE_SUMS`
/// `scale_groups[fl, blockIdx.x]` the [`ScalePartial`] sums, and under
/// `FUSED_BACKGROUND_SUMS` `background_ratio[fl, j]` is `obs / max(calc, 1e-12)`, 0 where
/// masked. Under `FUSED_PROJECT`, every mode is scaled by `sqrt(obs / max(calc, 1e-12)) /
/// n`, `1 / n` where masked (with `calc = I` the scale of `kernels::apply_amplitude`),
/// inverse column-FFT'd and stored back. `FUSED_GRAD` does the same with the scale `2 s (1
/// - obs / max(calc, 1e-12)) / n`, 0 where masked, which makes the stored waves the
/// gradient of the Poisson LLK sum of `calc` over `(Re psi, Im psi)`. Reads `counts`, and
/// `scale` and `background` under their flags; writes `work`, and `groups`, `scale_groups`
/// and `background_ratio` under theirs.
///
/// One 1D block of `nmode * FUSED_COLS * tpf` threads (`tpf = ny / fft_ept(ny)`): thread `tid`
/// is thread `f = (tid / FUSED_COLS) % tpf` of FFT `m * FUSED_COLS + c`, `c = tid %
/// FUSED_COLS`, `m = tid / (FUSED_COLS * tpf)`, so a warp's loads and stores cover whole
/// row segments. Grid `(nx / FUSED_COLS, nf_max)`; dynamic shared memory
/// `col_slot_bytes(flags, ny)` bytes of reduction slots, `2 * FUSED_COLS * ny` bytes of
/// counts, then the larger of `nmode * FUSED_COLS` column FFTs ([`FFT_SKEW`]) and
/// `4 * nmode * ny * FUSED_COLS` bytes.
/// `counts` must be 16-byte aligned. Of the [`FUSED_MODEL_TERMS`], the build honours those
/// in `TERMS` and ignores the rest of `cfg.flags`.
///
/// # Safety
///
/// `NY == g.ny`; `work`, `counts` and the buffers the flags name hold their shapes for `g`;
/// `ctl` one entry; launch as above.
#[device]
pub unsafe fn cols<const NY: u32, const TERMS: u32>(
    p: FusedPtrs,
    ctl: *const StackCtl,
    g: StackGeom,
    cfg: FusedCfg,
) {
    let (f0, nf) = unsafe { stack_frames(ctl, g) };
    let Some(fl) = walk_slot(cfg.flags, nf) else {
        return;
    };
    let (ny, nx, nmode) = (NY, g.nx, g.nmode);
    let n = ny as usize * nx as usize;
    let ept = const { fft_ept(NY) };
    let tpf = ny / ept;
    // Threads per mode, and reduction slots per term.
    let nslot = FUSED_COLS * tpf;
    let tid = thread::threadIdx_x();
    let c = tid % FUSED_COLS;
    let f = (tid / FUSED_COLS) % tpf;
    let m = tid / nslot;
    let b = m * FUSED_COLS + c;
    let bpb = nmode * FUSED_COLS;
    let u = thread::blockIdx_x() * FUSED_COLS + c;
    let plane = (fl * nmode + m) as usize * n;
    // A constant mask: a term outside `TERMS` folds away with its per-pixel work.
    let flags = cfg.flags & (TERMS | !FUSED_MODEL_TERMS);
    let llk = flags & FUSED_LLK != 0;
    let sums = flags & FUSED_SCALE_SUMS != 0;
    let background = flags & FUSED_BACKGROUND != 0;
    let bg_ratio = flags & FUSED_BACKGROUND_SUMS != 0;
    let project = flags & FUSED_PROJECT != 0;
    let grad = flags & FUSED_GRAD != 0;
    let mut r = [0.0f32; 2 * FFT_EPT_MAX as usize];
    unsafe {
        let smem = DynamicSharedArray::<u8, 16>::get();
        let slots = smem.cast::<f32>();
        // The frame's counts under this block's columns, by stored row, copied while the
        // forward FFT runs: read per pixel after it, they stalled the pass.
        let tile = smem.add(col_slot_bytes(flags, ny) as usize);
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
        for i in 0..ept {
            let v = load_c(p.work, plane + ((f + i * tpf) * nx + u) as usize);
            r[2 * i as usize] = v.0;
            r[2 * i as usize + 1] = v.1;
        }
        col_fft::<NY, false>(&mut r, fft_smem, b, f, bpb);
        thread::sync_threads();
        #[unroll]
        for i in 0..ept {
            let (re, im) = (r[2 * i as usize], r[2 * i as usize + 1]);
            let v = f + i * tpf;
            *power.add(((m * ny + v) * FUSED_COLS + c) as usize) = re * re + im * im;
        }
        async_copy::cp_async_wait_all();
        thread::sync_threads();
        let inv_n = 1.0 / n as f32;
        let tile = tile.cast::<u16>();
        // `1 * I == I`: without `FUSED_SCALE` every result is the unscaled pass's, bit for bit.
        let sf = if flags & FUSED_SCALE != 0 {
            *p.scale.add((f0 + fl) as usize)
        } else {
            1.0
        };
        let mut acc = LlkPartial::default();
        let mut sacc = ScalePartial::default();
        #[unroll]
        for i in 0..ept {
            let v = f + i * tpf;
            let mut s = 0.0f32;
            for mm in 0..nmode {
                s += *power.add(((mm * ny + v) * FUSED_COLS + c) as usize);
            }
            let intensity = s * inv_n;
            let y = (v + ny / 2) % ny;
            // Detector pixel: `col0 + c` is `(u + nx / 2) % nx`.
            let j = (y * nx + col0 + c) as usize;
            let mut calc = sf * intensity;
            if background {
                calc += *p.background.add(j);
            }
            let obs = *tile.add((y * FUSED_COLS + c) as usize);
            if m == 0 && bg_ratio {
                *p.background_ratio.add(fl as usize * n + j) = if obs == MASKED {
                    0.0
                } else {
                    obs as f32 / calc.max(1e-12)
                };
            }
            if m == 0 && obs != MASKED {
                let obs = obs as f32;
                if llk {
                    llk_add(&mut acc, obs, calc.max(0.0));
                }
                if sums {
                    sacc.intensity += intensity;
                    sacc.ratio += obs * intensity / calc.max(1e-12);
                    sacc.amplitude += (obs * intensity).sqrt();
                }
            }
            if project {
                let scale = if obs == MASKED {
                    inv_n
                } else {
                    (obs as f32 / calc.max(1e-12)).sqrt() * inv_n
                };
                r[2 * i as usize] *= scale;
                r[2 * i as usize + 1] *= scale;
            }
            if grad {
                let scale = if obs == MASKED {
                    0.0
                } else {
                    2.0 * sf * (1.0 - obs as f32 / calc.max(1e-12)) * inv_n
                };
                r[2 * i as usize] *= scale;
                r[2 * i as usize + 1] *= scale;
            }
        }
        if llk || sums {
            // Mode 0's threads are `tid < nslot`; a fixed tree over them per term reduced,
            // rows `lo..hi` of the slots `[7][nslot]`: the LLK terms 0 to 3, the scale sums
            // 4 to 6.
            let lo = if llk { 0 } else { 4 };
            let hi = if sums { 7 } else { 4 };
            let ns = nslot as usize;
            if m == 0 {
                // One store per term: an array indexed by the runtime `lo..hi` lives in
                // local memory.
                let s = slots.add(tid as usize);
                if llk {
                    *s = acc.poisson;
                    *s.add(ns) = acc.gaussian;
                    *s.add(2 * ns) = acc.euclidean;
                    *s.add(3 * ns) = acc.icalc;
                }
                if sums {
                    *s.add(4 * ns) = sacc.intensity;
                    *s.add(5 * ns) = sacc.ratio;
                    *s.add(6 * ns) = sacc.amplitude;
                }
            }
            let mut half = nslot / 2;
            while half > 0 {
                thread::sync_threads();
                if tid < half {
                    for j in lo..hi {
                        let r = slots.add(j * ns + tid as usize);
                        *r += *r.add(half as usize);
                    }
                }
                half /= 2;
            }
            thread::sync_threads();
            if tid == 0 {
                let at = (fl * (nx / FUSED_COLS) + thread::blockIdx_x()) as usize;
                if llk {
                    *p.groups.add(at) = LlkPartial {
                        poisson: *slots,
                        gaussian: *slots.add(ns),
                        euclidean: *slots.add(2 * ns),
                        icalc: *slots.add(3 * ns),
                    };
                }
                if sums {
                    *p.scale_groups.add(at) = ScalePartial {
                        intensity: *slots.add(4 * ns),
                        ratio: *slots.add(5 * ns),
                        amplitude: *slots.add(6 * ns),
                    };
                }
            }
        }
        if project || grad {
            // Every power is read before the inverse FFT reuses the region.
            thread::sync_threads();
            col_fft::<NY, true>(&mut r, fft_smem, b, f, bpb);
            #[unroll]
            for i in 0..ept {
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

/// Pass C: for frame `fl` of grid row `blockIdx.y` ([`FUSED_DOWN`]) of the stack and stored
/// row `ys`, the inverse row FFT of every mode `m` gives the projected exit wave `proj`.
/// The new exit wave is `proj`, or under `FUSED_DM` `proj + (1 - a) (PO - R)` with `R = (2
/// - a) PO - (1 - a) psi` (the arithmetic of `kernels::dm_reflect` then
/// `kernels::add_c32`); it goes to `new_waves`. Reads `work`, `views`, and under `FUSED_DM`
/// `obj`, `probe` and `psi`; writes `work` or `psi`. Grid `(ny / rows, nf_max)`, block
/// `FUSED_ROW_THREADS`, dynamic shared memory `rows` row FFTs ([`FFT_SKEW`]).
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
    let Some(fl) = walk_slot(cfg.flags, nf) else {
        return;
    };
    let (ny, nx) = (g.ny, NX);
    let n = ny as usize * nx as usize;
    let lane = row_lane::<NX>();
    let (ys, t) = (lane.ys, lane.f);
    let ept = const { fft_ept(NX) };
    let tpf = nx / ept;
    let y = (ys + ny / 2) % ny;
    let dm = cfg.flags & FUSED_DM != 0;
    let out = new_waves(p, cfg);
    let (a, b) = (2.0 - cfg.alpha, 1.0 - cfg.alpha);
    let k0 = (f0 + fl) * g.nmode;
    let mut r = [0.0f32; 2 * FFT_EPT_MAX as usize];
    unsafe {
        for m in 0..g.nmode {
            let row = (fl * g.nmode + m) as usize * n + (ys * nx) as usize;
            #[unroll]
            for i in 0..ept {
                let v = load_c(p.work, row + (t + i * tpf) as usize);
                r[2 * i as usize] = v.0;
                r[2 * i as usize + 1] = v.1;
            }
            row_fft::<NX, true>(&mut r, lane);
            let v = *p.views.add((k0 + m) as usize);
            #[unroll]
            for i in 0..ept {
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

/// Adds the stack's object-update sums from the exit waves pass C left in `new_waves`:
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
    unsafe { gather_with::<false>(p, tiles, sums, shared, ctl, g, cfg) }
}

/// [`gather`] with the shifted probe: `c_(f, o)` is read from `shift_obj` and the
/// denominator's `sum_p |W_p|^2` from `shift_power`, both as [`shift_rows_inv`] wrote them
/// under `FUSED_SHIFT_OBJ`, in place of the sums over `probe_q` and of `probe_power`.
///
/// # Safety
///
/// As [`gather`], with `shift_obj` and `shift_power` written for this stack in place of the
/// new waves, `probe_q` and `probe_power`.
#[device]
pub unsafe fn gather_shifted(
    p: FusedPtrs,
    tiles: TilePtrs,
    sums: SumPtrs,
    shared: *mut u32,
    ctl: *const StackCtl,
    g: StackGeom,
    cfg: FusedCfg,
) {
    unsafe { gather_with::<true>(p, tiles, sums, shared, ctl, g, cfg) }
}

#[device]
unsafe fn gather_with<const SHIFT: bool>(
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
    let (k0, nk) = (f0 * g.nmode, nf * g.nmode);
    let pixel = |lo: u32, end: u32, y: u32, x: u32| unsafe {
        if x >= g.nxo || y >= g.nyo {
            return;
        }
        let px = (iobj as usize * g.nyo as usize + y as usize) * g.nxo as usize + x as usize;
        let mut acc = load_c(sums.num, px);
        let mut norm = *sums.den.add(px);
        for j in lo..end {
            let k = *tiles.views.add(j as usize);
            let i = k.wrapping_sub(k0);
            if i >= nk {
                break;
            }
            let v = *p.views.add(k as usize);
            if v.obj != iobj || v.probe != 0 {
                continue;
            }
            // Mode `o * nprobe + pm` of the frame is view `k + pm`, plane `i + pm`.
            let plane = i as usize * n;
            obj_pixel_sources(v, y, x, g.ny, g.nx, |pry, prx, w| {
                let at = (pry * g.nx + prx) as usize;
                let ps = psi_index(pry, prx, g.ny, g.nx);
                let (c, power) = if SHIFT {
                    // Plane `i / nprobe` is `fl * nobj + o`, `i / nmode` is `fl`.
                    (
                        load_c(p.shift_obj, (i / g.nprobe) as usize * n + ps),
                        *p.shift_power.add((i / g.nmode) as usize * n + ps),
                    )
                } else {
                    let mut c = (0.0f32, 0.0f32);
                    for pm in 0..g.nprobe as usize {
                        let u = cmul_conj(
                            load_c(p.probe_q, pm * n + at),
                            load_c(waves, plane + pm * n + ps),
                        );
                        c = (c.0 + u.0, c.1 + u.1);
                    }
                    (c, *p.probe_power.add(at))
                };
                acc = (acc.0 + w * c.0, acc.1 + w * c.1);
                norm += w * power;
            });
        }
        store_c(sums.num, px, acc);
        *sums.den.add(px) = norm;
    };
    unsafe { crate::for_each_stack_tile(p.views, tiles, shared, ctl, g, pixel) }
}

// The position model: probe mode `p` of frame `f` is shifted by `(dy, dx) = offsets[f]`
// through its spectrum, `W_p = IDFT(A_p e_f)` (unnormalized inverse) with `A_p =
// DFT(stored(P_p q)) / n` and `e_f[ky, kx] = ramp(dy, ky, ny) ramp(dx, kx, nx)`, so that `W_p =
// stored(P_p q)` at zero shift. Exit waves are `psi_(o, p) = O_o W_p`, the object sampled
// as everywhere else. [`shift_pre`] column-transforms `A_p e_f` to `T`, [`shift_rows_fwd`]
// row-transforms `T` to `W` per view, [`cols`] is unchanged, and [`shift_rows_inv`] takes
// the new waves `x` (`psi_new`, or the gradient waves under `FUSED_GRAD`) to the object
// terms, to `X = DFT_x(Y_p)` with `Y_p = sum_o conj(O_o) x_(o, p)`, and with the slope rows
// `dW / d_ax` to the denominator and position sums; [`shift_post`] column-transforms `X` to
// `H = DFT(Y_p)` and the probe adjoint's `G = conj(e_f) H`.

/// Each frame's column half of the shifted probe: `T[fl * nprobe + p, ys, kx] =
/// IDFT_y(A_p e_f)` over `ky` for `f = f0 + fl` of grid row `blockIdx.y` ([`FUSED_DOWN`]),
/// and under `FUSED_SHIFT_DEN` or `FUSED_SHIFT_POSITION` `T_y` the same of `A_p e_f (-2 pi
/// i freq(ky, ny))` into `shift_adj`. Reads `offsets` and `spectrum`; writes `shift`, and
/// `shift_adj` under those flags.
///
/// One 1D block of `nprobe * FUSED_COLS * tpf` threads mapped as [`cols`] maps modes, with
/// probe modes for modes; grid `(nx / FUSED_COLS, nf_max)`; dynamic shared memory `8 * ny`
/// bytes of `e_y`, then `nprobe * FUSED_COLS` column FFTs ([`FFT_SKEW`]).
///
/// # Safety
///
/// `NY == g.ny`; `offsets`, `spectrum`, `shift` and under those flags `shift_adj` hold their
/// shapes for `g`; `ctl` one entry; launch as above.
#[device]
pub unsafe fn shift_pre<const NY: u32>(
    p: FusedPtrs,
    ctl: *const StackCtl,
    g: StackGeom,
    cfg: FusedCfg,
) {
    let (f0, nf) = unsafe { stack_frames(ctl, g) };
    let Some(fl) = walk_slot(cfg.flags, nf) else {
        return;
    };
    let (ny, nx) = (NY, g.nx);
    let n = ny as usize * nx as usize;
    let ept = const { fft_ept(NY) };
    let tpf = ny / ept;
    let nslot = FUSED_COLS * tpf;
    let tid = thread::threadIdx_x();
    let c = tid % FUSED_COLS;
    let f = (tid / FUSED_COLS) % tpf;
    let pm = tid / nslot;
    let (b, bpb) = (pm * FUSED_COLS + c, g.nprobe * FUSED_COLS);
    let kx = thread::blockIdx_x() * FUSED_COLS + c;
    let plane = (fl * g.nprobe + pm) as usize * n;
    let spectrum = pm as usize * n;
    let mut r = [0.0f32; 2 * FFT_EPT_MAX as usize];
    unsafe {
        let smem = DynamicSharedArray::<u8, 16>::get();
        let ey = smem.cast::<f32>();
        let fft_smem = smem.add(8 * ny as usize);
        let at = 2 * (f0 + fl) as usize;
        let (dy, dx) = (*p.offsets.add(at), *p.offsets.add(at + 1));
        let mut k = tid;
        while k < ny {
            store_c(ey, k as usize, ramp(dy, k, ny));
            k += g.nprobe * nslot;
        }
        thread::sync_threads();
        let ex = ramp(dx, kx, nx);
        if cfg.flags & (FUSED_SHIFT_DEN | FUSED_SHIFT_POSITION) != 0 {
            let tau = 2.0 * core::f32::consts::PI;
            #[unroll]
            for i in 0..ept {
                let ky = f + i * tpf;
                let e = cmul(load_c(ey, ky as usize), ex);
                let a = cmul(load_c(p.spectrum, spectrum + (ky * nx + kx) as usize), e);
                // `a (-i w)` with `w = 2 pi freq`.
                let w = tau * freq(ky, ny);
                r[2 * i as usize] = a.1 * w;
                r[2 * i as usize + 1] = -a.0 * w;
            }
            col_fft::<NY, true>(&mut r, fft_smem, b, f, bpb);
            #[unroll]
            for i in 0..ept {
                let k = 2 * i as usize;
                let at = plane + ((f + i * tpf) * nx + kx) as usize;
                store_c(p.shift_adj, at, (r[k], r[k + 1]));
            }
            thread::sync_threads();
        }
        #[unroll]
        for i in 0..ept {
            let ky = f + i * tpf;
            let e = cmul(load_c(ey, ky as usize), ex);
            let a = cmul(load_c(p.spectrum, spectrum + (ky * nx + kx) as usize), e);
            r[2 * i as usize] = a.0;
            r[2 * i as usize + 1] = a.1;
        }
        col_fft::<NY, true>(&mut r, fft_smem, b, f, bpb);
        #[unroll]
        for i in 0..ept {
            let k = 2 * i as usize;
            store_c(
                p.shift,
                plane + ((f + i * tpf) * nx + kx) as usize,
                (r[k], r[k + 1]),
            );
        }
    }
}

/// [`rows_inv`] with the shifted probe, per frame `fl` of grid row `blockIdx.y`
/// ([`FUSED_DOWN`]) and stored row `ys`, probe mode `p` outermost: `W_p` from the `T` row,
/// then for each object mode `o` the new wave `x` of mode `o * nprobe + p` as [`rows_inv`]
/// stores it (the DM add-back with `O W_p`). Under `FUSED_SHIFT_OBJ` `shift_power[fl] =
/// sum_p |W_p|^2` and `shift_obj[fl * nobj + o] = sum_p conj(W_p) x`, both in ascending
/// `p`. Under `FUSED_SHIFT_PROBE` `shift_adj`'s row gets `X = DFT_x(sum_o conj(O_o) x)`,
/// `o` ascending, in place of the `T_y` row it has read. With the slopes `dW / dx =
/// IDFT_x(T (-2 pi i freq(kx, nx)))` and `dW / dy = IDFT_x(T_y)`, under `FUSED_SHIFT_DEN`
/// `shift_den[fl, blockIdx.x]` gets the block's `sum |O_o|^2 |dW_p / d_ax|^2` over its
/// pixels, `o` and `p`, and under `FUSED_SHIFT_POSITION` `shift_grad[fl, blockIdx.x]` the
/// block's `Re sum conj(x - O_o W_p) O_o dW_p / d_ax`, `x` alone under `FUSED_GRAD`; the
/// difference is taken per pixel as `sum_o conj(O_o) x - sum_o |O_o|^2 W_p`, and each sum
/// is a fixed tree over the block. Reads `work`, `views`, `obj`, `shift`, `shift_adj` under
/// either sum's flag, and `psi` under `FUSED_DM`; writes the new waves and what the flags
/// name. Grid `(ny / rows, nf_max)`, block `FUSED_ROW_THREADS`, dynamic shared memory the
/// larger of `shift_stage_bytes(flags, nx)` bytes then `rows` row FFTs ([`FFT_SKEW`]), and
/// `16 * FUSED_ROW_THREADS` bytes.
///
/// # Safety
///
/// As [`rows_inv`], `shift` holding [`shift_pre`]'s `T` (and `shift_adj` its `T_y` under
/// `FUSED_SHIFT_DEN` or `FUSED_SHIFT_POSITION`) of this stack, and the buffers the flags name
/// holding their shapes.
#[device]
pub unsafe fn shift_rows_inv<const NX: u32>(
    p: FusedPtrs,
    ctl: *const StackCtl,
    g: StackGeom,
    cfg: FusedCfg,
) {
    unsafe { shift_rows_inv_with::<NX, false>(p, ctl, g, cfg) }
}

/// [`shift_rows_inv`]'s `FUSED_SHIFT_OBJ` and `FUSED_SHIFT_PROBE` terms of the waves `x` a
/// stack's `psi` holds, its stored rows as [`rows_inv`] stores them: no transform, no new
/// waves; the other flags are ignored. Reads `psi`, `views`, `shift`, and `obj` under
/// `FUSED_SHIFT_PROBE`; writes what the flags name. Grid `(ny / rows, nf_max)`, block
/// `FUSED_ROW_THREADS`, dynamic shared memory `rows` row FFTs ([`FFT_SKEW`]).
///
/// # Safety
///
/// As [`shift_rows_inv`], with `psi` holding the stack's waves in place of `work`.
#[device]
pub unsafe fn shift_sums<const NX: u32>(
    p: FusedPtrs,
    ctl: *const StackCtl,
    g: StackGeom,
    cfg: FusedCfg,
) {
    unsafe { shift_rows_inv_with::<NX, true>(p, ctl, g, cfg) }
}

/// [`shift_rows_inv`], or under `SUMS` [`shift_sums`].
#[device]
unsafe fn shift_rows_inv_with<const NX: u32, const SUMS: bool>(
    p: FusedPtrs,
    ctl: *const StackCtl,
    g: StackGeom,
    cfg: FusedCfg,
) {
    let (f0, nf) = unsafe { stack_frames(ctl, g) };
    let Some(fl) = walk_slot(cfg.flags, nf) else {
        return;
    };
    let (ny, nx) = (g.ny, NX);
    let n = ny as usize * nx as usize;
    let lane = row_lane::<NX>();
    let (ys, t) = (lane.ys, lane.f);
    let ept = const { fft_ept(NX) };
    let tpf = nx / ept;
    let y = (ys + ny / 2) % ny;
    let nobj = g.nmode / g.nprobe;
    // A constant mask: the sums build has no new waves and no slope rows.
    let flags = if SUMS {
        cfg.flags & (FUSED_SHIFT_OBJ | FUSED_SHIFT_PROBE)
    } else {
        cfg.flags
    };
    let dm = flags & FUSED_DM != 0;
    let obj_terms = flags & FUSED_SHIFT_OBJ != 0;
    let probe = flags & FUSED_SHIFT_PROBE != 0;
    let position = flags & FUSED_SHIFT_POSITION != 0;
    let den = flags & FUSED_SHIFT_DEN != 0;
    let grad = flags & FUSED_GRAD != 0;
    let adjoint = probe || position;
    let slopes = den || position;
    let out = new_waves(p, cfg);
    let (a, b) = (2.0 - cfg.alpha, 1.0 - cfg.alpha);
    let k0 = (f0 + fl) * g.nmode;
    let srow = (ys * nx) as usize;
    let mut w = [0.0f32; 2 * FFT_EPT_MAX as usize];
    let mut r = [0.0f32; 2 * FFT_EPT_MAX as usize];
    let (mut den_y, mut den_x) = (0.0f32, 0.0f32);
    let (mut pos_y, mut pos_x) = (0.0f32, 0.0f32);
    unsafe {
        let smem = DynamicSharedArray::<u8, 16>::get();
        let stage = smem.cast::<f32>();
        let fft_smem = smem.add(shift_stage_bytes(flags, nx) as usize);
        for pm in 0..g.nprobe {
            let trow = (fl * g.nprobe + pm) as usize * n + srow;
            if slopes {
                // `T_y`, copied while `W` and the waves transform: loaded beside the `X` that
                // overwrites it, it cost registers and stalled `X`'s stores.
                #[unroll]
                for i in 0..ept {
                    let e = 2 * (lane.b * nx + t + i * tpf) as usize;
                    async_copy::cp_async_ca_8(
                        stage.add(e).cast(),
                        p.shift_adj.add(2 * (trow + (t + i * tpf) as usize)).cast(),
                    );
                }
                async_copy::cp_async_commit_group();
            }
            #[unroll]
            for i in 0..ept {
                let v = load_c(p.shift, trow + (t + i * tpf) as usize);
                w[2 * i as usize] = v.0;
                w[2 * i as usize + 1] = v.1;
            }
            row_fft_at::<NX, true>(&mut w, fft_smem, lane);
            thread::sync_threads();
            if obj_terms {
                #[unroll]
                for i in 0..ept {
                    let k = 2 * i as usize;
                    let at = fl as usize * n + srow + (t + i * tpf) as usize;
                    let pw = w[k] * w[k] + w[k + 1] * w[k + 1];
                    let pw = if pm == 0 {
                        pw
                    } else {
                        *p.shift_power.add(at) + pw
                    };
                    *p.shift_power.add(at) = pw;
                }
            }
            let mut acc = [0.0f32; 2 * FFT_EPT_MAX as usize];
            for o in 0..nobj {
                let m = o * g.nprobe + pm;
                let row = (fl * g.nmode + m) as usize * n + srow;
                if !SUMS {
                    #[unroll]
                    for i in 0..ept {
                        let v = load_c(p.work, row + (t + i * tpf) as usize);
                        r[2 * i as usize] = v.0;
                        r[2 * i as usize + 1] = v.1;
                    }
                    row_fft_at::<NX, true>(&mut r, fft_smem, lane);
                    thread::sync_threads();
                }
                let v = *p.views.add((k0 + m) as usize);
                let cplane = (fl * nobj + o) as usize * n + srow;
                #[unroll]
                for i in 0..ept {
                    let k = 2 * i as usize;
                    let xs = t + i * tpf;
                    let x = (xs + nx / 2) % nx;
                    let wi = (w[k], w[k + 1]);
                    let at = row + xs as usize;
                    let mut s = if SUMS {
                        load_c(p.psi, at)
                    } else {
                        (r[k], r[k + 1])
                    };
                    let vo = if dm || adjoint {
                        sample_obj(p.obj, v, y, x, g.nyo, g.nxo)
                    } else {
                        (0.0, 0.0)
                    };
                    if dm {
                        let po = cmul(vo, wi);
                        let old = load_c(p.psi, at);
                        let refl = (a * po.0 - b * old.0, a * po.1 - b * old.1);
                        let d = (b * (po.0 - refl.0), b * (po.1 - refl.1));
                        s = (s.0 + d.0, s.1 + d.1);
                    }
                    if !SUMS {
                        store_c(out, at, s);
                    }
                    if obj_terms {
                        let ct = cplane + xs as usize;
                        let u = cmul_conj(wi, s);
                        let u = if pm == 0 {
                            u
                        } else {
                            let c = load_c(p.shift_obj, ct);
                            (c.0 + u.0, c.1 + u.1)
                        };
                        store_c(p.shift_obj, ct, u);
                    }
                    if adjoint {
                        let u = cmul_conj(vo, s);
                        acc[k] += u.0;
                        acc[k + 1] += u.1;
                    }
                }
            }
            let mut vn = [0.0f32; FFT_EPT_MAX as usize];
            if slopes {
                for o in 0..nobj {
                    let v = *p.views.add((k0 + o * g.nprobe + pm) as usize);
                    #[unroll]
                    for i in 0..ept {
                        let x = (t + i * tpf + nx / 2) % nx;
                        let vo = sample_obj(p.obj, v, y, x, g.nyo, g.nxo);
                        vn[i as usize] += vo.0 * vo.0 + vo.1 * vo.1;
                    }
                }
            }
            if position {
                // `W` is dead past here: it becomes the correction `c = sum_o conj(O_o) (x -
                // O_o W) = acc - vn W`, `vn = sum_o |O_o|^2`.
                #[unroll]
                for k in 0..2 * ept as usize {
                    w[k] = if grad {
                        acc[k]
                    } else {
                        acc[k] - vn[k / 2] * w[k]
                    };
                }
            }
            if probe {
                row_fft_at::<NX, false>(&mut acc, fft_smem, lane);
                thread::sync_threads();
            }
            if slopes {
                // `X` takes the pixels this thread copied `T_y` from.
                async_copy::cp_async_wait_all();
            }
            if probe {
                #[unroll]
                for i in 0..ept {
                    let k = 2 * i as usize;
                    store_c(
                        p.shift_adj,
                        trow + (t + i * tpf) as usize,
                        (acc[k], acc[k + 1]),
                    );
                }
            }
            if slopes {
                // This thread's `sum vn |r|^2` and `sum Re conj(c) r` over a slope row `r`.
                let terms = |r: &[f32; 2 * FFT_EPT_MAX as usize]| {
                    let (mut d, mut q) = (0.0f32, 0.0f32);
                    #[unroll]
                    for i in 0..ept {
                        let k = 2 * i as usize;
                        d += vn[i as usize] * (r[k] * r[k] + r[k + 1] * r[k + 1]);
                        q += w[k] * r[k] + w[k + 1] * r[k + 1];
                    }
                    (d, q)
                };
                #[unroll]
                for i in 0..ept {
                    let e = 2 * (lane.b * nx + t + i * tpf) as usize;
                    r[2 * i as usize] = *stage.add(e);
                    r[2 * i as usize + 1] = *stage.add(e + 1);
                }
                row_fft_at::<NX, true>(&mut r, fft_smem, lane);
                thread::sync_threads();
                let (d, q) = terms(&r);
                den_y += d;
                pos_y += q;
                let tau = 2.0 * core::f32::consts::PI;
                #[unroll]
                for i in 0..ept {
                    let kx = t + i * tpf;
                    let v = load_c(p.shift, trow + kx as usize);
                    // `T (-i w)` with `w = 2 pi freq`.
                    let wf = tau * freq(kx, nx);
                    r[2 * i as usize] = v.1 * wf;
                    r[2 * i as usize + 1] = -v.0 * wf;
                }
                row_fft_at::<NX, true>(&mut r, fft_smem, lane);
                thread::sync_threads();
                let (d, q) = terms(&r);
                den_x += d;
                pos_x += q;
            }
        }
        if slopes {
            let slots = DynamicSharedArray::<u8, 16>::get().cast::<f32>();
            let nt = FUSED_ROW_THREADS;
            // Slot of pixel `(b, f)`: the tree's order is the block's pixels', not its lanes'.
            let at = (lane.b * tpf + t) as usize;
            *slots.add(at) = den_y;
            *slots.add(nt as usize + at) = den_x;
            *slots.add(2 * nt as usize + at) = pos_y;
            *slots.add(3 * nt as usize + at) = pos_x;
            let tid = thread::threadIdx_x();
            let mut half = nt / 2;
            while half > 0 {
                thread::sync_threads();
                if tid < half {
                    #[unroll]
                    for j in 0..4 {
                        let s = slots.add((j * nt + tid) as usize);
                        *s += *s.add(half as usize);
                    }
                }
                half /= 2;
            }
            thread::sync_threads();
            if tid == 0 {
                let at = (fl * thread::gridDim_x() + thread::blockIdx_x()) as usize;
                let sum = |j: u32| PositionPartial {
                    dy: *slots.add((j * nt) as usize),
                    dx: *slots.add(((j + 1) * nt) as usize),
                };
                if den {
                    *p.shift_den.add(at) = sum(0);
                }
                if position {
                    *p.shift_grad.add(at) = sum(2);
                }
            }
        }
    }
}

/// The column FFT of [`shift_rows_inv`]'s `X`: `H = DFT_y(X)` over `ys` for frame `fl`
/// of grid row `blockIdx.y` ([`FUSED_DOWN`]), probe mode `p`, and the plane becomes the
/// probe adjoint's `G = conj(e_f) H`. Reads `offsets` and `shift_adj`; writes `shift_adj`.
///
/// Block and grid as [`shift_pre`]; dynamic shared memory `8 * ny` bytes of `e_y`, then
/// `nprobe * FUSED_COLS` column FFTs ([`FFT_SKEW`]).
///
/// # Safety
///
/// As [`shift_pre`], `shift_adj` holding [`shift_rows_inv`]'s `X` of this stack.
#[device]
pub unsafe fn shift_post<const NY: u32>(
    p: FusedPtrs,
    ctl: *const StackCtl,
    g: StackGeom,
    cfg: FusedCfg,
) {
    let (f0, nf) = unsafe { stack_frames(ctl, g) };
    let Some(fl) = walk_slot(cfg.flags, nf) else {
        return;
    };
    let (ny, nx) = (NY, g.nx);
    let n = ny as usize * nx as usize;
    let ept = const { fft_ept(NY) };
    let tpf = ny / ept;
    let nslot = FUSED_COLS * tpf;
    let tid = thread::threadIdx_x();
    let c = tid % FUSED_COLS;
    let f = (tid / FUSED_COLS) % tpf;
    let pm = tid / nslot;
    let (b, bpb) = (pm * FUSED_COLS + c, g.nprobe * FUSED_COLS);
    let kx = thread::blockIdx_x() * FUSED_COLS + c;
    let plane = (fl * g.nprobe + pm) as usize * n;
    let mut r = [0.0f32; 2 * FFT_EPT_MAX as usize];
    unsafe {
        let smem = DynamicSharedArray::<u8, 16>::get();
        let ey = smem.cast::<f32>();
        let fft_smem = smem.add(8 * ny as usize);
        let at = 2 * (f0 + fl) as usize;
        let (dy, dx) = (*p.offsets.add(at), *p.offsets.add(at + 1));
        let mut k = tid;
        while k < ny {
            store_c(ey, k as usize, ramp(dy, k, ny));
            k += g.nprobe * nslot;
        }
        thread::sync_threads();
        #[unroll]
        for i in 0..ept {
            let v = load_c(p.shift_adj, plane + ((f + i * tpf) * nx + kx) as usize);
            r[2 * i as usize] = v.0;
            r[2 * i as usize + 1] = v.1;
        }
        col_fft::<NY, false>(&mut r, fft_smem, b, f, bpb);
        let ex = ramp(dx, kx, nx);
        #[unroll]
        for i in 0..ept {
            let k = 2 * i as usize;
            let ky = f + i * tpf;
            let e = cmul(load_c(ey, ky as usize), ex);
            store_c(
                p.shift_adj,
                plane + (ky * nx + kx) as usize,
                cmul_conj(e, (r[k], r[k + 1])),
            );
        }
    }
}

/// Adds the stack's shifted probe sums: `sums.num[p, k] += G[fl * nprobe + p, k]` (FFT order)
/// and `sums.den[p, i] += |O_o(i)|^2` for natural pixel `i`, over the stack's frames `fl` in
/// ascending order and, for the denominator, object modes `o` ascending within a frame, so
/// that stacks taken in scan order give every pixel one sum whatever the split. Reads
/// `shift_adj`, `views` and `obj`; writes `sums`. One thread per pixel of `[nprobe, ny,
/// nx]`.
///
/// # Safety
///
/// `shift_adj` holds [`shift_post`]'s `G` of this stack; `sums` `[nprobe, ny, nx]`; `views` and
/// `obj` their shapes for `g`; `ctl` one entry.
#[device]
pub unsafe fn shift_probe_sums(p: FusedPtrs, sums: SumPtrs, ctl: *const StackCtl, g: StackGeom) {
    let (f0, nf) = unsafe { stack_frames(ctl, g) };
    let n = g.ny as usize * g.nx as usize;
    let t = crate::global_index();
    if t >= g.nprobe as usize * n || nf == 0 {
        return;
    }
    let (pm, i) = ((t / n) as u32, t % n);
    let (y, x) = ((i / g.nx as usize) as u32, (i % g.nx as usize) as u32);
    let nprobe = g.nprobe as usize;
    unsafe {
        let mut a = load_c(sums.num, t);
        let mut d = *sums.den.add(t);
        for fl in 0..nf {
            let u = load_c(p.shift_adj, (fl as usize * nprobe + pm as usize) * n + i);
            a = (a.0 + u.0, a.1 + u.1);
            for o in 0..g.nmode / g.nprobe {
                let v = *p
                    .views
                    .add(((f0 + fl) * g.nmode + o * g.nprobe + pm) as usize);
                let vo = sample_obj(p.obj, v, y, x, g.nyo, g.nxo);
                d += vo.0 * vo.0 + vo.1 * vo.1;
            }
        }
        store_c(sums.num, t, a);
        *sums.den.add(t) = d;
    }
}

/// The row half of the probe spectra: `spectrum[p, ys, :] = DFT_x(stored(P_p q)[ys, :])` for
/// `p = blockIdx.y`. Grid `(ny / rows, nprobe)`, block and shared memory as [`rows_fwd`].
///
/// # Safety
///
/// `NX` is the frame's `nx`; `probe` and `spectrum` hold `[nprobe, ny, nx]`; launch as above.
#[device]
pub unsafe fn spectrum_rows<const NX: u32>(
    probe: *const f32,
    spectrum: *mut f32,
    ny: u32,
    quad: f32,
) {
    let nx = NX;
    let n = ny as usize * nx as usize;
    let lane = row_lane::<NX>();
    let ys = lane.ys;
    let plane = thread::blockIdx_y() as usize * n;
    let t = lane.f;
    let ept = const { fft_ept(NX) };
    let tpf = nx / ept;
    let y = (ys + ny / 2) % ny;
    let mut r = [0.0f32; 2 * FFT_EPT_MAX as usize];
    unsafe {
        #[unroll]
        for i in 0..ept {
            let x = (t + i * tpf + nx / 2) % nx;
            let v = cmul(
                load_c(probe, plane + (y * nx + x) as usize),
                quad_phase(quad, y, x, ny, nx),
            );
            r[2 * i as usize] = v.0;
            r[2 * i as usize + 1] = v.1;
        }
        row_fft::<NX, false>(&mut r, lane);
        #[unroll]
        for i in 0..ept {
            let k = 2 * i as usize;
            let at = plane + (ys * nx + t + i * tpf) as usize;
            store_c(spectrum, at, (r[k], r[k + 1]));
        }
    }
}

/// In-place column FFT of every plane `blockIdx.y` of `[nplane, NY, nx]`, inverse when
/// `inverse != 0`, times `scale`. One 1D block of `FUSED_COLS * NY / fft_ept(NY)` threads over
/// columns `blockIdx.x * FUSED_COLS ..` (thread map of [`cols`] for one mode); grid `(nx /
/// FUSED_COLS, nplane)`; dynamic shared memory `FUSED_COLS` column FFTs
/// ([`FFT_SKEW`]).
///
/// # Safety
///
/// `buf` holds `[nplane, NY, nx]`; launch as above.
#[device]
pub unsafe fn shift_cols<const NY: u32>(buf: *mut f32, nx: u32, inverse: u32, scale: f32) {
    let ny = NY;
    let ept = const { fft_ept(NY) };
    let tpf = ny / ept;
    let tid = thread::threadIdx_x();
    let (c, f) = (tid % FUSED_COLS, tid / FUSED_COLS);
    let u = thread::blockIdx_x() * FUSED_COLS + c;
    let plane = thread::blockIdx_y() as usize * ny as usize * nx as usize;
    let mut r = [0.0f32; 2 * FFT_EPT_MAX as usize];
    unsafe {
        let smem = DynamicSharedArray::<u8, 16>::get();
        #[unroll]
        for i in 0..ept {
            let v = load_c(buf, plane + ((f + i * tpf) * nx + u) as usize);
            r[2 * i as usize] = v.0;
            r[2 * i as usize + 1] = v.1;
        }
        if inverse != 0 {
            col_fft::<NY, true>(&mut r, smem, c, f, FUSED_COLS);
        } else {
            col_fft::<NY, false>(&mut r, smem, c, f, FUSED_COLS);
        }
        #[unroll]
        for i in 0..ept {
            let k = 2 * i as usize;
            let at = plane + ((f + i * tpf) * nx + u) as usize;
            store_c(buf, at, (r[k] * scale, r[k + 1] * scale));
        }
    }
}

/// The probe update from shifted sums whose `num` [`shift_cols`] has inverse-transformed
/// along `y`: with `v = IDFT_x` of row `ys` of plane `p = blockIdx.y`, `out.num[p, y, x] +=
/// conj(q) v / n` and `out.den[p, y, x] += sums.den[p, y, x]` at the natural pixel of stored
/// `(ys, xs)`; then `sums` is zeroed there, consumed. Grid `(ny / rows, nprobe)`, block and
/// shared memory as [`rows_fwd`].
///
/// # Safety
///
/// `NX` is the frame's `nx`; `sums` and `out` hold `[nprobe, ny, nx]`; launch as above.
#[device]
pub unsafe fn shift_probe_rows<const NX: u32>(sums: SumPtrs, out: SumPtrs, ny: u32, quad: f32) {
    let nx = NX;
    let n = ny as usize * nx as usize;
    let lane = row_lane::<NX>();
    let ys = lane.ys;
    let plane = thread::blockIdx_y() as usize * n;
    let t = lane.f;
    let ept = const { fft_ept(NX) };
    let tpf = nx / ept;
    let y = (ys + ny / 2) % ny;
    let inv_n = 1.0 / n as f32;
    let mut r = [0.0f32; 2 * FFT_EPT_MAX as usize];
    unsafe {
        #[unroll]
        for i in 0..ept {
            let v = load_c(sums.num, plane + (ys * nx + t + i * tpf) as usize);
            r[2 * i as usize] = v.0;
            r[2 * i as usize + 1] = v.1;
        }
        row_fft::<NX, true>(&mut r, lane);
        #[unroll]
        for i in 0..ept {
            let k = 2 * i as usize;
            let xs = t + i * tpf;
            let x = (xs + nx / 2) % nx;
            let nat = plane + (y * nx + x) as usize;
            let u = cmul_conj(quad_phase(quad, y, x, ny, nx), (r[k], r[k + 1]));
            let a = load_c(out.num, nat);
            store_c(out.num, nat, (a.0 + u.0 * inv_n, a.1 + u.1 * inv_n));
            *out.den.add(nat) += *sums.den.add(nat);
            store_c(sums.num, plane + (ys * nx + xs) as usize, (0.0, 0.0));
            *sums.den.add(nat) = 0.0;
        }
    }
}
