//! M2 fused stack passes: a stack's exit waves, Fourier projection, LLK and object sums in
//! three in-kernel FFT passes over [`FusedBufs`], in place of the M1 sequence
//! `obj_probe_to_psi`, 2D FFT, `apply_amplitude` (and `llk`), inverse FFT, (`dm_reflect`,
//! `add_c32`), `psi_to_obj`. The conventions are those of the M1 operators (module `ops`);
//! the arithmetic per pixel is theirs, the FFTs and the sum order of the object update are
//! not, so results agree with M1 to rounding, not bit for bit.
//!
//! Frame sides are any of [`FUSED_SIDES`]; views must be canonical, which [`Views::new`]
//! checks.
//! The probe sums are M1's [`Kernels::psi_to_probe`], run by [`Kernels::fused_project`] on
//! the waves it leaves before the object gather can push them out of L2.
//!
//! With a [`Model::shift`] each frame's probe modes are moved by the frame's offsets through
//! the probe spectra of [`Kernels::probe_spectrum`]: a column pass of the spectra per frame
//! runs before pass A, which then row-transforms them, pass C takes the position sums, and a
//! column pass after it takes the probe sums' spectral terms, which [`ShiftedProbeSums`]
//! carries across stacks until [`Kernels::shifted_probe_sums`] turns them into probe sums.

use std::cell::Cell;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig, RawDeviceBuffer};
use ptyrs_kernels::fft_ept;
use ptyrs_kernels::fused::{
    FFT_SKEW, FUSED_BACKGROUND, FUSED_BACKGROUND_SUMS, FUSED_COLS, FUSED_DM, FUSED_DOWN,
    FUSED_GRAD, FUSED_LLK, FUSED_MODEL_TERMS, FUSED_PROJECT, FUSED_ROW_THREADS, FUSED_SCALE,
    FUSED_SCALE_SUMS, FUSED_SHIFT_DEN, FUSED_SHIFT_OBJ, FUSED_SHIFT_POSITION, FUSED_SHIFT_PROBE,
    FUSED_TO_PSI, FusedCfg, FusedPtrs, col_slot_bytes, fused_col_threads, shift_stage_bytes,
};
use ptyrs_kernels::pixel::SumPtrs;

use crate::gpu::ops::grid_1d;
use crate::gpu::{
    C32, Error, Kernels, LlkPartial, PositionPartial, ScalePartial, StackCtl, StackGeom, StackRef,
    UpdateSums, ViewDims, Views, block_fft, check_len, dptr,
};

/// Dynamic shared memory a block gets without opting in.
pub(crate) const SMEM_MAX: u32 = 48 * 1024;
/// Most blocks along grid y.
pub(crate) const GRID_Y_MAX: u64 = 65535;

/// Declares [`Side`] and [`Side::ALL`] from one list, so that they cannot disagree.
macro_rules! sides {
    ($($v:ident = $n:literal),+) => {
        /// A frame side the fused FFT passes have kernels for; the discriminant is the side.
        #[derive(Clone, Copy)]
        #[repr(u32)]
        pub(crate) enum Side {
            $($v = $n),+
        }

        impl Side {
            pub(crate) const ALL: &[Side] = &[$(Side::$v),+];
        }
    };
}
sides!(N64 = 64, N128 = 128, N256 = 256, N512 = 512);

/// Frame sides the fused passes take along either axis, ascending.
pub const FUSED_SIDES: &[u32] = &{
    let mut out = [0; Side::ALL.len()];
    let mut i = 0;
    while i < out.len() {
        out[i] = Side::ALL[i] as u32;
        i += 1;
    }
    out
};

/// Launch shapes of the fused passes for one frame and mode count.
#[derive(Clone, Copy)]
pub(crate) struct Shape {
    /// Sides along y (the column pass) and x (the row passes).
    pub(crate) ny: Side,
    pub(crate) nx: Side,
    /// Rows per row-pass block, of `nx / fft_ept(nx)` threads each.
    pub(crate) rows: u32,
    pub(crate) row_smem: u32,
    pub(crate) col_threads: u32,
    /// Column-pass shared memory past the reduction slots: counts tile and FFTs.
    pub(crate) col_tile_fft: u32,
    /// Shared memory of one column FFT, its skew included.
    pub(crate) col_fft: u32,
}

impl Shape {
    pub(crate) fn of(d: &ViewDims) -> Result<Self, Error> {
        d.check()?;
        let [ny, nx] = d.frame;
        let bad = |what: String| Error::Shape(format!("fused passes, {d:?}: {what}"));
        // The side's kernels and its shared memory per FFT, the larger of the two directions.
        let side = |n: u32| {
            let Some(side) = Side::ALL.iter().copied().find(|&s| s as u32 == n) else {
                return Err(bad(format!(
                    "no fused kernels for side {n}; fused sides are {FUSED_SIDES:?}"
                )));
            };
            match (block_fft(n, false), block_fft(n, true)) {
                (Some(f), Some(i))
                    if [f.ept, i.ept] == [fft_ept(n); 2]
                        && [f.threads_per_fft, i.threads_per_fft] == [n / fft_ept(n); 2] =>
                {
                    Ok((side, f.smem_per_fft.max(i.smem_per_fft)))
                }
                _ => Err(bad(format!("no {n}-point in-kernel FFT"))),
            }
        };
        let ((side_y, sy), (side_x, sx)) = (side(ny)?, side(nx)?);
        let rows = FUSED_ROW_THREADS / (nx / fft_ept(nx));
        let nslot = FUSED_COLS * (ny / fft_ept(ny));
        let col_threads = d.nmode as u64 * nslot as u64;
        if col_threads > fused_col_threads(ny) as u64 {
            return Err(bad(format!("{col_threads} threads per column block")));
        }
        let col_threads = col_threads as u32;
        let col_fft = sy + FFT_SKEW;
        let s = Self {
            ny: side_y,
            nx: side_x,
            rows,
            row_smem: rows * (sx + FFT_SKEW),
            col_threads,
            col_tile_fft: 2 * FUSED_COLS * ny
                + (col_threads / (ny / fft_ept(ny))) * col_fft.max(4 * ny),
            col_fft,
        };
        // The passes' most shared memory is under every flag that adds to it.
        if s.col_smem(FUSED_SCALE_SUMS) > SMEM_MAX
            || s.shift_row_smem(FUSED_SHIFT_DEN | FUSED_SHIFT_POSITION) > SMEM_MAX
        {
            return Err(bad("shared memory per block exceeds 48 KiB".into()));
        }
        Ok(s)
    }

    fn col_smem(&self, flags: u32) -> u32 {
        col_slot_bytes(flags, self.ny as u32) + self.col_tile_fft
    }

    fn shift_row_smem(&self, flags: u32) -> u32 {
        (shift_stage_bytes(flags, self.nx as u32) + self.row_smem).max(16 * FUSED_ROW_THREADS)
    }

    /// `rows_fwd` (one block row per view) or `rows_inv` (per frame).
    pub(crate) fn rows(&self, g: &StackGeom, per_view: bool) -> Result<LaunchConfig, Error> {
        let planes = g.nf_max as u64 * if per_view { g.nmode as u64 } else { 1 };
        if planes > GRID_Y_MAX {
            return Err(Error::Shape(format!("{planes} planes exceed grid y")));
        }
        Ok(LaunchConfig {
            grid_dim: (g.ny / self.rows, planes as u32, 1),
            block_dim: (FUSED_ROW_THREADS, 1, 1),
            shared_mem_bytes: self.row_smem,
        })
    }

    pub(crate) fn cols(&self, g: &StackGeom, flags: u32) -> Result<LaunchConfig, Error> {
        if g.nf_max as u64 > GRID_Y_MAX {
            return Err(Error::Shape(format!("{} frames exceed grid y", g.nf_max)));
        }
        Ok(LaunchConfig {
            grid_dim: (g.nx / FUSED_COLS, g.nf_max, 1),
            block_dim: (self.col_threads, 1, 1),
            shared_mem_bytes: self.col_smem(flags),
        })
    }

    /// `shift_pre` or `shift_post` for `nprobe` probe modes.
    fn shift_cols(&self, nprobe: u32, nf_max: u32) -> Result<LaunchConfig, Error> {
        let (ny, nx) = (self.ny as u32, self.nx as u32);
        let threads = nprobe * FUSED_COLS * (ny / fft_ept(ny));
        let smem = 8 * ny + nprobe * FUSED_COLS * self.col_fft;
        if smem > SMEM_MAX {
            return Err(Error::Shape(format!(
                "shift column pass: {smem} bytes of shared memory exceed 48 KiB"
            )));
        }
        if nf_max as u64 > GRID_Y_MAX {
            return Err(Error::Shape(format!("{nf_max} frames exceed grid y")));
        }
        Ok(LaunchConfig {
            grid_dim: (nx / FUSED_COLS, nf_max, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: smem,
        })
    }

    /// A row pass over `nplane` planes `[ny, nx]`: `spectrum_rows`, `shift_probe_rows`.
    fn plane_rows(&self, nplane: u32) -> LaunchConfig {
        let ny = self.ny as u32;
        LaunchConfig {
            grid_dim: (ny / self.rows, nplane, 1),
            block_dim: (FUSED_ROW_THREADS, 1, 1),
            shared_mem_bytes: self.row_smem,
        }
    }

    /// `shift_cols` over `nplane` planes `[ny, nx]`.
    fn plane_cols(&self, nplane: u32) -> LaunchConfig {
        let (ny, nx) = (self.ny as u32, self.nx as u32);
        LaunchConfig {
            grid_dim: (nx / FUSED_COLS, nplane, 1),
            block_dim: (FUSED_COLS * (ny / fft_ept(ny)), 1, 1),
            shared_mem_bytes: FUSED_COLS * self.col_fft,
        }
    }
}

/// Per-stack scratch of the fused passes for stacks of up to `nf_max` frames of one scan
/// geometry; one set serves every stack in turn.
pub struct FusedBufs {
    /// `[nf_max * nmode, ny, nx]`.
    work: DeviceBuffer<C32>,
    /// `[nprobe, ny, nx]`.
    probe_q: DeviceBuffer<C32>,
    /// `[ny, nx]`.
    probe_power: DeviceBuffer<f32>,
    /// `[nf_max, nx / FUSED_COLS]`.
    groups: DeviceBuffer<LlkPartial>,
    /// `[nf_max, nx / FUSED_COLS]`.
    scale_groups: DeviceBuffer<ScalePartial>,
    /// The scratch of a [`Model::shift`], from [`Self::with_shift`].
    shift: Option<ShiftScratch>,
}

/// [`FusedBufs`]' scratch of the position model.
struct ShiftScratch {
    /// `T` of the shift passes, `[nf_max * nprobe, ny, nx]`.
    t: DeviceBuffer<C32>,
    /// `T_y`, then the probe adjoint's `X` and `G`, as `t`: the kernels' `shift_adj`.
    adj: DeviceBuffer<C32>,
    /// `[nf_max * nobj, ny, nx]`.
    obj: DeviceBuffer<C32>,
    /// `[nf_max, ny, nx]`.
    power: DeviceBuffer<f32>,
    /// `[nf_max, ny / rows]`.
    grad_groups: DeviceBuffer<PositionPartial>,
    /// `[nf_max, ny / rows]`.
    den_groups: DeviceBuffer<PositionPartial>,
}

impl ShiftScratch {
    fn check(&self, d: &ViewDims, g: &StackGeom, sh: &Shape) -> Result<(), Error> {
        let (n, nf) = (d.n(), g.nf_max as usize);
        check_len("shift", &self.t, nf * d.nprobe as usize * n)?;
        check_len("shift_adj", &self.adj, nf * d.nprobe as usize * n)?;
        check_len("shift_obj", &self.obj, nf * d.nobj as usize * n)?;
        check_len("shift_power", &self.power, nf * n)?;
        let ngroup = nf * (g.ny / sh.rows) as usize;
        check_len("shift_grad", &self.grad_groups, ngroup)?;
        check_len("shift_den", &self.den_groups, ngroup)
    }
}

impl FusedBufs {
    /// Scratch for `dims`, whose frame sides must be in [`FUSED_SIDES`] and whose modes must
    /// fit one column block of [`fused_col_threads`] threads: at most `1024 / ny` of them.
    pub fn new(stream: &CudaStream, dims: &ViewDims, nf_max: u32) -> Result<Self, Error> {
        Shape::of(dims)?;
        if nf_max == 0 {
            return Err(Error::Shape("nf_max is zero".into()));
        }
        let (n, nf) = (dims.n(), nf_max as usize);
        let ngroup = nf * (dims.frame[1] / FUSED_COLS) as usize;
        Ok(Self {
            work: DeviceBuffer::zeroed(stream, nf * dims.nmode as usize * n)?,
            probe_q: DeviceBuffer::zeroed(stream, dims.nprobe as usize * n)?,
            probe_power: DeviceBuffer::zeroed(stream, n)?,
            groups: DeviceBuffer::zeroed(stream, ngroup)?,
            scale_groups: DeviceBuffer::zeroed(stream, ngroup)?,
            shift: None,
        })
    }

    /// [`Self::new`] with the scratch of a [`Model::shift`]: `(2 nprobe + nobj) nf_max`
    /// complex frames more, and `nf_max` real ones.
    pub fn with_shift(stream: &CudaStream, dims: &ViewDims, nf_max: u32) -> Result<Self, Error> {
        let mut bufs = Self::new(stream, dims, nf_max)?;
        let sh = Shape::of(dims)?;
        sh.shift_cols(dims.nprobe, nf_max)?;
        let (n, nf) = (dims.n(), nf_max as usize);
        let ngroup = nf * (dims.frame[0] / sh.rows) as usize;
        bufs.shift = Some(ShiftScratch {
            t: DeviceBuffer::zeroed(stream, nf * dims.nprobe as usize * n)?,
            adj: DeviceBuffer::zeroed(stream, nf * dims.nprobe as usize * n)?,
            obj: DeviceBuffer::zeroed(stream, nf * dims.nobj as usize * n)?,
            power: DeviceBuffer::zeroed(stream, nf * n)?,
            grad_groups: DeviceBuffer::zeroed(stream, ngroup)?,
            den_groups: DeviceBuffer::zeroed(stream, ngroup)?,
        });
        Ok(bufs)
    }

    /// The stack's transforms. After an AP [`Kernels::fused_project`], its projected exit
    /// waves in the layout of M1 `psi` (origin at `[0, 0]`, plane `kl` view `f0 * nmode +
    /// kl`), what [`Kernels::psi_to_probe`] takes.
    pub fn work(&self) -> &DeviceBuffer<C32> {
        &self.work
    }

    fn check(&self, d: &ViewDims, g: &StackGeom) -> Result<(), Error> {
        let (n, nf) = (d.n(), g.nf_max as usize);
        check_len("work", &self.work, nf * d.nmode as usize * n)?;
        check_len("probe_q", &self.probe_q, d.nprobe as usize * n)?;
        check_len("probe_power", &self.probe_power, n)?;
        let ngroup = nf * (g.nx / FUSED_COLS) as usize;
        check_len("groups", &self.groups, ngroup)?;
        check_len("scale_groups", &self.scale_groups, ngroup)
    }
}

/// What the fused passes read of the scan: its views, the whole scan's counts `[nframe,
/// ny, nx]`, the object `[nobj, nyo, nxo]`, the probe `[nprobe, ny, nx]` and the model of
/// the counts.
#[derive(Clone, Copy)]
pub struct FusedInputs<'a> {
    pub views: &'a Views,
    pub counts: &'a DeviceBuffer<u16>,
    pub obj: &'a DeviceBuffer<C32>,
    pub probe: &'a DeviceBuffer<C32>,
    pub model: Model<'a>,
}

/// The model intensity `calc` the counts are compared with, from the far field `Psi` of a
/// frame's exit waves: `Default` is `I = sum_m |Psi_m|^2 / n` and leaves every pass's
/// arithmetic as it is; each field set adds its term. Every pass that compares with the
/// counts uses `calc`: the LLK, the projection and the gradient.
#[derive(Clone, Copy, Default)]
pub struct Model<'a> {
    /// Per-frame intensity scale `s`, `[nframe]`: `calc = s[f] I` for frame `f`.
    pub scale: Option<&'a DeviceBuffer<f32>>,
    /// Incoherent background `b`, `[ny, nx]` in the counts' (detector) order, the same for
    /// every frame: `calc` plus `b[j]` at detector pixel `j`.
    pub background: Option<&'a DeviceBuffer<f32>>,
    /// The position model: each frame's probe modes moved by the frame's offsets.
    pub shift: Option<Shift<'a>>,
}

/// The position model. Probe mode `p` of frame `f` is `W_p = IDFT(A_p e_f)` (unnormalized
/// inverse) with `A_p = DFT(P_p q) / n`, `P_p q` in the exit waves' layout (origin at `[0,
/// 0]`), and `e_f[ky, kx] = exp(-2 pi i (fy dy + fx dx))`, `fy` and `fx` the FFT frequencies
/// in cycles per pixel and `(dy, dx) = offsets[f]` in pixels: the probe moved by `(dy, dx)`,
/// circularly. A view's exit wave is `O W_p`, the object sampled as without the model, so at
/// zero offsets `W_p` is `P_p q` up to rounding.
#[derive(Clone, Copy)]
pub struct Shift<'a> {
    /// `(dy, dx)` per frame, `[nframe, 2]`.
    pub offsets: &'a DeviceBuffer<f32>,
    /// The spectra of [`FusedInputs::probe`].
    pub spectrum: ProbeSpectrum<'a>,
}

/// Storage of a probe's spectra, `[nprobe, ny, nx]`, for [`Kernels::probe_spectrum`].
pub struct SpectrumBuf(DeviceBuffer<C32>);

impl SpectrumBuf {
    pub fn zeroed(stream: &CudaStream, dims: &ViewDims) -> Result<Self, Error> {
        Ok(Self(DeviceBuffer::zeroed(
            stream,
            dims.nprobe as usize * dims.n(),
        )?))
    }
}

/// A probe's spectra `A_p` of [`Shift`], borrowed with the probe and frame geometry they were
/// computed for: the probe cannot change while they are in use, and the fused passes refuse
/// them with any other probe.
#[derive(Clone, Copy)]
pub struct ProbeSpectrum<'a> {
    a: &'a DeviceBuffer<C32>,
    probe: &'a DeviceBuffer<C32>,
    /// `[ny, nx, nprobe, quad_phase bits]`.
    key: [u32; 4],
}

impl<'a> ProbeSpectrum<'a> {
    /// The spectra `A_p`, `[nprobe, ny, nx]`, FFT order.
    pub(crate) fn spectra(&self) -> &'a DeviceBuffer<C32> {
        self.a
    }

    fn key(d: &ViewDims) -> [u32; 4] {
        [d.frame[0], d.frame[1], d.nprobe, d.quad_phase.to_bits()]
    }

    fn check(&self, d: &ViewDims, probe: &DeviceBuffer<C32>) -> Result<(), Error> {
        if self.probe.cu_deviceptr() != probe.cu_deviceptr() || self.key != Self::key(d) {
            return Err(Error::Shape(
                "the probe spectrum is not of this probe and frame geometry".into(),
            ));
        }
        Ok(())
    }
}

/// The probe sums of a shifted probe over a scan's frames, kept in the spectral domain so
/// that the split into stacks does not change a bit: [`Kernels::fused_project`] adds a
/// stack's frames in ascending order and [`Kernels::shifted_probe_sums`] turns what was added
/// into probe sums and zeroes them; the only other reset is construction.
pub struct ShiftedProbeSums {
    /// `sum_f conj(e_f) DFT(Y_f)`, `[nprobe, ny, nx]`, FFT order.
    num: DeviceBuffer<C32>,
    /// `sum |O|^2`, `[nprobe, ny, nx]`.
    den: DeviceBuffer<f32>,
}

impl ShiftedProbeSums {
    pub fn zeroed(stream: &CudaStream, dims: &ViewDims) -> Result<Self, Error> {
        let len = dims.nprobe as usize * dims.n();
        Ok(Self {
            num: DeviceBuffer::zeroed(stream, len)?,
            den: DeviceBuffer::zeroed(stream, len)?,
        })
    }

    fn check(&self, d: &ViewDims) -> Result<(), Error> {
        let len = d.nprobe as usize * d.n();
        check_len("shifted probe sums num", &self.num, len)?;
        check_len("shifted probe sums den", &self.den, len)
    }

    fn ptrs(&self) -> SumPtrs {
        SumPtrs {
            num: dptr(&self.num),
            den: dptr(&self.den),
        }
    }

    fn storages(&self) -> [&cuda_core::RawDeviceBuffer; 2] {
        [self.num.storage(), self.den.storage()]
    }
}

/// The sums of a [`Model::shift`] that [`Kernels::fused_project`] writes, with `x` the new
/// exit waves of every view (under [`Exit::Grad`] the gradient waves) and `W` the shifted
/// probe; `Default` writes none. Each needs the shift model.
#[derive(Default)]
pub struct ShiftSums<'a> {
    /// The probe sums, in place of [`FusedSums::probe`].
    pub probe: Option<&'a mut ShiftedProbeSums>,
    /// The position sums of the exit-wave correction `c`: `position[f0 + fl] = Re sum
    /// conj(c) O dW / d(dy, dx)` over the frame's views and pixels, for `fl < nf`;
    /// `[nframe]`, no other entry written. `c = x - O W`, the step from the waves the pass
    /// started from, taken per pixel; under [`Exit::Grad`] `c = x`, and the sum is the LLK's
    /// derivative along the frame's offsets.
    pub position: Option<&'a mut DeviceBuffer<PositionPartial>>,
    /// `position_den[f0 + fl] = sum |O|^2 |dW / d(dy, dx)|^2` over the frame's views and
    /// pixels, as `position`.
    pub position_den: Option<&'a mut DeviceBuffer<PositionPartial>>,
}

/// Sums of the forward far field's `calc` against the counts that
/// [`Kernels::fused_project`] and [`Kernels::fused_llk`] write; `Default` writes none.
#[derive(Default)]
pub struct FarSums<'a> {
    /// The LLK partials `llk[fl]` for `fl < nf`, `[nf_max]`; entries past `nf` not written.
    pub llk: Option<&'a mut DeviceBuffer<LlkPartial>>,
    /// The scale sums `scale[f0 + fl]` for `fl < nf`, `[nframe]`; no other entry written.
    pub scale: Option<&'a mut DeviceBuffer<ScalePartial>>,
    /// The background sums, the stack's frames added.
    pub background: Option<&'a mut BackgroundSums>,
}

impl FarSums<'_> {
    /// The column-pass flags that write these sums.
    fn flags(&self) -> u32 {
        let flag = |on: bool, flag: u32| if on { flag } else { 0 };
        flag(self.llk.is_some(), FUSED_LLK)
            | flag(self.scale.is_some(), FUSED_SCALE_SUMS)
            | flag(self.background.is_some(), FUSED_BACKGROUND_SUMS)
    }
}

/// The background's sums over a scan's frames: `sum[j]` is `obs / max(calc, 1e-12)` at
/// detector pixel `j` summed over the frames where `j` is unmasked, which
/// [`Kernels::fused_project`] and [`Kernels::fused_llk`] add a stack at a time in ascending
/// frame order, so that the split into stacks does not change a bit. They hold what was
/// added since the last [`Kernels::background_ap`] or [`Kernels::background_grad`] consumed
/// them; the only other reset is construction.
pub struct BackgroundSums {
    /// `[ny, nx]`, detector order.
    pub sum: DeviceBuffer<f32>,
    /// Each frame's ratios, `[nf_max, ny, nx]`, the column pass writes for the sum.
    ratio: DeviceBuffer<f32>,
}

impl BackgroundSums {
    /// Zeroed sums for frames of `dims` in stacks of up to `nf_max` frames.
    pub fn zeroed(stream: &CudaStream, dims: &ViewDims, nf_max: u32) -> Result<Self, Error> {
        let n = dims.n();
        Ok(Self {
            sum: DeviceBuffer::zeroed(stream, n)?,
            ratio: DeviceBuffer::zeroed(stream, nf_max as usize * n)?,
        })
    }

    fn check(&self, n: usize, nf_max: u32) -> Result<(), Error> {
        check_len("background sums", &self.sum, n)?;
        check_len("background ratios", &self.ratio, nf_max as usize * n)
    }
}

/// The update sums [`Kernels::fused_project`] adds to; `Default` adds to neither.
#[derive(Default)]
pub struct FusedSums<'a> {
    /// `[nobj, nyo, nxo]`, added as [`Kernels::psi_to_obj`] adds them.
    pub obj: Option<&'a mut UpdateSums>,
    /// `[nprobe, ny, nx]`, added by [`Kernels::psi_to_probe`] against the object read;
    /// without a [`Model::shift`] only.
    pub probe: Option<&'a mut UpdateSums>,
    /// The sums of a [`Model::shift`].
    pub shift: ShiftSums<'a>,
}

/// The update sums [`Kernels::shifted_psi_sums`] adds to; `Default` adds to neither.
#[derive(Default)]
pub struct PsiSums<'a> {
    /// `[nobj, nyo, nxo]`, as [`FusedSums::obj`] under a [`Model::shift`]: `num += sum_p
    /// conj(W_p) x` and `den += sum_p |W_p|^2`, spread to the object.
    pub obj: Option<&'a mut UpdateSums>,
    /// As [`ShiftSums::probe`], against [`FusedInputs::obj`].
    pub probe: Option<&'a mut ShiftedProbeSums>,
}

/// Which iteration [`Kernels::fused_project`] runs and where its new exit waves go. `psi`
/// is a DM stack's exit waves, `[nf_max * nmode, ny, nx]` in the M1 layout.
pub enum Exit<'a> {
    /// AP: the projected exit waves `P(PO)` stay in [`FusedBufs::work`]. `far` gets the sums
    /// of the forward far field, the one [`Kernels::llk`] reads after `obj_probe_to_psi` and
    /// the FFT.
    Ap { far: FarSums<'a> },
    /// The Poisson LLK gradient, for ML: the far field of `PO` scaled at every pixel by
    /// `2 s (1 - obs / max(calc, 1e-12))` (`s` the [`Model::scale`] or 1), 0 where masked,
    /// as engine's `poisson_grad`, goes back to the detector plane in [`FusedBufs::work`].
    /// The object sums' `num` then gets the object gradient `2 dL/dO*` of `Ptycho::obj_grad`,
    /// and the probe sums' `num` the probe's, `Ptycho::probe_grad`. `far` as for
    /// [`Exit::Ap`].
    Grad { far: FarSums<'a> },
    /// The first DM cycle, from PyNX's start `psi = PO`: `psi = P(PO)`. A DM cycle's far
    /// sums are those of `PO`, which [`Kernels::fused_llk`] gives, not of the reflection.
    DmFirst { psi: &'a mut DeviceBuffer<C32> },
    /// A DM cycle: with `R = (2 - alpha) PO - (1 - alpha) psi`, `psi = P(R) + (1 - alpha)
    /// (PO - R)`, as `dm_reflect`, projection and `add_c32` compute it.
    Dm {
        psi: &'a mut DeviceBuffer<C32>,
        alpha: f32,
    },
}

impl Kernels {
    /// One stack's projection: from the current object and probe, the step `exit` names,
    /// then the update sums of the waves it leaves: the probe's as [`Self::psi_to_probe`]
    /// adds them, against `inputs.obj`, and the object's as [`Self::psi_to_obj`] does (each
    /// object pixel over the stack's frames in ascending order, so the split into stacks does
    /// not change a bit; only tiles whose run holds a view of the stack are read and
    /// written). With a
    /// [`Model::shift`], the shifted probe takes the place of `P_p q` and `sums.shift` gets
    /// its sums. Up to nine launches, twelve with a shift; a stack with `nf == 0` writes
    /// nothing but `bufs`.
    pub fn fused_project(
        &self,
        ctx: &ExecutionContext,
        inputs: &FusedInputs<'_>,
        stack: StackRef<'_>,
        exit: Exit<'_>,
        bufs: &mut FusedBufs,
        sums: FusedSums<'_>,
    ) -> Result<(), Error> {
        let bufs = &*bufs;
        let (run, shift) = Run::new(self, ctx, *inputs, stack, bufs)?;
        let d = inputs.views.dims();
        // `pass`: what the column pass does to the far field.
        let (psi, far, dm, alpha, pass) = match exit {
            Exit::Ap { far } => (None, far, false, 0.0, FUSED_PROJECT),
            Exit::Grad { far } => (None, far, false, 0.0, FUSED_GRAD),
            Exit::DmFirst { psi } => (Some(psi), FarSums::default(), false, 0.0, FUSED_PROJECT),
            Exit::Dm { psi, alpha } => (Some(psi), FarSums::default(), true, alpha, FUSED_PROJECT),
        };
        if let Some(psi) = &psi {
            check_len("psi", psi, run.g.nf_max as usize * d.nmode as usize * d.n())?;
        }
        run.check_far(&far)?;
        let FusedSums {
            obj: obj_sums,
            probe: probe_sums,
            shift:
                ShiftSums {
                    probe: shift_probe,
                    position,
                    position_den,
                },
        } = sums;
        if let Some(sums) = &obj_sums {
            sums.check(d.obj_len())?;
        }
        if let Some(sums) = &probe_sums {
            sums.check(d.nprobe as usize * d.n())?;
        }
        let shift_out = shift_probe.is_some() || position.is_some() || position_den.is_some();
        match &shift {
            Some(_) if probe_sums.is_some() => {
                return Err(Error::Shape(
                    "a shifted probe's sums go to FusedSums::shift".into(),
                ));
            }
            None if shift_out => {
                return Err(Error::Shape("FusedSums::shift needs a Model::shift".into()));
            }
            _ => {}
        }
        if let Some(sums) = &shift_probe {
            sums.check(d)?;
        }
        for out in [&position, &position_den].into_iter().flatten() {
            check_len("position sums", out, d.nframe as usize)?;
        }
        let psi = psi.as_deref();
        let p = FusedPtrs {
            psi: psi.map_or(std::ptr::null_mut(), dptr),
            ..run.ptrs(bufs, &far)
        };
        let dm_flag = if dm { FUSED_DM } else { 0 };
        if dm {
            // The inverse row pass walks with the column pass (`Run::walk_with`).
            run.start_up();
        }
        let flag = |on: bool, flag: u32| if on { flag } else { 0 };
        let shift_flags = match shift {
            Some(_) => {
                flag(obj_sums.is_some(), FUSED_SHIFT_OBJ)
                    | flag(shift_probe.is_some(), FUSED_SHIFT_PROBE)
                    | flag(position.is_some(), FUSED_SHIFT_POSITION)
                    | flag(position_den.is_some(), FUSED_SHIFT_DEN)
                    | (pass & FUSED_GRAD)
            }
            None => 0,
        };

        let cfg = FusedCfg {
            flags: dm_flag,
            alpha,
        };
        match &shift {
            Some(sh) => {
                let slopes = FUSED_SHIFT_DEN | FUSED_SHIFT_POSITION;
                let t_dy = (shift_flags & slopes != 0).then_some(&sh.scratch.adj);
                run.shift_pre(&sh.model, &sh.scratch.t, t_dy, p)?;
                run.shift_rows_fwd(&sh.scratch.t, &bufs.work, psi, p, cfg)?;
            }
            None => {
                if obj_sums.is_some() {
                    run.probe_terms(bufs, p)?;
                }
                run.rows_fwd(bufs, psi, p, cfg)?;
            }
        }
        let cfg = FusedCfg {
            flags: pass | run.model_flags() | far.flags(),
            alpha,
        };
        run.cols(bufs, &far, p, cfg)?;
        run.far_sums(bufs, far)?;
        let to_psi = if psi.is_some() { FUSED_TO_PSI } else { 0 };
        let cfg = FusedCfg {
            flags: dm_flag | to_psi | shift_flags,
            alpha,
        };
        let waves = psi.unwrap_or(&bufs.work);
        match &shift {
            Some(sh) => {
                run.shift_rows_inv(sh, bufs, psi, p, cfg)?;
                if shift_probe.is_some() {
                    run.shift_post(sh, p)?;
                }
                let ngroup = run.g.ny / run.sh.rows;
                if let Some(out) = position {
                    run.position_groups(&sh.scratch.grad_groups, ngroup, out)?;
                }
                if let Some(out) = position_den {
                    run.position_groups(&sh.scratch.den_groups, ngroup, out)?;
                }
                if let Some(sums) = shift_probe {
                    run.shift_probe_sums(sh, p, sums)?;
                }
                if let Some(sums) = obj_sums {
                    run.gather_shifted(sh, p, cfg, sums)?;
                }
            }
            None => {
                run.rows_inv(bufs, psi, p, cfg)?;
                if let Some(sums) = probe_sums {
                    debug_assert!(run.walked_down(), "psi_to_probe after an up walk");
                    self.psi_to_probe(ctx, inputs.views, stack, waves, inputs.obj, sums)?;
                }
                if let Some(sums) = obj_sums {
                    run.gather(bufs, waves, p, cfg, sums)?;
                }
            }
        }
        Ok(())
    }

    /// The update sums of a stack's exit waves `psi` (`[nf_max * nmode, ny, nx]`, as
    /// [`Exit::Dm`] leaves them) under the [`Model::shift`] of `inputs`, with no projection:
    /// what [`Self::fused_project`] adds to [`FusedSums::obj`] and [`ShiftSums::probe`] from
    /// the waves it leaves, in its order, so that the split into stacks does not change a
    /// bit. Another kernel build: the sums match `fused_project`'s to rounding, not bit for
    /// bit. Up to five launches; writes nothing but `sums` and `bufs`.
    pub fn shifted_psi_sums(
        &self,
        ctx: &ExecutionContext,
        inputs: &FusedInputs<'_>,
        stack: StackRef<'_>,
        psi: &DeviceBuffer<C32>,
        bufs: &mut FusedBufs,
        sums: PsiSums<'_>,
    ) -> Result<(), Error> {
        let bufs = &*bufs;
        let (run, shift) = Run::new(self, ctx, *inputs, stack, bufs)?;
        let Some(sh) = &shift else {
            return Err(Error::Shape("shifted_psi_sums needs a Model::shift".into()));
        };
        let d = inputs.views.dims();
        check_len("psi", psi, run.g.nf_max as usize * d.nmode as usize * d.n())?;
        let PsiSums { obj, probe } = sums;
        if let Some(sums) = &obj {
            sums.check(d.obj_len())?;
        }
        if let Some(sums) = &probe {
            sums.check(d)?;
        }
        let p = FusedPtrs {
            psi: dptr(psi),
            ..run.ptrs(bufs, &FarSums::default())
        };
        let flag = |on: bool, flag: u32| if on { flag } else { 0 };
        let cfg = FusedCfg {
            flags: flag(obj.is_some(), FUSED_SHIFT_OBJ) | flag(probe.is_some(), FUSED_SHIFT_PROBE),
            alpha: 0.0,
        };
        run.shift_pre(&sh.model, &sh.scratch.t, None, p)?;
        run.shift_sums(sh, psi, p, cfg)?;
        if let Some(sums) = probe {
            run.shift_post(sh, p)?;
            run.shift_probe_sums(sh, p, sums)?;
        }
        if let Some(sums) = obj {
            run.gather_shifted(sh, p, cfg, sums)?;
        }
        Ok(())
    }

    /// The stack's row-FFT'd exit waves `O W` into `work`, `[nf_max * nmode, ny, nx]`, the
    /// fused forward row pass with the shifted probe `W` of the [`Model::shift`] of `inputs`,
    /// as [`Self::fused_llk`] leaves them for its column pass; and `T` into `t`, `[nf_max *
    /// nprobe, ny, nx]`: `W_p` of frame `fl` is the unnormalized row IDFT of plane `fl *
    /// nprobe + p`, in the exit waves' layout. The model's scale and background are not read.
    /// Two launches.
    pub fn shifted_rows_fwd(
        &self,
        ctx: &ExecutionContext,
        inputs: &FusedInputs<'_>,
        stack: StackRef<'_>,
        t: &mut DeviceBuffer<C32>,
        work: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let run = Run::checked(self, ctx, *inputs, stack)?;
        let Some(model) = inputs.model.shift else {
            return Err(Error::Shape("shifted_rows_fwd needs a Model::shift".into()));
        };
        let d = inputs.views.dims();
        let (n, nf) = (d.n(), run.g.nf_max as usize);
        check_len("t", t, nf * d.nprobe as usize * n)?;
        check_len("work", work, nf * d.nmode as usize * n)?;
        let p = FusedPtrs {
            shift: dptr(t),
            work: dptr(work),
            ..run.input_ptrs()
        };
        run.shift_pre(&model, t, None, p)?;
        run.shift_rows_fwd(t, work, None, p, FusedCfg::default())
    }

    /// The stack's far sums of the far field of `PO`, the one `obj_probe_to_psi` and the FFT
    /// give (with a [`Model::shift`], of `O W`), the LLK partials as [`Self::llk`] gives
    /// them. Up to five launches, six with a shift; writes nothing but `far` and `bufs`.
    pub fn fused_llk(
        &self,
        ctx: &ExecutionContext,
        inputs: &FusedInputs<'_>,
        stack: StackRef<'_>,
        bufs: &mut FusedBufs,
        far: FarSums<'_>,
    ) -> Result<(), Error> {
        let bufs = &*bufs;
        let (run, shift) = Run::new(self, ctx, *inputs, stack, bufs)?;
        run.check_far(&far)?;
        let p = run.ptrs(bufs, &far);
        match &shift {
            Some(sh) => {
                run.shift_pre(&sh.model, &sh.scratch.t, None, p)?;
                run.shift_rows_fwd(&sh.scratch.t, &bufs.work, None, p, FusedCfg::default())?;
            }
            None => run.rows_fwd(bufs, None, p, FusedCfg::default())?,
        }
        let cfg = FusedCfg {
            flags: run.model_flags() | far.flags(),
            alpha: 0.0,
        };
        run.cols(bufs, &far, p, cfg)?;
        run.far_sums(bufs, far)
    }

    /// Poisson's fixed-point background step from `sums`, `b[j] = b[j] sum[j] / nobs[j]`
    /// where `nobs[j]`, the frames at which detector pixel `j` is unmasked, is not 0; then
    /// zeroes `sums`, which the step has consumed.
    pub fn background_ap(
        &self,
        ctx: &ExecutionContext,
        b: &mut DeviceBuffer<f32>,
        sums: &mut BackgroundSums,
        nobs: &DeviceBuffer<u32>,
    ) -> Result<(), Error> {
        let n = sums.sum.len();
        check_len("background", b, n)?;
        check_len("nobs", nobs, n)?;
        let config = grid_1d(n)?;
        let writes = [b.storage(), sums.sum.storage()];
        self.launch(ctx, &[nobs.storage()], &writes, |stream| {
            // SAFETY: all three hold `n` values.
            unsafe {
                self.module.background_ap(
                    stream,
                    config,
                    dptr(b),
                    dptr(&sums.sum),
                    dptr(nobs),
                    n as u32,
                )
            }
        })
    }

    /// The Poisson LLK gradient over the background from `sums`, `grad[j] = nobs[j] -
    /// sum[j]` (`nobs` as in [`Self::background_ap`]); then zeroes `sums`, which the
    /// gradient has consumed.
    pub fn background_grad(
        &self,
        ctx: &ExecutionContext,
        sums: &mut BackgroundSums,
        nobs: &DeviceBuffer<u32>,
        grad: &mut DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        let n = sums.sum.len();
        check_len("nobs", nobs, n)?;
        check_len("grad", grad, n)?;
        let config = grid_1d(n)?;
        let writes = [sums.sum.storage(), grad.storage()];
        self.launch(ctx, &[nobs.storage()], &writes, |stream| {
            // SAFETY: all three hold `n` values.
            unsafe {
                self.module.background_grad(
                    stream,
                    config,
                    dptr(&sums.sum),
                    dptr(nobs),
                    dptr(grad),
                    n as u32,
                )
            }
        })
    }
}

impl Kernels {
    /// The spectra `A_p = DFT(P_p q) / n` of `probe` `[nprobe, ny, nx]` for frames of `dims`
    /// (`P_p q` in the exit waves' layout, origin at `[0, 0]`; `A_p` in FFT order) into
    /// `buf`, for [`Shift::spectrum`]. Two launches.
    pub fn probe_spectrum<'a>(
        &self,
        ctx: &ExecutionContext,
        dims: &ViewDims,
        probe: &'a DeviceBuffer<C32>,
        buf: &'a mut SpectrumBuf,
    ) -> Result<ProbeSpectrum<'a>, Error> {
        let sh = Shape::of(dims)?;
        let len = dims.nprobe as usize * dims.n();
        check_len("probe", probe, len)?;
        check_len("spectrum", &buf.0, len)?;
        let config = sh.plane_rows(dims.nprobe);
        let (ny, quad) = (dims.frame[0], dims.quad_phase);
        let (src, a) = (dptr::<C32, f32>(probe).cast_const(), dptr(&buf.0));
        self.launch(ctx, &[probe.storage()], &[buf.0.storage()], |stream| {
            // SAFETY: both hold `[nprobe, ny, nx]`; the launch is the one `spectrum_rows`
            // documents.
            unsafe {
                match sh.nx {
                    Side::N64 => self
                        .module
                        .fused_spectrum_rows_64(stream, config, src, a, ny, quad),
                    Side::N128 => self
                        .module
                        .fused_spectrum_rows_128(stream, config, src, a, ny, quad),
                    Side::N256 => self
                        .module
                        .fused_spectrum_rows_256(stream, config, src, a, ny, quad),
                    Side::N512 => self
                        .module
                        .fused_spectrum_rows_512(stream, config, src, a, ny, quad),
                }
            }
        })?;
        self.plane_fft(ctx, &sh, &buf.0, dims.nprobe, false, 1.0 / dims.n() as f32)?;
        Ok(ProbeSpectrum {
            a: &buf.0,
            probe,
            key: ProbeSpectrum::key(dims),
        })
    }

    /// Adds the probe sums of a shifted probe, `sums`, to `out` `[nprobe, ny, nx]`: the
    /// adjoint of [`Shift`]'s `W` at each frame's offsets applied to `Y = sum_o conj(O_o) x`,
    /// `num += conj(q) IDFT(sum_f conj(e_f) DFT(Y_f)) / n`, which is [`Self::psi_to_probe`]'s
    /// at zero offsets, and `den += sum |O|^2`; then zeroes `sums`, which `out` has consumed.
    /// Two launches.
    pub fn shifted_probe_sums(
        &self,
        ctx: &ExecutionContext,
        dims: &ViewDims,
        sums: &mut ShiftedProbeSums,
        out: &mut UpdateSums,
    ) -> Result<(), Error> {
        let sh = Shape::of(dims)?;
        sums.check(dims)?;
        out.check(dims.nprobe as usize * dims.n())?;
        self.plane_fft(ctx, &sh, &sums.num, dims.nprobe, true, 1.0)?;
        let config = sh.plane_rows(dims.nprobe);
        let (ny, quad) = (dims.frame[0], dims.quad_phase);
        let (sp, op) = (sums.ptrs(), out.ptrs());
        let writes = [sums.storages(), out.storages()].concat();
        self.launch(ctx, &[], &writes, |stream| {
            // SAFETY: `check` sized both `[nprobe, ny, nx]`; the launch is the one
            // `shift_probe_rows` documents.
            unsafe {
                match sh.nx {
                    Side::N64 => self
                        .module
                        .fused_shift_probe_rows_64(stream, config, sp, op, ny, quad),
                    Side::N128 => self
                        .module
                        .fused_shift_probe_rows_128(stream, config, sp, op, ny, quad),
                    Side::N256 => self
                        .module
                        .fused_shift_probe_rows_256(stream, config, sp, op, ny, quad),
                    Side::N512 => self
                        .module
                        .fused_shift_probe_rows_512(stream, config, sp, op, ny, quad),
                }
            }
        })
    }

    /// In-place column FFT of `nplane` planes `[ny, nx]` of `buf`, times `scale`.
    fn plane_fft(
        &self,
        ctx: &ExecutionContext,
        sh: &Shape,
        buf: &DeviceBuffer<C32>,
        nplane: u32,
        inverse: bool,
        scale: f32,
    ) -> Result<(), Error> {
        let config = sh.plane_cols(nplane);
        let (b, nx, inverse) = (dptr(buf), sh.nx as u32, inverse as u32);
        self.launch(ctx, &[], &[buf.storage()], |stream| {
            // SAFETY: the callers checked `buf` holds `nplane` planes; the launch is the one
            // `shift_cols` documents.
            unsafe {
                match sh.ny {
                    Side::N64 => self
                        .module
                        .fused_shift_cols_64(stream, config, b, nx, inverse, scale),
                    Side::N128 => self
                        .module
                        .fused_shift_cols_128(stream, config, b, nx, inverse, scale),
                    Side::N256 => self
                        .module
                        .fused_shift_cols_256(stream, config, b, nx, inverse, scale),
                    Side::N512 => self
                        .module
                        .fused_shift_cols_512(stream, config, b, nx, inverse, scale),
                }
            }
        })
    }
}

/// A [`Model::shift`] and its scratch, checked.
struct Shifted<'a> {
    model: Shift<'a>,
    scratch: &'a ShiftScratch,
}

/// One fused call's inputs and stack, checked (a [`Model::shift`]'s offsets and spectrum
/// too): what every pass launch of it needs but the scratch.
struct Run<'a> {
    k: &'a Kernels,
    ctx: &'a ExecutionContext,
    inputs: FusedInputs<'a>,
    stack: StackRef<'a>,
    g: StackGeom,
    sh: Shape,
    /// Whether the next frame pass walks the stack down ([`FUSED_DOWN`]), opposite to the
    /// pass whose plane it reads back: the one before it, or the one before that under
    /// [`Self::walk_with`]. A call starts down, or up with one such pass, so the pass before
    /// an in-order frame sum walks down and the sum's first frames are the last touched.
    down: Cell<bool>,
}

impl<'a> Run<'a> {
    fn checked(
        k: &'a Kernels,
        ctx: &'a ExecutionContext,
        inputs: FusedInputs<'a>,
        stack: StackRef<'a>,
    ) -> Result<Self, Error> {
        let d = inputs.views.dims();
        let g = d.geom(stack)?;
        let sh = Shape::of(d)?;
        let n = d.n();
        check_len("obj", inputs.obj, d.obj_len())?;
        check_len("probe", inputs.probe, d.nprobe as usize * n)?;
        check_len("counts", inputs.counts, d.nframe as usize * n)?;
        if let Some(scale) = inputs.model.scale {
            check_len("scale", scale, d.nframe as usize)?;
        }
        if let Some(background) = inputs.model.background {
            check_len("background", background, n)?;
        }
        if let Some(model) = inputs.model.shift {
            check_len("offsets", model.offsets, 2 * d.nframe as usize)?;
            model.spectrum.check(d, inputs.probe)?;
            sh.shift_cols(d.nprobe, g.nf_max)?;
        }
        Ok(Self {
            k,
            ctx,
            inputs,
            stack,
            g,
            sh,
            down: Cell::new(true),
        })
    }

    /// `cfg` for the next frame pass, walking the stack opposite to the pass before it.
    fn walk(&self, cfg: FusedCfg) -> FusedCfg {
        let down = self.down.get();
        self.down.set(!down);
        let flags = if down { FUSED_DOWN } else { 0 };
        FusedCfg {
            flags: cfg.flags | flags,
            ..cfg
        }
    }

    /// `cfg` for a frame pass walking the way the pass before it did, opposite to the one
    /// before that, whose plane it reads back: under `FUSED_DM` the inverse row pass
    /// re-reads the `psi` the forward one read, of which L2 holds only the tail, while the
    /// column pass between leaves it all of `work` when one plane of the stack fits.
    fn walk_with(&self, cfg: FusedCfg) -> FusedCfg {
        self.down.set(!self.down.get());
        self.walk(cfg)
    }

    /// Starts the walk up, for a call with one [`Self::walk_with`] pass.
    fn start_up(&self) {
        self.down.set(false);
    }

    /// Whether the last frame pass walked down, as the one before an in-order frame sum must.
    fn walked_down(&self) -> bool {
        !self.down.get()
    }

    /// [`Self::checked`] with `bufs` checked, and the shift model with its scratch.
    fn new(
        k: &'a Kernels,
        ctx: &'a ExecutionContext,
        inputs: FusedInputs<'a>,
        stack: StackRef<'a>,
        bufs: &'a FusedBufs,
    ) -> Result<(Self, Option<Shifted<'a>>), Error> {
        let run = Self::checked(k, ctx, inputs, stack)?;
        let d = inputs.views.dims();
        bufs.check(d, &run.g)?;
        let Some(model) = inputs.model.shift else {
            return Ok((run, None));
        };
        let Some(scratch) = &bufs.shift else {
            return Err(Error::Shape(
                "a Model::shift needs FusedBufs::with_shift".into(),
            ));
        };
        scratch.check(d, &run.g, &run.sh)?;
        Ok((run, Some(Shifted { model, scratch })))
    }

    /// The inputs' addresses; every other is null.
    fn input_ptrs(&self) -> FusedPtrs {
        let input = |b: Option<&DeviceBuffer<f32>>| {
            b.map_or(std::ptr::null(), |b| dptr::<f32, f32>(b).cast_const())
        };
        let model = self.inputs.model.shift;
        let null = std::ptr::null_mut();
        FusedPtrs {
            obj: dptr(self.inputs.obj),
            probe: dptr(self.inputs.probe),
            views: dptr(self.inputs.views.addr()),
            counts: dptr(self.inputs.counts),
            psi: null,
            work: null,
            probe_q: null,
            probe_power: null,
            groups: std::ptr::null_mut(),
            scale: input(self.inputs.model.scale),
            scale_groups: std::ptr::null_mut(),
            background: input(self.inputs.model.background),
            background_ratio: null,
            offsets: input(model.map(|m| m.offsets)),
            spectrum: model.map_or(std::ptr::null(), |m| {
                dptr::<C32, f32>(m.spectrum.a).cast_const()
            }),
            shift: null,
            shift_adj: null,
            shift_obj: null,
            shift_power: null,
            shift_grad: std::ptr::null_mut(),
            shift_den: std::ptr::null_mut(),
        }
    }

    /// Every address but `psi`, which is null.
    fn ptrs(&self, bufs: &FusedBufs, far: &FarSums<'_>) -> FusedPtrs {
        let scratch =
            |f: fn(&ShiftScratch) -> *mut f32| bufs.shift.as_ref().map_or(std::ptr::null_mut(), f);
        let groups = |f: fn(&ShiftScratch) -> *mut PositionPartial| {
            bufs.shift.as_ref().map_or(std::ptr::null_mut(), f)
        };
        FusedPtrs {
            work: dptr(&bufs.work),
            probe_q: dptr(&bufs.probe_q),
            probe_power: dptr(&bufs.probe_power),
            groups: dptr(&bufs.groups),
            scale_groups: dptr(&bufs.scale_groups),
            background_ratio: (far.background.as_deref())
                .map_or(std::ptr::null_mut(), |b| dptr(&b.ratio)),
            shift: scratch(|b| dptr(&b.t)),
            shift_adj: scratch(|b| dptr(&b.adj)),
            shift_obj: scratch(|b| dptr(&b.obj)),
            shift_power: scratch(|b| dptr(&b.power)),
            shift_grad: groups(|b| dptr(&b.grad_groups)),
            shift_den: groups(|b| dptr(&b.den_groups)),
            ..self.input_ptrs()
        }
    }

    /// The column-pass flags of the model.
    fn model_flags(&self) -> u32 {
        let Model {
            scale, background, ..
        } = self.inputs.model;
        let flag = |on: bool, flag: u32| if on { flag } else { 0 };
        flag(scale.is_some(), FUSED_SCALE) | flag(background.is_some(), FUSED_BACKGROUND)
    }

    fn check_far(&self, far: &FarSums<'_>) -> Result<(), Error> {
        if let Some(out) = &far.llk {
            check_len("llk", out, self.g.nf_max as usize)?;
        }
        if let Some(out) = &far.scale {
            check_len("scale sums", out, self.g.nframe as usize)?;
        }
        if let Some(sums) = &far.background {
            sums.check(self.inputs.views.dims().n(), self.g.nf_max)?;
        }
        Ok(())
    }

    fn ctl(&self) -> *const StackCtl {
        self.stack.ctl() as *const StackCtl
    }

    /// `bufs.probe_q` and `bufs.probe_power` from the probe.
    fn probe_terms(&self, bufs: &FusedBufs, p: FusedPtrs) -> Result<(), Error> {
        let config = grid_1d(self.inputs.views.dims().n())?;
        let reads = [self.inputs.probe.storage()];
        let writes = [bufs.probe_q.storage(), bufs.probe_power.storage()];
        let g = self.g;
        self.k.launch(self.ctx, &reads, &writes, |stream| {
            // SAFETY: `Run::new` checked `probe` and `bufs` against `g`.
            unsafe { self.k.module.fused_probe_terms(stream, config, p, g) }
        })
    }

    /// Pass A into `bufs.work`; reads `psi` under `FUSED_DM`.
    fn rows_fwd(
        &self,
        bufs: &FusedBufs,
        psi: Option<&DeviceBuffer<C32>>,
        p: FusedPtrs,
        cfg: FusedCfg,
    ) -> Result<(), Error> {
        let cfg = self.walk(cfg);
        let config = self.sh.rows(&self.g, true)?;
        let mut reads = vec![
            self.inputs.obj.storage(),
            self.inputs.probe.storage(),
            self.inputs.views.addr().storage(),
            self.stack.storage(),
        ];
        if cfg.flags & FUSED_DM != 0 {
            reads.extend(psi.map(|b| b.storage()));
        }
        let (ctl, g) = (self.ctl(), self.g);
        self.k
            .launch(self.ctx, &reads, &[bufs.work.storage()], |stream| {
                // SAFETY: `Run::new` and the caller checked every buffer the flags name;
                // `Views::new` checked the views are canonical and inside the object; the
                // launch is the one `rows_fwd` documents.
                unsafe {
                    match self.sh.nx {
                        Side::N64 => self
                            .k
                            .module
                            .fused_rows_fwd_64(stream, config, p, ctl, g, cfg),
                        Side::N128 => self
                            .k
                            .module
                            .fused_rows_fwd_128(stream, config, p, ctl, g, cfg),
                        Side::N256 => self
                            .k
                            .module
                            .fused_rows_fwd_256(stream, config, p, ctl, g, cfg),
                        Side::N512 => self
                            .k
                            .module
                            .fused_rows_fwd_512(stream, config, p, ctl, g, cfg),
                    }
                }
            })
    }

    /// Pass B on `bufs.work`; reads the model's buffers and writes `bufs.groups`,
    /// `bufs.scale_groups` and the background ratios of `far` under their flags.
    fn cols(
        &self,
        bufs: &FusedBufs,
        far: &FarSums<'_>,
        p: FusedPtrs,
        cfg: FusedCfg,
    ) -> Result<(), Error> {
        let cfg = self.walk(cfg);
        let config = self.sh.cols(&self.g, cfg.flags)?;
        let mut reads = vec![self.inputs.counts.storage(), self.stack.storage()];
        let mut writes = vec![];
        if cfg.flags & (FUSED_PROJECT | FUSED_GRAD) != 0 {
            writes.push(bufs.work.storage());
        } else {
            reads.push(bufs.work.storage());
        }
        if cfg.flags & FUSED_SCALE != 0 {
            reads.extend(self.inputs.model.scale.map(|b| b.storage()));
        }
        if cfg.flags & FUSED_BACKGROUND != 0 {
            reads.extend(self.inputs.model.background.map(|b| b.storage()));
        }
        if cfg.flags & FUSED_LLK != 0 {
            writes.push(bufs.groups.storage());
        }
        if cfg.flags & FUSED_SCALE_SUMS != 0 {
            writes.push(bufs.scale_groups.storage());
        }
        if cfg.flags & FUSED_BACKGROUND_SUMS != 0 {
            writes.extend(far.background.as_deref().map(|b| b.ratio.storage()));
        }
        let (ctl, g) = (self.ctl(), self.g);
        let model = cfg.flags & FUSED_MODEL_TERMS != 0;
        self.k.launch(self.ctx, &reads, &writes, |stream| {
            let m = &self.k.module;
            // SAFETY: as in `rows_fwd`; the launch is the one `cols` documents, of the build
            // whose terms cover `cfg.flags`.
            unsafe {
                match (self.sh.ny, model) {
                    (Side::N64, false) => m.fused_cols_64(stream, config, p, ctl, g, cfg),
                    (Side::N128, false) => m.fused_cols_128(stream, config, p, ctl, g, cfg),
                    (Side::N256, false) => m.fused_cols_256(stream, config, p, ctl, g, cfg),
                    (Side::N512, false) => m.fused_cols_512(stream, config, p, ctl, g, cfg),
                    (Side::N64, true) => m.fused_cols_model_64(stream, config, p, ctl, g, cfg),
                    (Side::N128, true) => m.fused_cols_model_128(stream, config, p, ctl, g, cfg),
                    (Side::N256, true) => m.fused_cols_model_256(stream, config, p, ctl, g, cfg),
                    (Side::N512, true) => m.fused_cols_model_512(stream, config, p, ctl, g, cfg),
                }
            }
        })
    }

    /// Each frame's `far` sums from the column groups pass B wrote under `far.flags()`.
    fn far_sums(&self, bufs: &FusedBufs, far: FarSums<'_>) -> Result<(), Error> {
        let ngroup = self.g.nx / FUSED_COLS;
        if let Some(out) = far.llk {
            self.k
                .llk_groups(self.ctx, self.stack, self.g, &bufs.groups, ngroup, out)?;
        }
        if let Some(out) = far.scale {
            let config = grid_1d(self.g.nf_max as usize)?;
            let reads = [bufs.scale_groups.storage(), self.stack.storage()];
            let (groups, ctl, g) = (dptr(&bufs.scale_groups), self.ctl(), self.g);
            self.k
                .launch(self.ctx, &reads, &[out.storage()], |stream| {
                    // SAFETY: `Run::new` checked `bufs` and `check_far` `out` against `g`.
                    unsafe {
                        self.k.module.scale_groups(
                            stream,
                            config,
                            groups,
                            ngroup,
                            dptr(out),
                            ctl,
                            g,
                        )
                    }
                })?;
        }
        if let Some(sums) = far.background {
            let config = grid_1d(self.inputs.views.dims().n())?;
            let reads = [sums.ratio.storage(), self.stack.storage()];
            let (ctl, g) = (self.ctl(), self.g);
            self.k
                .launch(self.ctx, &reads, &[sums.sum.storage()], |stream| {
                    // SAFETY: `check_far` checked `sums` against `g`.
                    unsafe {
                        self.k.module.background_sums(
                            stream,
                            config,
                            dptr(&sums.ratio),
                            dptr(&sums.sum),
                            ctl,
                            g,
                        )
                    }
                })?;
        }
        Ok(())
    }

    /// Pass C from `bufs.work` to `psi` (under `FUSED_TO_PSI`) or back to `bufs.work`;
    /// reads `psi`, the probe and the object under `FUSED_DM`.
    fn rows_inv(
        &self,
        bufs: &FusedBufs,
        psi: Option<&DeviceBuffer<C32>>,
        p: FusedPtrs,
        cfg: FusedCfg,
    ) -> Result<(), Error> {
        let cfg = if cfg.flags & FUSED_DM != 0 {
            self.walk_with(cfg)
        } else {
            self.walk(cfg)
        };
        let config = self.sh.rows(&self.g, false)?;
        let mut reads = vec![self.inputs.views.addr().storage(), self.stack.storage()];
        let mut writes = vec![];
        match psi {
            Some(psi) if cfg.flags & FUSED_TO_PSI != 0 => {
                reads.push(bufs.work.storage());
                writes.push(psi.storage());
            }
            _ => writes.push(bufs.work.storage()),
        }
        if cfg.flags & FUSED_DM != 0 {
            reads.push(self.inputs.obj.storage());
            reads.push(self.inputs.probe.storage());
        }
        let (ctl, g) = (self.ctl(), self.g);
        self.k.launch(self.ctx, &reads, &writes, |stream| {
            // SAFETY: as in `rows_fwd`; the launch is the one `rows_inv` documents.
            unsafe {
                match self.sh.nx {
                    Side::N64 => self
                        .k
                        .module
                        .fused_rows_inv_64(stream, config, p, ctl, g, cfg),
                    Side::N128 => self
                        .k
                        .module
                        .fused_rows_inv_128(stream, config, p, ctl, g, cfg),
                    Side::N256 => self
                        .k
                        .module
                        .fused_rows_inv_256(stream, config, p, ctl, g, cfg),
                    Side::N512 => self
                        .k
                        .module
                        .fused_rows_inv_512(stream, config, p, ctl, g, cfg),
                }
            }
        })
    }

    /// The object sums from `waves`, the new exit waves pass C stored under `cfg`, and
    /// `bufs.probe_q` and `bufs.probe_power`.
    fn gather(
        &self,
        bufs: &FusedBufs,
        waves: &DeviceBuffer<C32>,
        p: FusedPtrs,
        cfg: FusedCfg,
        sums: &mut UpdateSums,
    ) -> Result<(), Error> {
        let views = self.inputs.views;
        let [start, len, list] = views.tile_storages();
        let reads = [
            waves.storage(),
            bufs.probe_q.storage(),
            bufs.probe_power.storage(),
            views.addr().storage(),
            start,
            len,
            list,
            self.stack.storage(),
        ];
        let config = views.dims().stack_tile_grid(self.g.nf_max)?;
        let (tiles, ctl, g) = (views.tile_ptrs(), self.ctl(), self.g);
        self.k.launch(self.ctx, &reads, &sums.storages(), |stream| {
            // SAFETY: as `psi_to_obj`, with `waves` where `cfg` has pass C store them and
            // `probe_q` and `probe_power` checked with the rest of `bufs`; the tile lists
            // were built from these views.
            unsafe {
                self.k
                    .module
                    .fused_gather(stream, config, p, tiles, sums.ptrs(), ctl, g, cfg)
            }
        })
    }

    /// `T` of the stack's frames into `t`, at `p.shift`, and with `t_dy` `T_y` into it, at
    /// `p.shift_adj`.
    fn shift_pre(
        &self,
        model: &Shift<'_>,
        t: &DeviceBuffer<C32>,
        t_dy: Option<&DeviceBuffer<C32>>,
        p: FusedPtrs,
    ) -> Result<(), Error> {
        let config = self.sh.shift_cols(self.g.nprobe, self.g.nf_max)?;
        let reads = [
            model.offsets.storage(),
            model.spectrum.a.storage(),
            self.stack.storage(),
        ];
        let mut writes = vec![t.storage()];
        writes.extend(t_dy.map(|b| b.storage()));
        let flags = match t_dy {
            Some(_) => FUSED_SHIFT_DEN | FUSED_SHIFT_POSITION,
            None => 0,
        };
        let cfg = self.walk(FusedCfg { flags, alpha: 0.0 });
        let (ctl, g) = (self.ctl(), self.g);
        self.k.launch(self.ctx, &reads, &writes, |stream| {
            // SAFETY: `Run::checked` checked the shift model against `g` and the callers `t`
            // and `t_dy`; the launch is the one `shift_pre` documents.
            unsafe {
                match self.sh.ny {
                    Side::N64 => self
                        .k
                        .module
                        .fused_shift_pre_64(stream, config, p, ctl, g, cfg),
                    Side::N128 => self
                        .k
                        .module
                        .fused_shift_pre_128(stream, config, p, ctl, g, cfg),
                    Side::N256 => self
                        .k
                        .module
                        .fused_shift_pre_256(stream, config, p, ctl, g, cfg),
                    Side::N512 => self
                        .k
                        .module
                        .fused_shift_pre_512(stream, config, p, ctl, g, cfg),
                }
            }
        })
    }

    /// Pass A with the shifted probe from `t` into `work`, at `p.shift` and `p.work`; reads
    /// `psi` under `FUSED_DM`.
    fn shift_rows_fwd(
        &self,
        t: &DeviceBuffer<C32>,
        work: &DeviceBuffer<C32>,
        psi: Option<&DeviceBuffer<C32>>,
        p: FusedPtrs,
        cfg: FusedCfg,
    ) -> Result<(), Error> {
        let cfg = self.walk(cfg);
        let config = self.sh.rows(&self.g, true)?;
        let mut reads = vec![
            self.inputs.obj.storage(),
            self.inputs.views.addr().storage(),
            self.stack.storage(),
            t.storage(),
        ];
        if cfg.flags & FUSED_DM != 0 {
            reads.extend(psi.map(|b| b.storage()));
        }
        let (ctl, g) = (self.ctl(), self.g);
        self.k
            .launch(self.ctx, &reads, &[work.storage()], |stream| {
                // SAFETY: as in `rows_fwd`, with the callers checking `t` and `work`, and
                // `shift_pre` wrote `T` of this stack into `t`; the launch is the one
                // `shift_rows_fwd` documents.
                unsafe {
                    match self.sh.nx {
                        Side::N64 => self
                            .k
                            .module
                            .fused_shift_rows_fwd_64(stream, config, p, ctl, g, cfg),
                        Side::N128 => self
                            .k
                            .module
                            .fused_shift_rows_fwd_128(stream, config, p, ctl, g, cfg),
                        Side::N256 => self
                            .k
                            .module
                            .fused_shift_rows_fwd_256(stream, config, p, ctl, g, cfg),
                        Side::N512 => self
                            .k
                            .module
                            .fused_shift_rows_fwd_512(stream, config, p, ctl, g, cfg),
                    }
                }
            })
    }

    /// Pass C with the shifted probe, as [`Self::rows_inv`], and the shift terms `cfg` names.
    fn shift_rows_inv(
        &self,
        sh: &Shifted<'_>,
        bufs: &FusedBufs,
        psi: Option<&DeviceBuffer<C32>>,
        p: FusedPtrs,
        cfg: FusedCfg,
    ) -> Result<(), Error> {
        let cfg = if cfg.flags & FUSED_DM != 0 {
            self.walk_with(cfg)
        } else {
            self.walk(cfg)
        };
        let config = LaunchConfig {
            shared_mem_bytes: self.sh.shift_row_smem(cfg.flags),
            ..self.sh.rows(&self.g, false)?
        };
        let f = cfg.flags;
        let (mut reads, mut writes) = self.shift_terms(sh, f);
        match psi {
            Some(psi) if f & FUSED_TO_PSI != 0 => {
                reads.push(bufs.work.storage());
                writes.push(psi.storage());
            }
            _ => writes.push(bufs.work.storage()),
        }
        let s = sh.scratch;
        // Under `FUSED_SHIFT_PROBE` `shift_terms` already lists `T_y`'s plane as written.
        if f & FUSED_SHIFT_PROBE == 0 && f & (FUSED_SHIFT_DEN | FUSED_SHIFT_POSITION) != 0 {
            reads.push(s.adj.storage());
        }
        if f & FUSED_SHIFT_DEN != 0 {
            writes.push(s.den_groups.storage());
        }
        if f & FUSED_SHIFT_POSITION != 0 {
            writes.push(s.grad_groups.storage());
        }
        let (ctl, g) = (self.ctl(), self.g);
        self.k.launch(self.ctx, &reads, &writes, |stream| {
            // SAFETY: as in `rows_inv`, and `shift_pre` wrote `T` (and `T_y` under
            // `FUSED_SHIFT_DEN` or `FUSED_SHIFT_POSITION`) of this stack; the launch is the
            // one `shift_rows_inv` documents.
            unsafe {
                match self.sh.nx {
                    Side::N64 => self
                        .k
                        .module
                        .fused_shift_rows_inv_64(stream, config, p, ctl, g, cfg),
                    Side::N128 => self
                        .k
                        .module
                        .fused_shift_rows_inv_128(stream, config, p, ctl, g, cfg),
                    Side::N256 => self
                        .k
                        .module
                        .fused_shift_rows_inv_256(stream, config, p, ctl, g, cfg),
                    Side::N512 => self
                        .k
                        .module
                        .fused_shift_rows_inv_512(stream, config, p, ctl, g, cfg),
                }
            }
        })
    }

    /// What a pass C of the shift reads and writes for its object and probe terms under
    /// `flags`: the views, the object and `T`, and the terms' scratch.
    fn shift_terms<'s>(
        &'s self,
        sh: &'s Shifted<'_>,
        flags: u32,
    ) -> (Vec<&'s RawDeviceBuffer>, Vec<&'s RawDeviceBuffer>) {
        let s = sh.scratch;
        let reads = vec![
            self.inputs.views.addr().storage(),
            self.stack.storage(),
            self.inputs.obj.storage(),
            s.t.storage(),
        ];
        let mut writes = vec![];
        if flags & FUSED_SHIFT_PROBE != 0 {
            writes.push(s.adj.storage());
        }
        if flags & FUSED_SHIFT_OBJ != 0 {
            writes.extend([s.obj.storage(), s.power.storage()]);
        }
        (reads, writes)
    }

    /// The object and probe terms `cfg` names of the waves `psi` holds, with the shifted
    /// probe.
    fn shift_sums(
        &self,
        sh: &Shifted<'_>,
        psi: &DeviceBuffer<C32>,
        p: FusedPtrs,
        cfg: FusedCfg,
    ) -> Result<(), Error> {
        let cfg = self.walk(cfg);
        let config = self.sh.rows(&self.g, false)?;
        let (mut reads, writes) = self.shift_terms(sh, cfg.flags);
        reads.push(psi.storage());
        let (ctl, g) = (self.ctl(), self.g);
        self.k.launch(self.ctx, &reads, &writes, |stream| {
            let m = &self.k.module;
            // SAFETY: as in `shift_rows_inv`, with `psi` checked to hold the stack's waves;
            // the launch is the one `shift_sums` documents.
            unsafe {
                match self.sh.nx {
                    Side::N64 => m.fused_shift_sums_64(stream, config, p, ctl, g, cfg),
                    Side::N128 => m.fused_shift_sums_128(stream, config, p, ctl, g, cfg),
                    Side::N256 => m.fused_shift_sums_256(stream, config, p, ctl, g, cfg),
                    Side::N512 => m.fused_shift_sums_512(stream, config, p, ctl, g, cfg),
                }
            }
        })
    }

    /// `G` from the `X` that [`Self::shift_rows_inv`] left.
    fn shift_post(&self, sh: &Shifted<'_>, p: FusedPtrs) -> Result<(), Error> {
        let config = self.sh.shift_cols(self.g.nprobe, self.g.nf_max)?;
        let reads = [sh.model.offsets.storage(), self.stack.storage()];
        let writes = [sh.scratch.adj.storage()];
        let (ctl, g, cfg) = (self.ctl(), self.g, self.walk(FusedCfg::default()));
        self.k.launch(self.ctx, &reads, &writes, |stream| {
            // SAFETY: as in `shift_pre`, and `shift_rows_inv` wrote `X` of this stack; the
            // launch is the one `shift_post` documents.
            unsafe {
                match self.sh.ny {
                    Side::N64 => self
                        .k
                        .module
                        .fused_shift_post_64(stream, config, p, ctl, g, cfg),
                    Side::N128 => self
                        .k
                        .module
                        .fused_shift_post_128(stream, config, p, ctl, g, cfg),
                    Side::N256 => self
                        .k
                        .module
                        .fused_shift_post_256(stream, config, p, ctl, g, cfg),
                    Side::N512 => self
                        .k
                        .module
                        .fused_shift_post_512(stream, config, p, ctl, g, cfg),
                }
            }
        })
    }

    /// `out[f0 + fl]` from `ngroup` position groups per frame.
    fn position_groups(
        &self,
        groups: &DeviceBuffer<PositionPartial>,
        ngroup: u32,
        out: &mut DeviceBuffer<PositionPartial>,
    ) -> Result<(), Error> {
        let config = grid_1d(self.g.nf_max as usize)?;
        let reads = [groups.storage(), self.stack.storage()];
        let (src, ctl, g) = (dptr(groups), self.ctl(), self.g);
        self.k.launch(self.ctx, &reads, &[out.storage()], |stream| {
            // SAFETY: `Run::new` checked the groups and `fused_project` `out` against
            // `g`.
            unsafe {
                self.k
                    .module
                    .position_groups(stream, config, src, ngroup, dptr(out), ctl, g)
            }
        })
    }

    /// Adds the stack's `G` and object power to `sums`.
    fn shift_probe_sums(
        &self,
        sh: &Shifted<'_>,
        p: FusedPtrs,
        sums: &mut ShiftedProbeSums,
    ) -> Result<(), Error> {
        debug_assert!(self.walked_down(), "shift_probe_sums after an up walk");
        let config = grid_1d(self.g.nprobe as usize * self.inputs.views.dims().n())?;
        let reads = [
            sh.scratch.adj.storage(),
            self.inputs.views.addr().storage(),
            self.inputs.obj.storage(),
            self.stack.storage(),
        ];
        let (ptrs, ctl, g) = (sums.ptrs(), self.ctl(), self.g);
        self.k.launch(self.ctx, &reads, &sums.storages(), |stream| {
            // SAFETY: `shift_post` wrote `G` of this stack; `fused_project` checked `sums`.
            unsafe {
                self.k
                    .module
                    .fused_shift_probe_sums(stream, config, p, ptrs, ctl, g)
            }
        })
    }

    /// The object sums from the shift terms of [`Self::shift_rows_inv`].
    fn gather_shifted(
        &self,
        sh: &Shifted<'_>,
        p: FusedPtrs,
        cfg: FusedCfg,
        sums: &mut UpdateSums,
    ) -> Result<(), Error> {
        let views = self.inputs.views;
        let [start, len, list] = views.tile_storages();
        let reads = [
            sh.scratch.obj.storage(),
            sh.scratch.power.storage(),
            views.addr().storage(),
            start,
            len,
            list,
            self.stack.storage(),
        ];
        let config = views.dims().stack_tile_grid(self.g.nf_max)?;
        let (tiles, ctl, g) = (views.tile_ptrs(), self.ctl(), self.g);
        self.k.launch(self.ctx, &reads, &sums.storages(), |stream| {
            // SAFETY: as `gather`, with `shift_obj` and `shift_power` written by
            // `shift_rows_inv` under `FUSED_SHIFT_OBJ` for this stack.
            unsafe {
                self.k.module.fused_gather_shifted(
                    stream,
                    config,
                    p,
                    tiles,
                    sums.ptrs(),
                    ctl,
                    g,
                    cfg,
                )
            }
        })
    }
}
