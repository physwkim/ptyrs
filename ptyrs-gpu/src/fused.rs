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

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use ptyrs_kernels::FFT_EPT;
use ptyrs_kernels::fused::{
    FUSED_COL_THREADS, FUSED_COLS, FUSED_DM, FUSED_GRAD, FUSED_LLK, FUSED_PROJECT,
    FUSED_ROW_THREADS, FUSED_SCALE, FUSED_SCALE_SUMS, FUSED_TO_PSI, FusedCfg, FusedPtrs,
    col_slot_bytes,
};

use crate::ops::grid_1d;
use crate::{
    C32, Error, Kernels, LlkPartial, ScalePartial, StackCtl, StackGeom, StackRef, UpdateSums,
    ViewDims, Views, block_fft, check_len, dptr,
};

/// Dynamic shared memory a block gets without opting in.
const SMEM_MAX: u32 = 48 * 1024;
/// Most blocks along grid y.
const GRID_Y_MAX: u64 = 65535;

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
    /// Rows per row-pass block, of `nx / FFT_EPT` threads each.
    pub(crate) rows: u32,
    pub(crate) row_smem: u32,
    pub(crate) col_threads: u32,
    /// Column-pass shared memory past the reduction slots: counts tile and FFTs.
    pub(crate) col_tile_fft: u32,
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
                    if [f.ept, i.ept] == [FFT_EPT; 2]
                        && [f.threads_per_fft, i.threads_per_fft] == [n / FFT_EPT; 2] =>
                {
                    Ok((side, f.smem_per_fft.max(i.smem_per_fft)))
                }
                _ => Err(bad(format!("no {n}-point in-kernel FFT"))),
            }
        };
        let ((side_y, sy), (side_x, sx)) = (side(ny)?, side(nx)?);
        let rows = FUSED_ROW_THREADS / (nx / FFT_EPT);
        let nslot = FUSED_COLS * (ny / FFT_EPT);
        let col_threads = d.nmode as u64 * nslot as u64;
        if col_threads > FUSED_COL_THREADS as u64 {
            return Err(bad(format!("{col_threads} threads per column block")));
        }
        let col_threads = col_threads as u32;
        let s = Self {
            ny: side_y,
            nx: side_x,
            rows,
            row_smem: rows * sx,
            col_threads,
            col_tile_fft: 2 * FUSED_COLS * ny + (col_threads / (ny / FFT_EPT)) * sy.max(4 * ny),
        };
        // The column pass's most shared memory is under every flag that adds to it.
        if s.col_smem(FUSED_SCALE_SUMS) > SMEM_MAX || s.row_smem > SMEM_MAX {
            return Err(bad("shared memory per block exceeds 48 KiB".into()));
        }
        Ok(s)
    }

    fn col_smem(&self, flags: u32) -> u32 {
        col_slot_bytes(flags, self.ny as u32) + self.col_tile_fft
    }

    /// `rows_fwd` (one block row per view) or `rows_inv` (per frame).
    pub(crate) fn rows(&self, g: &StackGeom, per_view: bool) -> Result<LaunchConfig, Error> {
        let planes = g.nf_max as u64 * if per_view { g.nmode as u64 } else { 1 };
        if planes > GRID_Y_MAX {
            return Err(Error::Shape(format!("{planes} planes exceed grid y")));
        }
        Ok(LaunchConfig {
            grid_dim: (g.ny / self.rows, planes as u32, 1),
            block_dim: (g.nx / FFT_EPT, self.rows, 1),
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
}

impl FusedBufs {
    /// Scratch for `dims`, whose frame sides must be in [`FUSED_SIDES`] and whose modes must
    /// fit one column block of [`FUSED_COL_THREADS`] threads: at most `1024 / ny` of them.
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
        })
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
}

/// Per-frame sums of the forward far field's `calc` against the counts that
/// [`Kernels::fused_project`] and [`Kernels::fused_llk`] write; `Default` writes none.
#[derive(Default)]
pub struct FarSums<'a> {
    /// The LLK partials `llk[fl]` for `fl < nf`, `[nf_max]`; entries past `nf` not written.
    pub llk: Option<&'a mut DeviceBuffer<LlkPartial>>,
    /// The scale sums `scale[f0 + fl]` for `fl < nf`, `[nframe]`; no other entry written.
    pub scale: Option<&'a mut DeviceBuffer<ScalePartial>>,
}

impl FarSums<'_> {
    /// The column-pass flags that write these sums.
    fn flags(&self) -> u32 {
        let llk = if self.llk.is_some() { FUSED_LLK } else { 0 };
        let scale = if self.scale.is_some() {
            FUSED_SCALE_SUMS
        } else {
            0
        };
        llk | scale
    }
}

/// The update sums [`Kernels::fused_project`] adds to; `Default` adds to neither.
#[derive(Default)]
pub struct FusedSums<'a> {
    /// `[nobj, nyo, nxo]`, added as [`Kernels::psi_to_obj`] adds them.
    pub obj: Option<&'a mut UpdateSums>,
    /// `[nprobe, ny, nx]`, added by [`Kernels::psi_to_probe`] against the object read.
    pub probe: Option<&'a mut UpdateSums>,
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
    /// not change a bit; only tiles the stack touches are read and written). Up to seven
    /// launches; a stack with `nf == 0` writes nothing but `bufs`.
    pub fn fused_project(
        &self,
        ctx: &ExecutionContext,
        inputs: &FusedInputs<'_>,
        stack: StackRef<'_>,
        exit: Exit<'_>,
        bufs: &mut FusedBufs,
        sums: FusedSums<'_>,
    ) -> Result<(), Error> {
        let run = Run::new(self, ctx, *inputs, stack, bufs)?;
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
        } = sums;
        if let Some(sums) = &obj_sums {
            sums.check(d.obj_len())?;
        }
        if let Some(sums) = &probe_sums {
            sums.check(d.nprobe as usize * d.n())?;
        }
        let psi = psi.as_deref();
        let p = FusedPtrs {
            psi: psi.map_or(std::ptr::null_mut(), dptr),
            ..run.ptrs(bufs)
        };
        let dm_flag = if dm { FUSED_DM } else { 0 };

        if obj_sums.is_some() {
            run.probe_terms(bufs, p)?;
        }
        run.rows_fwd(
            bufs,
            psi,
            p,
            FusedCfg {
                flags: dm_flag,
                alpha,
            },
        )?;
        let cfg = FusedCfg {
            flags: pass | run.model_flags() | far.flags(),
            alpha,
        };
        run.cols(bufs, p, cfg)?;
        run.far_sums(bufs, far)?;
        let to_psi = if psi.is_some() { FUSED_TO_PSI } else { 0 };
        let cfg = FusedCfg {
            flags: dm_flag | to_psi,
            alpha,
        };
        run.rows_inv(bufs, psi, p, cfg)?;
        let waves = psi.unwrap_or(&bufs.work);
        if let Some(sums) = probe_sums {
            self.psi_to_probe(ctx, inputs.views, stack, waves, inputs.obj, sums)?;
        }
        if let Some(sums) = obj_sums {
            run.gather(bufs, waves, p, cfg, sums)?;
        }
        Ok(())
    }

    /// The stack's far sums of the far field of `PO`, the one `obj_probe_to_psi` and the FFT
    /// give, the LLK partials as [`Self::llk`] gives them. Up to four launches; writes
    /// nothing but `far` and `bufs`.
    pub fn fused_llk(
        &self,
        ctx: &ExecutionContext,
        inputs: &FusedInputs<'_>,
        stack: StackRef<'_>,
        bufs: &mut FusedBufs,
        far: FarSums<'_>,
    ) -> Result<(), Error> {
        let run = Run::new(self, ctx, *inputs, stack, bufs)?;
        run.check_far(&far)?;
        let p = run.ptrs(bufs);
        run.rows_fwd(bufs, None, p, FusedCfg::default())?;
        let cfg = FusedCfg {
            flags: run.model_flags() | far.flags(),
            alpha: 0.0,
        };
        run.cols(bufs, p, cfg)?;
        run.far_sums(bufs, far)
    }
}

/// One fused call, checked: what every pass launch of it needs.
struct Run<'a> {
    k: &'a Kernels,
    ctx: &'a ExecutionContext,
    inputs: FusedInputs<'a>,
    stack: StackRef<'a>,
    g: StackGeom,
    sh: Shape,
}

impl<'a> Run<'a> {
    fn new(
        k: &'a Kernels,
        ctx: &'a ExecutionContext,
        inputs: FusedInputs<'a>,
        stack: StackRef<'a>,
        bufs: &FusedBufs,
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
        bufs.check(d, &g)?;
        Ok(Self {
            k,
            ctx,
            inputs,
            stack,
            g,
            sh,
        })
    }

    /// Every address but `psi`, which is null.
    fn ptrs(&self, bufs: &FusedBufs) -> FusedPtrs {
        FusedPtrs {
            obj: dptr(self.inputs.obj),
            probe: dptr(self.inputs.probe),
            views: dptr(self.inputs.views.addr()),
            counts: dptr(self.inputs.counts),
            psi: std::ptr::null_mut(),
            work: dptr(&bufs.work),
            probe_q: dptr(&bufs.probe_q),
            probe_power: dptr(&bufs.probe_power),
            groups: dptr(&bufs.groups),
            scale: (self.inputs.model.scale)
                .map_or(std::ptr::null(), |b| dptr::<f32, f32>(b).cast_const()),
            scale_groups: dptr(&bufs.scale_groups),
        }
    }

    /// The column-pass flags of the model.
    fn model_flags(&self) -> u32 {
        if self.inputs.model.scale.is_some() {
            FUSED_SCALE
        } else {
            0
        }
    }

    fn check_far(&self, far: &FarSums<'_>) -> Result<(), Error> {
        if let Some(out) = &far.llk {
            check_len("llk", out, self.g.nf_max as usize)?;
        }
        if let Some(out) = &far.scale {
            check_len("scale sums", out, self.g.nframe as usize)?;
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

    /// Pass B on `bufs.work`; reads the scale under `FUSED_SCALE`, writes `bufs.groups`
    /// under `FUSED_LLK` and `bufs.scale_groups` under `FUSED_SCALE_SUMS`.
    fn cols(&self, bufs: &FusedBufs, p: FusedPtrs, cfg: FusedCfg) -> Result<(), Error> {
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
        if cfg.flags & FUSED_LLK != 0 {
            writes.push(bufs.groups.storage());
        }
        if cfg.flags & FUSED_SCALE_SUMS != 0 {
            writes.push(bufs.scale_groups.storage());
        }
        let (ctl, g) = (self.ctl(), self.g);
        self.k.launch(self.ctx, &reads, &writes, |stream| {
            // SAFETY: as in `rows_fwd`; the launch is the one `cols` documents.
            unsafe {
                match self.sh.ny {
                    Side::N64 => self.k.module.fused_cols_64(stream, config, p, ctl, g, cfg),
                    Side::N128 => self.k.module.fused_cols_128(stream, config, p, ctl, g, cfg),
                    Side::N256 => self.k.module.fused_cols_256(stream, config, p, ctl, g, cfg),
                    Side::N512 => self.k.module.fused_cols_512(stream, config, p, ctl, g, cfg),
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
}
