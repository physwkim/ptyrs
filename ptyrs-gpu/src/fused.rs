//! M2 fused stack passes: a stack's exit waves, Fourier projection, LLK and object sums in
//! three in-kernel FFT passes over [`FusedBufs`], in place of the M1 sequence
//! `obj_probe_to_psi`, 2D FFT, `apply_amplitude` (and `llk`), inverse FFT, (`dm_reflect`,
//! `add_c32`), `psi_to_obj`. The conventions are those of the M1 operators (module `ops`);
//! the arithmetic per pixel is theirs, the FFTs and the sum order of the object update are
//! not, so results agree with M1 to rounding, not bit for bit.
//!
//! Frame sides are 128 or 256 each; views must be canonical, which [`Views::new`] checks.
//! The probe update stays M1: [`Kernels::psi_to_probe`] on [`FusedBufs::work`] after an AP
//! projection, or on `psi` after a DM one.

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use ptyrs_kernels::FFT_EPT;
use ptyrs_kernels::fused::{
    FUSED_COLS, FUSED_CONTRIB, FUSED_DM, FUSED_LLK, FUSED_PROJECT, FUSED_ROW_THREADS, FUSED_TO_PSI,
    FusedCfg, FusedPtrs,
};

use crate::ops::grid_1d;
use crate::{
    C32, Error, Kernels, LlkPartial, StackCtl, StackGeom, StackRef, UpdateSums, ViewDims, Views,
    block_fft, check_len, dptr,
};

/// Dynamic shared memory a block gets without opting in.
const SMEM_MAX: u32 = 48 * 1024;
/// Most blocks along grid y.
const GRID_Y_MAX: u64 = 65535;

/// A frame side the fused FFT passes have kernels for.
#[derive(Clone, Copy)]
enum Side {
    N128,
    N256,
}

/// Launch shapes of the fused passes for one frame and mode count.
#[derive(Clone, Copy)]
struct Shape {
    /// Sides along y (the column pass) and x (the row passes).
    ny: Side,
    nx: Side,
    /// Rows per row-pass block, of `nx / FFT_EPT` threads each.
    rows: u32,
    row_smem: u32,
    col_threads: u32,
    col_smem: u32,
}

impl Shape {
    fn of(d: &ViewDims) -> Result<Self, Error> {
        d.check()?;
        let [ny, nx] = d.frame;
        let bad = |what: String| Error::Shape(format!("fused passes, {d:?}: {what}"));
        // The side's kernels and its shared memory per FFT, the larger of the two directions.
        let side = |n: u32| {
            let side = match n {
                128 => Some(Side::N128),
                256 => Some(Side::N256),
                _ => None,
            };
            match (side, block_fft(n, false), block_fft(n, true)) {
                (Some(side), Some(f), Some(i))
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
        if col_threads > 1024 {
            return Err(bad(format!("{col_threads} threads per column block")));
        }
        let col_threads = col_threads as u32;
        let s = Self {
            ny: side_y,
            nx: side_x,
            rows,
            row_smem: rows * sx,
            col_threads,
            col_smem: 16 * nslot
                + 2 * FUSED_COLS * ny
                + (col_threads / (ny / FFT_EPT)) * sy.max(4 * ny),
        };
        if s.col_smem > SMEM_MAX || s.row_smem > SMEM_MAX {
            return Err(bad("shared memory per block exceeds 48 KiB".into()));
        }
        Ok(s)
    }

    /// `rows_fwd` (one block row per view) or `rows_inv` (per frame).
    fn rows(&self, g: &StackGeom, per_view: bool) -> Result<LaunchConfig, Error> {
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

    fn cols(&self, g: &StackGeom) -> Result<LaunchConfig, Error> {
        if g.nf_max as u64 > GRID_Y_MAX {
            return Err(Error::Shape(format!("{} frames exceed grid y", g.nf_max)));
        }
        Ok(LaunchConfig {
            grid_dim: (g.nx / FUSED_COLS, g.nf_max, 1),
            block_dim: (self.col_threads, 1, 1),
            shared_mem_bytes: self.col_smem,
        })
    }
}

/// Per-stack scratch of the fused passes for stacks of up to `nf_max` frames of one scan
/// geometry; one set serves every stack in turn.
pub struct FusedBufs {
    /// `[nf_max * nmode, ny, nx]`.
    work: DeviceBuffer<C32>,
    /// `[nf_max * nobj, ny, nx]`.
    contrib: DeviceBuffer<C32>,
    /// `[ny, nx]`.
    probe_power: DeviceBuffer<f32>,
    /// `[nf_max, nx / FUSED_COLS]`.
    groups: DeviceBuffer<LlkPartial>,
}

impl FusedBufs {
    /// Scratch for `dims`, whose frame sides must be 128 or 256 and whose mode count must fit
    /// one column block (8 modes of 128 rows, 4 of 256).
    pub fn new(stream: &CudaStream, dims: &ViewDims, nf_max: u32) -> Result<Self, Error> {
        Shape::of(dims)?;
        if nf_max == 0 {
            return Err(Error::Shape("nf_max is zero".into()));
        }
        let (n, nf) = (dims.n(), nf_max as usize);
        Ok(Self {
            work: DeviceBuffer::zeroed(stream, nf * dims.nmode as usize * n)?,
            contrib: DeviceBuffer::zeroed(stream, nf * dims.nobj as usize * n)?,
            probe_power: DeviceBuffer::zeroed(stream, n)?,
            groups: DeviceBuffer::zeroed(stream, nf * (dims.frame[1] / FUSED_COLS) as usize)?,
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
        check_len("contrib", &self.contrib, nf * d.nobj as usize * n)?;
        check_len("probe_power", &self.probe_power, n)?;
        check_len("groups", &self.groups, nf * (g.nx / FUSED_COLS) as usize)
    }
}

/// What the fused passes read of the scan: its views, the whole scan's counts `[nframe,
/// ny, nx]`, the object `[nobj, nyo, nxo]` and the probe `[nprobe, ny, nx]`.
#[derive(Clone, Copy)]
pub struct FusedInputs<'a> {
    pub views: &'a Views,
    pub counts: &'a DeviceBuffer<u16>,
    pub obj: &'a DeviceBuffer<C32>,
    pub probe: &'a DeviceBuffer<C32>,
}

/// Which iteration [`Kernels::fused_project`] runs and where its new exit waves go. `psi`
/// is a DM stack's exit waves, `[nf_max * nmode, ny, nx]` in the M1 layout.
pub enum Exit<'a> {
    /// AP: the projected exit waves `P(PO)` stay in [`FusedBufs::work`]. With `llk`, the
    /// LLK partials of the forward far field, which is the one [`Kernels::llk`] reads after
    /// `obj_probe_to_psi` and the FFT, go to `llk[fl]` for `fl < nf`.
    Ap {
        llk: Option<&'a mut DeviceBuffer<LlkPartial>>,
    },
    /// The first DM cycle, from PyNX's start `psi = PO`: `psi = P(PO)`.
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
    /// then with `obj_sums` the object-update sums of the new exit waves added as
    /// [`Self::psi_to_obj`] adds them (each object pixel over the stack's frames in
    /// ascending order, so the split into stacks does not change a bit; only tiles the
    /// stack touches are read and written). Up to six launches; a stack with `nf == 0`
    /// writes nothing but `bufs`.
    pub fn fused_project(
        &self,
        ctx: &ExecutionContext,
        inputs: &FusedInputs<'_>,
        stack: StackRef<'_>,
        exit: Exit<'_>,
        bufs: &mut FusedBufs,
        obj_sums: Option<&mut UpdateSums>,
    ) -> Result<(), Error> {
        let run = Run::new(self, ctx, *inputs, stack, bufs)?;
        let d = inputs.views.dims();
        let (psi, llk, dm, alpha) = match exit {
            Exit::Ap { llk } => (None, llk, false, 0.0),
            Exit::DmFirst { psi } => (Some(psi), None, false, 0.0),
            Exit::Dm { psi, alpha } => (Some(psi), None, true, alpha),
        };
        if let Some(psi) = &psi {
            check_len("psi", psi, run.g.nf_max as usize * d.nmode as usize * d.n())?;
        }
        if let Some(out) = &llk {
            check_len("llk", out, run.g.nf_max as usize)?;
        }
        if let Some(sums) = &obj_sums {
            sums.check(d.obj_len())?;
        }
        let psi = psi.as_deref();
        let p = FusedPtrs {
            psi: psi.map_or(std::ptr::null_mut(), dptr),
            ..run.ptrs(bufs)
        };
        let dm_flag = if dm { FUSED_DM } else { 0 };

        if obj_sums.is_some() {
            run.probe_power(bufs)?;
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
        let llk_flag = if llk.is_some() { FUSED_LLK } else { 0 };
        let cfg = FusedCfg {
            flags: FUSED_PROJECT | llk_flag,
            alpha,
        };
        run.cols(bufs, p, cfg)?;
        if let Some(out) = llk {
            run.llk_groups(bufs, out)?;
        }
        let to_psi = if psi.is_some() { FUSED_TO_PSI } else { 0 };
        let contrib = if obj_sums.is_some() { FUSED_CONTRIB } else { 0 };
        let cfg = FusedCfg {
            flags: dm_flag | to_psi | contrib,
            alpha,
        };
        run.rows_inv(bufs, psi, p, cfg)?;
        if let Some(sums) = obj_sums {
            run.gather(bufs, p, sums)?;
        }
        Ok(())
    }

    /// The stack's LLK partials, as `obj_probe_to_psi`, the FFT and [`Self::llk`] give them:
    /// `out[fl]` for `fl < nf`, entries past `nf` not written. Three launches; writes
    /// nothing but `out` and `bufs`.
    pub fn fused_llk(
        &self,
        ctx: &ExecutionContext,
        inputs: &FusedInputs<'_>,
        stack: StackRef<'_>,
        bufs: &mut FusedBufs,
        out: &mut DeviceBuffer<LlkPartial>,
    ) -> Result<(), Error> {
        let run = Run::new(self, ctx, *inputs, stack, bufs)?;
        check_len("out", out, run.g.nf_max as usize)?;
        let p = run.ptrs(bufs);
        run.rows_fwd(bufs, None, p, FusedCfg::default())?;
        let cfg = FusedCfg {
            flags: FUSED_LLK,
            alpha: 0.0,
        };
        run.cols(bufs, p, cfg)?;
        run.llk_groups(bufs, out)
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
            contrib: dptr(&bufs.contrib),
            probe_power: dptr(&bufs.probe_power),
            groups: dptr(&bufs.groups),
        }
    }

    fn ctl(&self) -> *const StackCtl {
        self.stack.ctl() as *const StackCtl
    }

    fn probe_power(&self, bufs: &FusedBufs) -> Result<(), Error> {
        let d = self.inputs.views.dims();
        let config = grid_1d(d.n())?;
        let (probe, pp) = (self.inputs.probe, &bufs.probe_power);
        let (nprobe, n) = (d.nprobe, d.n() as u32);
        self.k
            .launch(self.ctx, &[probe.storage()], &[pp.storage()], |stream| {
                // SAFETY: `probe` holds `nprobe` planes of `n`, `pp` one.
                unsafe {
                    self.k.module.fused_probe_power(
                        stream,
                        config,
                        dptr(probe),
                        dptr(pp),
                        nprobe,
                        n,
                    )
                }
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
                        Side::N128 => self
                            .k
                            .module
                            .fused_rows_fwd_128(stream, config, p, ctl, g, cfg),
                        Side::N256 => self
                            .k
                            .module
                            .fused_rows_fwd_256(stream, config, p, ctl, g, cfg),
                    }
                }
            })
    }

    /// Pass B on `bufs.work`; writes `bufs.groups` under `FUSED_LLK`.
    fn cols(&self, bufs: &FusedBufs, p: FusedPtrs, cfg: FusedCfg) -> Result<(), Error> {
        let config = self.sh.cols(&self.g)?;
        let mut reads = vec![self.inputs.counts.storage(), self.stack.storage()];
        let mut writes = vec![];
        if cfg.flags & FUSED_PROJECT != 0 {
            writes.push(bufs.work.storage());
        } else {
            reads.push(bufs.work.storage());
        }
        if cfg.flags & FUSED_LLK != 0 {
            writes.push(bufs.groups.storage());
        }
        let (ctl, g) = (self.ctl(), self.g);
        self.k.launch(self.ctx, &reads, &writes, |stream| {
            // SAFETY: as in `rows_fwd`; the launch is the one `cols` documents.
            unsafe {
                match self.sh.ny {
                    Side::N128 => self.k.module.fused_cols_128(stream, config, p, ctl, g, cfg),
                    Side::N256 => self.k.module.fused_cols_256(stream, config, p, ctl, g, cfg),
                }
            }
        })
    }

    fn llk_groups(
        &self,
        bufs: &FusedBufs,
        out: &mut DeviceBuffer<LlkPartial>,
    ) -> Result<(), Error> {
        let ngroup = self.g.nx / FUSED_COLS;
        self.k
            .llk_groups(self.ctx, self.stack, self.g, &bufs.groups, ngroup, out)
    }

    /// Pass C from `bufs.work` to `psi` (under `FUSED_TO_PSI`) or back to `bufs.work`;
    /// reads `psi` and the object under `FUSED_DM`, writes `bufs.contrib` under
    /// `FUSED_CONTRIB`.
    fn rows_inv(
        &self,
        bufs: &FusedBufs,
        psi: Option<&DeviceBuffer<C32>>,
        p: FusedPtrs,
        cfg: FusedCfg,
    ) -> Result<(), Error> {
        let config = self.sh.rows(&self.g, false)?;
        let mut reads = vec![
            self.inputs.probe.storage(),
            self.inputs.views.addr().storage(),
            self.stack.storage(),
        ];
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
        }
        if cfg.flags & FUSED_CONTRIB != 0 {
            writes.push(bufs.contrib.storage());
        }
        let (ctl, g) = (self.ctl(), self.g);
        self.k.launch(self.ctx, &reads, &writes, |stream| {
            // SAFETY: as in `rows_fwd`; the launch is the one `rows_inv` documents.
            unsafe {
                match self.sh.nx {
                    Side::N128 => self
                        .k
                        .module
                        .fused_rows_inv_128(stream, config, p, ctl, g, cfg),
                    Side::N256 => self
                        .k
                        .module
                        .fused_rows_inv_256(stream, config, p, ctl, g, cfg),
                }
            }
        })
    }

    /// The object sums from `bufs.contrib` and `bufs.probe_power`.
    fn gather(&self, bufs: &FusedBufs, p: FusedPtrs, sums: &mut UpdateSums) -> Result<(), Error> {
        let views = self.inputs.views;
        let [start, len, list] = views.tile_storages();
        let reads = [
            bufs.contrib.storage(),
            bufs.probe_power.storage(),
            views.addr().storage(),
            start,
            len,
            list,
            self.stack.storage(),
        ];
        let config = views.dims().tile_grid();
        let (tiles, ctl, g) = (views.tile_ptrs(), self.ctl(), self.g);
        self.k.launch(self.ctx, &reads, &sums.storages(), |stream| {
            // SAFETY: as `psi_to_obj`, with `contrib` and `probe_power` checked with the
            // rest of `bufs`; the tile lists were built from these views.
            unsafe {
                self.k
                    .module
                    .fused_gather(stream, config, p, tiles, sums.ptrs(), ctl, g)
            }
        })
    }
}
