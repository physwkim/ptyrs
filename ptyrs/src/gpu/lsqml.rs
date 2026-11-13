//! Checked launches of the LSQML kernels (`ptyrs_kernels::lsqml`), pty-chi's least-squares
//! maximum-likelihood step (ptyrs::engine `lsqml`) on the far-field Fourier path, with the
//! fused row passes of [`crate::gpu::FusedBufs`]' kernels around its column pass.
//!
//! # One LSQML batch in these calls
//!
//! A batch is a run of stacks whose table entries hold epoch positions, as in
//! [`crate::gpu::BhKernels`]; each stack has its own `work`. Per stack [`LsqmlKernels::forward`],
//! then for a Gaussian cost [`LsqmlKernels::chi`]; for a Poisson cost
//! [`LsqmlKernels::icalc`] per stack, [`LsqmlKernels::far`] over the batch, then
//! [`LsqmlKernels::chi`] per stack. With zeroed update sums, per stack
//! [`Kernels::psi_to_obj`] and [`Kernels::psi_to_probe`] of `work`, then
//! [`LsqmlKernels::dir_obj`] and the probe sums' `num` scaled by `1 / nb`; per stack
//! [`LsqmlKernels::dots`], then [`LsqmlKernels::step`] and [`LsqmlKernels::update`], which
//! consumes the sums. An epoch starts with [`LsqmlKernels::start`] on a new run, and
//! [`LsqmlKernels::precondition`] on pty-chi's schedule. The steps never leave the device.
//!
//! Under the position model the epoch's frames move as [`crate::gpu::PositionKernels`] lays
//! out: per stack [`Kernels::shifted_rows_fwd`] and its moved probe take the place of
//! [`LsqmlKernels::forward`], [`Kernels::shifted_psi_sums`] and then
//! [`Kernels::shifted_probe_sums`] that of the update sums, and [`LsqmlChi::Moved`] gives the
//! dots the moved probe.

use std::ops::Range;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig, RawDeviceBuffer};
use ptyrs_kernels::fused::{FUSED_COLS, FusedCfg, FusedPtrs};
use ptyrs_kernels::grad::ObjProbePtrs;
pub use ptyrs_kernels::lsqml::LSQML_REC;
use ptyrs_kernels::lsqml::{
    LSQML_CHI, LSQML_DOTS, LSQML_ICALC, LSQML_POISSON, LSQML_XI, LsqmlCols, LsqmlDots, LsqmlFar,
    LsqmlPre, lsqml_kernels,
};
use ptyrs_kernels::pixel::PIXEL_THREADS;

use crate::gpu::fused::{Shape, Side};
use crate::gpu::ops::grid_1d;
use crate::gpu::position::{GRID_Y_MAX, ONE_BLOCK, chunk_grid, chunks, same_dims, span};
use crate::gpu::{
    BhScan, C32, Error, Kernels, StackCtl, StackGeom, StackRef, UpdateSums, ViewAddr, ViewDims,
    Views, check_len, dptr,
};

/// The LSQML kernels, bound to the module a [`Kernels`] loaded.
#[derive(Clone, Debug)]
pub struct LsqmlKernels {
    base: Kernels,
    module: lsqml_kernels::LoadedModule,
}

/// The noise model of LSQML's residual and far-field step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LsqmlNoise {
    /// pty-chi's Gaussian model: `a_far = 1/2`.
    Gaussian,
    /// pty-chi's Poisson model: `a_far` solved per frame.
    Poisson,
}

/// LSQML's device state over one scan, every buffer sized by its dims: `a_far` by scan frame
/// and by epoch position, the running mean of the Poisson passes, the object
/// preconditioner, its maximum and the brightest frame's photons, the per-chunk sums, the
/// step records by epoch position (`(a_far, a_obj, a_probe)`, [`LSQML_REC`] floats each) and
/// the last batch's object and probe steps.
pub struct LsqmlScan {
    dims: ViewDims,
    a_scan: DeviceBuffer<f32>,
    a_pos: DeviceBuffer<f32>,
    prev: DeviceBuffer<f32>,
    pre: DeviceBuffer<f32>,
    pre_max: DeviceBuffer<u32>,
    count_max: DeviceBuffer<u64>,
    xi: DeviceBuffer<f32>,
    dots: DeviceBuffer<f32>,
    rec: DeviceBuffer<f32>,
    step: DeviceBuffer<f32>,
}

impl LsqmlScan {
    /// Zeroed state for a scan of `dims`; [`LsqmlKernels::start`] begins a run on it.
    pub fn zeroed(stream: &CudaStream, dims: &ViewDims) -> Result<Self, Error> {
        dims.check()?;
        let nframe = dims.nframe as usize;
        let parts = nframe * chunks(dims.n());
        Ok(Self {
            dims: *dims,
            a_scan: DeviceBuffer::zeroed(stream, nframe)?,
            a_pos: DeviceBuffer::zeroed(stream, nframe)?,
            prev: DeviceBuffer::zeroed(stream, 1)?,
            pre: DeviceBuffer::zeroed(stream, dims.obj_len())?,
            pre_max: DeviceBuffer::zeroed(stream, 1)?,
            count_max: DeviceBuffer::zeroed(stream, 1)?,
            xi: DeviceBuffer::zeroed(stream, LSQML_XI as usize * parts)?,
            dots: DeviceBuffer::zeroed(stream, LSQML_DOTS as usize * parts)?,
            rec: DeviceBuffer::zeroed(stream, LSQML_REC as usize * nframe)?,
            step: DeviceBuffer::zeroed(stream, 2)?,
        })
    }

    /// The step records by epoch position, [`LSQML_REC`] floats each: `a_far`, `a_obj`,
    /// `a_probe` of the position's last step.
    pub fn records(&self) -> &DeviceBuffer<f32> {
        &self.rec
    }

    fn check(&self, dims: &ViewDims) -> Result<(), Error> {
        same_dims(&self.dims, dims)
    }

    fn far(&self, order: &DeviceBuffer<u32>) -> LsqmlFar {
        LsqmlFar {
            order: dptr::<u32, u32>(order).cast_const(),
            a_scan: dptr(&self.a_scan),
            a_pos: dptr(&self.a_pos),
            prev: dptr(&self.prev),
        }
    }
}

/// A stack's exit-wave update `chi` ([`LsqmlKernels::chi`]) for [`LsqmlKernels::dots`]; under
/// the position model with the stack's moved probe `w`
/// ([`crate::gpu::PositionKernels::shift_w`]).
pub enum LsqmlChi<'a> {
    Unmoved(&'a DeviceBuffer<C32>),
    Moved {
        chi: &'a DeviceBuffer<C32>,
        w: &'a DeviceBuffer<C32>,
    },
}

/// Values of the object and of the probe.
fn lens(dims: &ViewDims) -> [usize; 2] {
    [dims.obj_len(), dims.nprobe as usize * dims.n()]
}

/// `n` as a kernel's element count.
fn to_u32(n: usize) -> Result<u32, Error> {
    u32::try_from(n).map_err(|_| Error::Shape(format!("{n} elements exceed u32")))
}

/// The object and probe a batch starts from, `x`, and their directions `d`, each `[object,
/// probe]`.
#[derive(Clone, Copy)]
pub struct LsqmlAt<'a> {
    pub x: [&'a DeviceBuffer<C32>; 2],
    pub d: [&'a DeviceBuffer<C32>; 2],
}

/// An array LSQML updates: the estimate and its update sums, whose `num` holds the direction.
pub struct LsqmlVar<'a> {
    pub x: &'a mut DeviceBuffer<C32>,
    pub sums: &'a mut UpdateSums,
}

impl LsqmlKernels {
    /// Binds the LSQML kernels of the cubin `kernels` already loaded; loads nothing.
    pub fn load(kernels: &Kernels) -> Result<Self, Error> {
        let module = lsqml_kernels::from_module(kernels.module.as_cuda_module().clone())?;
        Ok(Self {
            base: kernels.clone(),
            module,
        })
    }

    /// A new run on `st`: every frame's `a_far` 1/2, and the brightest frame's unmasked
    /// photons from `counts`, the scan's `[nframe, ny, nx]` counts, for
    /// [`Self::precondition`]'s first preconditioner.
    pub fn start(
        &self,
        ctx: &ExecutionContext,
        counts: &DeviceBuffer<u16>,
        st: &mut LsqmlScan,
    ) -> Result<(), Error> {
        let d = st.dims;
        let n = d.n();
        check_len("counts", counts, d.nframe as usize * n)?;
        crate::gpu::zero(ctx, &mut st.count_max)?;
        let config = LaunchConfig {
            grid_dim: (d.nframe, 1, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        self.base.launch(
            ctx,
            &[counts.storage()],
            &[st.count_max.storage()],
            |stream| {
                // SAFETY: `counts` holds `nframe` frames of `n` (`ViewDims::check` bounds it
                // by u32); one PIXEL_THREADS block per frame.
                unsafe {
                    self.module.lsqml_count_max(
                        stream,
                        config,
                        dptr::<u16, u16>(counts).cast_const(),
                        n as u32,
                        dptr(&st.count_max),
                    )
                }
            },
        )?;
        for buf in [&mut st.a_scan, &mut st.a_pos] {
            let config = grid_1d(d.nframe as usize)?;
            self.base.launch(ctx, &[], &[buf.storage()], |stream| {
                // SAFETY: `buf` holds `nframe` floats.
                unsafe { self.module.lsqml_half(stream, config, dptr(buf), d.nframe) }
            })?;
        }
        Ok(())
    }

    /// pty-chi's object preconditioner from the illumination of every view of `views` at
    /// `probe`: with `init`, from the [`Self::start`] photons, else from the last; and its
    /// maximum, for [`Self::dir_obj`].
    pub fn precondition(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        probe: &DeviceBuffer<C32>,
        st: &mut LsqmlScan,
        init: bool,
    ) -> Result<(), Error> {
        let d = *views.dims();
        st.check(&d)?;
        check_len("probe", probe, lens(&d)[1])?;
        crate::gpu::zero(ctx, &mut st.pre_max)?;
        let mut reads = vec![
            probe.storage(),
            views.addr().storage(),
            st.count_max.storage(),
        ];
        reads.extend(views.tile_storages());
        let writes = [st.pre.storage(), st.pre_max.storage()];
        let config = d.tile_grid();
        self.base.launch(ctx, &reads, &writes, |stream| {
            let pre = LsqmlPre {
                pre: dptr(&st.pre),
                max_bits: dptr(&st.pre_max),
                count_max: dptr::<u64, u64>(&st.count_max).cast_const(),
            };
            // SAFETY: `pre` holds the object's pixels, `probe` its planes; `Views::new` built
            // the tile runs over every view and checked each lies inside the object.
            unsafe {
                self.module.lsqml_pre(
                    stream,
                    config,
                    dptr(probe),
                    dptr::<ViewAddr, ViewAddr>(views.addr()),
                    views.tile_ptrs(),
                    pre,
                    init.into(),
                    d.scan_geom(),
                )
            }
        })
    }

    /// The stack's row-FFT'd exit waves `O P q` into `work`, `[nf_max * nmode, ny, nx]`, the
    /// fused forward row pass; `views` in epoch order.
    pub fn forward(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        obj: &DeviceBuffer<C32>,
        probe: &DeviceBuffer<C32>,
        work: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let d = views.dims();
        let (sh, g) = (Shape::of(d)?, d.geom(stack)?);
        let [no, np] = lens(d);
        check_len("obj", obj, no)?;
        check_len("probe", probe, np)?;
        check_len("work", work, g.nf_max as usize * d.nmode as usize * d.n())?;
        let config = sh.rows(&g, true)?;
        let p = FusedPtrs {
            obj: dptr::<C32, f32>(obj).cast_const(),
            probe: dptr::<C32, f32>(probe).cast_const(),
            views: dptr::<ViewAddr, ViewAddr>(views.addr()).cast_const(),
            work: dptr(work),
            ..null_ptrs()
        };
        let reads = [
            obj.storage(),
            probe.storage(),
            views.addr().storage(),
            stack.storage(),
        ];
        let (ctl, cfg) = (stack.ctl() as *const StackCtl, FusedCfg::default());
        let m = &self.base.module;
        self.base.launch(ctx, &reads, &[work.storage()], |stream| {
            // SAFETY: lengths checked above; `Shape::of` checked the side's kernels and shared
            // memory; `Views::new` checked the views are inside the object.
            unsafe {
                match sh.nx {
                    Side::N64 => m.fused_rows_fwd_64(stream, config, p, ctl, g, cfg),
                    Side::N128 => m.fused_rows_fwd_128(stream, config, p, ctl, g, cfg),
                    Side::N256 => m.fused_rows_fwd_256(stream, config, p, ctl, g, cfg),
                    Side::N512 => m.fused_rows_fwd_512(stream, config, p, ctl, g, cfg),
                }
            }
        })
    }

    /// The stack's model intensity `icalc`, `[nf_max, ny, nx]` in detector order, from the
    /// row transforms [`Self::forward`] left in `work`, which it only reads.
    pub fn icalc(
        &self,
        ctx: &ExecutionContext,
        scan: BhScan<'_>,
        stack: StackRef<'_>,
        work: &DeviceBuffer<C32>,
        icalc: &mut DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        let d = *scan.check()?;
        let obs = scan.obs();
        let g = d.geom(stack)?;
        check_len("icalc", icalc, g.nf_max as usize * d.n())?;
        let p = LsqmlCols {
            work: dptr(work),
            icalc: dptr(icalc),
            obs,
            a_far: std::ptr::null(),
        };
        let l = ColsLaunch::new(&d, stack, work)?;
        let reads = [work.storage(), stack.storage()];
        self.cols(ctx, l, p, LSQML_ICALC, &reads, &[icalc.storage()])
    }

    /// The stack's exit-wave update `chi` from the row transforms [`Self::forward`] left in
    /// `work`, into `work` in the layout [`Kernels::psi_to_obj`] reads: the column pass
    /// scales each far-field pixel by `−a_far c / n` at the epoch position's `a_far` of `st`
    /// (`c` `noise`'s residual), then the inverse fused row pass.
    pub fn chi(
        &self,
        ctx: &ExecutionContext,
        scan: BhScan<'_>,
        stack: StackRef<'_>,
        noise: LsqmlNoise,
        st: &LsqmlScan,
        work: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let d = *scan.check()?;
        let obs = scan.obs();
        st.check(&d)?;
        let l = ColsLaunch::new(&d, stack, work)?;
        let p = LsqmlCols {
            work: dptr(work),
            icalc: std::ptr::null_mut(),
            obs,
            a_far: dptr::<f32, f32>(&st.a_pos).cast_const(),
        };
        let flags = match noise {
            LsqmlNoise::Gaussian => LSQML_CHI,
            LsqmlNoise::Poisson => LSQML_CHI | LSQML_POISSON,
        };
        let reads = [
            scan.counts.storage(),
            scan.order.ids().storage(),
            st.a_pos.storage(),
            stack.storage(),
        ];
        self.cols(ctx, l, p, flags, &reads, &[work.storage()])?;
        let (sh, g) = (Shape::of(&d)?, d.geom(stack)?);
        let config = sh.rows(&g, false)?;
        let p = FusedPtrs {
            views: dptr::<ViewAddr, ViewAddr>(scan.order.views().addr()).cast_const(),
            work: dptr(work),
            ..null_ptrs()
        };
        let reads = [scan.order.views().addr().storage(), stack.storage()];
        let (ctl, cfg) = (stack.ctl() as *const StackCtl, FusedCfg::default());
        let m = &self.base.module;
        self.base.launch(ctx, &reads, &[work.storage()], |stream| {
            // SAFETY: `ColsLaunch::new` checked `work`; as in `forward`.
            unsafe {
                match sh.nx {
                    Side::N64 => m.fused_rows_inv_64(stream, config, p, ctl, g, cfg),
                    Side::N128 => m.fused_rows_inv_128(stream, config, p, ctl, g, cfg),
                    Side::N256 => m.fused_rows_inv_256(stream, config, p, ctl, g, cfg),
                    Side::N512 => m.fused_rows_inv_512(stream, config, p, ctl, g, cfg),
                }
            }
        })
    }

    /// The Poisson `a_far` of the batch's epoch positions `batch` from the `icalc`
    /// ([`Self::icalc`]) of each of its stacks: pty-chi's two passes, each from the batch
    /// mean `a′` of the values before it (first the frames' own), into `st`'s `a_far` by
    /// position, and after the second by frame.
    pub fn far(
        &self,
        ctx: &ExecutionContext,
        scan: BhScan<'_>,
        icalc: &[(StackRef<'_>, &DeviceBuffer<f32>)],
        batch: Range<u32>,
        st: &mut LsqmlScan,
    ) -> Result<(), Error> {
        let d = *scan.check()?;
        let obs = scan.obs();
        st.check(&d)?;
        let span = span(&st.dims, &batch)?;
        let ids = scan.order.ids();
        let far = st.far(ids);
        let mut stacks = Vec::with_capacity(icalc.len());
        for &(stack, buf) in icalc {
            let g = d.geom(stack)?;
            check_len("icalc", buf, g.nf_max as usize * d.n())?;
            stacks.push((stack, buf, g, chunk_grid(&g, d.n())?));
        }
        let reads = [ids.storage(), st.a_scan.storage()];
        self.base
            .launch(ctx, &reads, &[st.prev.storage()], |stream| {
                // SAFETY: `span` lies in the scan's positions, which `order` maps to frames.
                unsafe { self.module.lsqml_far_mean(stream, ONE_BLOCK, far, span) }
            })?;
        for last in [0, 1] {
            for &(stack, buf, g, config) in &stacks {
                let reads = [
                    buf.storage(),
                    scan.counts.storage(),
                    ids.storage(),
                    st.prev.storage(),
                    stack.storage(),
                ];
                self.base
                    .launch(ctx, &reads, &[st.xi.storage()], |stream| {
                        // SAFETY: `buf` checked above, `st.xi` sized for every position's chunks;
                        // one PIXEL_THREADS block per chunk and frame.
                        unsafe {
                            self.module.lsqml_xi(
                                stream,
                                config,
                                dptr::<f32, f32>(buf).cast_const(),
                                obs,
                                dptr::<f32, f32>(&st.prev).cast_const(),
                                dptr(&st.xi),
                                stack.ctl() as *const StackCtl,
                                g,
                            )
                        }
                    })?;
            }
            let reads = [st.xi.storage(), ids.storage()];
            let writes = [st.a_scan.storage(), st.a_pos.storage(), st.prev.storage()];
            self.base.launch(ctx, &reads, &writes, |stream| {
                // SAFETY: as `lsqml_far_mean`; `st.xi` holds the span's chunks.
                unsafe {
                    self.module.lsqml_far_step(
                        stream,
                        ONE_BLOCK,
                        dptr::<f32, f32>(&st.xi).cast_const(),
                        span,
                        far,
                        last,
                    )
                }
            })?;
        }
        Ok(())
    }

    /// The preconditioned object direction in `sums.num`: divided by `√(pre² + (0.1 max
    /// pre)²)` at `st`'s last [`Self::precondition`].
    pub fn dir_obj(
        &self,
        ctx: &ExecutionContext,
        st: &LsqmlScan,
        sums: &mut UpdateSums,
    ) -> Result<(), Error> {
        let n = st.dims.obj_len();
        sums.check(n)?;
        let config = grid_1d(n)?;
        let n = to_u32(n)?;
        let reads = [st.pre.storage(), st.pre_max.storage()];
        self.base
            .launch(ctx, &reads, &[sums.num.storage()], |stream| {
                // SAFETY: `sums` and `st.pre` hold the object's `n` values.
                unsafe {
                    self.module.lsqml_dir_obj(
                        stream,
                        config,
                        dptr(&sums.num),
                        dptr::<f32, f32>(&st.pre).cast_const(),
                        dptr::<u32, u32>(&st.pre_max).cast_const(),
                        n,
                    )
                }
            })
    }

    /// The step-size sums of the stack's frames into `st` from its exit-wave update `chi`;
    /// `views` in epoch order. Under [`LsqmlChi::Moved`] the object direction's waves take
    /// the moved probe.
    pub fn dots(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        at: LsqmlAt<'_>,
        chi: LsqmlChi<'_>,
        st: &mut LsqmlScan,
    ) -> Result<(), Error> {
        let d = views.dims();
        st.check(d)?;
        let g = d.geom(stack)?;
        let c = match &chi {
            LsqmlChi::Unmoved(c) | LsqmlChi::Moved { chi: c, .. } => *c,
        };
        check_len("chi", c, g.nf_max as usize * d.nmode as usize * d.n())?;
        for (pair, what) in [(at.x, ["obj", "probe"]), (at.d, ["d_obj", "d_probe"])] {
            for ((b, len), what) in pair.into_iter().zip(lens(d)).zip(what) {
                check_len(what, b, len)?;
            }
        }
        let config = chunk_grid(&g, d.n())?;
        let mut reads = vec![c.storage(), views.addr().storage(), stack.storage()];
        reads.extend(at.x.into_iter().chain(at.d).map(|b| b.storage()));
        let pair = |p: [&DeviceBuffer<C32>; 2]| ObjProbePtrs {
            obj: dptr::<C32, f32>(p[0]).cast_const(),
            probe: dptr::<C32, f32>(p[1]).cast_const(),
        };
        let p = LsqmlDots {
            chi: dptr::<C32, f32>(c).cast_const(),
            x: pair(at.x),
            d: pair(at.d),
            views: dptr::<ViewAddr, ViewAddr>(views.addr()).cast_const(),
            out: dptr(&st.dots),
        };
        let ctl = stack.ctl() as *const StackCtl;
        let LsqmlChi::Moved { w, .. } = chi else {
            return self
                .base
                .launch(ctx, &reads, &[st.dots.storage()], |stream| {
                    // SAFETY: lengths checked above, `st.dots` sized for every position's
                    // chunks; `Views::new` checked the views are inside the object.
                    unsafe { self.module.lsqml_dots(stream, config, p, ctl, g) }
                });
        };
        check_len("w", w, g.nf_max as usize * d.nprobe as usize * d.n())?;
        reads.push(w.storage());
        let w = dptr::<C32, f32>(w).cast_const();
        self.base
            .launch(ctx, &reads, &[st.dots.storage()], |stream| {
                // SAFETY: as `lsqml_dots`; `w` checked above.
                unsafe { self.module.lsqml_dots_shifted(stream, config, p, w, ctl, g) }
            })
    }

    /// The step sizes of the batch's epoch positions `batch` from the [`Self::dots`] sums of
    /// all its stacks: each position's record, and the object step (the mean `a_obj` without
    /// the `⌈0.05 nb − 0.5⌉` smallest and largest) and the probe step (the mean `a_probe`)
    /// that [`Self::update`] takes.
    pub fn step(
        &self,
        ctx: &ExecutionContext,
        batch: Range<u32>,
        st: &mut LsqmlScan,
    ) -> Result<(), Error> {
        let span = span(&st.dims, &batch)?;
        // pty-chi's `trim_mean`, the cut saturating at 0 for small batches.
        let trim = (span.nb as f64 * 0.1 / 2.0 - 0.5).ceil() as u32;
        let g = st.dims.scan_geom();
        let writes = [st.dots.storage(), st.rec.storage(), st.step.storage()];
        self.base
            .launch(ctx, &[st.a_pos.storage()], &writes, |stream| {
                // SAFETY: `span` lies in the scan's positions, `2 trim < nb`; one block.
                unsafe {
                    self.module.lsqml_step(
                        stream,
                        ONE_BLOCK,
                        dptr(&st.dots),
                        span,
                        trim,
                        dptr::<f32, f32>(&st.a_pos).cast_const(),
                        dptr(&st.rec),
                        dptr(&st.step),
                        g,
                    )
                }
            })
    }

    /// `x += s num` with `s` the last [`Self::step`]'s object step for `obj` and its probe
    /// step for `probe`; consumes both update sums (zero after).
    pub fn update(
        &self,
        ctx: &ExecutionContext,
        st: &LsqmlScan,
        obj: LsqmlVar<'_>,
        probe: LsqmlVar<'_>,
    ) -> Result<(), Error> {
        for (var, n, at) in [(obj, lens(&st.dims)[0], 0), (probe, lens(&st.dims)[1], 1)] {
            check_len("x", var.x, n)?;
            var.sums.check(n)?;
            let config = grid_1d(n)?;
            let n = to_u32(n)?;
            let step = dptr::<f32, f32>(&st.step).wrapping_add(at).cast_const();
            let writes = [
                var.x.storage(),
                var.sums.num.storage(),
                var.sums.den.storage(),
            ];
            self.base
                .launch(ctx, &[st.step.storage()], &writes, |stream| {
                    // SAFETY: lengths checked above; `st.step` holds both steps.
                    unsafe {
                        self.module.lsqml_update(
                            stream,
                            config,
                            dptr(var.x),
                            var.sums.ptrs(),
                            n,
                            step,
                        )
                    }
                })?;
        }
        Ok(())
    }

    /// The column pass under `flags`, `reads` and `writes` its leases.
    fn cols(
        &self,
        ctx: &ExecutionContext,
        l: ColsLaunch,
        p: LsqmlCols,
        flags: u32,
        reads: &[&RawDeviceBuffer],
        writes: &[&RawDeviceBuffer],
    ) -> Result<(), Error> {
        let ColsLaunch {
            side,
            config,
            g,
            ctl,
        } = l;
        let m = &self.module;
        self.base.launch(ctx, reads, writes, |stream| {
            // SAFETY: `ColsLaunch::new` checked `work` and the caller every other buffer
            // `flags` names; `Shape::of` checked the side and that `col_tile_fft` fits;
            // `EpochOrder` holds frame ids below `nframe`.
            unsafe {
                match side {
                    Side::N64 => m.lsqml_cols_64(stream, config, p, ctl, g, flags),
                    Side::N128 => m.lsqml_cols_128(stream, config, p, ctl, g, flags),
                    Side::N256 => m.lsqml_cols_256(stream, config, p, ctl, g, flags),
                    Side::N512 => m.lsqml_cols_512(stream, config, p, ctl, g, flags),
                }
            }
        })
    }
}

/// The launch of [`LsqmlKernels`]' column pass on one stack.
struct ColsLaunch {
    side: Side,
    config: LaunchConfig,
    g: StackGeom,
    ctl: *const StackCtl,
}

impl ColsLaunch {
    /// For `stack` of a scan of `d`, its transforms in `work`.
    fn new(d: &ViewDims, stack: StackRef<'_>, work: &DeviceBuffer<C32>) -> Result<Self, Error> {
        let (sh, g) = (Shape::of(d)?, d.geom(stack)?);
        check_len("work", work, g.nf_max as usize * d.nmode as usize * d.n())?;
        if g.nf_max > GRID_Y_MAX {
            return Err(Error::Shape(format!("{} frames exceed grid y", g.nf_max)));
        }
        Ok(Self {
            side: sh.ny,
            config: LaunchConfig {
                grid_dim: (g.nx / FUSED_COLS, g.nf_max, 1),
                block_dim: (sh.col_threads, 1, 1),
                shared_mem_bytes: sh.col_tile_fft,
            },
            g,
            ctl: stack.ctl() as *const StackCtl,
        })
    }
}

fn null_ptrs() -> FusedPtrs {
    FusedPtrs {
        obj: std::ptr::null(),
        probe: std::ptr::null(),
        views: std::ptr::null(),
        counts: std::ptr::null(),
        psi: std::ptr::null_mut(),
        work: std::ptr::null_mut(),
        probe_q: std::ptr::null_mut(),
        probe_power: std::ptr::null_mut(),
        groups: std::ptr::null_mut(),
        scale: std::ptr::null(),
        scale_groups: std::ptr::null_mut(),
        background: std::ptr::null(),
        background_ratio: std::ptr::null_mut(),
        offsets: std::ptr::null(),
        spectrum: std::ptr::null(),
        shift: std::ptr::null_mut(),
        shift_dy: std::ptr::null_mut(),
        shift_obj: std::ptr::null_mut(),
        shift_power: std::ptr::null_mut(),
        shift_grad: std::ptr::null_mut(),
        shift_den: std::ptr::null_mut(),
        shift_adj: std::ptr::null_mut(),
    }
}
