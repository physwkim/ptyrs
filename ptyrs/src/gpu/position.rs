//! Checked launches of the position kernels (`ptyrs_kernels::position`): pty-chi's gradient
//! position correction under the moved probe of [`Kernels::shifted_probe`], for any
//! method whose stacks hold epoch positions and whose exit-wave update `chi` is at the exit
//! wave's stored pixels.
//!
//! # One epoch's position correction in these calls
//!
//! The epoch starts with [`PositionKernels::gather`]; the shifted passes read
//! [`Positions::offsets`] as [`crate::gpu::Shift::offsets`] for the whole epoch, so each frame
//! moves once, in its batch. Per stack, [`PositionKernels::shift_w`] turns the probe rows
//! [`Kernels::shifted_probe`] left into the moved probe `W`; with the stack's `chi`,
//! [`PositionKernels::sums`] takes the object slopes and the stack's position sums against
//! the object the batch started from into a [`PositionStep`]. After the batch's update,
//! [`PositionKernels::step`] steps its frames' offsets. Where each view samples its object at
//! its offset instead and the probe stays put, [`PositionKernels::derivative_sums`] takes the
//! sums from the sampled windows' derivatives. A held position's epoch runs the gather alone.

use std::ops::Range;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use ptyrs_kernels::fft_ept;
use ptyrs_kernels::fused::FUSED_COLS;
use ptyrs_kernels::grad::GAMMA_CHUNK;
use ptyrs_kernels::optics::WavePlanes;
use ptyrs_kernels::pixel::PIXEL_THREADS;
pub use ptyrs_kernels::position::PositionRule;
use ptyrs_kernels::position::{BatchSpan, POSITION_SUMS, PositionSums, position_kernels};

use crate::gpu::fused::{GRID_Y_MAX, SMEM_MAX, Shape, Side};
use crate::gpu::ops::grid_1d;
use crate::gpu::{
    C32, EpochOrder, Error, FrameWaves, Kernels, SliceSource, StackCtl, StackGeom, StackRef,
    ViewAddr, ViewDims, Views, check_len, dptr,
};

/// How [`PositionKernels::step`] solves and takes a shift.
#[derive(Clone, Copy)]
pub struct Solve {
    /// The fraction of the solved shift taken and the largest step.
    pub rule: PositionRule,
    /// What pty-chi's `1e-6` on the slopes' denominator is times: the intensity scale of the
    /// waves the slopes are of.
    pub scale: f32,
}

/// The position kernels, bound to the module a [`Kernels`] loaded.
#[derive(Clone, Debug)]
pub struct PositionKernels {
    base: Kernels,
    module: position_kernels::LoadedModule,
}

/// The offsets `[nframe, 2]` by epoch position over one scan of its dims.
pub struct Positions {
    dims: ViewDims,
    offsets: DeviceBuffer<f32>,
}

impl Positions {
    /// Zeroed offsets for a scan of `dims`.
    pub fn zeroed(stream: &CudaStream, dims: &ViewDims) -> Result<Self, Error> {
        dims.check()?;
        Ok(Self {
            dims: *dims,
            offsets: DeviceBuffer::zeroed(stream, 2 * dims.nframe as usize)?,
        })
    }

    /// The offsets by epoch position, as [`PositionKernels::gather`] set them.
    pub fn offsets(&self) -> &DeviceBuffer<f32> {
        &self.offsets
    }
}

/// What [`PositionKernels::step`] reads over one scan, sized by its dims and the stacks'
/// `nf_max`: every position's [`POSITION_SUMS`] per chunk, and one stack's derivatives of the
/// object at its frames' views along `y` then `x`, `[2, nf_max, ny, nx]`.
pub struct PositionStep {
    dims: ViewDims,
    sums: DeviceBuffer<f32>,
    slopes: DeviceBuffer<C32>,
}

impl PositionStep {
    /// Zeroed state for a scan of `dims` in stacks of up to `nf_max` frames.
    pub fn zeroed(stream: &CudaStream, dims: &ViewDims, nf_max: u32) -> Result<Self, Error> {
        dims.check()?;
        Ok(Self {
            dims: *dims,
            sums: DeviceBuffer::zeroed(
                stream,
                POSITION_SUMS as usize * dims.nframe as usize * chunks(dims.n()),
            )?,
            slopes: DeviceBuffer::zeroed(stream, 2 * nf_max as usize * dims.n())?,
        })
    }
}

/// Errors unless state built for `have` serves views of `dims`.
pub(super) fn same_dims(have: &ViewDims, dims: &ViewDims) -> Result<(), Error> {
    if have != dims {
        return Err(Error::Shape(format!(
            "state for {have:?}, views have {dims:?}"
        )));
    }
    Ok(())
}

/// The batch `p` over the epoch positions of a scan of `dims`.
pub(super) fn span(dims: &ViewDims, p: &Range<u32>) -> Result<BatchSpan, Error> {
    if p.start >= p.end || p.end > dims.nframe {
        return Err(Error::Shape(format!(
            "batch {p:?} outside {} frames",
            dims.nframe
        )));
    }
    Ok(BatchSpan {
        p0: p.start,
        nb: p.end - p.start,
        // Every per-chunk buffer was sized by it, so it fits.
        nchunk: chunks(dims.n()) as u32,
    })
}

/// Chunks of [`GAMMA_CHUNK`] pixels per frame of `n` pixels.
pub(super) fn chunks(n: usize) -> usize {
    n.div_ceil(GAMMA_CHUNK as usize)
}

/// One block of [`PIXEL_THREADS`] per chunk of each of the stack's frames.
pub(super) fn chunk_grid(g: &StackGeom, n: usize) -> Result<LaunchConfig, Error> {
    if g.nf_max as u64 > GRID_Y_MAX {
        return Err(Error::Shape(format!("{} frames exceed grid y", g.nf_max)));
    }
    Ok(LaunchConfig {
        grid_dim: (chunks(n) as u32, g.nf_max, 1),
        block_dim: (PIXEL_THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// One block of [`PIXEL_THREADS`].
pub(super) const ONE_BLOCK: LaunchConfig = LaunchConfig {
    grid_dim: (1, 1, 1),
    block_dim: (PIXEL_THREADS, 1, 1),
    shared_mem_bytes: 0,
};

/// `bytes` of dynamic shared memory, if a block may take them.
fn smem(bytes: u32) -> Result<u32, Error> {
    if bytes > SMEM_MAX {
        return Err(Error::Shape(format!(
            "{bytes} bytes of shared memory per block exceed {SMEM_MAX}"
        )));
    }
    Ok(bytes)
}

impl PositionKernels {
    /// Binds the position kernels of the cubin `kernels` already loaded; loads nothing.
    pub fn load(kernels: &Kernels) -> Result<Self, Error> {
        let module = position_kernels::from_module(kernels.module.as_cuda_module().clone())?;
        Ok(Self {
            base: kernels.clone(),
            module,
        })
    }

    /// The stack's moved probe in place in `t`, `[nf_max * nprobe, ny, nx]` as
    /// [`Kernels::shifted_probe`] left it: plane `fl * nprobe + p` becomes mode `p` of
    /// frame `fl` moved by its offset, at the exit wave's stored pixels.
    pub fn shift_w(
        &self,
        ctx: &ExecutionContext,
        dims: &ViewDims,
        stack: StackRef<'_>,
        t: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let (sh, g) = (Shape::of(dims)?, dims.geom(stack)?);
        let planes = g.nf_max as u64 * g.nprobe as u64;
        if planes > GRID_Y_MAX {
            return Err(Error::Shape(format!("{planes} planes exceed grid y")));
        }
        check_len("t", t, planes as usize * dims.n())?;
        let config = LaunchConfig {
            grid_dim: (g.ny / FUSED_COLS, planes as u32, 1),
            block_dim: (FUSED_COLS * (g.nx / fft_ept(g.nx)), 1, 1),
            shared_mem_bytes: smem(FUSED_COLS * (sh.row_smem / sh.rows))?,
        };
        let ctl = stack.ctl() as *const StackCtl;
        let m = &self.module;
        self.base
            .launch(ctx, &[stack.storage()], &[t.storage()], |stream| {
                // SAFETY: `t` checked above; `Shape::of` checked the side's FFT.
                unsafe {
                    let t = dptr::<C32, f32>(t);
                    match sh.nx {
                        Side::N64 => m.position_shift_w_64(stream, config, t, ctl, g),
                        Side::N128 => m.position_shift_w_128(stream, config, t, ctl, g),
                        Side::N256 => m.position_shift_w_256(stream, config, t, ctl, g),
                        Side::N512 => m.position_shift_w_512(stream, config, t, ctl, g),
                    }
                }
            })
    }

    /// The position sums of the stack's frames into `step`: each frame's first view (mode 0)
    /// of its exit-wave update `chi`, `[nf_max * nmode, ny, nx]`, against `u = ∂O W`, the
    /// slopes of `obj` at that view, which it writes to `step`'s, times its wave of `w`: the
    /// moved probe ([`Self::shift_w`]), or a multislice object's wave entering the slice
    /// `obj`; `chi` at the exit wave's stored pixels, `views` in epoch order.
    pub fn sums(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        obj: &DeviceBuffer<C32>,
        (chi, w): (&DeviceBuffer<C32>, FrameWaves<'_>),
        step: &mut PositionStep,
    ) -> Result<(), Error> {
        let d = views.dims();
        check_len("obj", obj, d.obj_len())?;
        self.check_slopes(views, stack, step)?;
        self.slopes(ctx, views, stack, obj, &mut step.slopes)?;
        let w = SliceSource::Waves(w);
        self.launch_sums(ctx, views, stack, &step.slopes, (chi, w), &mut step.sums)
    }

    /// The position sums of the stack's frames into `step` where each view samples its object
    /// at its offset and the probe stays put: [`Self::sums`] against the slopes `derivatives`
    /// writes for this stack into `step`'s, the derivatives along `y` then `x` of each
    /// frame's sampled window (object mode 0) in the natural layout. `w` is the wave entering
    /// the slice: with [`SliceSource::Probe`] probe mode 0 times the exit's quadratic phase.
    pub fn derivative_sums(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        derivatives: impl FnOnce(&mut DeviceBuffer<C32>) -> Result<(), Error>,
        (chi, w): (&DeviceBuffer<C32>, SliceSource<'_>),
        step: &mut PositionStep,
    ) -> Result<(), Error> {
        self.check_slopes(views, stack, step)?;
        derivatives(&mut step.slopes)?;
        self.launch_sums(ctx, views, stack, &step.slopes, (chi, w), &mut step.sums)
    }

    /// Errors unless `step` serves `views` and has room for the stack's slopes.
    fn check_slopes(
        &self,
        views: &Views,
        stack: StackRef<'_>,
        step: &PositionStep,
    ) -> Result<(), Error> {
        let d = views.dims();
        same_dims(&step.dims, d)?;
        let len = 2 * d.geom(stack)?.nf_max as usize * d.n();
        check_len("slopes", &step.slopes, len)
    }

    /// [`position_kernels::LoadedModule::position_sums`] of the stack against `slopes`, `[2,
    /// nf_max, ny, nx]` checked by the caller, into `sums`, sized by [`PositionStep::zeroed`]
    /// for `views`' dims.
    fn launch_sums(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        slopes: &DeviceBuffer<C32>,
        (chi, w): (&DeviceBuffer<C32>, SliceSource<'_>),
        sums: &mut DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        let d = views.dims();
        let g = d.geom(stack)?;
        check_len("chi", chi, g.nf_max as usize * d.nmode as usize * d.n())?;
        let (w, waves, probe) = match w {
            SliceSource::Probe(p) => {
                check_len("probe", p, d.nprobe as usize * d.n())?;
                let waves = WavePlanes {
                    w: dptr::<C32, f32>(p).cast_const(),
                    nw: d.nprobe,
                };
                (p, waves, 1)
            }
            SliceSource::Waves(w) => {
                let (w, waves) = w.planes(d, g.nf_max)?;
                (w, waves, 0)
            }
        };
        let config = chunk_grid(&g, d.n())?;
        let reads = [
            chi.storage(),
            w.storage(),
            slopes.storage(),
            stack.storage(),
        ];
        let p = PositionSums {
            chi: dptr::<C32, f32>(chi).cast_const(),
            w: waves,
            probe,
            slopes: dptr::<C32, f32>(slopes).cast_const(),
            sums: dptr(sums),
        };
        let ctl = stack.ctl() as *const StackCtl;
        self.base.launch(ctx, &reads, &[sums.storage()], |stream| {
            // SAFETY: lengths checked above and by the caller, `sums` sized for every
            // position's chunks.
            unsafe { self.module.position_sums(stream, config, p, ctl, g) }
        })
    }

    /// The object slopes of each of the stack's frames, at its first view's window of `obj`,
    /// into `slopes` `[2, nf_max, ny, nx]`; `views` in epoch order.
    fn slopes(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        obj: &DeviceBuffer<C32>,
        slopes: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let d = views.dims();
        let (sh, g) = (Shape::of(d)?, d.geom(stack)?);
        if g.nf_max as u64 > GRID_Y_MAX {
            return Err(Error::Shape(format!("{} frames exceed grid y", g.nf_max)));
        }
        let reads = [obj.storage(), views.addr().storage(), stack.storage()];
        let (o, v) = (
            dptr::<C32, f32>(obj).cast_const(),
            dptr::<ViewAddr, ViewAddr>(views.addr()).cast_const(),
        );
        let ctl = stack.ctl() as *const StackCtl;
        let m = &self.module;
        // Along `y` the lines are the columns, `nx` of them; along `x` the rows.
        for (axis, side, across, fft) in [
            (0, sh.ny, g.nx, sh.col_fft),
            (1, sh.nx, g.ny, sh.row_smem / sh.rows),
        ] {
            let config = LaunchConfig {
                grid_dim: (across / FUSED_COLS, g.nf_max, 1),
                block_dim: (FUSED_COLS * (side as u32 / fft_ept(side as u32)), 1, 1),
                shared_mem_bytes: smem(FUSED_COLS * fft)?,
            };
            let out = dptr::<C32, f32>(slopes);
            self.base
                .launch(ctx, &reads, &[slopes.storage()], |stream| {
                    // SAFETY: the caller checked `slopes`, `obj` and `views`; `Shape::of` checked
                    // the side's FFT; `Views::new` checked every view's window is inside the
                    // object.
                    unsafe {
                        match side {
                            Side::N64 => {
                                m.position_slopes_64(stream, config, o, v, out, axis, ctl, g)
                            }
                            Side::N128 => {
                                m.position_slopes_128(stream, config, o, v, out, axis, ctl, g)
                            }
                            Side::N256 => {
                                m.position_slopes_256(stream, config, o, v, out, axis, ctl, g)
                            }
                            Side::N512 => {
                                m.position_slopes_512(stream, config, o, v, out, axis, ctl, g)
                            }
                        }
                    }
                })?;
        }
        Ok(())
    }

    /// `pos`' offsets by epoch position from `offsets`, `[nframe, 2]` by frame, through
    /// `order`'s permutation.
    pub fn gather(
        &self,
        ctx: &ExecutionContext,
        order: &EpochOrder,
        offsets: &DeviceBuffer<f32>,
        pos: &mut Positions,
    ) -> Result<(), Error> {
        let nframe = pos.dims.nframe;
        same_dims(&pos.dims, order.views().dims())?;
        check_len("offsets", offsets, 2 * nframe as usize)?;
        let ids = order.ids();
        let config = grid_1d(2 * nframe as usize)?;
        let reads = [offsets.storage(), ids.storage()];
        self.base
            .launch(ctx, &reads, &[pos.offsets.storage()], |stream| {
                // SAFETY: both offsets hold `2 nframe` floats, `ids` the epoch's `nframe` frames.
                unsafe {
                    self.module.position_gather(
                        stream,
                        config,
                        dptr::<f32, f32>(offsets).cast_const(),
                        dptr::<u32, u32>(ids).cast_const(),
                        dptr(&pos.offsets),
                        nframe,
                    )
                }
            })
    }

    /// pty-chi's position step of the batch's epoch positions `batch` from the [`Self::sums`]
    /// of all its stacks in `step`, into `offsets`, `[nframe, 2]` by frame through `order`;
    /// `solve` how the shift is solved and taken. The [`Positions`] the epoch gathered stay:
    /// the epoch moves each frame once.
    pub fn step(
        &self,
        ctx: &ExecutionContext,
        batch: Range<u32>,
        order: &EpochOrder,
        offsets: &mut DeviceBuffer<f32>,
        solve: Solve,
        step: &mut PositionStep,
    ) -> Result<(), Error> {
        let span = span(&step.dims, &batch)?;
        same_dims(&step.dims, order.views().dims())?;
        let nframe = step.dims.nframe;
        check_len("offsets", offsets, 2 * nframe as usize)?;
        let ids = order.ids();
        let writes = [step.sums.storage(), offsets.storage()];
        self.base.launch(ctx, &[ids.storage()], &writes, |stream| {
            // SAFETY: `span` lies in the scan's positions, `step.sums` sized for every
            // position's chunks, `offsets` checked above; one block.
            unsafe {
                self.module.position_step(
                    stream,
                    ONE_BLOCK,
                    dptr(&step.sums),
                    span,
                    dptr::<u32, u32>(ids).cast_const(),
                    dptr(offsets),
                    nframe,
                    solve.rule,
                    solve.scale,
                )
            }
        })
    }
}
