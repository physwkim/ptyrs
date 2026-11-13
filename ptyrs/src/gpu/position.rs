//! Checked launches of the position kernels (`ptyrs_kernels::position`): pty-chi's gradient
//! position correction under the moved probe of [`Kernels::shifted_rows_fwd`], for any
//! method whose stacks hold epoch positions and whose exit-wave update `chi` is at the exit
//! wave's stored pixels.
//!
//! # One epoch's position correction in these calls
//!
//! The epoch starts with [`PositionKernels::gather`]; the shifted passes read
//! [`Positions::offsets`] as [`crate::gpu::Shift::offsets`] for the whole epoch, so each frame
//! moves once, in its batch. Per stack, [`PositionKernels::shift_w`] turns the probe rows
//! [`Kernels::shifted_rows_fwd`] left into the moved probe `W`; with the stack's `chi`,
//! [`PositionKernels::sums`] takes the object slopes and the stack's position sums against
//! the object the batch started from. After the batch's update, [`PositionKernels::step`]
//! steps its frames' offsets.

use std::ops::Range;

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
use ptyrs_kernels::fft_ept;
use ptyrs_kernels::fused::FUSED_COLS;
use ptyrs_kernels::grad::GAMMA_CHUNK;
use ptyrs_kernels::pixel::PIXEL_THREADS;
pub use ptyrs_kernels::position::PositionRule;
use ptyrs_kernels::position::{BatchSpan, POSITION_SUMS, PositionSums, position_kernels};

use crate::gpu::fused::{GRID_Y_MAX, SMEM_MAX, Shape, Side};
use crate::gpu::ops::grid_1d;
use crate::gpu::{
    C32, EpochOrder, Error, Kernels, StackCtl, StackGeom, StackRef, ViewAddr, ViewDims, Views,
    check_len, dptr,
};

/// The position kernels, bound to the module a [`Kernels`] loaded.
#[derive(Clone, Debug)]
pub struct PositionKernels {
    base: Kernels,
    module: position_kernels::LoadedModule,
}

/// The position state over one scan, sized by its dims and the stacks' `nf_max`: the offsets
/// `[nframe, 2]` by epoch position, every position's [`POSITION_SUMS`] per chunk, and one
/// stack's object slopes `[nf_max, 2, ny, nx]`.
pub struct Positions {
    dims: ViewDims,
    offsets: DeviceBuffer<f32>,
    sums: DeviceBuffer<f32>,
    slopes: DeviceBuffer<C32>,
}

impl Positions {
    /// Zeroed state for a scan of `dims` in stacks of up to `nf_max` frames.
    pub fn zeroed(stream: &CudaStream, dims: &ViewDims, nf_max: u32) -> Result<Self, Error> {
        dims.check()?;
        let nframe = dims.nframe as usize;
        Ok(Self {
            dims: *dims,
            offsets: DeviceBuffer::zeroed(stream, 2 * nframe)?,
            sums: DeviceBuffer::zeroed(stream, POSITION_SUMS as usize * nframe * chunks(dims.n()))?,
            slopes: DeviceBuffer::zeroed(stream, 2 * nf_max as usize * dims.n())?,
        })
    }

    /// The offsets by epoch position, as [`PositionKernels::gather`] set them.
    pub fn offsets(&self) -> &DeviceBuffer<f32> {
        &self.offsets
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
    /// [`Kernels::shifted_rows_fwd`] left it: plane `fl * nprobe + p` becomes mode `p` of
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

    /// The position sums of the stack's frames into `pos`: each frame's first view (mode 0)
    /// of its exit-wave update `chi`, `[nf_max * nmode, ny, nx]`, against `u = ∂O W`, the
    /// slopes of `obj` at that view times its moved probe `w`, `[nf_max * nprobe, ny, nx]`
    /// ([`Self::shift_w`]); `chi` and `w` at the exit wave's stored pixels, `views` in epoch
    /// order.
    pub fn sums(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        obj: &DeviceBuffer<C32>,
        [chi, w]: [&DeviceBuffer<C32>; 2],
        pos: &mut Positions,
    ) -> Result<(), Error> {
        let d = views.dims();
        same_dims(&pos.dims, d)?;
        let g = d.geom(stack)?;
        let nf = g.nf_max as usize;
        check_len("obj", obj, d.obj_len())?;
        check_len("chi", chi, nf * d.nmode as usize * d.n())?;
        check_len("w", w, nf * d.nprobe as usize * d.n())?;
        check_len("slopes", &pos.slopes, 2 * nf * d.n())?;
        self.slopes(ctx, views, stack, obj, &mut pos.slopes)?;
        let config = chunk_grid(&g, d.n())?;
        let reads = [
            chi.storage(),
            w.storage(),
            pos.slopes.storage(),
            views.addr().storage(),
            stack.storage(),
        ];
        let p = PositionSums {
            chi: dptr::<C32, f32>(chi).cast_const(),
            w: dptr::<C32, f32>(w).cast_const(),
            slopes: dptr::<C32, f32>(&pos.slopes).cast_const(),
            views: dptr::<ViewAddr, ViewAddr>(views.addr()).cast_const(),
            sums: dptr(&pos.sums),
        };
        let ctl = stack.ctl() as *const StackCtl;
        self.base
            .launch(ctx, &reads, &[pos.sums.storage()], |stream| {
                // SAFETY: lengths checked above, `pos.sums` sized for every position's
                // chunks.
                unsafe { self.module.position_sums(stream, config, p, ctl, g) }
            })
    }

    /// The object slopes of each of the stack's frames, at its first view's window of `obj`,
    /// into `slopes` `[nf_max, 2, ny, nx]`; `views` in epoch order.
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
    /// of all its stacks, into `offsets`, `[nframe, 2]` by frame through `order`; `rule` the
    /// fraction of the solved shift taken and the largest step. `pos`' own offsets stay as
    /// gathered: the epoch moves each frame once.
    pub fn step(
        &self,
        ctx: &ExecutionContext,
        batch: Range<u32>,
        order: &EpochOrder,
        offsets: &mut DeviceBuffer<f32>,
        rule: PositionRule,
        pos: &mut Positions,
    ) -> Result<(), Error> {
        let span = span(&pos.dims, &batch)?;
        same_dims(&pos.dims, order.views().dims())?;
        let nframe = pos.dims.nframe;
        check_len("offsets", offsets, 2 * nframe as usize)?;
        let ids = order.ids();
        let g = pos.dims.scan_geom();
        let writes = [pos.sums.storage(), offsets.storage()];
        self.base.launch(ctx, &[ids.storage()], &writes, |stream| {
            // SAFETY: `span` lies in the scan's positions, `pos.sums` sized for every
            // position's chunks, `offsets` checked above; one block.
            unsafe {
                self.module.position_step(
                    stream,
                    ONE_BLOCK,
                    dptr(&pos.sums),
                    span,
                    dptr::<u32, u32>(ids).cast_const(),
                    dptr(offsets),
                    nframe,
                    rule,
                    g,
                )
            }
        })
    }
}
