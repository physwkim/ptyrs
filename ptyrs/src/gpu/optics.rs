//! Checked launches of the propagator stages (`ptyrs_kernels::optics`): what a propagator
//! other than the Fourier transform adds around the fused passes' transform, and the
//! in-object propagation between the slices of a multislice object.
//!
//! The fused row pass leaves a stack's row transforms in `work`. [`OpticsKernels::exit`]
//! turns them into the row transforms of the staged waves before the column pass, and after
//! it, as the adjoint, the correction's back; the far field and a forward Fresnel transform
//! have no stage and launch nothing.

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{DeviceBuffer, LaunchConfig};
use ptyrs_kernels::optics::{
    STAGE_COLS, STAGE_CONJ, STAGE_MUL, STAGE_POST_FWD, STAGE_POST_INV, STAGE_PRE_FWD,
    STAGE_PRE_INV, optics_kernels,
};

use crate::gpu::fused::Side;
use crate::gpu::ops::grid_1d;
use crate::gpu::{
    C32, Error, Kernels, StackCtl, StackGeom, StackRef, ViewDims, block_fft, check_len, dptr,
};

/// What a propagator applies to the exit-wave plane ahead of the fused transform `F`, the
/// propagator being `D F V` with `D` a unit-modulus detector factor.
pub enum ExitStage {
    /// `V = 1`: the far field, and a Fresnel transform forward in distance.
    None,
    /// `V` the index reversal: a Fresnel transform back in distance.
    Reverse,
    /// `V = F⁻¹ F⁻¹ T F`, angular spectrum of the transfer function `T`, `[ny, nx]` on the
    /// unshifted frequency grid and even in both frequencies.
    Transfer(DeviceBuffer<C32>),
}

/// An [`ExitStage`] lent to a pass with the kernels that run it.
#[derive(Clone, Copy)]
pub struct ExitPass<'a> {
    pub kernels: &'a OpticsKernels,
    pub stage: &'a ExitStage,
}

/// The propagator stage kernels, bound to the module a [`Kernels`] loaded.
#[derive(Clone, Debug)]
pub struct OpticsKernels {
    base: Kernels,
    module: optics_kernels::LoadedModule,
}

impl OpticsKernels {
    /// Binds the stage kernels of the cubin `kernels` already loaded; loads nothing.
    pub fn load(kernels: &Kernels) -> Result<Self, Error> {
        let module = optics_kernels::from_module(kernels.module.as_cuda_module().clone())?;
        Ok(Self {
            base: kernels.clone(),
            module,
        })
    }

    /// `stage` on `stack`'s row transforms in `work`, `[nf_max * nmode, ny, nx]`: the row
    /// transforms of `V w` from those of the exit waves `w`, or with `adjoint` those of
    /// `Vᴴ c` from the correction's.
    pub fn exit(
        &self,
        ctx: &ExecutionContext,
        d: &ViewDims,
        stack: StackRef<'_>,
        stage: &ExitStage,
        work: &mut DeviceBuffer<C32>,
        adjoint: bool,
    ) -> Result<(), Error> {
        self.exit_on(ctx, d, stack, stage, work, adjoint)
    }

    /// [`Self::exit`] on a `work` the caller holds for writing.
    pub(crate) fn exit_on(
        &self,
        ctx: &ExecutionContext,
        d: &ViewDims,
        stack: StackRef<'_>,
        stage: &ExitStage,
        work: &DeviceBuffer<C32>,
        adjoint: bool,
    ) -> Result<(), Error> {
        if matches!(stage, ExitStage::None) {
            return Ok(());
        }
        let l = Launch::new(d, stack, work)?;
        // Each transform is unnormalized, and a stage is a column and a row one.
        let scale = 1.0 / (d.n() as f32).sqrt();
        match stage {
            ExitStage::None => Ok(()),
            ExitStage::Reverse => {
                let config = grid_1d(l.planes * d.n())?;
                let (m, buf, ctl, g) = (&self.module, dptr(work), l.ctl, l.g);
                self.base
                    .launch(ctx, &[stack.storage()], &[work.storage()], |stream| {
                        // SAFETY: `Launch::new` checked `work` for `g`.
                        unsafe { m.optics_reverse(stream, config, buf, ctl, g) }
                    })
            }
            // `F_x V w = F_x⁻¹ T F_y⁻¹ (F_x w)`, `T` being even, and its adjoint.
            ExitStage::Transfer(tf) if adjoint => {
                self.stage(ctx, &l, stack, work, None, STAGE_PRE_FWD)?;
                let flags = STAGE_COLS | STAGE_MUL | STAGE_CONJ | STAGE_POST_FWD;
                self.stage(ctx, &l, stack, work, Some((tf, scale)), flags)
            }
            ExitStage::Transfer(tf) => {
                let flags = STAGE_COLS | STAGE_PRE_INV | STAGE_MUL;
                self.stage(ctx, &l, stack, work, Some((tf, scale)), flags)?;
                self.stage(ctx, &l, stack, work, None, STAGE_PRE_INV)
            }
        }
    }

    /// The in-object propagation of `stack`'s planes of `waves`, `[nf_max * nmode, ny, nx]`,
    /// in place: `F⁻¹ T F` of each for the transfer function `tf`, `[ny, nx]` on the
    /// unshifted frequency grid, or with `adjoint` `F⁻¹ conj(T) F`.
    pub fn gap(
        &self,
        ctx: &ExecutionContext,
        d: &ViewDims,
        stack: StackRef<'_>,
        tf: &DeviceBuffer<C32>,
        waves: &mut DeviceBuffer<C32>,
        adjoint: bool,
    ) -> Result<(), Error> {
        let l = Launch::new(d, stack, waves)?;
        let conj = if adjoint { STAGE_CONJ } else { 0 };
        let flags = STAGE_COLS | STAGE_PRE_FWD | STAGE_MUL | conj | STAGE_POST_INV;
        self.stage(ctx, &l, stack, waves, None, STAGE_PRE_FWD)?;
        self.stage(ctx, &l, stack, waves, Some((tf, 1.0 / d.n() as f32)), flags)?;
        self.stage(ctx, &l, stack, waves, None, STAGE_PRE_INV)
    }

    /// One line pass of `flags` over `buf`, multiplying by `mul`'s plane times its scale.
    fn stage(
        &self,
        ctx: &ExecutionContext,
        l: &Launch,
        stack: StackRef<'_>,
        buf: &DeviceBuffer<C32>,
        mul: Option<(&DeviceBuffer<C32>, f32)>,
        flags: u32,
    ) -> Result<(), Error> {
        let cols = flags & STAGE_COLS != 0;
        let (side, config) = if cols { l.cols } else { l.rows };
        let (plane, scale) = match mul {
            Some((tf, scale)) => {
                check_len("transfer function", tf, l.n)?;
                (dptr(tf) as *const f32, scale)
            }
            None => (std::ptr::null(), 0.0),
        };
        let (m, p, ctl, g) = (&self.module, dptr(buf), l.ctl, l.g);
        let mut reads = vec![stack.storage()];
        reads.extend(mul.map(|(tf, _)| tf.storage()));
        self.base.launch(ctx, &reads, &[buf.storage()], |stream| {
            // SAFETY: `Launch::new` checked `buf` for `g` and the side's FFT and shared
            // memory; `mul` holds a plane wherever `flags` name it.
            unsafe {
                match side {
                    Side::N64 => m.optics_stage_64(stream, config, p, plane, scale, ctl, g, flags),
                    Side::N128 => {
                        m.optics_stage_128(stream, config, p, plane, scale, ctl, g, flags)
                    }
                    Side::N256 => {
                        m.optics_stage_256(stream, config, p, plane, scale, ctl, g, flags)
                    }
                    Side::N512 => {
                        m.optics_stage_512(stream, config, p, plane, scale, ctl, g, flags)
                    }
                }
            }
        })
    }
}

/// The launches of the stages on one stack's planes.
struct Launch {
    /// The row and column line passes: a side's kernel and its launch shape.
    rows: (Side, LaunchConfig),
    cols: (Side, LaunchConfig),
    g: StackGeom,
    ctl: *const StackCtl,
    /// Planes the buffer holds for the stack, and a plane's values.
    planes: usize,
    n: usize,
}

impl Launch {
    /// For `stack` of a scan of `d`, its planes in `buf`.
    fn new(d: &ViewDims, stack: StackRef<'_>, buf: &DeviceBuffer<C32>) -> Result<Self, Error> {
        let g = d.geom(stack)?;
        let planes = g.nf_max as usize * d.nmode as usize;
        check_len("planes", buf, planes * d.n())?;
        let [ny, nx] = d.frame;
        // Lines of `n` points, `per` to a plane.
        let pass = |n: u32, per: u32| {
            let bad = |what: &str| Error::Shape(format!("propagator stage, {d:?}: {what} {n}"));
            let side = Side::ALL.iter().copied().find(|&s| s as u32 == n);
            let (Some(side), Some(f), Some(i)) = (side, block_fft(n, false), block_fft(n, true))
            else {
                return Err(bad("no in-kernel FFT for side"));
            };
            let bpb = f.suggested_ffts_per_block.min(i.suggested_ffts_per_block);
            let blocks = (planes as u64 * per as u64).div_ceil(bpb as u64);
            let blocks = u32::try_from(blocks)
                .ok()
                .filter(|&b| b <= i32::MAX as u32)
                .ok_or_else(|| bad("the lines exceed a 1D grid at side"))?;
            let config = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (f.threads_per_fft, bpb, 1),
                shared_mem_bytes: f.smem_per_fft.max(i.smem_per_fft) * bpb,
            };
            Ok((side, config))
        };
        Ok(Self {
            rows: pass(nx, ny)?,
            cols: pass(ny, nx)?,
            g,
            ctl: stack.ctl() as *const StackCtl,
            planes,
            n: d.n(),
        })
    }
}
