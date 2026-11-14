//! Checked launches of the propagator stages (`ptyrs_kernels::optics`): what a propagator
//! other than the Fourier transform adds around the fused passes' transform, and the
//! in-object propagation between the slices of a multislice object.
//!
//! The fused row pass leaves a stack's row transforms in `work`. [`OpticsKernels::exit`]
//! turns them into the row transforms of the staged waves before the column pass, and after
//! it, as the adjoint, the correction's back; the far field and a forward Fresnel transform
//! have no stage and launch nothing.
//!
//! A multislice object's waves never pass the fused row pass, which multiplies one object
//! by the probe: [`OpticsKernels::slice_mul`] is a slice's product and its adjoint,
//! [`OpticsKernels::slice_phase`] the exit plane's phase, [`OpticsKernels::rows`] the row
//! transform the column pass takes and gives, and [`OpticsKernels::slice_pair`] the object
//! adjoint's integrand.

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{DeviceBuffer, LaunchConfig};
use ptyrs_kernels::optics::{
    SLICE_CONJ, SLICE_OFFSET, SLICE_PROBE, STAGE_COLS, STAGE_CONJ, STAGE_MUL, STAGE_POST_FWD,
    STAGE_POST_INV, STAGE_PRE_FWD, STAGE_PRE_INV, WavePlanes, optics_kernels,
};

use crate::gpu::fused::Side;
use crate::gpu::ops::grid_1d;
use crate::gpu::{
    C32, Error, Kernels, StackCtl, StackGeom, StackRef, ViewAddr, ViewDims, Views, block_fft,
    check_len, dptr,
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

/// A stack's waves at the exit wave's stored pixels, by what each frame's planes hold: view
/// `m` of a frame reads the `m % nw`th of its `nw` planes, its own or its probe mode's
/// (`m % nprobe`).
#[derive(Clone, Copy)]
pub enum FrameWaves<'a> {
    /// A plane per probe mode, `[nf_max * nprobe, ny, nx]`: the moved probe
    /// ([`crate::gpu::PositionKernels::shift_w`]).
    Probes(&'a DeviceBuffer<C32>),
    /// A plane per view, `[nf_max * nmode, ny, nx]`: a multislice object's wave entering a
    /// slice, or an exit-wave update.
    Views(&'a DeviceBuffer<C32>),
}

impl<'a> FrameWaves<'a> {
    /// The buffer and its planes as a kernel reads them, checked to hold `nf_max` frames
    /// under `d`.
    pub(super) fn planes(
        self,
        d: &ViewDims,
        nf_max: u32,
    ) -> Result<(&'a DeviceBuffer<C32>, WavePlanes), Error> {
        let (w, nw) = match self {
            Self::Probes(w) => (w, d.nprobe),
            Self::Views(w) => (w, d.nmode),
        };
        check_len("waves", w, nf_max as usize * nw as usize * d.n())?;
        let planes = WavePlanes {
            w: dptr::<C32, f32>(w).cast_const(),
            nw,
        };
        Ok((w, planes))
    }
}

/// What a slice multiplies: [`OpticsKernels::slice_mul`]'s source.
#[derive(Clone, Copy)]
pub enum SliceSource<'a> {
    /// The probe, `[nprobe, ny, nx]` in the natural layout; each view reads its mode.
    Probe(&'a DeviceBuffer<C32>),
    /// The stack's waves.
    Waves(FrameWaves<'a>),
}

/// The exit plane's phase of a multislice object's last slice: `exp(i quad |r|²)`, `r` the
/// pixel from the frame's centre.
#[derive(Clone, Copy)]
pub struct SlicePhase<'a> {
    /// Radians per squared pixel.
    pub quad: f32,
    /// Each frame's probe move `(dy, dx)` in pixels, `[nframe, 2]` by epoch position: the
    /// phase is then displaced with the probe, times `exp(-2i quad r·d)`. The frame's
    /// constant `exp(i quad |d|²)` is left out; no intensity sees it and the conjugate takes
    /// back what was applied.
    pub offsets: Option<&'a DeviceBuffer<f32>>,
}

/// An [`ExitStage`] lent to a pass with the kernels that run it.
#[derive(Clone, Copy)]
pub struct ExitPass<'a> {
    pub kernels: &'a OpticsKernels,
    pub stage: &'a ExitStage,
    /// The propagator's intensity scale `k`, `ny nx` in the far field: pty-chi's waves are
    /// the passes' over `√(ny nx / k)`.
    pub scale: f32,
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

    /// The unnormalized row transforms of `stack`'s planes of `waves`, `[nf_max * nmode, ny,
    /// nx]`, in place: forward what the fused row pass leaves of its product, with `inverse`
    /// what it makes of the column pass's back.
    pub fn rows(
        &self,
        ctx: &ExecutionContext,
        d: &ViewDims,
        stack: StackRef<'_>,
        waves: &mut DeviceBuffer<C32>,
        inverse: bool,
    ) -> Result<(), Error> {
        let l = Launch::new(d, stack, waves)?;
        let flags = if inverse {
            STAGE_PRE_INV
        } else {
            STAGE_PRE_FWD
        };
        self.stage(ctx, &l, stack, waves, None, flags)
    }

    /// A slice's product over `stack`'s views into `dst`, `[nf_max * nmode, ny, nx]` at the
    /// exit wave's stored pixels: each view's bilinear window of `obj` times `src`.
    pub fn slice_mul(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        obj: &DeviceBuffer<C32>,
        src: SliceSource<'_>,
        dst: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        self.slice(ctx, views, stack, obj, (src, 0), dst)
    }

    /// The adjoint of [`Self::slice_mul`] over waves: each view's window of `obj`,
    /// conjugated, times `waves` into `dst`, both `[nf_max * nmode, ny, nx]`.
    pub fn slice_adjoint(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        obj: &DeviceBuffer<C32>,
        waves: &DeviceBuffer<C32>,
        dst: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let src = (SliceSource::Waves(FrameWaves::Views(waves)), SLICE_CONJ);
        self.slice(ctx, views, stack, obj, src, dst)
    }

    fn slice(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        obj: &DeviceBuffer<C32>,
        (src, conj): (SliceSource<'_>, u32),
        dst: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let d = views.dims();
        let g = d.geom(stack)?;
        let len = g.nf_max as usize * d.nmode as usize * d.n();
        check_len("obj", obj, d.obj_len())?;
        check_len("dst", dst, len)?;
        let (buf, src, probe) = match src {
            SliceSource::Probe(p) => {
                check_len("probe", p, d.nprobe as usize * d.n())?;
                let src = WavePlanes {
                    w: dptr::<C32, f32>(p).cast_const(),
                    nw: d.nprobe,
                };
                (p, src, SLICE_PROBE)
            }
            SliceSource::Waves(w) => {
                let (w, src) = w.planes(d, g.nf_max)?;
                (w, src, 0)
            }
        };
        let flags = probe | conj;
        let config = grid_1d(len)?;
        let reads = [
            buf.storage(),
            obj.storage(),
            views.addr().storage(),
            stack.storage(),
        ];
        let (m, ctl) = (&self.module, stack.ctl() as *const StackCtl);
        self.base.launch(ctx, &reads, &[dst.storage()], |stream| {
            // SAFETY: the lengths checked above; `Views` checked each window.
            unsafe {
                m.optics_slice_mul(
                    stream,
                    config,
                    dptr(dst),
                    src,
                    dptr::<C32, f32>(obj).cast_const(),
                    dptr::<ViewAddr, ViewAddr>(views.addr()).cast_const(),
                    ctl,
                    g,
                    flags,
                )
            }
        })
    }

    /// `phase` on `stack`'s planes of `waves`, `[nf_max * nmode, ny, nx]` at the exit wave's
    /// stored pixels, in place; with `adjoint` its conjugate.
    pub fn slice_phase(
        &self,
        ctx: &ExecutionContext,
        d: &ViewDims,
        stack: StackRef<'_>,
        phase: SlicePhase<'_>,
        waves: &mut DeviceBuffer<C32>,
        adjoint: bool,
    ) -> Result<(), Error> {
        let g = d.geom(stack)?;
        let len = g.nf_max as usize * d.nmode as usize * d.n();
        check_len("waves", waves, len)?;
        let mut reads = vec![stack.storage()];
        let (at, offset) = match phase.offsets {
            Some(o) => {
                check_len("offsets", o, 2 * d.nframe as usize)?;
                reads.push(o.storage());
                (dptr::<f32, f32>(o).cast_const(), SLICE_OFFSET)
            }
            None => (std::ptr::null(), 0),
        };
        let flags = offset | if adjoint { SLICE_CONJ } else { 0 };
        let config = grid_1d(len)?;
        let (m, ctl, quad) = (&self.module, stack.ctl() as *const StackCtl, phase.quad);
        self.base.launch(ctx, &reads, &[waves.storage()], |stream| {
            // SAFETY: the lengths checked above.
            unsafe { m.optics_slice_phase(stream, config, dptr(waves), at, ctl, g, quad, flags) }
        })
    }

    /// `dst = conj(a) b` over `stack`'s planes, all `[nf_max * nmode, ny, nx]`.
    pub fn slice_pair(
        &self,
        ctx: &ExecutionContext,
        d: &ViewDims,
        stack: StackRef<'_>,
        [a, b]: [&DeviceBuffer<C32>; 2],
        dst: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let g = d.geom(stack)?;
        let len = g.nf_max as usize * d.nmode as usize * d.n();
        for (what, buf) in [("a", a), ("b", b), ("dst", &*dst)] {
            check_len(what, buf, len)?;
        }
        let config = grid_1d(len)?;
        let reads = [a.storage(), b.storage(), stack.storage()];
        let (m, ctl) = (&self.module, stack.ctl() as *const StackCtl);
        self.base.launch(ctx, &reads, &[dst.storage()], |stream| {
            // SAFETY: the lengths checked above.
            unsafe {
                m.optics_slice_pair(
                    stream,
                    config,
                    dptr(dst),
                    dptr::<C32, f32>(a).cast_const(),
                    dptr::<C32, f32>(b).cast_const(),
                    ctl,
                    g,
                )
            }
        })
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
