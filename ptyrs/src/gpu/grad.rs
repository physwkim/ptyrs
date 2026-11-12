//! Checked launches of the Poisson ML conjugate-gradient kernels (`ptyrs_kernels::grad`).
//!
//! # One CG cycle (PyNX `ML`) in these calls
//!
//! Per updated array (object, probe): the gradient `g` accumulates into zeroed sums (per
//! stack [`Kernels::obj_probe_to_psi`], forward FFT, [`GradKernels::poisson_grad`], inverse
//! FFT, then [`Kernels::psi_to_obj`] and [`Kernels::psi_to_probe`]), `h` holds the last
//! cycle's gradient (zeros before the first) and `d` the direction (zeros).
//! [`GradKernels::beta_parts`] of every array into its own `CG_BLOCKS` pairs of one parts
//! buffer, then [`GradKernels::sum_parts`] of them all gives the Polak–Ribière pair; it is
//! `(sum |g|^2, 0)` on the first cycle, so `β = 0`. [`GradKernels::direction`] per array,
//! then the line-search pair `γ`: per stack [`GradKernels::gamma_fields`], a forward FFT of
//! each of its three stacks and [`GradKernels::gamma_frames`], then
//! [`GradKernels::sum_parts`] of the [`gamma_parts`] pairs. Then [`GradKernels::step`] per
//! array. The coefficients never leave the device, so a cycle is one host-sync-free sequence.

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
pub use ptyrs_kernels::grad::{CG_BLOCKS, GAMMA_CHUNK};
use ptyrs_kernels::grad::{GammaPtrs, NormPtrs, ObjProbePtrs, grad_kernels};
use ptyrs_kernels::pixel::{PIXEL_THREADS, PositionPartial, SumPtrs};

use crate::gpu::ops::grid_1d;
use crate::gpu::{
    C32, Error, Kernels, LlkPartial, Model, ScalePartial, StackCtl, StackRef, UpdateSums, ViewDims,
    Views, check_len, dptr,
};

/// The line-search exit-wave stacks of [`GradKernels::gamma_fields`], each of `nf_max *
/// nmode` planes like a stack's `psi`: `po = O P`, `a = dO P + O dP`, `b = dO dP` (times the
/// far-field phase).
pub struct GammaStacks {
    pub po: DeviceBuffer<C32>,
    pub a: DeviceBuffer<C32>,
    pub b: DeviceBuffer<C32>,
}

/// The counts `[nframe, ny, nx]` that [`GradKernels::gamma_frames`] compares the line
/// search's far fields with, and their model intensity as the fused passes take it.
#[derive(Clone, Copy)]
pub struct GammaCounts<'a> {
    pub counts: &'a DeviceBuffer<u16>,
    pub model: Model<'a>,
}

impl GammaStacks {
    /// Three zeroed stacks of `len` values.
    pub fn zeroed(stream: &CudaStream, len: usize) -> Result<Self, Error> {
        Ok(Self {
            po: DeviceBuffer::zeroed(stream, len)?,
            a: DeviceBuffer::zeroed(stream, len)?,
            b: DeviceBuffer::zeroed(stream, len)?,
        })
    }

    fn check(&self, len: usize) -> Result<(), Error> {
        check_len("po", &self.po, len)?;
        check_len("a", &self.a, len)?;
        check_len("b", &self.b, len)
    }

    fn ptrs(&self) -> GammaPtrs {
        GammaPtrs {
            po: dptr(&self.po),
            a: dptr(&self.a),
            b: dptr(&self.b),
        }
    }
}

/// The factor `s` by which [`GradKernels::scale_obj_probe`] scales `O P`.
#[derive(Clone, Copy)]
pub enum ObjProbeScale<'a> {
    /// PyNX `ScaleObjProbe(absolute=False)`: `s = 1`.
    Relative,
    /// PyNX `ScaleObjProbe(absolute=True)`: `s = sqrt((total - background) / icalc)`, with
    /// `icalc` the unmasked calculated photons of [`GradKernels::icalc_add`], `total` the
    /// unmasked observed photons and `background` the background's of
    /// [`GradKernels::background_photons`] (0 for `None`).
    Absolute {
        icalc: &'a DeviceBuffer<f32>,
        total: f32,
        background: Option<&'a DeviceBuffer<f32>>,
    },
}

/// The CG kernels, bound to the module a [`Kernels`] loaded.
#[derive(Clone, Debug)]
pub struct GradKernels {
    base: Kernels,
    module: grad_kernels::LoadedModule,
}

/// The line-search pairs [`GradKernels::gamma_frames`] writes over a scan of `dims`, one per
/// chunk of [`GAMMA_CHUNK`] pixels of each frame: its `out` holds twice as many floats, and
/// [`GradKernels::sum_parts`] of this many pairs is `γ`.
pub fn gamma_parts(dims: &ViewDims) -> Result<u32, Error> {
    let npart = dims.nframe as usize * gamma_chunks(dims.n()) as usize;
    u32::try_from(npart).map_err(|_| Error::Shape(format!("{npart} line-search pairs exceed u32")))
}

/// Chunks of [`GAMMA_CHUNK`] pixels per frame of `n` pixels.
fn gamma_chunks(n: usize) -> u32 {
    // `n <= 32768^2`, so at most 2^19 chunks.
    n.div_ceil(GAMMA_CHUNK as usize) as u32
}

impl GradKernels {
    /// Binds the CG kernels of the cubin `kernels` already loaded; loads nothing.
    pub fn load(kernels: &Kernels) -> Result<Self, Error> {
        let module = grad_kernels::from_module(kernels.module.as_cuda_module().clone())?;
        Ok(Self {
            base: kernels.clone(),
            module,
        })
    }

    /// One array's Polak–Ribière partial sums `(sum Re(conj(g) (g - h)), sum |h|^2)` into
    /// the `CG_BLOCKS` pairs of `parts` from pair `first`, then `h = g`.
    pub fn beta_parts(
        &self,
        ctx: &ExecutionContext,
        g: &DeviceBuffer<C32>,
        h: &mut DeviceBuffer<C32>,
        n: u32,
        parts: &mut DeviceBuffer<f32>,
        first: u32,
    ) -> Result<(), Error> {
        check_len("g", g, n as usize)?;
        check_len("h", h, n as usize)?;
        check_len("parts", parts, 2 * (first as usize + CG_BLOCKS as usize))?;
        let config = LaunchConfig {
            grid_dim: (CG_BLOCKS, 1, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let out = dptr::<f32, f32>(parts).wrapping_add(2 * first as usize);
        let writes = [h.storage(), parts.storage()];
        self.base.launch(ctx, &[g.storage()], &writes, |stream| {
            // SAFETY: lengths checked above; CG_BLOCKS blocks of PIXEL_THREADS.
            unsafe {
                self.module
                    .cg_beta_parts(stream, config, dptr(g), dptr(h), n, out)
            }
        })
    }

    /// `pair = sum` of the first `npart` pairs of `parts`, in a fixed order.
    pub fn sum_parts(
        &self,
        ctx: &ExecutionContext,
        parts: &DeviceBuffer<f32>,
        npart: u32,
        pair: &mut DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        check_len("parts", parts, 2 * npart as usize)?;
        check_len("pair", pair, 2)?;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        self.base
            .launch(ctx, &[parts.storage()], &[pair.storage()], |stream| {
                // SAFETY: lengths checked above; one PIXEL_THREADS block.
                unsafe {
                    self.module
                        .cg_sum_parts(stream, config, dptr(parts), npart, dptr(pair))
                }
            })
    }

    /// Adds `(sum |x|, sum |x|^2)` over the first `n` values of `x` to the first `CG_BLOCKS`
    /// pairs of `parts`, as block partials that [`Self::sum_parts`] totals; zero `parts` to
    /// start a sum, then add arrays or stacks in a fixed order.
    pub fn norm_parts(
        &self,
        ctx: &ExecutionContext,
        x: &DeviceBuffer<C32>,
        n: u32,
        parts: &mut DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        check_len("x", x, n as usize)?;
        check_len("parts", parts, 2 * CG_BLOCKS as usize)?;
        let config = LaunchConfig {
            grid_dim: (CG_BLOCKS, 1, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        self.base
            .launch(ctx, &[x.storage()], &[parts.storage()], |stream| {
                // SAFETY: lengths checked above; CG_BLOCKS blocks of PIXEL_THREADS.
                unsafe {
                    self.module
                        .cg_norm_parts(stream, config, dptr(x), n, dptr(parts))
                }
            })
    }

    /// Adds the calculated photons of the stack's frames, the `icalc` of the `partials`
    /// [`Kernels::llk`] wrote for it, each frame's times its intensity `scale` (`[nframe]`)
    /// if given, to `out[0]`; zero `out` to start a scan, then add its stacks in order.
    /// Masked pixels are left out, as in the LLK.
    pub fn icalc_add(
        &self,
        ctx: &ExecutionContext,
        dims: &ViewDims,
        stack: StackRef<'_>,
        partials: &DeviceBuffer<LlkPartial>,
        scale: Option<&DeviceBuffer<f32>>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        let g = dims.geom(stack)?;
        check_len("partials", partials, g.nf_max as usize)?;
        if let Some(scale) = scale {
            check_len("scale", scale, dims.nframe as usize)?;
        }
        check_len("out", out, 1)?;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut reads = vec![partials.storage(), stack.storage()];
        reads.extend(scale.map(|b| b.storage()));
        self.base.launch(ctx, &reads, &[out.storage()], |stream| {
            // SAFETY: lengths checked above; one PIXEL_THREADS block.
            unsafe {
                self.module.cg_icalc_add(
                    stream,
                    config,
                    dptr(partials),
                    scale.map_or(std::ptr::null(), |b| dptr::<f32, f32>(b).cast_const()),
                    stack.ctl() as *const StackCtl,
                    g,
                    dptr(out),
                )
            }
        })
    }

    /// The background's photons over the unmasked pixels of every frame into `out[0]`,
    /// `sum_j b[j] nobs[j]` in a fixed order, `nobs[j]` the frames where detector pixel `j`
    /// is unmasked, for [`ObjProbeScale::Absolute`].
    pub fn background_photons(
        &self,
        ctx: &ExecutionContext,
        b: &DeviceBuffer<f32>,
        nobs: &DeviceBuffer<u32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        let n = u32::try_from(b.len())
            .map_err(|_| Error::Shape(format!("{} background pixels exceed u32", b.len())))?;
        check_len("nobs", nobs, b.len())?;
        check_len("out", out, 1)?;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let reads = [b.storage(), nobs.storage()];
        self.base.launch(ctx, &reads, &[out.storage()], |stream| {
            // SAFETY: `b` and `nobs` hold `n` values, `out` one; one PIXEL_THREADS block.
            unsafe {
                self.module.cg_background_photons(
                    stream,
                    config,
                    dptr::<f32, f32>(b).cast_const(),
                    dptr::<u32, u32>(nobs).cast_const(),
                    n,
                    dptr(out),
                )
            }
        })
    }

    /// PyNX `ScaleObjProbe` of `obj` and `probe` from the [`Self::sum_parts`] totals of
    /// [`Self::norm_parts`] `norms = [object, probe]`: with `s` of `scale`,
    /// `obj *= sqrt(sum |P| / sum |O| * s)` and `probe *= sqrt(sum |O| / sum |P| * s)`.
    pub fn scale_obj_probe(
        &self,
        ctx: &ExecutionContext,
        obj: &mut DeviceBuffer<C32>,
        probe: &mut DeviceBuffer<C32>,
        norms: [&DeviceBuffer<f32>; 2],
        scale: ObjProbeScale<'_>,
    ) -> Result<(), Error> {
        for (what, pair) in ["obj", "probe"].into_iter().zip(norms) {
            check_len(what, pair, 2)?;
        }
        let (icalc, total, background) = match scale {
            ObjProbeScale::Relative => (None, 0.0, None),
            ObjProbeScale::Absolute {
                icalc,
                total,
                background,
            } => {
                check_len("icalc", icalc, 1)?;
                if let Some(b) = background {
                    check_len("background photons", b, 1)?;
                }
                (Some(icalc), total, background)
            }
        };
        let len = |what: &str, b: &DeviceBuffer<C32>| {
            u32::try_from(b.len())
                .map_err(|_| Error::Shape(format!("{what} holds {} values, over u32", b.len())))
        };
        let (nobj, nprobe) = (len("obj", obj)?, len("probe", probe)?);
        let config = grid_1d(nobj.max(nprobe) as usize)?;
        let reads: Vec<_> = norms
            .iter()
            .chain(icalc.as_slice())
            .chain(background.as_slice())
            .map(|b| b.storage())
            .collect();
        let writes = [obj.storage(), probe.storage()];
        self.base.launch(ctx, &reads, &writes, |stream| {
            let [o, p] = norms.map(dptr::<f32, f32>);
            let norms = NormPtrs {
                obj: o,
                probe: p,
                icalc: icalc.map_or(std::ptr::null(), |b| dptr::<f32, f32>(b).cast_const()),
                background: (background)
                    .map_or(std::ptr::null(), |b| dptr::<f32, f32>(b).cast_const()),
            };
            // SAFETY: each pair holds two floats, `icalc` and `background` one or are null; one
            // thread per element of the longer array.
            unsafe {
                self.module.cg_scale_obj_probe(
                    stream,
                    config,
                    dptr(obj),
                    nobj,
                    dptr(probe),
                    nprobe,
                    norms,
                    total,
                )
            }
        })
    }

    /// `d = β d - g` with `β = beta[0] / beta[1]`, or 0 where that is negative or not finite;
    /// consumes the gradient sums `g` (zero after).
    pub fn direction(
        &self,
        ctx: &ExecutionContext,
        d: &mut DeviceBuffer<C32>,
        g: &mut UpdateSums,
        n: u32,
        beta: &DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        check_len("d", d, n as usize)?;
        check_len("g.num", &g.num, n as usize)?;
        check_len("g.den", &g.den, n as usize)?;
        check_len("beta", beta, 2)?;
        if n == 0 {
            return Ok(());
        }
        let config = LaunchConfig::for_num_elems(n);
        let writes = [d.storage(), g.num.storage(), g.den.storage()];
        self.base.launch(ctx, &[beta.storage()], &writes, |stream| {
            let g = SumPtrs {
                num: dptr(&g.num),
                den: dptr(&g.den),
            };
            // SAFETY: lengths checked above.
            unsafe {
                self.module
                    .cg_direction(stream, config, dptr(d), g, n, dptr(beta))
            }
        })
    }

    /// The stack's Poisson gradient in Fourier space, in place on its forward-FFT'd exit
    /// waves (PyNX `GradPoissonFourier`, no Hann filter): `psi *= 2 (1 - obs / max(calc,
    /// 1e-12)) / (ny * nx)`, `calc = sum_m |psi|^2 / (ny * nx)`, and masked pixels zeroed.
    /// The unnormalized inverse FFT of the result, back-projected by
    /// [`Kernels::psi_to_obj`] and [`Kernels::psi_to_probe`], gives the sums' `num` as the
    /// gradients `2 dL/dO*` and `2 dL/dP*` of the unnormalised Poisson sum.
    pub fn poisson_grad(
        &self,
        ctx: &ExecutionContext,
        dims: &ViewDims,
        stack: StackRef<'_>,
        counts: &DeviceBuffer<u16>,
        psi: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let g = dims.geom(stack)?;
        let (n, nf_max) = (dims.n(), g.nf_max as usize);
        check_len("counts", counts, dims.nframe as usize * n)?;
        check_len("psi", psi, nf_max * dims.nmode as usize * n)?;
        let config = grid_1d(nf_max * n)?;
        let reads = [counts.storage(), stack.storage()];
        self.base.launch(ctx, &reads, &[psi.storage()], |stream| {
            // SAFETY: lengths checked above.
            unsafe {
                self.module.cg_poisson_grad(
                    stream,
                    config,
                    dptr(psi),
                    dptr(counts),
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// The stack's line-search exit waves along `d = [dO, dP]` from `x = [O, P]` into `out`,
    /// as [`Kernels::obj_probe_to_psi`] lays out and computes `psi`: `po` is its `psi` of
    /// `(O, P)` bit for bit, `b` of `(dO, dP)`, and `a = (dO P + O dP) q`.
    pub fn gamma_fields(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        x: [&DeviceBuffer<C32>; 2],
        d: [&DeviceBuffer<C32>; 2],
        out: &mut GammaStacks,
    ) -> Result<(), Error> {
        let dims = views.dims();
        let g = dims.geom(stack)?;
        let (n, nk_max) = (dims.n(), g.nf_max as usize * dims.nmode as usize);
        let [obj, probe] = x;
        let [d_obj, d_probe] = d;
        check_len("obj", obj, dims.obj_len())?;
        check_len("d_obj", d_obj, dims.obj_len())?;
        check_len("probe", probe, dims.nprobe as usize * n)?;
        check_len("d_probe", d_probe, dims.nprobe as usize * n)?;
        out.check(nk_max * n)?;
        let config = grid_1d(nk_max * n)?;
        let reads = [
            views.addr().storage(),
            obj.storage(),
            d_obj.storage(),
            probe.storage(),
            d_probe.storage(),
            stack.storage(),
        ];
        let writes = [out.po.storage(), out.a.storage(), out.b.storage()];
        self.base.launch(ctx, &reads, &writes, |stream| {
            // SAFETY: lengths checked above; `Views::new` checked every view lies inside the
            // object with valid mode indices.
            unsafe {
                let pair = |obj, probe| ObjProbePtrs {
                    obj: dptr(obj),
                    probe: dptr(probe),
                };
                self.module.cg_gamma_fields(
                    stream,
                    config,
                    pair(obj, probe),
                    pair(d_obj, d_probe),
                    dptr(views.addr()),
                    out.ptrs(),
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// Line-search pairs of the stack's frames from its forward-FFT'd (unnormalized)
    /// [`GammaStacks`]: PyNX `CG_Poisson_Gamma`'s numerator and denominator (its cross-term
    /// slip corrected) over each chunk of [`GAMMA_CHUNK`] pixels, a fixed-order block sum, at
    /// pair `f * nchunk + c` of `out` for chunk `c` of frame `f`; other frames' pairs are left
    /// as they are. The model intensity is `obs`'s, which takes a [`Model::scale`] and a
    /// [`Model::background`] but no [`Model::shift`]. `out` holds the scan's [`gamma_parts`]
    /// pairs, which [`Self::sum_parts`] totals into `γ`.
    pub fn gamma_frames(
        &self,
        ctx: &ExecutionContext,
        dims: &ViewDims,
        stack: StackRef<'_>,
        obs: GammaCounts<'_>,
        x: &GammaStacks,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        let g = dims.geom(stack)?;
        let n = dims.n();
        let GammaCounts { counts, model } = obs;
        check_len("counts", counts, dims.nframe as usize * n)?;
        if model.shift.is_some() {
            return Err(Error::Shape(
                "the line search does not model position shifts".into(),
            ));
        }
        let (scale, background) = (model.scale, model.background);
        if let Some(scale) = scale {
            check_len("scale", scale, dims.nframe as usize)?;
        }
        if let Some(b) = background {
            check_len("background", b, n)?;
        }
        x.check(g.nf_max as usize * dims.nmode as usize * n)?;
        check_len("out", out, 2 * gamma_parts(dims)? as usize)?;
        if g.nf_max > u16::MAX as u32 {
            return Err(Error::Shape(format!("{} frames exceed grid y", g.nf_max)));
        }
        let config = LaunchConfig {
            grid_dim: (gamma_chunks(n), g.nf_max, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut reads = vec![
            counts.storage(),
            x.po.storage(),
            x.a.storage(),
            x.b.storage(),
            stack.storage(),
        ];
        reads.extend(scale.map(|b| b.storage()));
        reads.extend(background.map(|b| b.storage()));
        let opt = |b: Option<&DeviceBuffer<f32>>| {
            b.map_or(std::ptr::null(), |b| dptr::<f32, f32>(b).cast_const())
        };
        self.base.launch(ctx, &reads, &[out.storage()], |stream| {
            // SAFETY: lengths checked above; one block of PIXEL_THREADS per chunk and frame.
            unsafe {
                self.module.cg_gamma_frames(
                    stream,
                    config,
                    x.ptrs(),
                    dptr(counts),
                    opt(scale),
                    opt(background),
                    dptr(out),
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// With `γ = gamma[0] / gamma[1]`: `x += γ d` if `γ` is finite, else `x` stays and
    /// `d = -h`, `h` being this cycle's gradient as [`Self::beta_parts`] leaves it.
    pub fn step(
        &self,
        ctx: &ExecutionContext,
        x: &mut DeviceBuffer<C32>,
        d: &mut DeviceBuffer<C32>,
        h: &DeviceBuffer<C32>,
        n: u32,
        gamma: &DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        check_len("x", x, n as usize)?;
        check_len("d", d, n as usize)?;
        check_len("h", h, n as usize)?;
        check_len("gamma", gamma, 2)?;
        if n == 0 {
            return Ok(());
        }
        let config = LaunchConfig::for_num_elems(n);
        let reads = [h.storage(), gamma.storage()];
        let writes = [x.storage(), d.storage()];
        self.base.launch(ctx, &reads, &writes, |stream| {
            // SAFETY: lengths checked above.
            unsafe {
                self.module
                    .cg_step(stream, config, dptr(x), dptr(d), dptr(h), n, dptr(gamma))
            }
        })
    }

    /// Poisson's fixed-point step of the per-frame intensity scale `s` (`[nframe]`, as
    /// [`Model::scale`]) from the frames' far sums ([`crate::gpu::FarSums::scale`]):
    /// `s[f] *= sums[f].ratio / sums[f].intensity` where the intensity is above 0, then divided
    /// by its mean.
    pub fn scale_step(
        &self,
        ctx: &ExecutionContext,
        scale: &mut DeviceBuffer<f32>,
        sums: &DeviceBuffer<ScalePartial>,
    ) -> Result<(), Error> {
        let nframe = scale.len();
        check_len("sums", sums, nframe)?;
        let n = u32::try_from(nframe)
            .map_err(|_| Error::Shape(format!("{nframe} frames exceed u32")))?;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        self.base
            .launch(ctx, &[sums.storage()], &[scale.storage()], |stream| {
                // SAFETY: both hold `nframe` entries; one PIXEL_THREADS block.
                unsafe {
                    self.module
                        .cg_scale_step(stream, config, dptr(scale), dptr(sums), n)
                }
            })
    }

    /// pty-chi's gradient step of the offsets `[nframe, 2]` (as [`crate::gpu::Shift`]'s) from
    /// each frame's position sums ([`crate::gpu::ShiftSums`]): per axis, `s = clamp(step num /
    /// (den + max den), -limit, limit)`, 0 where that denominator is not above 0, then `d +=
    /// s - mean s`.
    pub fn position_step(
        &self,
        ctx: &ExecutionContext,
        offsets: &mut DeviceBuffer<f32>,
        sums: [&DeviceBuffer<PositionPartial>; 2],
        [step, limit]: [f32; 2],
    ) -> Result<(), Error> {
        let [num, den] = sums;
        let nframe = num.len();
        check_len("offsets", offsets, 2 * nframe)?;
        check_len("den", den, nframe)?;
        let n = u32::try_from(nframe)
            .map_err(|_| Error::Shape(format!("{nframe} frames exceed u32")))?;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let reads = [num.storage(), den.storage()];
        self.base
            .launch(ctx, &reads, &[offsets.storage()], |stream| {
                // SAFETY: `offsets` holds `2 nframe` floats, `num` and `den` `nframe`
                // entries; one PIXEL_THREADS block.
                unsafe {
                    self.module.cg_position_step(
                        stream,
                        config,
                        dptr(offsets),
                        dptr(num),
                        dptr(den),
                        n,
                        step,
                        limit,
                    )
                }
            })
    }
}
