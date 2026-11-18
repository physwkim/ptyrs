//! Checked launches of the rPIE kernels (`ptyrs_kernels::rpie`).
//!
//! # rPIE in these calls (per batch of frames, one object mode)
//!
//! pty-chi's `RPIEReconstructor` but for the object step's weight. `obj_probe_to_psi` into
//! `psi`; forward FFT of a copy `dlt`; [`RpieKernels::amplitude`]; inverse FFT; `dlt -= psi`
//! (`scale_c32(psi, -1)`, `add_c32`), so `dlt` is pty-chi's `Δ`. Object step, normalised by
//! the batch's overlap: per probe mode [`RpieKernels::overlap`] places `|P|^2` at every view
//! of the batch as the step places it and [`RpieKernels::overlap_read`] reads it back per
//! view into `d`; [`RpieKernels::sum_max`] takes `max sum_m d` over the batch,
//! [`RpieKernels::overlap_weight`] leaves `Δ / ((1 - a) d + a max)` in `psi`, and
//! `psi_to_obj(psi, P)` into the (consumed, so zero) sums yields the object step's direction
//! in `num`. At one frame per batch `d = |P|^2`, pty-chi's weight. Probe step:
//! `obj_probe_to_psi` with a probe of ones gives the windows `O_v q`;
//! [`RpieKernels::power_max`] per plane and [`RpieKernels::weight_inplace`] of `dlt` by
//! them, then `psi_to_probe(dlt, O)`. Both steps read the batch-start arrays: apply
//! [`RpieKernels::step`], which consumes the sums, only after both sums are taken.
//!
//! Under the position model `W = T_d(P q)` is each frame's moved probe, which
//! `shifted_probe` and [`crate::gpu::PositionKernels::shift_w`] leave; `psi` is
//! [`RpieKernels::obj_wave_to_psi`] of it. The object step's overlap is
//! [`RpieKernels::overlap_moved`] of `W`, and its weighted `Δ` goes through
//! `shifted_psi_sums`; the probe step takes the weighted `dlt` through `shifted_psi_sums`
//! and `shifted_probe_sums`, which move it back by `-d`. Under the full-object sampling the
//! sampler places the overlap from [`RpieKernels::mode_power`]'s patches, and
//! [`RpieKernels::overlap_read`] reads it back from the sampler's windows.

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{DeviceBuffer, LaunchConfig};
use ptyrs_kernels::pixel::PIXEL_THREADS;
use ptyrs_kernels::rpie::{OverlapOf, WeightGeom, rpie_kernels};

pub use ptyrs_kernels::rpie::POWER_CHUNK;

use crate::gpu::{BhScan, C32, Error, Kernels, StackCtl, StackRef, Views, check_len, dptr, zero};

/// The rPIE kernels, bound to the module a [`Kernels`] loaded.
#[derive(Clone, Debug)]
pub struct RpieKernels {
    base: Kernels,
    module: rpie_kernels::LoadedModule,
}

/// A 1D launch over `total` elements; errors if it does not fit `u32`.
fn elems(total: usize) -> Result<(u32, LaunchConfig), Error> {
    let n = u32::try_from(total)
        .map_err(|_| Error::Shape(format!("{total} elements exceed a 1D launch")))?;
    Ok((n, LaunchConfig::for_num_elems(n)))
}

impl RpieKernels {
    /// Binds the rPIE kernels of the cubin `kernels` already loaded; loads nothing.
    pub fn load(kernels: &Kernels) -> Result<Self, Error> {
        let module = rpie_kernels::from_module(kernels.module.as_cuda_module().clone())?;
        Ok(Self {
            base: kernels.clone(),
            module,
        })
    }

    /// pty-chi's modulus constraint on the forward-FFT'd stack: each pixel's modes are
    /// scaled by `sqrt(obs + 1e-7) / (sqrt(icalc) + 1e-7) / n`, `icalc = sum_m |Psi_m|^2 /
    /// n`, or by `1 / n` where the count is `MASKED`; the unnormalized inverse FFT then
    /// yields pty-chi's corrected exit wave. The stack's frames are `scan`'s epoch positions.
    pub fn amplitude(
        &self,
        ctx: &ExecutionContext,
        scan: BhScan<'_>,
        stack: StackRef<'_>,
        psi: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let dims = scan.check()?;
        let g = dims.geom(stack)?;
        let (n, nf_max) = (g.ny as usize * g.nx as usize, g.nf_max as usize);
        check_len("psi", psi, nf_max * dims.nmode as usize * n)?;
        let (_, config) = elems(nf_max * n)?;
        let reads = [
            scan.counts.storage(),
            scan.order.ids().storage(),
            stack.storage(),
        ];
        self.base.launch(ctx, &reads, &[psi.storage()], |stream| {
            // SAFETY: lengths checked above; the ids are a permutation of the frames.
            unsafe {
                self.module.rpie_amplitude(
                    stream,
                    config,
                    dptr(psi),
                    scan.obs(),
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// Exit waves of each frame's own probe `w`, `[nf_max * nprobe, ny, nx]` in the exit
    /// waves' layout with plane `fl * nprobe + p` mode `p` of frame `fl`, as
    /// [`crate::gpu::PositionKernels::shift_w`] leaves the moved probe: `psi[kl] = O_v w[fl *
    /// nprobe + v.probe]` for view `v = f0 * nmode + kl`, `O_v` and `psi` as in
    /// [`Kernels::obj_probe_to_psi`].
    pub fn obj_wave_to_psi(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        obj: &DeviceBuffer<C32>,
        w: &DeviceBuffer<C32>,
        psi: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let d = views.dims();
        let g = d.geom(stack)?;
        let (n, nf_max) = (d.n(), g.nf_max as usize);
        check_len("obj", obj, d.obj_len())?;
        check_len("w", w, nf_max * d.nprobe as usize * n)?;
        check_len("psi", psi, nf_max * d.nmode as usize * n)?;
        let (_, config) = elems(nf_max * d.nmode as usize * n)?;
        let reads = [
            views.addr().storage(),
            obj.storage(),
            w.storage(),
            stack.storage(),
        ];
        self.base.launch(ctx, &reads, &[psi.storage()], |stream| {
            // SAFETY: lengths checked above; `Views::new` checked every view lies inside the
            // object with valid mode indices.
            unsafe {
                self.module.rpie_obj_wave_to_psi(
                    stream,
                    config,
                    dptr(obj),
                    dptr(w),
                    dptr(views.addr()),
                    dptr(psi),
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// `out[b] = max_i sum_{m < nsum} |x[b * nsum + m, i]|^2` over the groups of `spec`,
    /// through the per-chunk maxes in `chunks` ([`PowerMax::chunks`] floats).
    pub fn power_max(
        &self,
        ctx: &ExecutionContext,
        x: &DeviceBuffer<C32>,
        spec: PowerMax,
        chunks: &mut DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        let PowerMax { ngroup, nsum, n } = spec;
        let config = spec.grid()?;
        check_len("x", x, ngroup as usize * nsum as usize * n as usize)?;
        check_len("chunks", chunks, spec.chunks())?;
        check_len("out", out, ngroup as usize)?;
        let nchunk = config.grid_dim.0;
        self.base
            .launch(ctx, &[x.storage()], &[chunks.storage()], |stream| {
                // SAFETY: lengths checked above; one PIXEL_THREADS block per (chunk, group).
                unsafe {
                    self.module
                        .rpie_power_max(stream, config, dptr(x), nsum, n, dptr(chunks))
                }
            })?;
        let config = LaunchConfig::for_num_elems(ngroup);
        self.base
            .launch(ctx, &[chunks.storage()], &[out.storage()], |stream| {
                // SAFETY: lengths checked above; one thread per group.
                unsafe {
                    self.module.rpie_max_chunks(
                        stream,
                        config,
                        dptr(chunks),
                        nchunk,
                        ngroup,
                        dptr(out),
                    )
                }
            })
    }

    /// `x[k, i] /= (1 - alpha) |w[k, i]|^2 + alpha mx[k]` over the planes of `spec`.
    pub fn weight_inplace(
        &self,
        ctx: &ExecutionContext,
        x: &mut DeviceBuffer<C32>,
        w: &DeviceBuffer<C32>,
        mx: &DeviceBuffer<f32>,
        spec: Weight,
    ) -> Result<(), Error> {
        let total = spec.len()?;
        check_len("x", x, total)?;
        check_len("w", w, total)?;
        check_len("mx", mx, spec.nk as usize)?;
        let (_, config) = elems(total)?;
        let geom = WeightGeom {
            nk: spec.nk,
            n: spec.n,
            alpha: spec.alpha,
        };
        let reads = [w.storage(), mx.storage()];
        self.base.launch(ctx, &reads, &[x.storage()], |stream| {
            // SAFETY: lengths checked above.
            unsafe {
                self.module
                    .rpie_weight(stream, config, dptr(x), dptr(w), dptr(mx), geom)
            }
        })
    }

    /// Probe mode `mode`'s overlap over the stack into `d` `[nyo, nxo]`, the views' one
    /// object mode: `sum_k |P_k|^2` over the stack's views `k` of that mode, each placed as
    /// [`Kernels::psi_to_obj`] places the object step, `P_k` mode `mode` of `probe`
    /// `[nprobe, ny, nx]`. Overwrites the pixels of the object tiles whose run holds a view
    /// of the stack, which hold every pixel such a view samples, and leaves the rest.
    pub fn overlap(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        probe: &DeviceBuffer<C32>,
        mode: u32,
        d: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let dims = views.dims();
        check_len("probe", probe, dims.nprobe as usize * dims.n())?;
        let of = OverlapOf { mode, per_frame: 0 };
        self.launch_overlap(ctx, views, stack, probe, of, d)
    }

    /// [`Self::overlap`] of each frame's own probe `w`, laid out as
    /// [`Self::obj_wave_to_psi`] reads it.
    pub fn overlap_moved(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        w: &DeviceBuffer<C32>,
        mode: u32,
        d: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let dims = views.dims();
        let nf_max = dims.geom(stack)?.nf_max as usize;
        check_len("w", w, nf_max * dims.nprobe as usize * dims.n())?;
        let of = OverlapOf { mode, per_frame: 1 };
        self.launch_overlap(ctx, views, stack, w, of, d)
    }

    /// Launches `rpie_overlap` of `p`, which the caller checked.
    fn launch_overlap(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        p: &DeviceBuffer<C32>,
        of: OverlapOf,
        d: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        let dims = views.dims();
        let g = dims.geom(stack)?;
        if dims.nobj != 1 || of.mode >= dims.nprobe {
            return Err(Error::Shape(format!(
                "overlap of mode {} under {dims:?}",
                of.mode
            )));
        }
        check_len("d", d, dims.obj_len())?;
        let config = dims.stack_tile_grid(g.nf_max)?;
        let [start, len, list] = views.tile_storages();
        let reads = [
            p.storage(),
            views.addr().storage(),
            start,
            len,
            list,
            stack.storage(),
        ];
        self.base.launch(ctx, &reads, &[d.storage()], |stream| {
            // SAFETY: lengths checked above; the tile lists were built from these views.
            unsafe {
                self.module.rpie_overlap(
                    stream,
                    config,
                    dptr(p),
                    of,
                    dptr(views.addr()),
                    views.tile_ptrs(),
                    dptr(d),
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// The overlap each view of probe mode `mode` reads back: `dens[kl, j] = max(0, Re
    /// d(y, x))` for view `f0 * nmode + kl` of the stack, `d` the one-mode object `[nyo,
    /// nxo]` of `views` sampled as [`Kernels::obj_probe_to_psi`] samples it, `j` the exit
    /// waves' pixel of `(y, x)`. The other views' entries are left.
    pub fn overlap_read(
        &self,
        ctx: &ExecutionContext,
        views: &Views,
        stack: StackRef<'_>,
        d: &DeviceBuffer<C32>,
        mode: u32,
        dens: &mut DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        let dims = views.dims();
        let g = dims.geom(stack)?;
        if dims.nobj != 1 || mode >= dims.nprobe {
            return Err(Error::Shape(format!(
                "overlap of mode {mode} under {dims:?}"
            )));
        }
        check_len("d", d, dims.obj_len())?;
        let total = g.nf_max as usize * dims.nmode as usize * dims.n();
        check_len("dens", dens, total)?;
        let (_, config) = elems(total)?;
        let reads = [views.addr().storage(), d.storage(), stack.storage()];
        self.base.launch(ctx, &reads, &[dens.storage()], |stream| {
            // SAFETY: lengths checked above; `Views::new` checked every view lies inside the
            // object with valid mode indices.
            unsafe {
                self.module.rpie_overlap_read(
                    stream,
                    config,
                    dptr(d),
                    mode,
                    dptr(views.addr()),
                    dptr(dens),
                    stack.ctl() as *const StackCtl,
                    g,
                )
            }
        })
    }

    /// `out[0] = max_{b, i} sum_{m < nsum} x[b * nsum + m, i]` over the groups of `spec`,
    /// `x` non-negative.
    pub fn sum_max(
        &self,
        ctx: &ExecutionContext,
        x: &DeviceBuffer<f32>,
        spec: PowerMax,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), Error> {
        let PowerMax { ngroup, nsum, n } = spec;
        let config = spec.grid()?;
        check_len("x", x, ngroup as usize * nsum as usize * n as usize)?;
        check_len("out", out, 1)?;
        zero(ctx, out)?;
        self.base
            .launch(ctx, &[x.storage()], &[out.storage()], |stream| {
                // SAFETY: lengths checked above; one PIXEL_THREADS block per (chunk, group), and
                // `out` holds zero, a non-negative float's bits.
                unsafe {
                    self.module
                        .rpie_sum_max(stream, config, dptr(x), nsum, n, dptr(out))
                }
            })
    }

    /// The object step's weighted `Δ`: `out[i] = x[i] / ((1 - alpha) dens[i] + alpha
    /// mx[0])` over the `nk * n` values of `spec`.
    pub fn overlap_weight(
        &self,
        ctx: &ExecutionContext,
        (x, dens, mx): (&DeviceBuffer<C32>, &DeviceBuffer<f32>, &DeviceBuffer<f32>),
        out: &mut DeviceBuffer<C32>,
        spec: Weight,
    ) -> Result<(), Error> {
        let total = spec.len()?;
        check_len("x", x, total)?;
        check_len("dens", dens, total)?;
        check_len("mx", mx, 1)?;
        check_len("out", out, total)?;
        let (len, config) = elems(total)?;
        let reads = [x.storage(), dens.storage(), mx.storage()];
        self.base.launch(ctx, &reads, &[out.storage()], |stream| {
            // SAFETY: lengths checked above.
            unsafe {
                self.module.rpie_overlap_weight(
                    stream,
                    config,
                    dptr(x),
                    dptr(dens),
                    dptr(mx),
                    dptr(out),
                    len,
                    spec.alpha,
                )
            }
        })
    }

    /// Every frame's patch of probe mode `mode` of `probe` `[nprobe, n]`: `out[fl, i] =
    /// |probe[mode, i]|^2` over the whole of `out`, planes of `n`.
    pub fn mode_power(
        &self,
        ctx: &ExecutionContext,
        probe: &DeviceBuffer<C32>,
        mode: u32,
        n: u32,
        out: &mut DeviceBuffer<C32>,
    ) -> Result<(), Error> {
        check_len("probe", probe, (mode as usize + 1) * n as usize)?;
        let (len, config) = elems(out.len())?;
        if n == 0 || len % n != 0 {
            return Err(Error::Shape(format!("{len} values in planes of {n}")));
        }
        self.base
            .launch(ctx, &[probe.storage()], &[out.storage()], |stream| {
                // SAFETY: lengths checked above.
                unsafe {
                    self.module.rpie_mode_power(
                        stream,
                        config,
                        dptr(probe),
                        mode,
                        n,
                        len,
                        dptr(out),
                    )
                }
            })
    }

    /// The rPIE step of the `n` values of `x` from the numerator `num`, consuming it:
    /// `x += s * num`, then `num` is zero.
    pub fn step(
        &self,
        ctx: &ExecutionContext,
        x: &mut DeviceBuffer<C32>,
        num: &mut DeviceBuffer<C32>,
        n: u32,
        s: f32,
    ) -> Result<(), Error> {
        check_len("x", x, n as usize)?;
        check_len("num", num, n as usize)?;
        if n == 0 {
            return Ok(());
        }
        let (_, config) = elems(n as usize)?;
        let writes = [x.storage(), num.storage()];
        self.base.launch(ctx, &[], &writes, |stream| {
            // SAFETY: lengths checked above.
            unsafe {
                self.module
                    .rpie_step(stream, config, dptr(x), dptr(num), n, s)
            }
        })
    }
}

/// Groups of [`RpieKernels::power_max`]: `ngroup` groups of `nsum` planes of `n` values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PowerMax {
    pub ngroup: u32,
    pub nsum: u32,
    pub n: u32,
}

impl PowerMax {
    /// The floats of the `chunks` scratch: one per group and `POWER_CHUNK` values of a plane.
    pub fn chunks(&self) -> usize {
        self.ngroup as usize * self.n.div_ceil(POWER_CHUNK) as usize
    }

    /// One `PIXEL_THREADS` block per chunk of `POWER_CHUNK` values and group.
    fn grid(&self) -> Result<LaunchConfig, Error> {
        let PowerMax { ngroup, nsum, n } = *self;
        if ngroup == 0 || nsum == 0 || n == 0 || ngroup > u16::MAX as u32 {
            return Err(Error::Shape(format!("empty or too many groups {self:?}")));
        }
        Ok(LaunchConfig {
            grid_dim: (n.div_ceil(POWER_CHUNK), ngroup, 1),
            block_dim: (PIXEL_THREADS, 1, 1),
            shared_mem_bytes: 0,
        })
    }
}

/// Planes and regulariser of [`RpieKernels::weight_inplace`] and
/// [`RpieKernels::overlap_weight`]: `nk` planes of `n` values and the `alpha` of their
/// denominators.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Weight {
    pub nk: u32,
    pub n: u32,
    pub alpha: f32,
}

impl Weight {
    /// The elements of the planes, at least one and launchable.
    fn len(&self) -> Result<usize, Error> {
        if self.nk == 0 || self.n == 0 {
            return Err(Error::Shape(format!("empty weight {self:?}")));
        }
        Ok(elems(self.nk as usize * self.n as usize)?.0 as usize)
    }
}
